#include "class_runtime.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace class_runtime {
namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
struct Node { std::int32_t feature; u32 payload, left, right; };
static_assert(sizeof(Node) == 16 && std::endian::native == std::endian::little);
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(cudaError_t code, const char* message) {
  if (code != cudaSuccess) throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(code));
}
void finish() { check(cudaGetLastError(), "runtime launch"); check(cudaDeviceSynchronize(), "runtime synchronize"); }
std::string digest(const std::string& bytes) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> out{}; unsigned n = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), out.data(), &n, EVP_sha256(), nullptr), "runtime SHA256");
  std::ostringstream s; s << std::hex << std::setfill('0');
  for (unsigned i = 0; i < n; ++i) s << std::setw(2) << unsigned(out[i]);
  return s.str();
}
u32 word(const std::string& s, std::size_t at) {
  require(at <= s.size() && s.size() - at >= 4, "runtime short header");
  u32 out; std::memcpy(&out, s.data() + at, 4); return out;
}
void append(std::string& s, u32 v) { s.append(reinterpret_cast<const char*>(&v), 4); }
u32 width(u32 n) { u32 bits = 0; do { ++bits; n >>= 1; } while (n); return bits; }
u32 grid(u64 n, u32 block = 256) { return u32(std::min<u64>((n - 1) / block + 1, 65535)); }
template<class T> struct Buffer {
  T* p = nullptr; u64 n = 0;
  Buffer() = default;
  explicit Buffer(u64 count) : n(count) {
    require(n <= SIZE_MAX / sizeof(T), "runtime allocation overflow");
    if (n) check(cudaMalloc(&p, std::size_t(n) * sizeof(T)), "runtime allocate");
  }
  ~Buffer() { if (p) cudaFree(p); }
  Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : p(std::exchange(other.p, nullptr)), n(std::exchange(other.n, 0)) {}
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) { if (p) cudaFree(p); p = std::exchange(other.p, nullptr); n = std::exchange(other.n, 0); }
    return *this;
  }
  void zero() { if (n) check(cudaMemset(p, 0, std::size_t(n) * sizeof(T)), "runtime zero"); }
  void upload(const char* bytes) { if (n) check(cudaMemcpy(p, bytes, std::size_t(n) * sizeof(T), cudaMemcpyHostToDevice), "runtime upload"); }
  T scalar() const { require(n == 1, "runtime scalar extent"); T v; check(cudaMemcpy(&v, p, sizeof(T), cudaMemcpyDeviceToHost), "runtime scalar"); return v; }
};
__host__ __device__ u64 mask(u32 b) { return (u64(1) << b) - 1; }
__device__ bool valid_node(Node v, u32 i, u32 F, u32 K) {
  if (v.feature == -1) return v.payload < K && v.left == 0 && v.right == 0;
  std::int64_t f = v.feature >= 0 ? v.feature : -std::int64_t(v.feature) - 2;
  return f >= 0 && u64(f) < F && !isnan(__uint_as_float(v.payload)) && v.left < i && v.right < i && v.left != v.right;
}
__global__ void validate_canonical(const Node* nodes, u32 count, u32 F, u32 K, u32* bad) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += u64(blockDim.x) * gridDim.x)
    if (!valid_node(nodes[i], u32(i), F, K)) atomicOr(bad, 1u);
}
__global__ void predicate_keys(const Node* nodes, u32 count, u64* out) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += u64(blockDim.x) * gridDim.x) {
    Node v = nodes[i]; out[i] = v.feature == -1 ? UINT64_MAX : (u64(u32(v.feature)) << 32) | v.payload;
  }
}
__global__ void pack(const Node* nodes, u32 count, const u64* dict, u32 nd, u32 cb, u64* words, u32* bad) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += u64(blockDim.x) * gridDim.x) {
    Node v = nodes[i]; u32 p = nd + v.payload;
    if (v.feature != -1) {
      const u64 key = (u64(u32(v.feature)) << 32) | v.payload;
      u32 l = 0, r = nd; while (l < r) { u32 m = l + (r-l)/2; if (dict[m] < key) l = m+1; else r = m; }
      p = l; if (p >= nd || dict[p] != key) { atomicOr(bad, 1u); continue; }
    }
    words[i] = (u64(p) << (2*cb)) | (u64(v.right) << cb) | v.left;
  }
}
__device__ bool unpack(u64 w, const u64* dict, u32 nd, u32 K, u32 cb, u32 pb, Node& v) {
  u32 total = 2*cb + pb; if (total < 64 && w >> total) return false;
  u32 p = u32(w >> (2*cb)), l = u32(w & mask(cb)), r = u32((w >> cb) & mask(cb));
  if (p >= nd) { if (p-nd >= K || l || r) return false; v = {-1,p-nd,0,0}; }
  else { u64 key = dict[p]; v = {std::int32_t(u32(key >> 32)),u32(key),l,r}; }
  return true;
}
__global__ void validate_compact(const Node* nodes, const u64* words, const u64* dict, u32 count, u32 nd, u32 F, u32 K, u32 cb, u32 pb, u32* bad) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += u64(blockDim.x) * gridDim.x) {
    Node v{}, original = nodes[i];
    if (!unpack(words[i],dict,nd,K,cb,pb,v) || !valid_node(v,u32(i),F,K) || v.feature != original.feature || v.payload != original.payload || v.left != original.left || v.right != original.right) atomicOr(bad,1u);
  }
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < nd; i += u64(blockDim.x) * gridDim.x) {
    const u64 key = dict[i]; const std::int32_t sf = std::int32_t(u32(key >> 32));
    const std::int64_t f = sf >= 0 ? sf : -std::int64_t(sf)-2;
    if (sf == -1 || f < 0 || u64(f) >= F || isnan(__uint_as_float(u32(key))) || (i && dict[i-1] >= key)) atomicOr(bad,1u);
  }
}
struct Validated { const Node* nodes; const u64* words; const u64* dict; u32 count, root, F, K, nd, cb, pb; };
template<bool Packed, bool Checked, bool Traced = false>
__global__ void traverse(Validated c, const float* input, u64 rows, u64 stride, u32* output, u32* bad,
                         u32* paths = nullptr, u64 capacity = 0, u32* lengths = nullptr) {
  for (u64 row = u64(blockIdx.x)*blockDim.x + threadIdx.x; row < rows; row += u64(blockDim.x)*gridDim.x) {
    const float* x = input + row*stride; u32 at = c.root, steps = 0, path_steps = 0;
    if constexpr (Traced) lengths[row] = 0;
    for (;;) {
      if constexpr (Checked) { if (at >= c.count || steps++ >= c.count) { atomicOr(bad,1u); output[row] = UINT32_MAX; break; } }
      Node v;
      if constexpr (Packed) {
        const u64 w = c.words[at]; const u32 p = u32(w >> (2*c.cb));
        if constexpr (Checked) { if (!unpack(w,c.dict,c.nd,c.K,c.cb,c.pb,v) || !valid_node(v,at,c.F,c.K)) { atomicOr(bad,1u); output[row] = UINT32_MAX; break; } }
        else { if (p >= c.nd) v = {-1,p-c.nd,0,0}; else { u64 key = c.dict[p]; v = {std::int32_t(u32(key >> 32)),u32(key),u32(w & mask(c.cb)),u32((w >> c.cb) & mask(c.cb))}; } }
      } else { v = c.nodes[at]; if constexpr (Checked) { if (!valid_node(v,at,c.F,c.K)) { atomicOr(bad,1u); output[row] = UINT32_MAX; break; } } }
      if constexpr (Traced) {
        if (path_steps >= capacity) { atomicOr(bad,1u); output[row] = UINT32_MAX; break; }
        paths[row*capacity+path_steps++] = at;
      }
      if (v.feature == -1) { output[row] = v.payload; if constexpr (Traced) lengths[row] = path_steps; break; }
      const u32 f = v.feature >= 0 ? u32(v.feature) : u32(-std::int64_t(v.feature)-2);
      const float value = x[f]; at = (isnan(value) ? v.feature < 0 : value < __uint_as_float(v.payload)) ? v.left : v.right;
    }
  }
}
__global__ void compare(const u32* actual, const u32* expected, u64 rows, u32* bad) {
  for (u64 row = u64(blockIdx.x)*blockDim.x + threadIdx.x; row < rows; row += u64(blockDim.x)*gridDim.x)
    if (actual[row] != expected[row]) atomicOr(bad,1u);
}
void allocation(const void* p, u64 bytes, int device) {
  require(p && reinterpret_cast<uintptr_t>(p)%4 == 0 && bytes && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(p), "runtime borrowed extent overflow");
  cudaPointerAttributes a{}; check(cudaPointerGetAttributes(&a,p), "runtime borrowed pointer");
  require(a.type == cudaMemoryTypeDevice && a.device == device, "runtime borrowed allocation device/type");
  CUdeviceptr base = 0; std::size_t size = 0; const auto address = reinterpret_cast<CUdeviceptr>(p);
  require(cuMemGetAddressRange(&base,&size,address) == CUDA_SUCCESS && address >= base && bytes <= size && address-base <= size-bytes, "runtime borrowed span exceeds CUDA allocation");
}
void disjoint(const void* a, u64 an, const void* b, u64 bn) {
  auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  require(x + an <= y || y + bn <= x, "runtime input/output alias");
}
} // namespace

struct Runtime::Impl {
  Buffer<Node> nodes; Buffer<u64> words, dictionary;
  Metadata meta{}; std::string header; int device = -1;
  Validated capability() const { return {nodes.p,words.p,dictionary.p,meta.nodes,meta.root,meta.features,meta.classes,meta.predicates,meta.child_bits,meta.predicate_bits}; }
  void input(DenseBatch x, u32 block) const {
    int current = -1; check(cudaGetDevice(&current), "runtime active device"); require(current == device, "runtime graph device differs");
    require(block == 64 || block == 128 || block == 256, "runtime block size");
    require(x.rows && x.row_stride >= meta.features && x.elements <= UINT64_MAX/4 && x.first_row <= UINT64_MAX-(x.rows-1), "runtime row metadata");
    u64 last = x.first_row + x.rows-1;
    require(last <= (UINT64_MAX-meta.features)/x.row_stride && last*x.row_stride+meta.features <= x.elements, "runtime input row extent");
    allocation(x.device_values,x.elements*4,device);
  }
  void output(DenseBatch x, ClassOutput y) const {
    require(y.elements >= x.rows && y.elements <= UINT64_MAX/4, "runtime output extent"); allocation(y.device_classes,y.elements*4,device);
    disjoint(x.device_values,x.elements*4,y.device_classes,y.elements*4);
  }
  void launch(DenseBatch x, u32* y, Layout layout, Traversal mode, u32 block, u32* bad) const {
    const float* begin = x.device_values + x.first_row*x.row_stride; const u32 blocks = grid(x.rows,block); const auto c = capability();
    if (layout == Layout::compact8) { if (mode == Traversal::validated) traverse<true,false><<<blocks,block>>>(c,begin,x.rows,x.row_stride,y,bad); else traverse<true,true><<<blocks,block>>>(c,begin,x.rows,x.row_stride,y,bad); }
    else { if (mode == Traversal::validated) traverse<false,false><<<blocks,block>>>(c,begin,x.rows,x.row_stride,y,bad); else traverse<false,true><<<blocks,block>>>(c,begin,x.rows,x.row_stride,y,bad); }
  }
};
Runtime::Runtime(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
Runtime::~Runtime() { cudaStreamSynchronize(nullptr); }
const Metadata& Runtime::metadata() const noexcept { return impl_->meta; }
std::unique_ptr<Runtime> Runtime::load(const std::string& bytes, const std::string& pin, const std::string& source, const std::string& compact, const std::string& compact_pin, Residency residency) {
  require(residency==Residency::dual || residency==Residency::canonical_only || residency==Residency::compact_only, "runtime residency selector");
  require(pin.size() == 64 && digest(bytes) == pin, "runtime canonical pin differs");
  require(bytes.size() >= 80 && bytes.compare(0,8,"CLSGDAG1") == 0 && word(bytes,8) == 1 && word(bytes,28) == 1, "runtime canonical header");
  auto p = std::make_unique<Impl>(); auto& m = p->meta;
  m.features=word(bytes,12); m.classes=word(bytes,16); m.root=word(bytes,20); m.nodes=word(bytes,24);
  require(m.features && m.features <= INT32_MAX && m.classes >= 2 && m.nodes && m.root < m.nodes && bytes.size() == 64+u64(m.nodes)*16, "runtime canonical dimensions");
  std::ostringstream identity; identity << std::hex << std::setfill('0'); for (unsigned i=32;i<64;++i) identity << std::setw(2) << unsigned(static_cast<unsigned char>(bytes[i]));
  require(source.size() == 64 && source == identity.str(), "runtime source identity differs");
  m.source_sha256=source; m.canonical_sha256=pin; m.canonical_bytes=bytes.size(); p->header=bytes.substr(0,64);
  check(cudaGetDevice(&p->device), "runtime graph device");
  p->nodes=Buffer<Node>(m.nodes); p->nodes.upload(bytes.data()+64); Buffer<u32> bad(1); bad.zero();
  validate_canonical<<<grid(m.nodes),256>>>(p->nodes.p,m.nodes,m.features,m.classes,bad.p); finish(); require(!bad.scalar(), "runtime invalid canonical nodes");
  if (residency==Residency::canonical_only && compact.empty()) {
    require(compact_pin.empty(), "runtime compact bytes missing");
  } else if (compact.empty()) {
    require(compact_pin.empty(), "runtime compact bytes missing"); Buffer<u64> keys(m.nodes);
    predicate_keys<<<grid(m.nodes),256>>>(p->nodes.p,m.nodes,keys.p); finish();
    auto begin=thrust::device_pointer_cast(keys.p); thrust::sort(thrust::device,begin,begin+m.nodes);
    auto end=thrust::unique(thrust::device,begin,begin+m.nodes); finish(); u32 nd=u32(end-begin);
    if (nd) { u64 last; check(cudaMemcpy(&last,keys.p+nd-1,8,cudaMemcpyDeviceToHost), "runtime dictionary sentinel"); if (last==UINT64_MAX) --nd; }
    m.predicates=nd; m.child_bits=width(m.nodes-1); const bool class_field_fits=u64(nd)+m.classes-1<=UINT32_MAX; m.predicate_bits=class_field_fits ? width(nd+m.classes-1) : 0;
    if (!class_field_fits || 2*m.child_bits+m.predicate_bits>64) {
      require(residency!=Residency::compact_only, "runtime exceeds compact8 capacity");
    } else {
    p->dictionary=Buffer<u64>(nd); if(nd) check(cudaMemcpy(p->dictionary.p,keys.p,u64(nd)*8,cudaMemcpyDeviceToDevice), "runtime dictionary copy");
    p->words=Buffer<u64>(m.nodes); pack<<<grid(m.nodes),256>>>(p->nodes.p,m.nodes,p->dictionary.p,nd,m.child_bits,p->words.p,bad.p); finish(); require(!bad.scalar(), "runtime packing failed");
    }
  } else {
    require(compact_pin.size()==64 && digest(compact)==compact_pin, "runtime compact pin differs");
    require(compact.size()>=104 && compact.compare(0,8,"CLSG64B1")==0 && word(compact,8)==1 && word(compact,24)==0 && word(compact,28)==0 && compact.substr(32,64)==p->header, "runtime compact header differs");
    m.child_bits=word(compact,12); m.predicate_bits=word(compact,16); m.predicates=word(compact,20);
    require(m.predicates<=m.nodes && u64(m.predicates)+m.classes-1<=UINT32_MAX && m.child_bits==width(m.nodes-1) && m.predicate_bits==width(m.predicates+m.classes-1) && 2*m.child_bits+m.predicate_bits<=64 && compact.size()==96+u64(m.predicates)*8+u64(m.nodes)*8, "runtime compact dimensions");
    p->dictionary=Buffer<u64>(m.predicates); p->dictionary.upload(compact.data()+96); p->words=Buffer<u64>(m.nodes); p->words.upload(compact.data()+96+u64(m.predicates)*8);
  }
  if (p->words.n) {
  validate_compact<<<grid(m.nodes),256>>>(p->nodes.p,p->words.p,p->dictionary.p,m.nodes,m.predicates,m.features,m.classes,m.child_bits,m.predicate_bits,bad.p); finish(); require(!bad.scalar(), "runtime compact validation or cross-layout mismatch");
  m.cross_layout_validated=true; m.compact_bytes=96+u64(m.predicates)*8+u64(m.nodes)*8;
  }
  if (residency==Residency::canonical_only) { p->words=Buffer<u64>(); p->dictionary=Buffer<u64>(); }
  if (residency==Residency::compact_only) { require(p->words.n!=0, "runtime compact layout unavailable"); p->nodes=Buffer<Node>(); }
  m.canonical_resident=p->nodes.n!=0; m.compact_resident=p->words.n!=0;
  m.device_bytes=p->nodes.n*16+p->words.n*8+p->dictionary.n*8;
  auto result=std::unique_ptr<Runtime>(new Runtime(std::move(p))); if(result->impl_->meta.compact_resident) result->impl_->meta.compact_sha256=digest(result->compact_bytes()); return result;
}
void Runtime::retain(Residency keep) {
  require(keep==Residency::dual || keep==Residency::canonical_only || keep==Residency::compact_only, "runtime residency selector");
  auto& p=*impl_; int current=-1; check(cudaGetDevice(&current), "runtime active device");
  require(current==p.device, "runtime graph device differs");
  require(keep==Residency::compact_only || p.meta.canonical_resident, "runtime canonical layout is not resident");
  require(keep==Residency::canonical_only || p.meta.compact_resident, "runtime compact layout is not resident");
  check(cudaDeviceSynchronize(), "runtime release synchronize");
  if(keep==Residency::canonical_only){p.words=Buffer<u64>();p.dictionary=Buffer<u64>();}
  if(keep==Residency::compact_only)p.nodes=Buffer<Node>();
  p.meta.canonical_resident=p.nodes.n!=0;p.meta.compact_resident=p.words.n!=0;
  p.meta.device_bytes=p.nodes.n*16+p.words.n*8+p.dictionary.n*8;
}
void Runtime::predict(DenseBatch x, ClassOutput y, Layout layout, Traversal mode, u32 block) const {
  require((layout==Layout::canonical16 || layout==Layout::compact8) && (mode==Traversal::checked || mode==Traversal::validated), "runtime traversal selector");
  require(layout==Layout::canonical16 ? impl_->meta.canonical_resident : impl_->meta.compact_resident, "runtime requested layout is not resident");
  impl_->input(x,block); impl_->output(x,y); Buffer<u32> bad(mode==Traversal::checked ? 1 : 0); bad.zero(); impl_->launch(x,y.device_classes,layout,mode,block,bad.p); finish(); if (bad.n) require(!bad.scalar(), "runtime checked route invalid");
}
void Runtime::trace(DenseBatch x, ClassOutput y, PathOutput path, Layout layout, u32 block) const {
  require(layout==Layout::canonical16 || layout==Layout::compact8, "runtime trace layout selector");
  require(layout==Layout::canonical16 ? impl_->meta.canonical_resident : impl_->meta.compact_resident, "runtime trace layout is not resident");
  impl_->input(x,block); impl_->output(x,y);
  require(path.capacity_per_row && path.capacity_per_row<=UINT32_MAX && x.rows<=UINT64_MAX/path.capacity_per_row &&
          path.node_elements>=x.rows*path.capacity_per_row && path.node_elements<=UINT64_MAX/4 &&
          path.length_elements>=x.rows && path.length_elements<=UINT64_MAX/4, "runtime trace output extent");
  allocation(path.device_nodes,path.node_elements*4,impl_->device); allocation(path.device_lengths,path.length_elements*4,impl_->device);
  disjoint(x.device_values,x.elements*4,path.device_nodes,path.node_elements*4);
  disjoint(x.device_values,x.elements*4,path.device_lengths,path.length_elements*4);
  disjoint(y.device_classes,y.elements*4,path.device_nodes,path.node_elements*4);
  disjoint(y.device_classes,y.elements*4,path.device_lengths,path.length_elements*4);
  disjoint(path.device_nodes,path.node_elements*4,path.device_lengths,path.length_elements*4);
  Buffer<u32> bad(1); bad.zero(); const float* begin=x.device_values+x.first_row*x.row_stride;
  const auto c=impl_->capability(); const u32 blocks=grid(x.rows,block);
  if(layout==Layout::compact8) traverse<true,true,true><<<blocks,block>>>(c,begin,x.rows,x.row_stride,y.device_classes,bad.p,path.device_nodes,path.capacity_per_row,path.device_lengths);
  else traverse<false,true,true><<<blocks,block>>>(c,begin,x.rows,x.row_stride,y.device_classes,bad.p,path.device_nodes,path.capacity_per_row,path.device_lengths);
  finish(); require(!bad.scalar(), "runtime trace invalid route or insufficient path capacity");
}
void Runtime::verify(DenseBatch x, const u32* expected, u64 expected_elements, u32 block) const {
  impl_->input(x,block); require(expected_elements>=x.rows && expected_elements<=UINT64_MAX/4, "runtime expected extent"); allocation(expected,expected_elements*4,impl_->device);
  Buffer<u32> output(x.rows),bad(1);
  for (auto layout : {Layout::canonical16,Layout::compact8}) for (auto mode : {Traversal::checked,Traversal::validated}) {
    if (layout==Layout::canonical16 ? !impl_->meta.canonical_resident : !impl_->meta.compact_resident) continue;
    bad.zero(); impl_->launch(x,output.p,layout,mode,block,bad.p); compare<<<grid(x.rows),256>>>(output.p,expected,x.rows,bad.p); finish(); require(!bad.scalar(), "runtime four-variant prediction mismatch");
  }
}
std::string Runtime::compact_bytes() const {
  const auto& p=*impl_; require(p.meta.compact_resident, "runtime compact layout is not resident"); std::string out="CLSG64B1";
  for (u32 v : {1u,p.meta.child_bits,p.meta.predicate_bits,p.meta.predicates,0u,0u}) append(out,v); out+=p.header;
  auto download=[&](const Buffer<u64>& b) { std::size_t at=out.size(); out.resize(at+std::size_t(b.n)*8); if(b.n) check(cudaMemcpy(out.data()+at,b.p,std::size_t(b.n)*8,cudaMemcpyDeviceToHost), "runtime packed download"); };
  download(p.dictionary); download(p.words); return out;
}
} // namespace class_runtime
