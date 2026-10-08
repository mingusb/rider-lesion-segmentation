#pragma once
#include "adaptive_domain.cuh"
#include "adaptive_growth_plan.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The maintained converter's adaptive construction. Host work owns opaque
// transport, capacities and callback scheduling; all model decisions are CUDA.
namespace class_conversion_adaptive {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
namespace domain = class_conversion_adaptive_domain;
namespace growth = class_conversion_growth;
constexpr u32 none = UINT32_MAX;

inline void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline void check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
inline void synchronize() {
  check(cudaGetLastError(), "adaptive launch");
  check(cudaDeviceSynchronize(), "adaptive synchronize");
}
inline u64 multiply(u64 a, u64 b) {
  require(!b || a <= UINT64_MAX / b, "adaptive byte extent overflow");
  return a * b;
}
inline u32 buckets(u32 capacity) {
  require(capacity && capacity <= 0x3fffffffu, "adaptive index capacity");
  u32 result = 1;
  while (result < u64(capacity) * 2) result *= 2;
  return result;
}
// Only these resource refusals may trigger a smaller growth retry. CUDA launch,
// copy, rehash and invariant failures keep their original failure semantics.
struct AllocationRefusal : std::runtime_error {
  cudaError_t error;
  u64 requested_bytes;
  AllocationRefusal(cudaError_t code,u64 bytes,const char* operation)
      :std::runtime_error(std::string(operation)+(code==cudaSuccess?"":std::string(": ")+cudaGetErrorString(code))),
       error(code),requested_bytes(bytes) {}
};
using GrowthObserver=std::function<void(const growth::Outcome&)>;
using GrowthAllocationHook=std::function<void(const char*)>;
struct Budget {
  u64 limit, used = 0, peak = 0;
  void reserve(u64 bytes) {
    if(used>limit||bytes>limit-used)
      throw AllocationRefusal(cudaSuccess,bytes,"adaptive owned GPU byte budget exhausted");
    used += bytes;
    peak = std::max(peak, used);
  }
};
template<class T> struct Buffer {
  T* data = nullptr;
  u64 size = 0;
  Budget* budget = nullptr;
  Buffer() = default;
  Buffer(Budget& owner, u64 count) : size(count), budget(&owner) {
    auto bytes = multiply(count, sizeof(T));
    owner.reserve(bytes);
    if (bytes) {
      auto error = cudaMalloc(&data, bytes);
      if (error != cudaSuccess) {
        owner.used -= bytes;
        budget = nullptr;
        if(error==cudaErrorMemoryAllocation) {
          // A recoverable allocation refusal must not poison the next launch
          // check. Preserve any different runtime error rather than masking it.
          auto last=cudaGetLastError();
          if(last!=cudaSuccess&&last!=cudaErrorMemoryAllocation)
            check(last,"adaptive allocation prior runtime error");
          throw AllocationRefusal(error,bytes,"adaptive allocation");
        }
        check(error, "adaptive allocation");
      }
    }
  }
  ~Buffer() { reset(); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept { swap(other); }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) { reset(); swap(other); }
    return *this;
  }
  void swap(Buffer& other) noexcept {
    std::swap(data, other.data); std::swap(size, other.size);
    std::swap(budget, other.budget);
  }
  void reset() noexcept {
    if (data) cudaFree(data);
    if (budget) budget->used -= size * sizeof(T);
    data = nullptr; size = 0; budget = nullptr;
  }
  void zero() { if (size) check(cudaMemset(data, 0, size * sizeof(T)), "adaptive zero"); }
  void empty_table() {
    if (size) check(cudaMemset(data, 0xff, size * sizeof(T)), "adaptive table initialization");
  }
  void upload(const std::vector<T>& bytes) {
    require(bytes.size() == size, "adaptive opaque transport extent");
    if (size) check(cudaMemcpy(data, bytes.data(), size * sizeof(T), cudaMemcpyHostToDevice),
                    "adaptive upload");
  }
  std::vector<T> download(u64 count) const {
    require(count <= size, "adaptive readback extent");
    std::vector<T> result(count);
    if (count) check(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost),
                     "adaptive readback");
    return result;
  }
};

struct SourceView {
  const std::int32_t *feature, *left, *right, *roots, *channels;
  const float *cut, *value, *bias;
  const std::uint8_t* missing;
  u32 features, classes, nodes, trees;
  // Zero audits every channel. Mixed sources append exact structural guards
  // after this native margin prefix; they are never passed to native models.
  u32 native_margin_classes = 0;
};
struct State {
  u32 phase = 0, predicate = none, left = none, right = none, node = none;
  u32 reserved = 0;
  u64 hash = 0;
};
constexpr u32 free_phase = 4;
struct Node { std::int32_t feature; u32 payload, left, right; };
static_assert(sizeof(Node) == 16);
struct Status {
  // states is the slot high-water mark, not cumulative work after recycling.
  u64 states = 0, nodes = 0, expansions = 0, normalization_steps = 0;
  u64 prefix_additions = 0, state_hits = 0, node_hits = 0, terminal_hits = 0;
  u64 native_terminals = 0, key_collisions = 0;
  u64 class_pruned_states = 0, terminal_gap_pruned_states = 0;
  u64 state_creations = 0, state_reuses = 0, state_evictions = 0;
  u32 depth = 0, root = none, terminal = none, request = 0, error = 0, complete = 0;
  u32 free_head = none, free_count = 0;
};
// Requests are recoverable synchronization boundaries, not partial commits:
// 1 native terminal; 2 reserve one state; 3 reserve one graph node.
struct ArenaView {
  State* states;
  u32 *words, *positions, *lower, *upper, *missing, *stack;
  u32 *witness_lower, *witness_upper, *witness_missing;
  std::int32_t* residual;
  u64* allowed;
  u64* witness_allowed;
  u32 *table, *terminal_table;
  Node* nodes;
  u32 state_capacity, node_capacity, state_buckets, node_buckets;
  u32* node_table;
};
struct EngineView {
  SourceView source;
  domain::DomainView domain;
  ArenaView arena;
  State* draft;
  u32 *draft_words, *draft_positions;
  u32* blocked;
  std::int32_t* draft_residual;
  domain::RegionView draft_region;
  domain::RegionView draft_witness;
  const u64* support;
  u64* active_support;
  u32 support_words;
  const u32 *minimum, *maximum;
  float *range_lower, *range_upper;
  bool qualified_gap = false;
  Status* status;
  u32 refinement_stack_capacity = 0;
  u32 refinement_maximum_visits = 0;
  // Score storage remains source.classes wide. A composed source can have
  // several independent teacher score vectors but fewer public class labels.
  // Zero preserves the established single-source behavior for all callers.
  u32 output_classes = 0;
};
__host__ __device__ inline u32 output_class_count(const EngineView& e) {
  return e.output_classes ? e.output_classes : e.source.classes;
}
__device__ domain::RegionView region(const EngineView& e, u32 id) {
  u64 offset = u64(id) * e.domain.numeric_features;
  return {e.domain.numeric_features ? e.arena.lower + offset : nullptr,
          e.domain.numeric_features ? e.arena.upper + offset : nullptr,
          e.domain.numeric_features ? e.arena.missing + offset : nullptr,
          e.domain.mask_words ? e.arena.allowed + u64(id) * e.domain.mask_words : nullptr};
}
__device__ domain::RegionView witness_region(const EngineView& e, u32 id) {
  u64 offset = u64(id) * e.domain.numeric_features;
  return {e.domain.numeric_features ? e.arena.witness_lower + offset : nullptr,
          e.domain.numeric_features ? e.arena.witness_upper + offset : nullptr,
          e.domain.numeric_features ? e.arena.witness_missing + offset : nullptr,
          e.domain.mask_words ? e.arena.witness_allowed + u64(id) * e.domain.mask_words : nullptr};
}
// Erase constraints only where no residual source subtree can inspect them.
// An exactly-one group is coupled: retain its entire mask if any member is used.
// The full path witness remains separate for native prediction callbacks.
__device__ void project_context(EngineView e, domain::RegionView r,
                               const std::int32_t* roots) {
  for (u32 w = 0; w < e.support_words; ++w) e.active_support[w] = 0;
  for (u32 t = 0; t < e.source.trees; ++t) if (roots[t] >= 0)
    for (u32 w = 0; w < e.support_words; ++w)
      e.active_support[w] |= e.support[u64(roots[t]) * e.support_words + w];
  for (u32 f = 0; f < e.source.features; ++f)
    if (e.domain.feature_group[f] < 0 &&
        !(e.active_support[f / 64] & (u64(1) << (f % 64)))) {
      u32 numeric = u32(e.domain.feature_numeric[f]);
      r.lower[numeric] = domain::finite_min_key;
      r.upper[numeric] = domain::finite_max_key;
      r.missing[numeric] = e.domain.allow_nan;
    }
  for (u32 g = 0; g < e.domain.groups; ++g) {
    bool used = false;
    for (u32 j = e.domain.group_feature_offsets[g];
         j < e.domain.group_feature_offsets[g + 1]; ++j) {
      u32 f = e.domain.group_features[j];
      used |= (e.active_support[f / 64] & (u64(1) << (f % 64))) != 0;
    }
    if (!used)
      for (u32 w = e.domain.group_word_offsets[g]; w < e.domain.group_word_offsets[g + 1]; ++w)
        r.allowed[w] = e.domain.initial_masks[w];
  }
}
__device__ u64 hash_word(u64 h, u32 value) { return (h ^ value) * 1099511628211ull; }
__device__ u32 hash_bucket(u64 h, u32 count) {
  // FP32 integers and aligned source indices often have zero low bits. Mix the
  // whole hash before masking into a power-of-two table; equality stays exact.
  h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
  h ^= h >> 33;
  return u32(h) & (count - 1);
}
__device__ u64 region_hash(const EngineView& e, domain::RegionView r, u64 h) {
  for (u32 f = 0; f < e.domain.numeric_features; ++f) {
    h = hash_word(h, r.lower[f]); h = hash_word(h, r.upper[f]);
    h = hash_word(h, r.missing[f]);
  }
  for (u32 w = 0; w < e.domain.mask_words; ++w) {
    h = hash_word(h, u32(r.allowed[w])); h = hash_word(h, u32(r.allowed[w] >> 32));
  }
  return h;
}
__device__ u64 score_hash(const EngineView& e, const u32* words) {
  u64 h = 1469598103934665603ull;
  for (u32 c = 0; c < e.source.classes; ++c) h = hash_word(h, words[c]);
  return h;
}
__device__ u64 state_hash(const EngineView& e, domain::RegionView r,
                          const u32* words, const u32* positions,
                          const std::int32_t* roots) {
  u64 h = region_hash(e, r, 1469598103934665603ull);
  for (u32 c = 0; c < e.source.classes; ++c) {
    h = hash_word(h, words[c]); h = hash_word(h, positions[c]);
  }
  for (u32 t = 0; t < e.source.trees; ++t) h = hash_word(h, u32(roots[t]));
  return h;
}
__device__ bool scores_equal(const EngineView& e, u32 id, const u32* words) {
  for (u32 c = 0; c < e.source.classes; ++c)
    if (e.arena.words[u64(id) * e.source.classes + c] != words[c]) return false;
  return true;
}
__device__ bool state_equal(const EngineView& e, u32 id,
                            domain::RegionView r, const u32* words,
                            const u32* positions, const std::int32_t* roots) {
  auto original = region(e, id);
  for (u32 f = 0; f < e.domain.numeric_features; ++f)
    if (original.lower[f] != r.lower[f] || original.upper[f] != r.upper[f] ||
        original.missing[f] != r.missing[f]) return false;
  for (u32 w = 0; w < e.domain.mask_words; ++w)
    if (original.allowed[w] != r.allowed[w]) return false;
  for (u32 c = 0; c < e.source.classes; ++c)
    if (e.arena.words[u64(id) * e.source.classes + c] != words[c] ||
        e.arena.positions[u64(id) * e.source.classes + c] != positions[c]) return false;
  for (u32 t = 0; t < e.source.trees; ++t)
    if (e.arena.residual[u64(id) * e.source.trees + t] != roots[t]) return false;
  return true;
}
__device__ void copy_region(const EngineView& e, domain::RegionView to,
                            domain::RegionView from) {
  for (u32 f = 0; f < e.domain.numeric_features; ++f) {
    to.lower[f] = from.lower[f]; to.upper[f] = from.upper[f];
    to.missing[f] = from.missing[f];
  }
  for (u32 w = 0; w < e.domain.mask_words; ++w) to.allowed[w] = from.allowed[w];
}
// Same adaptive prefix rule: visit trees in original global order; a class
// cannot consume a later leaf until its earlier unresolved tree is consumed.
__device__ u32 normalize(EngineView e, domain::RegionView r, u32* words,
                         u32* positions, std::int32_t* roots) {
  if (!domain::region_valid(e.domain, r)) { e.status->error = 11; return none; }
  u32 first = none;
  for (u32 c = 0; c < e.source.classes; ++c) e.blocked[c] = 0;
  for (u32 t = 0; t < e.source.trees; ++t) {
    int at = roots[t];
    if (at < 0) continue;
    u32 steps = 0;
    while (e.source.left[at] >= 0) {
      if (++steps > e.source.nodes) { e.status->error = 12; return none; }
      bool right = false;
      if (!domain::forced_side(e.domain, r, u32(e.source.feature[at]),
                              __float_as_uint(e.source.cut[at]),
                              bool(e.source.missing[at]), right)) break;
      at = right ? e.source.right[at] : e.source.left[at];
      ++e.status->normalization_steps;
    }
    roots[t] = at;
    u32 channel = u32(e.source.channels[t]);
    if (e.source.left[at] < 0 && !e.blocked[channel]) {
      float sum = __fadd_rn(__uint_as_float(words[channel]), e.source.value[at]);
      if (!isfinite(sum)) { e.status->error = 13; return none; }
      words[channel] = __float_as_uint(sum); ++positions[channel];
      roots[t] = -1; ++e.status->prefix_additions;
    } else {
      e.blocked[channel] = 1;
      if (first == none && e.source.left[at] >= 0) first = u32(at);
    }
  }
  return first;
}
struct StateAdmission {
  u32 id = none, slot = none;
  bool fresh = false;
};
// Sole-writer lookup/reservation. The caller prepares the same projected key
// before entering this helper; hashes only screen exact equality. Reserving a
// free/new slot does not publish its State or table entry before payload copy.
__device__ StateAdmission reserve_state(EngineView e, State draft,
    domain::RegionView r, const u32* words, const u32* positions,
    const std::int32_t* roots) {
  u32 slot = hash_bucket(draft.hash, e.arena.state_buckets);
  for (u32 probe = 0; probe < e.arena.state_buckets; ++probe,
       slot = (slot + 1) & (e.arena.state_buckets - 1)) {
    u32 id = e.arena.table[slot];
    if (id == none) {
      if (e.status->free_head != none) {
        id = e.status->free_head;
        if (!e.status->free_count || id >= e.status->states ||
            e.arena.states[id].phase != free_phase) { e.status->error = 75; return {}; }
        e.status->free_head = e.arena.states[id].reserved;
        --e.status->free_count; ++e.status->state_reuses;
      } else {
        if (e.status->free_count) { e.status->error = 75; return {}; }
        if (e.status->states == e.arena.state_capacity) { e.status->request = 2; return {}; }
        id = u32(e.status->states++);
      }
      ++e.status->state_creations;
      return {id, slot, true};
    }
    if (id >= e.status->states || e.arena.states[id].phase == free_phase) {
      e.status->error = 76; return {};
    }
    if (e.arena.states[id].hash == draft.hash && state_equal(e, id, r, words, positions, roots)) {
      ++e.status->state_hits; return {id, slot, false};
    }
    ++e.status->key_collisions;
  }
  e.status->error = 21; return {};
}
// Opaque bit transport only. Scalar and warp admission use this exact payload
// definition; projection, normalization and ordered FP32 folding remain outside.
// Empty numeric/category/tree extents do not dereference nullable slab pointers.
__device__ void copy_state_payload(EngineView e, u32 id, domain::RegionView r,
    const u32* words, const u32* positions, const std::int32_t* roots,
    u32 worker, u32 width) {
  auto key = region(e, id), witness = witness_region(e, id);
  for (u64 f = worker; f < e.domain.numeric_features; f += width) {
    key.lower[f] = r.lower[f]; key.upper[f] = r.upper[f]; key.missing[f] = r.missing[f];
    witness.lower[f] = e.draft_witness.lower[f];
    witness.upper[f] = e.draft_witness.upper[f];
    witness.missing[f] = e.draft_witness.missing[f];
  }
  for (u64 w = worker; w < e.domain.mask_words; w += width) {
    key.allowed[w] = r.allowed[w]; witness.allowed[w] = e.draft_witness.allowed[w];
  }
  for (u64 c = worker; c < e.source.classes; c += width) {
    e.arena.words[u64(id) * e.source.classes + c] = words[c];
    e.arena.positions[u64(id) * e.source.classes + c] = positions[c];
  }
  for (u64 t = worker; t < e.source.trees; t += width)
    e.arena.residual[u64(id) * e.source.trees + t] = roots[t];
}
__device__ void publish_state(EngineView e, State draft, StateAdmission admission) {
  e.arena.states[admission.id] = draft;
  e.arena.table[admission.slot] = admission.id;
}
__device__ u32 intern_state(EngineView e, State draft, domain::RegionView r,
                            const u32* words, const u32* positions,
                            const std::int32_t* roots, bool key_prepared = false) {
  // Initialization and scalar checks retain the same sole-writer path.
  if (!key_prepared) {
    project_context(e, r, roots);
    draft.hash = state_hash(e, r, words, positions, roots);
  }
  auto admission = reserve_state(e, draft, r, words, positions, roots);
  if (admission.id == none || !admission.fresh) return admission.id;
  copy_state_payload(e, admission.id, r, words, positions, roots, 0, 1);
  publish_state(e, draft, admission);
  return admission.id;
}
// Collective caller contract: exactly one block in the grid, block y/z = 1,
// and a power-of-two block x. Every thread enters uniformly with the same
// immutable input pointers; no other state-cache writer runs concurrently.
// The launch-shape refusal is uniform; it cannot legalize divergent calls in
// a valid block. Hardware launch validation supplies the device thread limit.
// Thread 0 preserves lookup/probe/equality/free-list order. Other threads only
// copy the accepted opaque payload. State/table publication follows a full
// block barrier, and the final barrier precedes any sibling lookup.
__device__ u32 intern_state_block(EngineView e, State draft, domain::RegionView r,
    const u32* words, const u32* positions, const std::int32_t* roots,
    bool key_prepared = false) {
  if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
      !blockDim.x || (blockDim.x & (blockDim.x - 1)) ||
      blockDim.y != 1 || blockDim.z != 1) {
    if (!blockIdx.x && !blockIdx.y && !blockIdx.z &&
        !threadIdx.x && !threadIdx.y && !threadIdx.z && !e.status->error)
      e.status->error = 83;
    return none;
  }
  __shared__ u32 admission_words[3];
  const u32 worker = threadIdx.x;
  StateAdmission admission{};
  if (!worker) {
    if (!key_prepared) {
      project_context(e, r, roots);
      draft.hash = state_hash(e, r, words, positions, roots);
    }
    admission = reserve_state(e, draft, r, words, positions, roots);
    admission_words[0] = admission.id;
    admission_words[1] = admission.slot;
    admission_words[2] = u32(admission.fresh);
  }
  __syncthreads();
  admission = {admission_words[0], admission_words[1], admission_words[2] != 0};
  if (admission.id == none || !admission.fresh) return admission.id;
  copy_state_payload(e, admission.id, r, words, positions, roots, worker, blockDim.x);
  __syncthreads();
  if (!worker) publish_state(e, draft, admission);
  __syncthreads();
  return admission.id;
}
__device__ u64 node_hash(Node n) {
  u64 h = hash_word(1469598103934665603ull, u32(n.feature));
  h = hash_word(h, n.payload); h = hash_word(h, n.left); return hash_word(h, n.right);
}
__device__ u32 intern_node(EngineView e, Node n) {
  if (n.feature != -1 && n.left == n.right) { ++e.status->node_hits; return n.left; }
  u32 slot = hash_bucket(node_hash(n), e.arena.node_buckets);
  for (u32 probe = 0; probe < e.arena.node_buckets; ++probe,
       slot = (slot + 1) & (e.arena.node_buckets - 1)) {
    u32 id = e.arena.node_table[slot];
    if (id == none) {
      if (e.status->nodes == e.arena.node_capacity) { e.status->request = 3; return none; }
      id = u32(e.status->nodes++); e.arena.nodes[id] = n; e.arena.node_table[slot] = id;
      return id;
    }
    auto other = e.arena.nodes[id];
    if (n.feature == other.feature && n.payload == other.payload &&
        n.left == other.left && n.right == other.right) { ++e.status->node_hits; return id; }
  }
  e.status->error = 23; return none;
}
__device__ u32 terminal_lookup(EngineView e, const u32* words, bool insert, u32 state) {
  u32 slot = hash_bucket(score_hash(e, words), e.arena.state_buckets);
  for (u32 probe = 0; probe < e.arena.state_buckets; ++probe,
       slot = (slot + 1) & (e.arena.state_buckets - 1)) {
    u32 id = e.arena.terminal_table[slot];
    if (id == none) {
      if (insert) e.arena.terminal_table[slot] = state;
      return none;
    }
    if (id >= e.status->states || e.arena.states[id].phase != 3 ||
        e.arena.states[id].predicate != none) { e.status->error = 77; return none; }
    if (scores_equal(e, id, words)) return id;
  }
  e.status->error = 24; return none;
}
// Serial cluster repair removes a cache entry without leaving tombstones.
// Moving an entry is legal exactly when the hole lies on its linear probe path.
// All callers are at a scheduler boundary, after parallel readers have finished.
__device__ bool erase_cached_state_entry(EngineView e, u32 victim, bool terminal) {
  u32* table = terminal ? e.arena.terminal_table : e.arena.table;
  const u32 count = e.arena.state_buckets, mask = count - 1;
  const u64 key = terminal ? score_hash(e, e.arena.words + u64(victim) * e.source.classes)
                           : e.arena.states[victim].hash;
  u32 hole = hash_bucket(key, count);
  bool found = false;
  for (u32 probe = 0; probe < count; ++probe, hole = (hole + 1) & mask) {
    u32 at = table[hole];
    if (at == none) {
      // An equivalent terminal vector may be represented by another state.
      if (!terminal) e.status->error = 71;
      return terminal;
    }
    if (at >= e.status->states || e.arena.states[at].phase == free_phase) {
      e.status->error = 72; return false;
    }
    if (at == victim) { found = true; break; }
  }
  if (!found) { e.status->error = 71; return false; }
  u32 next = (hole + 1) & mask;
  for (u32 probe = 0; probe < count; ++probe, next = (next + 1) & mask) {
    u32 at = table[next];
    if (at == none) { table[hole] = none; return true; }
    if (at >= e.status->states || e.arena.states[at].phase == free_phase || at == victim) {
      e.status->error = 72; return false;
    }
    u64 h = terminal ? score_hash(e, e.arena.words + u64(at) * e.source.classes)
                     : e.arena.states[at].hash;
    u32 ideal = hash_bucket(h, count);
    if (((next - ideal) & mask) >= ((next - hole) & mask)) {
      table[hole] = at;
      hole = next;
    }
  }
  e.status->error = 73; return false;
}
// The scheduler must first deliver this state's node to every waiting parent
// and consume its completion event. Graph nodes themselves are never recycled.
__device__ bool evict_completed_state(EngineView e, u32 id) {
  if (!id || id >= e.status->states || e.arena.states[id].phase != 3 ||
      e.arena.states[id].node >= e.status->nodes) { e.status->error = 70; return false; }
  if (e.arena.states[id].predicate == none && !erase_cached_state_entry(e, id, true)) return false;
  if (!erase_cached_state_entry(e, id, false)) return false;
  auto& state = e.arena.states[id];
  state.phase = free_phase; state.predicate = state.left = state.right = state.node = none;
  state.reserved = e.status->free_head;
  e.status->free_head = id; ++e.status->free_count; ++e.status->state_evictions;
  return true;
}
__global__ void initialize(EngineView e) {
  if (blockIdx.x || threadIdx.x) return;
  *e.status = Status{};
  for (u32 i = 0; i < e.source.nodes; ++i) {
    bool leaf = e.source.left[i] < 0;
    if ((leaf && !isfinite(e.source.value[i])) ||
        (!leaf && (isnan(e.source.cut[i]) || e.source.feature[i] < 0 ||
                   u32(e.source.feature[i]) >= e.source.features))) {
      e.status->error = 1; return;
    }
  }
  if (!domain::initial_domain(e.domain, e.draft_witness)) { e.status->error = 2; return; }
  copy_region(e, e.draft_region, e.draft_witness);
  for (u32 c = 0; c < e.source.classes; ++c) {
    e.draft_words[c] = __float_as_uint(e.source.bias[c]); e.draft_positions[c] = 0;
  }
  for (u32 t = 0; t < e.source.trees; ++t) e.draft_residual[t] = e.source.roots[t];
  State s{};
  s.predicate = normalize(e, e.draft_region, e.draft_words, e.draft_positions, e.draft_residual);
  if (e.status->error) return;
  u32 id = intern_state(e, s, e.draft_region, e.draft_words, e.draft_positions, e.draft_residual);
  if (e.status->error || id == none) return;
  e.arena.stack[0] = id; e.status->depth = 1;
}
// The existing adaptive4 unconditional subtree enclosure. Context restriction
// only removes leaves, so these extrema remain conservative in every state.
__global__ void subtree_extrema(SourceView v, u32* minimum, u32* maximum, u32* stack,
                                u64* support, u32 support_words, u32* walk_shape=nullptr) {
  if (blockIdx.x || threadIdx.x) return;
  u32 peak=0;u64 visits=0;
  for (u32 t = 0; t < v.trees; ++t) {
    u32 depth = 0; stack[depth++] = u32(v.roots[t]);
    while (depth) {
      peak=max(peak,depth);
      u32 item = stack[--depth], at = item & 0x7fffffffu;
      if(!(item&0x80000000u))++visits;
      if (v.left[at] < 0) {
        minimum[at] = maximum[at] = __float_as_uint(v.value[at]);
        for (u32 w = 0; w < support_words; ++w) support[u64(at) * support_words + w] = 0;
      } else if (item & 0x80000000u) {
        for (u32 w = 0; w < support_words; ++w)
          support[u64(at) * support_words + w] = support[u64(v.left[at]) * support_words + w] |
                                               support[u64(v.right[at]) * support_words + w];
        support[u64(at) * support_words + u32(v.feature[at]) / 64] |= u64(1) << (u32(v.feature[at]) % 64);
        minimum[at] = __float_as_uint(fminf(__uint_as_float(minimum[v.left[at]]),
                                           __uint_as_float(minimum[v.right[at]])));
        maximum[at] = __float_as_uint(fmaxf(__uint_as_float(maximum[v.left[at]]),
                                           __uint_as_float(maximum[v.right[at]])));
      } else {
        stack[depth++] = at | 0x80000000u;
        stack[depth++] = u32(v.right[at]); stack[depth++] = u32(v.left[at]);
      }
    }
  }
  if(walk_shape){walk_shape[0]=peak;walk_shape[1]=u32(visits>UINT32_MAX?UINT32_MAX:visits);}
}
// An optional read-only enclosure refinement. The caller selects its work
// budget without changing RuntimeGate authority. R stays unchanged along this walk:
// ignoring ancestor correlations keeps a superset of reachable leaves. A fixed
// visit/stack limit replaces unexplored frontiers with their static enclosures.
struct ConditionedExtrema {
  u32 minimum = 0, maximum = 0;
  u32 visited = 0, fallbacks = 0;
};
__device__ ConditionedExtrema conditioned_subtree_extrema(
    EngineView e, domain::RegionView r, u32 root, u32* private_stack,
    u32 stack_capacity, u32 visit_budget) {
  ConditionedExtrema out{e.minimum[root], e.maximum[root], 0, 0};
  if (!stack_capacity || !visit_budget) { out.fallbacks = 1; return out; }
  u32 pending = 1; private_stack[0] = root; bool collected = false;
  auto absorb = [&](u32 node) {
    if (!collected) {
      out.minimum = e.minimum[node]; out.maximum = e.maximum[node];
      collected = true;
    } else {
      // These are numeric enclosures, not rewrites of signed-zero leaf words.
      out.minimum = __float_as_uint(fminf(__uint_as_float(out.minimum),
                                        __uint_as_float(e.minimum[node])));
      out.maximum = __float_as_uint(fmaxf(__uint_as_float(out.maximum),
                                        __uint_as_float(e.maximum[node])));
    }
  };
  while (pending && out.visited < visit_budget) {
    u32 node = private_stack[--pending]; ++out.visited;
    if (e.source.left[node] < 0) { absorb(node); continue; }
    bool right = false;
    if (domain::forced_side(e.domain, r, u32(e.source.feature[node]),
          __float_as_uint(e.source.cut[node]), bool(e.source.missing[node]), right)) {
      private_stack[pending++] = u32(right ? e.source.right[node] : e.source.left[node]);
    } else if (stack_capacity - pending >= 2) {
      private_stack[pending++] = u32(e.source.left[node]);
      private_stack[pending++] = u32(e.source.right[node]);
    } else { absorb(node); ++out.fallbacks; }
  }
  while (pending) { absorb(private_stack[--pending]); ++out.fallbacks; }
  return out;
}
__device__ int qualified_range_label(EngineView e) {
  if (!e.qualified_gap) return -1;
  for (u32 c = 0; c < e.source.classes; ++c) {
    float lo = e.range_lower[c], hi = e.range_upper[c];
    if (!isfinite(lo) || !isfinite(hi) || lo > hi || lo < -10.f || hi > 10.f) return -1;
  }
  for (u32 winner = 0; winner < e.source.classes; ++winner) {
    bool wins = true;
    for (u32 c = 0; c < e.source.classes; ++c) if (c != winner)
      if (__dsub_rn(double(e.range_lower[winner]), double(e.range_upper[c])) < 0x1p-10)
        wins = false;
    if (wins) return int(winner);
  }
  return -1;
}
__device__ int qualified_interval_label(EngineView e, u32 id) {
  if (!e.qualified_gap) return -1;
  const u32* words = e.arena.words + u64(id) * e.source.classes;
  const auto* roots = e.source.trees ? e.arena.residual + u64(id) * e.source.trees : nullptr;
  for (u32 c = 0; c < e.source.classes; ++c)
    e.range_lower[c] = e.range_upper[c] = __uint_as_float(words[c]);
  for (u32 t = 0; t < e.source.trees; ++t) if (roots[t] >= 0) {
    u32 c = u32(e.source.channels[t]);
    e.range_lower[c] = __fadd_rn(e.range_lower[c], __uint_as_float(e.minimum[roots[t]]));
    e.range_upper[c] = __fadd_rn(e.range_upper[c], __uint_as_float(e.maximum[roots[t]]));
  }
  return qualified_range_label(e);
}
__global__ void rehash_states(EngineView e, u32 count) {
  if (blockIdx.x || threadIdx.x) return;
  for (u32 id = 0; id < count; ++id) {
    if (e.arena.states[id].phase == free_phase) continue;
    u32 slot = hash_bucket(e.arena.states[id].hash, e.arena.state_buckets);
    while (e.arena.table[slot] != none) slot = (slot + 1) & (e.arena.state_buckets - 1);
    e.arena.table[slot] = id;
    if (e.arena.states[id].predicate == none && e.arena.states[id].phase == 3)
      terminal_lookup(e, e.arena.words + u64(id) * e.source.classes, true, id);
  }
}
__global__ void rehash_nodes(EngineView e, u32 count) {
  if (blockIdx.x || threadIdx.x) return;
  for (u32 id = 0; id < count; ++id) {
    u32 slot = hash_bucket(node_hash(e.arena.nodes[id]), e.arena.node_buckets);
    while (e.arena.node_table[slot] != none) slot = (slot + 1) & (e.arena.node_buckets - 1);
    e.arena.node_table[slot] = id;
  }
}
// Interning creates children before parents. Descending reachability followed
// by ascending compaction therefore needs no recursion or expanded tree.
__global__ void collect_root(EngineView e, u32* remap, u32 count) {
  if (blockIdx.x || threadIdx.x) return;
  for (u32 i = 0; i < count; ++i) remap[i] = none;
  remap[e.status->root] = 0;
  for (u32 i = count; i-- > 0;) if (remap[i] != none) {
    auto n = e.arena.nodes[i];
    if (n.feature != -1) {
      if (n.left >= i || n.right >= i) { e.status->error = 40; return; }
      remap[n.left] = 0; remap[n.right] = 0;
    }
  }
  u32 kept = 0;
  for (u32 i = 0; i < count; ++i) if (remap[i] != none) {
    auto n = e.arena.nodes[i];
    remap[i] = kept;
    if (n.feature != -1) { n.left = remap[n.left]; n.right = remap[n.right]; }
    e.arena.nodes[kept++] = n;
  }
  e.status->root = remap[e.status->root]; e.status->nodes = kept;
}
struct StateStorage {
  Buffer<State> states;
  Buffer<u32> words, positions, lower, upper, missing, stack, table, terminal_table;
  Buffer<u32> witness_lower, witness_upper, witness_missing;
  Buffer<std::int32_t> residual;
  Buffer<u64> allowed;
  Buffer<u64> witness_allowed;
  u32 capacity;
  StateStorage(Budget& b, u32 n, u32 F, u32 K, u32 T, u32 W)
      : states(b, n), words(b, multiply(n, K)), positions(b, multiply(n, K)),
        lower(b, multiply(n, F)), upper(b, multiply(n, F)), missing(b, multiply(n, F)),
        stack(b, n), table(b, buckets(n)), terminal_table(b, buckets(n)),
        witness_lower(b, multiply(n, F)), witness_upper(b, multiply(n, F)), witness_missing(b, multiply(n, F)),
        residual(b, multiply(n, T)), allowed(b, multiply(n, W)), witness_allowed(b, multiply(n, W)), capacity(n) {
    table.empty_table(); terminal_table.empty_table();
  }
};
struct NodeStorage {
  Buffer<Node> nodes;
  Buffer<u32> table;
  u32 capacity;
  NodeStorage(Budget& b, u32 n) : nodes(b, n), table(b, buckets(n)), capacity(n) {
    table.empty_table();
  }
};
inline void bind(EngineView& e, StateStorage& s, NodeStorage& n) {
  e.arena = {s.states.data, s.words.data, s.positions.data, s.lower.data, s.upper.data,
             s.missing.data, s.stack.data, s.witness_lower.data, s.witness_upper.data, s.witness_missing.data,
             s.residual.data, s.allowed.data, s.witness_allowed.data,
             s.table.data, s.terminal_table.data, n.nodes.data, s.capacity, n.capacity,
             buckets(s.capacity), buckets(n.capacity), n.table.data};
}
template<class T> inline void copy(Buffer<T>& to, const Buffer<T>& from, u64 count) {
  require(count <= to.size && count <= from.size, "adaptive growth copy extent");
  if (count) check(cudaMemcpy(to.data, from.data, count * sizeof(T), cudaMemcpyDeviceToDevice),
                   "adaptive growth copy");
}
inline u32 next_capacity(u32 current, u64 ceiling) {
  require(current < ceiling && ceiling <= 0x3fffffffu, "adaptive plan index limit reached");
  return u32(std::min<u64>(ceiling, u64(current) * 2));
}
inline growth::Shape growth_shape(const EngineView& e) {
  return {e.domain.numeric_features,e.source.classes,e.source.trees,e.domain.mask_words,
          sizeof(State),sizeof(Node)};
}
inline u64 growth_headroom(const Budget& budget,u64 device_reserve=0) {
  std::size_t free_bytes=0,total_bytes=0;
  check(cudaMemGetInfo(&free_bytes,&total_bytes),"adaptive growth device headroom");
  u64 owned=budget.used<=budget.limit?budget.limit-budget.used:0;
  u64 device=u64(free_bytes)>=device_reserve?u64(free_bytes)-device_reserve:0;
  return std::min(owned,device);
}
template<class Cost>inline void plan_growth(growth::Outcome& out,const Budget& budget,
    u32 current,u64 minimum,u64 ceiling,Cost&& cost,u32 retry_ceiling,
    u64 device_reserve,const GrowthObserver& observer) {
  for(;;) {
    out.plan=growth::choose(current,minimum,ceiling,growth_headroom(budget,device_reserve),cost,retry_ceiling);
    if(observer)observer(out);
    if(!out.plan.affordable)
      throw AllocationRefusal(cudaSuccess,out.plan.minimum_bytes,"adaptive required growth exceeds resource headroom");
    // A progress hook may itself allocate. A fresh query prevents publishing a
    // reserve estimate and then knowingly allocating against obsolete headroom.
    if(out.plan.additional_bytes<=growth_headroom(budget,device_reserve))return;
  }
}
// Caller has allocated every coupled replacement owner first. Copy and rehash
// use only the staged view; live bindings are unchanged on any thrown failure.
inline EngineView stage_state_growth(const EngineView& e,const StateStorage& old,
                                     StateStorage& fresh,NodeStorage& nodes,const Status& h) {
  copy(fresh.states,old.states,h.states);
  copy(fresh.words,old.words,h.states*e.source.classes);
  copy(fresh.positions,old.positions,h.states*e.source.classes);
  copy(fresh.residual,old.residual,h.states*e.source.trees);
  copy(fresh.lower,old.lower,h.states*e.domain.numeric_features);
  copy(fresh.upper,old.upper,h.states*e.domain.numeric_features);
  copy(fresh.missing,old.missing,h.states*e.domain.numeric_features);
  copy(fresh.allowed,old.allowed,h.states*e.domain.mask_words);
  copy(fresh.witness_lower,old.witness_lower,h.states*e.domain.numeric_features);
  copy(fresh.witness_upper,old.witness_upper,h.states*e.domain.numeric_features);
  copy(fresh.witness_missing,old.witness_missing,h.states*e.domain.numeric_features);
  copy(fresh.witness_allowed,old.witness_allowed,h.states*e.domain.mask_words);
  copy(fresh.stack,old.stack,h.depth);
  EngineView staged=e;bind(staged,fresh,nodes);
  rehash_states<<<1,1>>>(staged,u32(h.states));synchronize();
  Status checked{};check(cudaMemcpy(&checked,e.status,sizeof(checked),cudaMemcpyDeviceToHost),"adaptive staged state status");
  require(!checked.error,"adaptive staged state rehash invariant failure");
  return staged;
}
inline growth::Outcome grow_nodes(EngineView& e,StateStorage& states,
    std::unique_ptr<NodeStorage>& storage,Budget& budget,const Status& h,u64 ceiling,
    u64 minimum=0,const GrowthObserver& observer={},const GrowthAllocationHook& hook={},
    u64 device_reserve=0) {
  growth::Outcome out;u32 retry=0;const auto shape=growth_shape(e);
  if(!minimum)minimum=h.nodes+1;
  for(;;) {
    plan_growth(out,budget,storage->capacity,minimum,ceiling,
      [&](u32 capacity){return growth::node_bytes(shape,capacity);},retry,device_reserve,observer);
    std::unique_ptr<NodeStorage> fresh;
    try {
      if(hook)hook("nodes");
      fresh=std::make_unique<NodeStorage>(budget,out.plan.selected);
    }catch(const AllocationRefusal&) {
      ++out.allocation_refusals;++out.transaction_rollbacks;
      if(observer)observer(out);
      retry=growth::retry_ceiling(out.plan);
      if(!retry)throw;
      continue;
    }
    copy(fresh->nodes,storage->nodes,h.nodes);
    EngineView staged=e;bind(staged,states,*fresh);
    rehash_nodes<<<1,1>>>(staged,u32(h.nodes));synchronize();
    storage.swap(fresh);e=staged;fresh.reset();out.complete=true;
    if(observer)observer(out);return out;
  }
}
} // namespace class_conversion_adaptive
