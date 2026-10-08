#pragma once
// Opaque idle-boundary transport for the sole adaptive converter. The caller
// validates source/domain/native identity and allocates fresh owners before
// restore; no model arithmetic, device addresses or RuntimeGate is serialized.
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/crypto.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace class_conversion_adaptive::checkpoint {
using u32=std::uint32_t;
using u64=std::uint64_t;
using Json=nlohmann::json;
struct Range {u64 offset=0,bytes=0;};
struct DeviceSection {
  std::string name;const void* data=nullptr;u64 bytes=0;
  // Empty sparse ranges mean a completely zero-filled fresh destination.
  // Empty nonsparse ranges mean the entire section is initialized/copied.
  bool sparse=false;std::vector<Range> ranges;
};
struct MutableSection {std::string name;void* data=nullptr;u64 bytes=0;};
struct LayoutSection {std::string name;u64 bytes=0;bool sparse=false;std::vector<Range> ranges;};
struct IdleBoundary {u32 stage=0,jobs=0,native_count=0,native_cursor=0;};
struct Options {
  u64 chunk_bytes=1ull<<20;
  u64 maximum_header_bytes=4ull<<20;
  u64 maximum_payload_bytes=UINT64_MAX;
  u64 maximum_sections=1024;
  std::function<void(u64 copied,u64 total)> progress;
};
struct Receipt {
  std::string sha256;u64 file_bytes=0,payload_bytes=0,sections=0,host_chunk_bytes=0;
  bool integrity_verified=false;
};
namespace detail {
inline void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
inline u64 add(u64 x,u64 y){require(x<=UINT64_MAX-y,"checkpoint extent overflow");return x+y;}
inline void cuda_check(cudaError_t error,const char* what){
  if(error!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(error));
}
inline void io_error(const char* what){throw std::runtime_error(std::string(what)+": "+std::strerror(errno));}
struct Fd {
  int value=-1;Fd()=default;explicit Fd(int v):value(v){}
  ~Fd(){if(value>=0)::close(value);}Fd(const Fd&)=delete;Fd&operator=(const Fd&)=delete;
  Fd(Fd&&x)noexcept:value(std::exchange(x.value,-1)){}
};
struct Digest {
  std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),EVP_MD_CTX_free};
  bool ended=false;
  Digest(){require(bool(ctx)&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"checkpoint digest initialization");}
  void update(const void* p,std::size_t n){require(!ended&&EVP_DigestUpdate(ctx.get(),p,n)==1,"checkpoint digest update");}
  std::array<unsigned char,32> finish(){std::array<unsigned char,32> out{};unsigned int n=0;
    require(!ended&&EVP_DigestFinal_ex(ctx.get(),out.data(),&n)==1&&n==out.size(),"checkpoint digest finalization");ended=true;return out;}
};
inline std::string hex(const std::array<unsigned char,32>& bytes){
  constexpr char digits[]="0123456789abcdef";std::string out(64,'0');
  for(std::size_t i=0;i<bytes.size();++i){out[2*i]=digits[bytes[i]>>4];out[2*i+1]=digits[bytes[i]&15];}return out;
}
inline void write_all(int fd,const void* p,std::size_t n){
  auto q=static_cast<const unsigned char*>(p);
  while(n){const auto k=::write(fd,q,n);if(k<0){if(errno==EINTR)continue;io_error("checkpoint write");}
    require(k>0,"checkpoint short write");q+=k;n-=std::size_t(k);}
}
inline void read_all(int fd,void* p,std::size_t n){
  auto q=static_cast<unsigned char*>(p);
  while(n){const auto k=::read(fd,q,n);if(k<0){if(errno==EINTR)continue;io_error("checkpoint read");}
    require(k>0,"checkpoint truncated file");q+=k;n-=std::size_t(k);}
}
inline void durable(int fd,const char* what){while(::fsync(fd)<0){if(errno==EINTR)continue;io_error(what);}}
inline u64 integer(const Json& j,const char* what){
  require(j.is_number_integer(),what);
  if(j.is_number_unsigned())return j.get<u64>();
  const auto x=j.get<std::int64_t>();require(x>=0,what);return u64(x);
}
inline void validate_options(const Options& o){
  require(o.chunk_bytes&&o.chunk_bytes<=u64(std::numeric_limits<std::size_t>::max()),"checkpoint invalid host chunk extent");
  require(o.maximum_header_bytes&&o.maximum_header_bytes<=u64(std::numeric_limits<std::size_t>::max()),"checkpoint invalid header extent");
  require(o.maximum_sections,"checkpoint invalid section cap");
}
inline void validate_boundary(const IdleBoundary& b){
  require(!b.stage&&!b.jobs&&!b.native_count&&!b.native_cursor,"checkpoint requires synchronized idle boundary");
}
inline void validate_ranges(const LayoutSection& s){
  require(!s.name.empty()&&s.name.find('\0')==std::string::npos,"checkpoint invalid section name");
  u64 end=0;
  for(const auto& r:s.ranges){require(r.bytes&&r.offset>=end&&r.offset<=s.bytes&&r.bytes<=s.bytes-r.offset,
    "checkpoint unordered, overlapping or invalid section range");end=r.offset+r.bytes;}
  if(!s.sparse)require(s.bytes? s.ranges.size()==1&&s.ranges[0].offset==0&&s.ranges[0].bytes==s.bytes:s.ranges.empty(),
    "checkpoint nonsparse section must contain exactly its full extent");
}
inline std::array<unsigned char,40> prefix(u64 header,u64 payload,u64 sections){
  std::array<unsigned char,40> out{};std::memcpy(out.data(),"ADPCKP01",8);
  const u64 words[4]{1,header,payload,sections};
  for(u32 w=0;w<4;++w)for(u32 byte=0;byte<8;++byte)out[8+8*w+byte]=static_cast<unsigned char>(words[w]>>(8*byte));return out;
}
inline u64 prefix_word(const std::array<unsigned char,40>& p,u32 index){
  u64 value=0;for(u32 b=0;b<8;++b)value|=u64(p[8+8*index+b])<<(8*b);return value;
}
inline void unique_names(const std::vector<LayoutSection>& sections){
  for(std::size_t i=0;i<sections.size();++i)for(std::size_t j=0;j<i;++j)
    require(sections[i].name!=sections[j].name,"checkpoint duplicate section name");
}
inline void payload_layout(const std::vector<LayoutSection>& sections,u64& bytes){
  bytes=0;for(const auto& s:sections){validate_ranges(s);for(const auto&r:s.ranges)bytes=add(bytes,r.bytes);}unique_names(sections);
}
inline const unsigned char* offset(const void* p,u64 byte){
  return reinterpret_cast<const unsigned char*>(reinterpret_cast<std::uintptr_t>(p)+byte);
}
inline unsigned char* offset(void* p,u64 byte){
  return reinterpret_cast<unsigned char*>(reinterpret_cast<std::uintptr_t>(p)+byte);
}
} // namespace detail

inline Receipt write_atomic(const std::filesystem::path& path,const Json& identity,
    const Json& metadata,const std::vector<DeviceSection>& sections,
    IdleBoundary boundary={},const Options& options={}) {
  using namespace detail;validate_options(options);validate_boundary(boundary);
  require(identity.is_object()&&!identity.empty()&&metadata.is_object(),"checkpoint identity/metadata must be objects");
  require(!path.empty()&&path.has_filename(),"checkpoint invalid destination");
  require(sections.size()<=options.maximum_sections,"checkpoint excessive sections");
  std::vector<LayoutSection> layout;layout.reserve(sections.size());
  for(const auto& s:sections){require(!s.bytes||s.data,"checkpoint null device section");
    require(s.bytes<=u64(std::numeric_limits<std::uintptr_t>::max())-reinterpret_cast<std::uintptr_t>(s.data),"checkpoint device address extent overflow");
    LayoutSection l{s.name,s.bytes,s.sparse,s.ranges};
    if(!s.sparse){require(s.ranges.empty(),"checkpoint nonsparse input must not offer ranges");if(s.bytes)l.ranges.push_back({0,s.bytes});}layout.push_back(std::move(l));}
  u64 payload=0;payload_layout(layout,payload);require(payload<=options.maximum_payload_bytes,"checkpoint payload cap");
  Json description=Json::array();for(const auto& s:layout){Json ranges=Json::array();for(const auto&r:s.ranges)ranges.push_back({{"offset",r.offset},{"bytes",r.bytes}});
    description.push_back({{"name",s.name},{"bytes",s.bytes},{"sparse",s.sparse},{"ranges",std::move(ranges)}});}
  Json header{{"format","adaptive-idle-checkpoint-1"},{"identity",identity},{"metadata",metadata},
    {"boundary",{{"stage",0},{"jobs",0},{"native_count",0},{"native_cursor",0}}},{"sections",std::move(description)}};
  const std::string encoded=header.dump();require(encoded.size()<=options.maximum_header_bytes,"checkpoint header cap");
  auto first=prefix(encoded.size(),payload,layout.size());const u64 file_bytes=add(add(add(first.size(),encoded.size()),payload),32);
  require(file_bytes<=u64(std::numeric_limits<off_t>::max()),"checkpoint file offset extent");
  std::filesystem::path parent=path.parent_path();if(parent.empty())parent=".";
  static std::atomic<u64> sequence{0};
  const auto temporary=parent/(path.filename().string()+".tmp."+std::to_string(::getpid())+"."+std::to_string(sequence.fetch_add(1)));
  Fd file(::open(temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600));if(file.value<0)io_error("checkpoint temporary open");
  bool published=false;
  try{
    Digest digest;write_all(file.value,first.data(),first.size());digest.update(first.data(),first.size());
    write_all(file.value,encoded.data(),encoded.size());digest.update(encoded.data(),encoded.size());
    const auto chunk_size=std::size_t(std::min<u64>(options.chunk_bytes,std::max<u64>(1,payload)));
    std::vector<unsigned char> chunk(chunk_size);u64 copied=0;
    for(std::size_t i=0;i<layout.size();++i)for(const auto& range:layout[i].ranges){
      for(u64 at=0;at<range.bytes;){const auto count=std::size_t(std::min<u64>(chunk.size(),range.bytes-at));
        cuda_check(cudaMemcpy(chunk.data(),offset(sections[i].data,range.offset+at),count,cudaMemcpyDeviceToHost),"checkpoint snapshot copy");
        write_all(file.value,chunk.data(),count);digest.update(chunk.data(),count);at+=count;copied+=count;
        if(options.progress)options.progress(copied,payload);}
    }
    const auto sum=digest.finish();write_all(file.value,sum.data(),sum.size());durable(file.value,"checkpoint file fsync");
    Fd directory(::open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC));if(directory.value<0)io_error("checkpoint directory open");
    if(::rename(temporary.c_str(),path.c_str())<0)io_error("checkpoint atomic rename");published=true;
    durable(directory.value,"checkpoint directory fsync");
    return{hex(sum),file_bytes,payload,u64(layout.size()),u64(chunk.size()),true};
  }catch(...){if(!published)::unlink(temporary.c_str());throw;}
}

// One descriptor is held from metadata inspection through complete restore.
// No second file hash pass; checksum verification occurs in the payload read.
// restore_exact may mutate ONLY unpublished fresh owners. A failed restore
// leaves those owners disposable and must never be swapped into live state.
class Reader {
  Options options_;detail::Fd file_;detail::Digest digest_;
  Json header_;std::vector<LayoutSection> layout_;u64 payload_=0,file_bytes_=0;
  bool consumed_=false;
public:
  Reader(const std::filesystem::path& path,const Json& expected_identity,const Options& options={})
      :options_(options),file_(::open(path.c_str(),O_RDONLY|O_CLOEXEC)) {
    using namespace detail;validate_options(options_);if(file_.value<0)io_error("checkpoint open");
    require(expected_identity.is_object()&&!expected_identity.empty(),"checkpoint expected identity must be an object");
    struct stat info{};if(::fstat(file_.value,&info)<0)io_error("checkpoint fstat");
    require(S_ISREG(info.st_mode)&&info.st_size>=72,"checkpoint file type/size");
    std::array<unsigned char,40> first{};read_all(file_.value,first.data(),first.size());digest_.update(first.data(),first.size());
    require(!std::memcmp(first.data(),"ADPCKP01",8)&&prefix_word(first,0)==1,"checkpoint magic/version");
    const auto header_bytes=prefix_word(first,1);payload_=prefix_word(first,2);const auto count=prefix_word(first,3);
    require(header_bytes&&header_bytes<=options_.maximum_header_bytes&&payload_<=options_.maximum_payload_bytes&&count<=options_.maximum_sections,"checkpoint declared extent cap");
    file_bytes_=add(add(add(first.size(),header_bytes),payload_),32);require(file_bytes_==u64(info.st_size),"checkpoint declared file size/truncation");
    std::string encoded(std::size_t(header_bytes),'\0');read_all(file_.value,encoded.data(),encoded.size());digest_.update(encoded.data(),encoded.size());
    header_=Json::parse(encoded);require(header_.is_object()&&header_.at("format")=="adaptive-idle-checkpoint-1","checkpoint header format");
    require(header_.at("identity")==expected_identity,"checkpoint source/domain/native identity differs");
    require(header_.at("metadata").is_object(),"checkpoint metadata type");
    const auto& b=header_.at("boundary");require(b.is_object(),"checkpoint boundary type");
    for(const char* field:{"stage","jobs","native_count","native_cursor"})require(integer(b.at(field),"checkpoint boundary extent")==0,"checkpoint nonidle snapshot");
    const auto& table=header_.at("sections");require(table.is_array()&&table.size()==count,"checkpoint section table extent");
    layout_.reserve(std::size_t(count));
    for(const auto& s:table){require(s.is_object()&&s.at("name").is_string()&&s.at("sparse").is_boolean()&&s.at("ranges").is_array(),"checkpoint section types");
      LayoutSection l{s.at("name").get<std::string>(),integer(s.at("bytes"),"checkpoint section extent"),s.at("sparse").get<bool>(),{}};
      require(l.bytes<=u64(std::numeric_limits<std::size_t>::max()),"checkpoint destination extent");
      for(const auto&r:s.at("ranges")){require(r.is_object(),"checkpoint range type");l.ranges.push_back({integer(r.at("offset"),"checkpoint range offset"),integer(r.at("bytes"),"checkpoint range bytes")});}
      layout_.push_back(std::move(l));}
    u64 declared=0;payload_layout(layout_,declared);require(declared==payload_,"checkpoint payload/section extents differ");
  }
  Reader(const Reader&)=delete;Reader&operator=(const Reader&)=delete;
  const Json& identity()const{return header_.at("identity");}
  const Json& metadata()const{return header_.at("metadata");}
  const std::vector<LayoutSection>& layout()const{return layout_;}
  u64 payload_bytes()const{return payload_;}
  Receipt restore_exact(const std::vector<MutableSection>& targets) {
    using namespace detail;require(!consumed_,"checkpoint reader already consumed");consumed_=true;
    require(targets.size()==layout_.size(),"checkpoint fresh section count differs");
    // Validate EVERY target before the first write into fresh owners.
    for(std::size_t i=0;i<layout_.size();++i){const auto&t=targets[i];const auto&s=layout_[i];
      require(t.name==s.name&&t.bytes==s.bytes&&(!t.bytes||t.data),"checkpoint fresh section identity/extent differs");
      require(t.bytes<=u64(std::numeric_limits<std::uintptr_t>::max())-reinterpret_cast<std::uintptr_t>(t.data),"checkpoint target address extent overflow");}
    const auto chunk_size=std::size_t(std::min<u64>(options_.chunk_bytes,std::max<u64>(1,payload_)));
    std::vector<unsigned char> chunk(chunk_size);u64 copied=0;
    for(std::size_t i=0;i<layout_.size();++i){const auto&s=layout_[i];auto& t=targets[i];
      if(s.sparse&&s.bytes)cuda_check(cudaMemset(t.data,0,std::size_t(s.bytes)),"checkpoint fresh sparse zero");
      for(const auto& r:s.ranges)for(u64 at=0;at<r.bytes;){const auto n=std::size_t(std::min<u64>(chunk.size(),r.bytes-at));
        read_all(file_.value,chunk.data(),n);digest_.update(chunk.data(),n);
        cuda_check(cudaMemcpy(offset(t.data,r.offset+at),chunk.data(),n,cudaMemcpyHostToDevice),"checkpoint fresh restore copy");
        at+=n;copied+=n;if(options_.progress)options_.progress(copied,payload_);}
    }
    std::array<unsigned char,32> offered{};read_all(file_.value,offered.data(),offered.size());const auto actual=digest_.finish();
    require(CRYPTO_memcmp(actual.data(),offered.data(),actual.size())==0,"checkpoint integrity checksum differs");
    unsigned char extra=0;ssize_t n;do{n=::read(file_.value,&extra,1);}while(n<0&&errno==EINTR);
    if(n<0)io_error("checkpoint final read");require(!n,"checkpoint trailing bytes");
    return{hex(actual),file_bytes_,payload_,u64(layout_.size()),u64(chunk.size()),true};
  }
};
// Background writes own only these immutable host bytes. Capturing is the sole
// GPU-touching operation and MUST finish at an idle synchronized boundary before
// search resumes. There is no deferred device pointer in an OwnedSnapshot.
struct CaptureOptions : Options {u64 host_byte_budget=0;}; // 0 uses current MemAvailable
class AsyncWriter;
class OwnedSnapshot {
  std::array<unsigned char,40> prefix_{};std::string header_;
  std::unique_ptr<unsigned char[]> payload_;u64 payload_bytes_=0,file_bytes_=0;
  u64 sections_=0,chunk_bytes_=0,captured_bytes_=0;double capture_seconds_=0;
  friend OwnedSnapshot capture(const Json&,const Json&,const std::vector<DeviceSection>&,IdleBoundary,const CaptureOptions&);
  friend class AsyncWriter;
public:
  OwnedSnapshot()=default;OwnedSnapshot(const OwnedSnapshot&)=delete;OwnedSnapshot&operator=(const OwnedSnapshot&)=delete;
  OwnedSnapshot(OwnedSnapshot&&)=default;OwnedSnapshot&operator=(OwnedSnapshot&&)=default;
  u64 captured_bytes()const{return captured_bytes_;}u64 payload_bytes()const{return payload_bytes_;}
  double capture_seconds()const{return capture_seconds_;}
};
inline u64 available_host_bytes(){
  // Host resource metadata only. The explicit caller budget is also enforced;
  // this observation is not a reservation against concurrent host allocation.
  std::ifstream in("/proc/meminfo");detail::require(bool(in),"checkpoint host availability unavailable");
  std::string key,unit;u64 amount=0;
  while(in>>key>>amount>>unit){if(key=="MemAvailable:"){detail::require(unit=="kB"&&amount<=UINT64_MAX/1024,"checkpoint host availability extent");return amount*1024;}
    in.ignore(std::numeric_limits<std::streamsize>::max(),'\n');}
  throw std::runtime_error("checkpoint MemAvailable unavailable");
}
inline OwnedSnapshot capture(const Json& identity,const Json& metadata,
    const std::vector<DeviceSection>& sections,IdleBoundary boundary={},const CaptureOptions& options={}) {
  using namespace detail;validate_options(options);validate_boundary(boundary);
  require(identity.is_object()&&!identity.empty()&&metadata.is_object(),"checkpoint identity/metadata must be objects");
  require(sections.size()<=options.maximum_sections,"checkpoint excessive sections");
  const auto start=std::chrono::steady_clock::now();
  std::vector<LayoutSection> layout;layout.reserve(sections.size());
  for(const auto& s:sections){require(!s.bytes||s.data,"checkpoint null capture section");
    require(s.bytes<=u64(std::numeric_limits<std::uintptr_t>::max())-reinterpret_cast<std::uintptr_t>(s.data),"checkpoint capture address extent overflow");
    LayoutSection l{s.name,s.bytes,s.sparse,s.ranges};
    if(!s.sparse){require(s.ranges.empty(),"checkpoint nonsparse input must not offer ranges");if(s.bytes)l.ranges.push_back({0,s.bytes});}layout.push_back(std::move(l));}
  u64 payload=0;payload_layout(layout,payload);require(payload<=options.maximum_payload_bytes&&payload<=u64(std::numeric_limits<std::size_t>::max()),"checkpoint capture payload cap");
  Json description=Json::array();for(const auto&s:layout){Json ranges=Json::array();for(const auto&r:s.ranges)ranges.push_back({{"offset",r.offset},{"bytes",r.bytes}});
    description.push_back({{"name",s.name},{"bytes",s.bytes},{"sparse",s.sparse},{"ranges",std::move(ranges)}});}
  Json header{{"format","adaptive-idle-checkpoint-1"},{"identity",identity},{"metadata",metadata},
    {"boundary",{{"stage",0},{"jobs",0},{"native_count",0},{"native_cursor",0}}},{"sections",std::move(description)}};
  OwnedSnapshot out;out.header_=header.dump();require(out.header_.size()<=options.maximum_header_bytes,"checkpoint capture header cap");
  out.prefix_=prefix(out.header_.size(),payload,layout.size());out.payload_bytes_=payload;
  out.file_bytes_=add(add(add(out.prefix_.size(),out.header_.size()),payload),32);
  require(out.file_bytes_<=u64(std::numeric_limits<off_t>::max()),"checkpoint capture file offset extent");
  out.captured_bytes_=add(add(payload,out.header_.capacity()),sizeof(OwnedSnapshot));
  const u64 available=available_host_bytes();
  require(out.captured_bytes_<=available&&(!options.host_byte_budget||out.captured_bytes_<=options.host_byte_budget),"checkpoint full host snapshot is unaffordable");
  out.sections_=layout.size();out.chunk_bytes_=options.chunk_bytes;
  if(payload)out.payload_=std::unique_ptr<unsigned char[]>(new unsigned char[std::size_t(payload)]);
  u64 copied=0;
  for(std::size_t i=0;i<layout.size();++i)for(const auto& range:layout[i].ranges)
    for(u64 at=0;at<range.bytes;){const auto n=std::size_t(std::min<u64>(options.chunk_bytes,range.bytes-at));
      cuda_check(cudaMemcpy(out.payload_.get()+copied,offset(sections[i].data,range.offset+at),n,cudaMemcpyDeviceToHost),"checkpoint immutable capture copy");
      copied+=n;at+=n;if(options.progress)options.progress(copied,payload);}
  require(copied==payload,"checkpoint capture internal payload extent");
  out.capture_seconds_=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return out;
}
struct AsyncStatus {
  bool inflight=false,committed=false,failed=false;
  u64 captured_bytes=0,payload_bytes=0,written_bytes=0,total_bytes=0;
  double capture_seconds=0,write_seconds=0;
  std::string error;Receipt receipt;
};
class AsyncWriter {
  mutable std::mutex mutex_;std::mutex submission_mutex_;AsyncStatus status_;std::jthread worker_;
  static Receipt write_owned(const std::filesystem::path& path,const OwnedSnapshot& snapshot,
      const std::function<void(u64)>& progress) {
    using namespace detail;require(!path.empty()&&path.has_filename()&&!snapshot.header_.empty(),"checkpoint invalid asynchronous snapshot/destination");
    std::filesystem::path parent=path.parent_path();if(parent.empty())parent=".";
    static std::atomic<u64> sequence{0};
    const auto temporary=parent/(path.filename().string()+".async.tmp."+std::to_string(::getpid())+"."+std::to_string(sequence.fetch_add(1)));
    Fd file(::open(temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600));if(file.value<0)io_error("checkpoint asynchronous temporary open");
    bool published=false;
    try{
      Digest digest;u64 written=0;
      auto part=[&](const void*p,std::size_t n){write_all(file.value,p,n);digest.update(p,n);written+=n;progress(written);};
      part(snapshot.prefix_.data(),snapshot.prefix_.size());part(snapshot.header_.data(),snapshot.header_.size());
      for(u64 at=0;at<snapshot.payload_bytes_;){const auto n=std::size_t(std::min<u64>(snapshot.chunk_bytes_,snapshot.payload_bytes_-at));
        part(snapshot.payload_.get()+at,n);at+=n;}
      const auto sum=digest.finish();write_all(file.value,sum.data(),sum.size());written+=sum.size();progress(written);
      durable(file.value,"checkpoint asynchronous file fsync");
      Fd directory(::open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC));if(directory.value<0)io_error("checkpoint asynchronous directory open");
      if(::rename(temporary.c_str(),path.c_str())<0)io_error("checkpoint asynchronous atomic rename");published=true;
      durable(directory.value,"checkpoint asynchronous directory fsync");
      return{hex(sum),snapshot.file_bytes_,snapshot.payload_bytes_,snapshot.sections_,snapshot.chunk_bytes_,true};
    }catch(...){if(!published)::unlink(temporary.c_str());throw;}
  }
public:
  AsyncWriter()=default;AsyncWriter(const AsyncWriter&)=delete;AsyncWriter&operator=(const AsyncWriter&)=delete;
  ~AsyncWriter(){if(worker_.joinable())worker_.join();}
  bool try_submit(const std::filesystem::path& path,OwnedSnapshot&& snapshot,
      std::function<void()> prepare_storage={},std::function<void()> publish={},
      std::function<void(const std::string&)> abort={}){
    std::lock_guard submission_lock(submission_mutex_);
    {std::lock_guard lock(mutex_);if(status_.inflight)return false;}
    if(worker_.joinable())worker_.join();
    detail::require(!snapshot.header_.empty(),"checkpoint empty asynchronous submission");
    {std::lock_guard lock(mutex_);status_=AsyncStatus{};status_.inflight=true;status_.captured_bytes=snapshot.captured_bytes_;
      status_.payload_bytes=snapshot.payload_bytes_;status_.total_bytes=snapshot.file_bytes_;status_.capture_seconds=snapshot.capture_seconds_;}
    try{worker_=std::jthread([this,path,snapshot=std::move(snapshot),
                            prepare_storage=std::move(prepare_storage),publish=std::move(publish),abort]() mutable {
      const auto start=std::chrono::steady_clock::now();Receipt receipt;std::string error;
      try{if(prepare_storage)prepare_storage();
        receipt=write_owned(path,snapshot,[this](u64 written){std::lock_guard lock(mutex_);status_.written_bytes=written;});
        if(publish)publish();}
      catch(const std::exception& e){error=e.what();}catch(...){error="checkpoint asynchronous unknown failure";}
      if(!error.empty()&&abort){try{abort(error);}catch(...){error+="; checkpoint transaction abort callback failed";}}
      const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      // Release full host ownership before publishing idle/commit status.
      snapshot=OwnedSnapshot{};
      std::lock_guard lock(mutex_);status_.write_seconds=elapsed;status_.receipt=std::move(receipt);status_.error=std::move(error);
      status_.failed=!status_.error.empty();status_.committed=!status_.failed;status_.inflight=false;
    });}catch(...){if(abort){try{abort("checkpoint background thread creation failed");}catch(...){}}
      std::lock_guard lock(mutex_);status_.inflight=false;status_.failed=true;status_.error="checkpoint background thread creation failed";throw;}
    return true;
  }
  AsyncStatus poll()const{std::lock_guard lock(mutex_);return status_;}
  AsyncStatus finish(){std::lock_guard submission_lock(submission_mutex_);if(worker_.joinable())worker_.join();return poll();}
};
} // namespace class_conversion_adaptive::checkpoint
