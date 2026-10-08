#pragma once
// Host-only coordinator payload. Native/model bytes are opaque; no CPU model math.
#include "class_study_convert.hpp"
#include <openssl/evp.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <streambuf>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
namespace class_study::checkpoint {
using Json=nlohmann::json;using u64=std::uint64_t;namespace fs=std::filesystem;
inline void require(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
inline Json binary(const std::string& s){return Json::binary(std::vector<std::uint8_t>(s.begin(),s.end()));}
inline std::string bytes(const Json& j){require(j.is_binary(),"study checkpoint byte field is not binary");const auto& b=j.get_binary();return std::string(b.begin(),b.end());}
namespace detail {
inline void io_error(const char*w){throw std::runtime_error(std::string(w)+": "+std::strerror(errno));}
struct Fd {int n=-1;explicit Fd(int v):n(v){}~Fd(){if(n>=0)::close(n);}Fd(const Fd&)=delete;};
inline void write_all(int f,const void*p,std::size_t n){auto*b=static_cast<const unsigned char*>(p);while(n){auto k=::write(f,b,n);if(k<0){if(errno==EINTR)continue;io_error("study checkpoint write");}require(k>0,"study checkpoint short write");b+=k;n-=std::size_t(k);}}
inline void read_all(int f,void*p,std::size_t n){auto*b=static_cast<unsigned char*>(p);while(n){auto k=::read(f,b,n);if(k<0){if(errno==EINTR)continue;io_error("study checkpoint read");}require(k>0,"study checkpoint truncated");b+=k;n-=std::size_t(k);}}
inline void sync(int f){while(::fsync(f)<0){if(errno==EINTR)continue;io_error("study checkpoint fsync");}}
inline void sync_dir(const fs::path&p){Fd d(::open(p.c_str(),O_RDONLY|O_DIRECTORY));if(d.n<0)io_error("study checkpoint directory open");sync(d.n);}
struct Digest{std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> c{EVP_MD_CTX_new(),EVP_MD_CTX_free};Digest(){require(bool(c)&&EVP_DigestInit_ex(c.get(),EVP_sha256(),nullptr)==1,"study digest init");}void put(const void*p,std::size_t n){require(EVP_DigestUpdate(c.get(),p,n)==1,"study digest update");}std::array<unsigned char,32>end(){std::array<unsigned char,32>b{};unsigned n=0;require(EVP_DigestFinal_ex(c.get(),b.data(),&n)==1&&n==32,"study digest final");return b;}};
inline std::string hex(const std::array<unsigned char,32>&b){const char*d="0123456789abcdef";std::string s(64,'0');for(std::size_t i=0;i<32;++i){s[2*i]=d[b[i]>>4];s[2*i+1]=d[b[i]&15];}return s;}
inline u64 available(){std::ifstream f("/proc/meminfo");std::string k,unit;u64 n=0;while(f>>k>>n>>unit)if(k=="MemAvailable:"){require(n<=UINT64_MAX/1024,"host available extent overflow");return n*1024;}throw std::runtime_error("host available memory unavailable");}
inline void bound(u64 n,u64 budget){require(n<=std::numeric_limits<std::size_t>::max(),"study checkpoint host extent");require(n<=(budget?std::min(budget,available()):available()),"study checkpoint host memory budget refusal");}
struct CountingBuffer:std::streambuf {u64 count=0;std::streamsize xsputn(const char*,std::streamsize n)override{require(n>=0&&u64(n)<=UINT64_MAX-count,"study CBOR extent overflow");count+=u64(n);return n;}int_type overflow(int_type c)override{if(!traits_type::eq_int_type(c,traits_type::eof())){require(count<UINT64_MAX,"study CBOR extent overflow");++count;}return traits_type::not_eof(c);}};
inline u64 cbor_extent(const Json&j){CountingBuffer b;std::ostream out(&b);out.exceptions(std::ios::badbit|std::ios::failbit);Json::to_cbor(j,nlohmann::detail::output_adapter<char>(out));return b.count;}
inline std::array<unsigned char,16> prefix(u64 n){std::array<unsigned char,16>b{};std::memcpy(b.data(),"STUDYCP1",8);for(unsigned i=0;i<8;++i)b[8+i]=static_cast<unsigned char>(n>>(8*i));return b;}
inline u64 extent(const std::array<unsigned char,16>&p){u64 n=0;for(unsigned i=0;i<8;++i)n|=u64(p[8+i])<<(8*i);return n;}
inline std::string write_bundle(const fs::path&p,const std::vector<std::uint8_t>&data){Fd f(::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600));if(f.n<0)io_error("study bundle open");auto h=prefix(data.size());Digest d;d.put(h.data(),h.size());write_all(f.n,h.data(),h.size());for(std::size_t offset=0;offset<data.size();){auto n=std::min<std::size_t>(1u<<20,data.size()-offset);d.put(data.data()+offset,n);write_all(f.n,data.data()+offset,n);offset+=n;}auto hash=d.end();write_all(f.n,hash.data(),hash.size());sync(f.n);return hex(hash);}
inline void atomic_manifest(const fs::path&root,const Json&j,const std::string&generation){auto p=root/(".current-"+generation);auto data=j.dump()+"\n";bool renamed=false;try{Fd f(::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600));if(f.n<0)io_error("study manifest open");write_all(f.n,data.data(),data.size());sync(f.n);if(::rename(p.c_str(),(root/"current.json").c_str())<0)io_error("study manifest rename");renamed=true;sync_dir(root);}catch(...){if(!renamed)::unlink(p.c_str());throw;}}
inline Json read_manifest(const fs::path&p){Fd f(::open(p.c_str(),O_RDONLY));if(f.n<0)io_error("study manifest read");struct stat s{};require(::fstat(f.n,&s)==0&&s.st_size>0&&s.st_size<(1<<20),"study manifest extent");std::string b(std::size_t(s.st_size),'\0');read_all(f.n,b.data(),b.size());return Json::parse(b);}
} // detail
struct Loaded {Json state;std::string engine_snapshot_path,generation;};
struct Status {bool in_flight=false,committed=false,failed=false;std::string generation,error;u64 host_bytes=0;};
inline constexpr const char* study_state_format="resident-study-checkpoint-1";
class Coordinator {
 struct Shared {fs::path root;std::string state_format;u64 host_budget=0;std::atomic<u64> serial{0};mutable std::mutex mutex;Status status;};
 struct Capture {std::shared_ptr<Shared> owner;std::string generation;fs::path directory;std::vector<std::uint8_t> bytes;bool engine=false;std::string hash;};
 std::shared_ptr<Shared> p_;std::jthread writer_;
 static void mark(const std::shared_ptr<Shared>&p,const Status&s){std::lock_guard lock(p->mutex);p->status=s;}
 std::shared_ptr<Capture> snapshot(const Json&state,bool engine,u64 gpu_bytes=0){require(state.is_object()&&state.at("format")==p_->state_format,"experiment checkpoint state format");auto c=std::make_shared<Capture>();c->owner=p_;c->engine=engine;auto t=std::chrono::steady_clock::now().time_since_epoch().count();c->generation="generation-"+std::to_string(::getpid())+"-"+std::to_string(t)+"-"+std::to_string(p_->serial.fetch_add(1));c->directory=p_->root/c->generation;
  // The owned CBOR snapshot is created while its producing thread is stopped at
  // the stage boundary. Writers never read changing study/trainer objects.
    const auto n=detail::cbor_extent(state);u64 remaining=p_->host_budget;if(remaining){require(gpu_bytes<remaining,"study checkpoint combined host memory budget refusal");remaining-=gpu_bytes;}detail::bound(n,remaining);c->bytes.reserve(std::size_t(n));Json::to_cbor(state,c->bytes);require(c->bytes.size()==n,"study CBOR preflight extent differs");return c;}
 static void prepare(const std::shared_ptr<Capture>&c){mark(c->owner,{true,false,false,c->generation,{},u64(c->bytes.size())});try{fs::create_directories(c->owner->root);require(fs::create_directory(c->directory),"study generation already exists");c->hash=detail::write_bundle(c->directory/"host.cbor",c->bytes);detail::sync_dir(c->directory);detail::sync_dir(c->owner->root);}catch(const std::exception&e){mark(c->owner,{false,false,true,c->generation,e.what(),u64(c->bytes.size())});throw;}}
 static void publish(const std::shared_ptr<Capture>&c){try{if(c->engine){struct stat s{};require(::stat((c->directory/"engine.ckpt").c_str(),&s)==0&&s.st_size>0,"study generation engine snapshot absent");}Json m={{"format","resident-study-generation-1"},{"generation",c->generation},{"host_bundle","host.cbor"},{"host_sha256",c->hash},{"host_payload_bytes",c->bytes.size()},{"engine_snapshot",c->engine?Json("engine.ckpt"):Json(nullptr)}};detail::sync_dir(c->directory);detail::atomic_manifest(c->owner->root,m,c->generation);mark(c->owner,{false,true,false,c->generation,{},u64(c->bytes.size())});}catch(const std::exception&e){mark(c->owner,{false,false,true,c->generation,e.what(),u64(c->bytes.size())});throw;}}
 public:
 // Each workflow owns its versioned payload schema. The common transport does
 // not reinterpret one workflow's state as another's or invent absent state.
 explicit Coordinator(fs::path root,u64 host_budget=0,std::string state_format=study_state_format):p_(std::make_shared<Shared>()){
   require(root.is_absolute(),"experiment checkpoint path must be absolute");
   require(!state_format.empty()&&state_format.find('\0')==std::string::npos,"experiment checkpoint format identifier");
   p_->root=std::move(root);p_->host_budget=host_budget;p_->state_format=std::move(state_format);
 }
 ~Coordinator(){finish();}Coordinator(const Coordinator&)=delete;
 Status poll()const{std::lock_guard lock(p_->mutex);return p_->status;}
 void finish(){if(writer_.joinable())writer_.join();}
 CheckpointPublication transaction(const Json&state,u64 gpu_bytes=0){finish();auto c=snapshot(state,true,gpu_bytes);return{(c->directory/"engine.ckpt").string(),[c]{prepare(c);},[c]{publish(c);},[c](const std::string&error){mark(c->owner,{false,false,true,c->generation,error,u64(c->bytes.size())});}};}
 bool save_host(const Json&state){if(poll().in_flight)return false;finish();auto c=snapshot(state,false);mark(p_,{true,false,false,c->generation,{},u64(c->bytes.size())});try{writer_=std::jthread([c]{try{prepare(c);publish(c);}catch(...){/* failure is durable status; previous manifest remains */}});}catch(const std::exception&e){mark(p_,{false,false,true,c->generation,e.what(),u64(c->bytes.size())});throw;}return true;}
 static Loaded load(const fs::path&root,u64 host_budget=0,const std::string& state_format=study_state_format){require(root.is_absolute(),"experiment resume path must be absolute");auto m=detail::read_manifest(root/"current.json");require(m.at("format")=="resident-study-generation-1","study generation format");const auto generation=m.at("generation").get<std::string>();require(generation.starts_with("generation-")&&generation.find_first_of("/\\")==std::string::npos,"study generation path");require(m.at("host_bundle")=="host.cbor","study host bundle path");auto dir=root/generation;detail::Fd f(::open((dir/"host.cbor").c_str(),O_RDONLY));if(f.n<0)detail::io_error("study bundle read");std::array<unsigned char,16> h{};detail::read_all(f.n,h.data(),h.size());require(std::memcmp(h.data(),"STUDYCP1",8)==0,"study bundle magic");u64 n=detail::extent(h);require(n<=UINT64_MAX-48&&m.at("host_payload_bytes").is_number_unsigned()&&m.at("host_payload_bytes").get<u64>()==n,"study bundle payload extent");detail::bound(n,host_budget);struct stat s{};require(::fstat(f.n,&s)==0&&s.st_size>=0&&u64(s.st_size)==n+48,"study bundle exact file extent");std::vector<std::uint8_t>data(std::size_t(n),0);detail::Digest hash;hash.put(h.data(),h.size());for(std::size_t offset=0;offset<data.size();){auto count=std::min<std::size_t>(1u<<20,data.size()-offset);detail::read_all(f.n,data.data()+offset,count);hash.put(data.data()+offset,count);offset+=count;}std::array<unsigned char,32>stored{};detail::read_all(f.n,stored.data(),stored.size());const auto actual=hash.end();require(stored==actual&&m.at("host_sha256")==detail::hex(actual),"study bundle integrity");Loaded out;out.generation=generation;out.state=Json::from_cbor(data,true,true);require(out.state.is_object()&&out.state.at("format")==state_format,"experiment state format differs");if(!m.at("engine_snapshot").is_null()){require(m.at("engine_snapshot")=="engine.ckpt","study engine snapshot path");out.engine_snapshot_path=(dir/"engine.ckpt").string();}return out;}
};
} // namespace class_study::checkpoint
