#pragma once
// Stable proposal ABI and host-only registry. Proposals never contain labels.
#include <cuda.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#ifdef __linux__
#include <poll.h>
#include <sys/inotify.h>
#include <unistd.h>
#endif

namespace class_conversion_adaptive::proof_modules {
using u32=std::uint32_t;using u64=std::uint64_t;
constexpr u32 interface_version=1;
struct Session {u64 token=0;u32 device=0,features=0,classes=0,nodes=0,trees=0;};
struct InputV1 {
  u32 abi_version=interface_version,struct_bytes=0;
  u64 session_token=0;
  u32 features=0,classes=0,nodes=0,trees=0,state_capacity=0,reserved=0;
  const std::int32_t *feature=nullptr,*left=nullptr,*right=nullptr,*channels=nullptr,*residual=nullptr;
  const u32 *minimum=nullptr,*maximum=nullptr,*state_ids=nullptr,*parent_predicates=nullptr;
};
struct ProposalV1 {u32 predicate=UINT32_MAX,flags=0;};
static_assert(sizeof(InputV1)==112&&sizeof(ProposalV1)==8);
static_assert(std::is_standard_layout_v<InputV1>&&std::is_trivially_copyable_v<InputV1>);
// Cubin entry: extern "C" __global__ void entry(InputV1,u32 jobs,ProposalV1*).
// parent_predicates/state_ids/output are per job; residual is [state_id*T+t].
// All input pointers are borrowed read-only for that synchronized batch. This
// is an audited/trusted CUDA plugin ABI, not a sandbox for hostile GPU code.

namespace host_detail {
using Json=nlohmann::json;
inline void driver(CUresult code,const char* operation){
  if(code==CUDA_SUCCESS)return;const char*text=nullptr;cuGetErrorString(code,&text);
  throw std::runtime_error(std::string(operation)+": "+(text?text:"CUDA Driver error"));
}
inline bool power2(u32 n){return n&&!(n&(n-1));}
inline u32 floor_power2(u32 n){u32 p=1;while(p<=n/2)p*=2;return p;}
inline u32 ceil_power2(u32 n){u32 p=1;while(p<n&&p<=UINT32_MAX/2)p*=2;return p;}
inline u64 integer(const Json& j,const char*key,u64 fallback,bool optional){
  if(!j.contains(key)){if(optional)return fallback;throw std::runtime_error(std::string("missing module field ")+key);}
  const auto&v=j.at(key);
  if(!v.is_number_unsigned()&&!v.is_number_integer())throw std::runtime_error(std::string("noninteger module field ")+key);
  if(v.is_number_integer()&&!v.is_number_unsigned()&&v.get<std::int64_t>()<0)throw std::runtime_error(std::string("negative module field ")+key);
  return v.get<u64>();
}
inline std::string text(const Json&j,const char*key){
  if(!j.contains(key)||!j.at(key).is_string())throw std::runtime_error(std::string("invalid module string ")+key);
  auto s=j.at(key).get<std::string>();if(s.empty()||s.size()>4096)throw std::runtime_error(std::string("module string extent ")+key);return s;
}
inline std::vector<char> read_once(const std::filesystem::path&p,u64 maximum){
  const auto n=std::filesystem::file_size(p);if(!n||n>maximum)throw std::runtime_error("module file extent");
  std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("module file open");
  std::vector<char>b(static_cast<std::size_t>(n));in.read(b.data(),std::streamsize(n));
  if(in.gcount()!=std::streamsize(n))throw std::runtime_error("module file short read");
  char extra;if(in.get(extra))throw std::runtime_error("module file changed size during read");return b;
}
struct Loaded {
  CUmodule module=nullptr;CUfunction function=nullptr;CUcontext context=nullptr;
  std::string name,entry;u64 version=0,bytes=0;
  u32 threads=0,automatic_threads=1,maximum_threads=1,maximum_grid=1;
  ~Loaded(){if(module)cuModuleUnload(module);}
};
} // namespace host_detail

class Registry {
 public:
  explicit Registry(Session binding,u64 maximum_module_bytes=16ull<<20,u32 maximum_jobs=65536)
      : binding_(binding),maximum_module_bytes_(maximum_module_bytes),maximum_jobs_(maximum_jobs){
    if(!binding_.features||!binding_.classes||!maximum_module_bytes_||!maximum_jobs_)
      throw std::invalid_argument("proof-module session/limits");
    if(!binding_.token){static std::atomic<u64>next{1};binding_.token=next.fetch_add(1);if(!binding_.token)throw std::runtime_error("proof-module token exhaustion");}
  }
  Registry(const Registry&)=delete;Registry&operator=(const Registry&)=delete;
  ~Registry(){stop_watching();if(active_){if(active_->context){CUcontext current=nullptr;if(cuCtxGetCurrent(&current)==CUDA_SUCCESS&&current==active_->context)cuCtxSynchronize();}active_.reset();}}
  Session session()const{return binding_;}
  bool active()const{return bool(active_);}
  void request(const std::filesystem::path&manifest){
    std::lock_guard lock(mutex_);
    if(std::find(pending_.begin(),pending_.end(),manifest)!=pending_.end())return;
    if(pending_.size()==64){pending_.pop_front();++dropped_requests_;}
    pending_.push_back(manifest);
  }
  // Event notifications only. Existing files must be explicitly requested.
  // No CUDA calls, file reads or hashes occur in the notification thread.
  void start_watching(const std::filesystem::path&directory){
    if(watcher_.joinable())throw std::runtime_error("proof-module watcher already active");
#ifdef __linux__
    if(!std::filesystem::is_directory(directory))throw std::runtime_error("proof-module watch directory");
    const int fd=inotify_init1(IN_NONBLOCK|IN_CLOEXEC);if(fd<0)throw std::runtime_error("proof-module inotify initialization");
    if(inotify_add_watch(fd,directory.c_str(),IN_CLOSE_WRITE|IN_MOVED_TO)<0){close(fd);throw std::runtime_error("proof-module inotify watch");}
    stop_=false;watcher_=std::thread([this,directory,fd]{
      alignas(inotify_event)char bytes[8192];
      while(!stop_.load()){
        pollfd p{fd,POLLIN,0};const int ready=poll(&p,1,250);if(ready<=0)continue;
        const auto n=read(fd,bytes,sizeof(bytes));if(n<=0)continue;
        for(std::size_t at=0;at<static_cast<std::size_t>(n);){
          const auto*e=reinterpret_cast<const inotify_event*>(bytes+at);
          if(e->len&&!(e->mask&IN_ISDIR)&&(e->mask&(IN_CLOSE_WRITE|IN_MOVED_TO))){
            std::filesystem::path name(e->name);if(name.extension()==".json"){++notifications_;request(directory/name);}
          }
          at+=sizeof(inotify_event)+e->len;
        }
      }
      close(fd);
    });
    watch_directory_=directory.string();
#else
    (void)directory;throw std::runtime_error("proof-module directory watch unsupported; use explicit request");
#endif
  }
  void stop_watching(){stop_=true;if(watcher_.joinable())watcher_.join();}
  // Call only at an idle synchronized host boundary with native outputs consumed.
  // Invalid manifests/cubins/functions preserve the active owner and queued work.
  bool activate_pending_at_idle(){
    std::filesystem::path path;
    {std::lock_guard lock(mutex_);if(pending_.empty())return false;path=pending_.front();pending_.pop_front();}
    ++attempts_;last_request_=path.string();
    CUcontext current=nullptr;host_detail::driver(cuCtxGetCurrent(&current),"proof-module current context");
    if(!current){reject("no active CUDA context");return false;}
    CUdevice device;host_detail::driver(cuCtxGetDevice(&device),"proof-module device");
    if(u32(device)!=binding_.device||(active_&&active_->context!=current)){reject("proof-module session CUDA context mismatch");return false;}
    // A prior CUDA failure is a conversion failure, not an invalid-module refusal.
    host_detail::driver(cuCtxSynchronize(),"proof-module activation boundary");
    try{
      const auto manifest=host_detail::read_once(path,64ull<<10);++manifest_reads_;
      const auto j=host_detail::Json::parse(manifest.begin(),manifest.end());
      const std::vector<std::string>keys={"format","abi_version","name","version","cubin_path","entry","threads"};
      if(!j.is_object())throw std::runtime_error("module manifest object");
      for(auto it=j.begin();it!=j.end();++it)if(std::find(keys.begin(),keys.end(),it.key())==keys.end())throw std::runtime_error("unsupported module manifest field "+it.key());
      if(host_detail::text(j,"format")!="adaptive-proof-module-v1"||host_detail::integer(j,"abi_version",0,false)!=interface_version)throw std::runtime_error("incompatible proof-module ABI");
      auto next=std::make_unique<host_detail::Loaded>();next->context=current;
      next->name=host_detail::text(j,"name");next->entry=host_detail::text(j,"entry");next->version=host_detail::integer(j,"version",0,false);
      if(!next->version||next->name.size()>128||next->entry.size()>128)throw std::runtime_error("module name/version/entry extent");
      for(std::size_t i=0;i<next->entry.size();++i){const char c=next->entry[i];if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||c=='_'||(i&&c>='0'&&c<='9')))throw std::runtime_error("module entry identifier");}
      if(active_&&next->name==active_->name&&next->version<=active_->version)throw std::runtime_error("module version is not newer");
      const auto requested=host_detail::integer(j,"threads",0,true);if(requested>UINT32_MAX||(requested&&!host_detail::power2(u32(requested))))throw std::runtime_error("module threads must be zero or a power of two");next->threads=u32(requested);
      std::filesystem::path cubin=host_detail::text(j,"cubin_path");if(cubin.is_relative())cubin=path.parent_path()/cubin;
      if(cubin.extension()!=".cubin")throw std::runtime_error("module must be a precompiled cubin");
      const auto bytes=host_detail::read_once(cubin,maximum_module_bytes_);++module_reads_;next->bytes=bytes.size();
      host_detail::driver(cuModuleLoadDataEx(&next->module,bytes.data(),0,nullptr,nullptr),"proof-module load");
      host_detail::driver(cuModuleGetFunction(&next->function,next->module,next->entry.c_str()),"proof-module entry");
      const std::size_t offsets[]={0,sizeof(InputV1),sizeof(InputV1)+8};
      const std::size_t sizes[]={sizeof(InputV1),sizeof(u32),sizeof(ProposalV1*)};
      for(std::size_t i=0;i<3;++i){std::size_t offset=0,size=0;host_detail::driver(cuFuncGetParamInfo(next->function,i,&offset,&size),"proof-module parameter ABI");if(offset!=offsets[i]||size!=sizes[i])throw std::runtime_error("proof-module kernel parameter ABI mismatch");}
      {std::size_t count=0;host_detail::driver(cuFuncGetParamCount(next->function,&count),"proof-module parameter count");if(count!=3)throw std::runtime_error("proof-module kernel has incompatible parameter count");}
      int function_threads=0,device_threads=0,grid=0;host_detail::driver(cuFuncGetAttribute(&function_threads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,next->function),"proof-module function threads");
      host_detail::driver(cuDeviceGetAttribute(&device_threads,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,device),"proof-module device threads");
      host_detail::driver(cuDeviceGetAttribute(&grid,CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,device),"proof-module grid bound");
      if(function_threads<=0||device_threads<=0||grid<=0)throw std::runtime_error("module CUDA resource bounds");
      next->maximum_threads=host_detail::floor_power2(u32(std::min(function_threads,device_threads)));next->maximum_grid=u32(grid);
      if(next->threads>next->maximum_threads)throw std::runtime_error("module fixed thread width exceeds device/function resources");
      int minimum_grid=0,suggested=0;host_detail::driver(cuOccupancyMaxPotentialBlockSize(&minimum_grid,&suggested,next->function,nullptr,0,int(next->maximum_threads)),"proof-module occupancy");
      if(suggested<=0)throw std::runtime_error("module has no occupancy-legal launch width");
      next->automatic_threads=host_detail::floor_power2(std::min(u32(suggested),next->maximum_threads));
      int blocks=0;host_detail::driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,next->function,int(next->threads?next->threads:next->automatic_threads),0),"proof-module occupancy bound");
      if(blocks<=0)throw std::runtime_error("module has no active blocks at requested width");
      active_=std::move(next);++activations_;++generation_;last_error_.clear();return true;
    }catch(const std::exception&e){reject(e.what());return false;}
  }
  bool propose(InputV1 input,u32 jobs,ProposalV1*output,CUstream stream=nullptr){
    if(!active_||!jobs)return false;
    if(input.abi_version!=interface_version||input.struct_bytes!=sizeof(InputV1)||input.reserved||input.session_token!=binding_.token||
       input.features!=binding_.features||input.classes!=binding_.classes||input.nodes!=binding_.nodes||input.trees!=binding_.trees||
       jobs>maximum_jobs_||!output||!input.state_ids||!input.parent_predicates||!input.state_capacity||
       (input.nodes&&(!input.feature||!input.left||!input.right||!input.minimum||!input.maximum))||
       (input.trees&&(!input.residual||!input.channels)))throw std::invalid_argument("proof-module proposal binding/extent");
    CUcontext current=nullptr;host_detail::driver(cuCtxGetCurrent(&current),"proof-module proposal context");if(current!=active_->context)throw std::runtime_error("proof-module proposal context changed");
    const u32 threads=active_->threads?active_->threads:std::min(active_->automatic_threads,host_detail::ceil_power2(jobs));
    const u64 blocks=(u64(jobs)+threads-1)/threads;if(blocks>active_->maximum_grid)throw std::invalid_argument("proof-module launch grid extent");
    void*args[]={&input,&jobs,&output};host_detail::driver(cuLaunchKernel(active_->function,u32(blocks),1,1,threads,1,1,0,stream,args,nullptr),"proof-module proposal launch");
    ++launches_;proposed_jobs_+=jobs;last_threads_=threads;return true;
  }
  nlohmann::json diagnostics()const{
    std::lock_guard lock(mutex_);
    return {{"interface_version",interface_version},{"session_token",binding_.token},{"active",bool(active_)},{"generation",generation_},
      {"name",active_?active_->name:""},{"version",active_?active_->version:0},{"entry",active_?active_->entry:""},
      {"cubin_bytes",active_?active_->bytes:0},{"requested_threads",active_?active_->threads:0},{"last_launch_threads",last_threads_},
      {"maximum_jobs",maximum_jobs_},{"activation_attempts",attempts_},{"activations",activations_},{"rejections",rejections_},
      {"manifest_content_reads",manifest_reads_},{"module_content_reads",module_reads_},{"launches",launches_},{"proposed_jobs",proposed_jobs_},
      {"watch_notifications",notifications_.load()},{"dropped_requests",dropped_requests_},{"pending_requests",pending_.size()},
      {"watch_directory",watch_directory_},{"last_request",last_request_},{"last_error",last_error_},
      {"class_authority",false},{"driver_module_memory_not_in_arena_budget",true}};
  }
 private:
  void reject(const std::string&why){++rejections_;last_error_=why;}
  Session binding_;u64 maximum_module_bytes_;u32 maximum_jobs_;
  std::unique_ptr<host_detail::Loaded>active_;
  mutable std::mutex mutex_;std::deque<std::filesystem::path>pending_;
  std::atomic<bool>stop_{false};std::atomic<u64>notifications_{0};std::thread watcher_;
  std::string watch_directory_,last_request_,last_error_;
  u64 attempts_=0,activations_=0,generation_=0,rejections_=0,manifest_reads_=0,module_reads_=0,launches_=0,proposed_jobs_=0,dropped_requests_=0;
  u32 last_threads_=0;
};
} // namespace class_conversion_adaptive::proof_modules
