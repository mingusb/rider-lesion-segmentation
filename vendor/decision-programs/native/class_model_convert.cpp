#include "class_study_convert.hpp"
#include "class_study_options.hpp"
#include "class_study_native_gate.hpp"
#include "class_io.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <utility>

namespace {
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
using Size=std::uint64_t;
volatile std::sig_atomic_t interruption=0;
void record_interruption(int signal){interruption=signal;}
volatile std::sig_atomic_t checkpoint_request=0;
void record_checkpoint_request(int){checkpoint_request=1;}
void require(bool condition,const std::string& message) {if(!condition)throw std::runtime_error(message);}
void cuda_check(cudaError_t status,const char* operation) {
  if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void require_path(const std::string& text,const char* role) {
  require(std::filesystem::path(text).is_absolute(),std::string(role)+" must be absolute");
  require(text.find('\0')==std::string::npos,std::string(role)+" contains NUL");
}
void require_sha(const std::string& sha,const char* role) {
  require(sha.size()==64,std::string(role)+" must contain 64 lowercase hexadecimal characters");
  for(char c:sha)require((c>='0'&&c<='9')||(c>='a'&&c<='f'),std::string(role)+" must contain lowercase hexadecimal characters");
}
Size decimal(const Json& value,const char* role) {
  require(value.is_string(),std::string(role)+" must be a decimal string");auto text=value.get<std::string>();Size result=0;
  auto parsed=std::from_chars(text.data(),text.data()+text.size(),result);
  require(!text.empty()&&parsed.ec==std::errc{}&&parsed.ptr==text.data()+text.size(),std::string(role)+" is invalid");return result;
}
struct SourceShape {std::uint32_t features,classes;std::string objective;};
SourceShape source_shape(const std::string& bytes) {
  auto source=Json::parse(bytes);const auto& learner=source.at("learner");const auto& shape=learner.at("learner_model_param");
  require(learner.at("gradient_booster").at("name")=="gbtree","source must be gbtree");
  auto objective=learner.at("objective").at("name").get<std::string>();
  require(objective=="multi:softprob"||objective=="multi:softmax","source must use multiclass softprob or softmax");
  Size F=decimal(shape.at("num_feature"),"num_feature"),K=decimal(shape.at("num_class"),"num_class");
  require(F>0&&F<=INT32_MAX&&K>=2&&K<=INT32_MAX,"source feature/class metadata exceeds supported representation");
  if(shape.contains("num_target"))require(decimal(shape.at("num_target"),"num_target")==1,"source must have one target");
  const auto& version=source.at("version");require(version.is_array()&&version.size()==3,"source version must have three integers");
  std::array<Size,3> expected{3,4,1};
  for(std::size_t i=0;i<3;++i)require(class_study::option_integer(version[i],"source version",INT32_MAX)==expected[i],"source requires XGBoost 3.4.1");
  return {std::uint32_t(F),std::uint32_t(K),std::move(objective)};
}
// One stable initial read per file. Library metadata is checked around dlopen;
// the conversion loop performs no filesystem reads, stat calls or hashes.
class BoundFile {
 public:
  explicit BoundFile(std::string path):path_(std::move(path)) {
    fd_=open(path_.c_str(),O_RDONLY|O_CLOEXEC);require(fd_>=0,"cannot open "+path_+": "+std::strerror(errno));
    try{require(fstat(fd_,&initial_)==0&&S_ISREG(initial_.st_mode)&&initial_.st_size>=0,"input must be a readable regular file");}
    catch(...){close(std::exchange(fd_,-1));throw;}
  }
  ~BoundFile(){if(fd_>=0)close(fd_);}
  BoundFile(const BoundFile&)=delete;BoundFile&operator=(const BoundFile&)=delete;
  std::string bytes() {
    require(!consumed_,"bound file already consumed");consumed_=true;
    require(Size(initial_.st_size)<=SIZE_MAX,"file exceeds host extent");std::string output(std::size_t(initial_.st_size),'\0');
    std::size_t done=0;while(done<output.size()) {
      auto count=read(fd_,output.data()+done,std::min<std::size_t>(output.size()-done,1u<<20));
      if(count<0&&errno==EINTR)continue;require(count>0,"short/error input read: "+path_);done+=std::size_t(count);
    }
    char excess;ssize_t extra;do{extra=read(fd_,&excess,1);}while(extra<0&&errno==EINTR);
    require(extra==0,"input grew or failed during read: "+path_);unchanged();return output;
  }
  void unchanged()const {
    struct stat held{},named{};require(fstat(fd_,&held)==0&&stat(path_.c_str(),&named)==0,"bound input metadata unavailable: "+path_);
    auto same=[&](const struct stat& now){return now.st_dev==initial_.st_dev&&now.st_ino==initial_.st_ino&&now.st_size==initial_.st_size&&
      now.st_mtim.tv_sec==initial_.st_mtim.tv_sec&&now.st_mtim.tv_nsec==initial_.st_mtim.tv_nsec&&
      now.st_ctim.tv_sec==initial_.st_ctim.tv_sec&&now.st_ctim.tv_nsec==initial_.st_ctim.tv_nsec;};
    require(same(held)&&same(named),"input changed during initial binding: "+path_);
  }
  Size size()const{return Size(initial_.st_size);}
 private:
  std::string path_;int fd_=-1;struct stat initial_{};bool consumed_=false;
};
class ResidentModel {
 public:
  ResidentModel(const std::string& library,const std::string& model,SourceShape shape):shape_(std::move(shape)) {
    static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559);
    library_=dlopen(library.c_str(),RTLD_NOW|RTLD_LOCAL);
    require(library_!=nullptr,std::string("dlopen native library: ")+(library_?"":dlerror()));
    try {
      last_error_=symbol<const char*(*)()>("XGBGetLastError");std::array<int,3> version{};
      symbol<void(*)(int*,int*,int*)>("XGBoostVersion")(&version[0],&version[1],&version[2]);
      require(version==std::array<int,3>{3,4,1},"native library requires XGBoost 3.4.1 exactly");
      const char* info=nullptr;check(symbol<int(*)(const char**)>("XGBuildInfo")(&info),"XGBuildInfo");
      require(info!=nullptr,"native build information missing");build_info_=Json::parse(info);
      require(build_info_.at("USE_CUDA").is_boolean()&&build_info_.at("USE_CUDA").get<bool>(),"native library has no CUDA support");
      free_=symbol<int(*)(void*)>("XGBoosterFree");cuda_check(cudaSetDevice(0),"select native CUDA device");
      check(symbol<int(*)(const void*[],Size,void**)>("XGBoosterCreate")(nullptr,0,&booster_),"XGBoosterCreate");
      check(symbol<int(*)(void*,const void*,Size)>("XGBoosterLoadModelFromBuffer")(booster_,model.data(),model.size()),"XGBoosterLoadModelFromBuffer");
      auto parameter=symbol<int(*)(void*,const char*,const char*)>("XGBoosterSetParam");
      check(parameter(booster_,"device","cuda:0"),"set native CUDA device");check(parameter(booster_,"nthread","4"),"set native threads");
      Size features=0;check(symbol<int(*)(void*,Size*)>("XGBoosterGetNumFeature")(booster_,&features),"XGBoosterGetNumFeature");
      require(features==shape_.features,"loaded native feature count differs from source");
      Size config_size=0;const char* config=nullptr;
      check(symbol<int(*)(void*,Size*,const char**)>("XGBoosterSaveJsonConfig")(booster_,&config_size,&config),"XGBoosterSaveJsonConfig");
      require(config&&config_size>0&&config_size<=SIZE_MAX,"native configuration missing");
      auto configuration=Json::parse(std::string(config,std::size_t(config_size)));const auto& learner=configuration.at("learner");
      require(learner.at("generic_param").at("device")=="cuda:0"&&learner.at("gradient_booster").at("name")=="gbtree"&&
        learner.at("objective").at("name")==shape_.objective&&
        decimal(learner.at("learner_model_param").at("num_class"),"native num_class")==shape_.classes,
        "loaded native objective/classes/booster/device differ from source");
      predict_=symbol<Predict>("XGBoosterPredictFromCudaArray");cuda_check(cudaDeviceSynchronize(),"synchronize native model setup");
    } catch(...){release();throw;}
  }
  ~ResidentModel(){release();}
  ResidentModel(const ResidentModel&)=delete;ResidentModel&operator=(const ResidentModel&)=delete;
  const float* predict(const float* input,Size rows,bool margin) {
    require(input&&rows>0&&rows<=UINT64_MAX/shape_.features&&rows<=UINT64_MAX/shape_.classes,"native prediction shape invalid");
    require(rows*shape_.features<=SIZE_MAX/sizeof(float),"native input byte extent overflows");
    int device=-1;cuda_check(cudaGetDevice(&device),"read native CUDA device");require(device==0,"native CUDA device changed");
    device_pointer(input,"native input");
    CUdeviceptr base=0;std::size_t capacity=0;auto address=reinterpret_cast<CUdeviceptr>(input);
    require(cuMemGetAddressRange(&base,&capacity,address)==CUDA_SUCCESS&&address>=base&&address-base<=capacity&&
      rows*shape_.features*sizeof(float)<=capacity-(address-base),"native input extent exceeds allocation");
    cuda_check(cudaDeviceSynchronize(),"synchronize native inputs and prior borrowed outputs");
    auto array=Json{{"data",Json::array({reinterpret_cast<std::uintptr_t>(input),true})},{"shape",Json::array({rows,shape_.features})},
      {"strides",nullptr},{"typestr","<f4"},{"version",3},{"stream",1}}.dump();
    auto options=Json{{"type",margin?1:0},{"training",false},{"iteration_begin",0},{"iteration_end",0},{"strict_shape",true},{"cache_id",0}}.dump();
    options=options.substr(0,options.size()-1)+",\"missing\":NaN}";
    const Size* shape=nullptr;Size dimensions=0;const float* output=nullptr;
    check(predict_(booster_,array.c_str(),options.c_str(),nullptr,&shape,&dimensions,&output),"XGBoosterPredictFromCudaArray");
    auto columns=(margin||shape_.objective=="multi:softprob")?shape_.classes:1u;
    require(shape&&dimensions==2&&shape[0]==rows&&shape[1]==columns&&output,"native prediction output shape differs");
    device_pointer(output,"native output");cuda_check(cudaDeviceSynchronize(),"synchronize native output producer");return output;
  }
  const Json& build_info()const{return build_info_;}
 private:
  using Predict=int(*)(void*,const char*,const char*,void*,const Size**,Size*,const float**);
  template<class Function>Function symbol(const char* name) {
    dlerror();void* pointer=dlsym(library_,name);const char* error=dlerror();
    require(!error&&pointer,std::string("missing native C API symbol: ")+name);return reinterpret_cast<Function>(pointer);
  }
  void check(int status,const char* operation)const {
    if(status!=0){const char* error=last_error_?last_error_():nullptr;throw std::runtime_error(std::string(operation)+": "+(error?error:"unknown native error"));}
  }
  void device_pointer(const float* pointer,const char* role)const {
    require(reinterpret_cast<std::uintptr_t>(pointer)%alignof(float)==0,std::string(role)+" is misaligned");
    cudaPointerAttributes attributes{};cuda_check(cudaPointerGetAttributes(&attributes,pointer),role);
    require(attributes.type==cudaMemoryTypeDevice&&attributes.device==0,std::string(role)+" must be CUDA0 device memory");
  }
  void release()noexcept {if(booster_&&free_)free_(std::exchange(booster_,nullptr));if(library_)dlclose(std::exchange(library_,nullptr));}
  void* library_=nullptr;void* booster_=nullptr;const char*(*last_error_)()=nullptr;int(*free_)(void*)=nullptr;Predict predict_=nullptr;
  SourceShape shape_;Json build_info_;
};
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
} // namespace

int main(int argc,char** argv) {
  auto start=Clock::now();Json report={{"format","class-model-convert-result-1"},{"completed",false}};
  std::filesystem::path output;bool output_ready=false;std::string phase="arguments";
  try {
    require(argc==4,"usage: class_model_convert MODEL OPTIONS_JSON FRESH_OUTPUT_DIRECTORY");
    for(int i=1;i<4;++i)require_path(argv[i],"CLI path");output=argv[3];
    require(!std::filesystem::exists(output),"output must be a fresh directory");output_ready=true;
    report["source_path"]=argv[1];report["options_path"]=argv[2];phase="initial_input_binding";
    BoundFile options_file(argv[2]);auto option_bytes=options_file.bytes();auto settings=Json::parse(option_bytes);
    require(settings.is_object(),"options must be an object");report["options_sha256"]=dpnative::sha256(option_bytes);
    require(settings.at("native_library_path").is_string()&&settings.at("native_library_sha256").is_string(),"native library path and SHA must be strings");
    auto library=settings.at("native_library_path").get<std::string>(),library_pin=settings.at("native_library_sha256").get<std::string>();
    require_path(library,"native library path");require_sha(library_pin,"native_library_sha256");
    settings.erase("native_library_path");settings.erase("native_library_sha256");
    Json work_settings=nullptr;
    class_study::WorkEstimateOptions work_options;
    if(settings.contains("work_estimate")) {
      work_settings=settings.at("work_estimate");settings.erase("work_estimate");
      require(work_settings.is_object(),"work_estimate must be an object");
      for(auto item=work_settings.begin();item!=work_settings.end();++item) {
        const auto& key=item.key();const auto& value=item.value();
        if(key=="paths")work_options.paths=class_study::option_integer(value,key,UINT32_MAX);
        else if(key=="seed")work_options.seed=class_study::option_integer(value,key);
        else if(key=="maximum_decisions")work_options.maximum_decisions=class_study::option_integer(value,key,UINT32_MAX);
        else if(key=="decisions_per_chunk")work_options.decisions_per_chunk=class_study::option_integer(value,key,UINT32_MAX);
        else if(key=="refinement_visit_budget")work_options.refinement_visit_budget=class_study::option_integer(value,key,UINT32_MAX);
        else if(key=="maximum_seconds") {
          require(value.is_number(),"work_estimate.maximum_seconds must be numeric");
          work_options.maximum_seconds=value.get<double>();
          require(std::isfinite(work_options.maximum_seconds)&&work_options.maximum_seconds>0,"work_estimate.maximum_seconds must be finite and positive");
        } else throw std::invalid_argument("unsupported work_estimate option: "+key);
      }
      require(work_options.paths&&work_options.decisions_per_chunk,"work_estimate paths and chunk size must be positive");
      report["format"]="class-model-work-estimate-result-1";
      report["model_conversion_performed"]=false;
      report["work_estimate_options"]=work_settings;
    }
    Json gate_settings=nullptr;
    if(settings.contains("native_gate")) {
      gate_settings=settings.at("native_gate");settings.erase("native_gate");require(gate_settings.is_object(),"native_gate must be an object");
      for(auto item=gate_settings.begin();item!=gate_settings.end();++item) {
        require(item.key()=="helper_path"||item.key()=="snapshot_path"||item.key()=="qualification_path","unsupported native_gate option: "+item.key());
        require(item.value().is_string(),"native_gate paths must be strings");require_path(item.value().get<std::string>(),"native_gate path");
      }
      for(const char* key:{"helper_path","snapshot_path","qualification_path"})require(gate_settings.contains(key),std::string("native_gate missing ")+key);
    }
    std::string offered_source;
    if(settings.contains("source_sha256")){require(settings.at("source_sha256").is_string(),"source_sha256 must be a string");offered_source=settings.at("source_sha256").get<std::string>();require_sha(offered_source,"source_sha256");settings.erase("source_sha256");}
    BoundFile source_file(argv[1]);auto source_bytes=source_file.bytes();auto source_pin=dpnative::sha256(source_bytes);
    require(offered_source.empty()||offered_source==source_pin,"source SHA differs from offered pin");auto shape=source_shape(source_bytes);
    auto options=class_study::parse_conversion_options(settings,shape.features);
    std::signal(SIGINT,record_interruption);std::signal(SIGTERM,record_interruption);
    options.stop_requested=[] {return interruption!=0;};
    if(!options.checkpoint_path.empty()) {
      std::signal(SIGUSR1,record_checkpoint_request);
      options.checkpoint_requested=[] {if(!checkpoint_request)return false;checkpoint_request=0;return true;};
    }
    options.progress=[throttle=class_study::ProgressThrottle{},estimate_mode=!work_settings.is_null(),
                      checkpoint_enabled=!options.checkpoint_path.empty()](const Json& statistics)mutable{
      if(interruption&&(!checkpoint_enabled||estimate_mode)){auto partial=statistics;partial["interrupted_signal"]=int(interruption);throw class_study::ConversionFailure("conversion interrupted at a host progress boundary",std::move(partial));}
      if(throttle.emit(statistics))std::cout<<Json{{"event",estimate_mode?"work_estimate_progress":"conversion_progress"},{"statistics",statistics}}.dump()<<'\n'<<std::flush;
    };
    report.update({{"source_sha256",source_pin},{"source_bytes",source_bytes.size()},{"features",shape.features},{"classes",shape.classes},
      {"objective",shape.objective},{"native_library_path",library},{"native_library_sha256",library_pin},
      {"conversion_options",class_study::describe_conversion_options(options)}});
    BoundFile library_file(library);{auto library_bytes=library_file.bytes();require(dpnative::sha256(library_bytes)==library_pin,"native library SHA differs from offered pin");}
    phase="native_resident_model_setup";ResidentModel native(library,source_bytes,shape);library_file.unchanged();
    report["native_build_info"]=native.build_info();report["initial_binding_and_setup_seconds"]=seconds(start);
    class_study::NativeOracle oracle;oracle.features=shape.features;oracle.classes=shape.classes;oracle.objective=shape.objective;
    oracle.library_sha256=library_pin;oracle.source_sha256=source_pin;
    oracle.predict=[&native](const float* input,Size rows,bool margin){return native.predict(input,rows,margin);};
    report["native_gate"]={{"enabled",false},{"reason","optional setup not requested"}};
    if(!gate_settings.is_null()) {
      phase="native_gate_setup";
      auto setup=class_study::try_qualify_native_gap(source_pin,library,gate_settings.at("helper_path").get<std::string>(),
        gate_settings.at("snapshot_path").get<std::string>(),gate_settings.at("qualification_path").get<std::string>(),
        shape.classes,shape.objective);
      oracle.softprob_gap_gate=std::move(setup.gate);report["native_gate"]=std::move(setup.receipt);report["native_gate_settings"]=gate_settings;
      library_file.unchanged();
      std::cout<<Json{{"event","native_gate_setup"},{"evidence",report.at("native_gate")}}.dump()<<'\n'<<std::flush;
    }
    report["initial_binding_and_setup_seconds"]=seconds(start);
    class_study::ConversionSource source{source_bytes,shape.features,shape.classes,source_pin};
    if(!work_settings.is_null()) {
      phase="work_estimation";
      report["work_estimate"]=class_study::estimate_model_work(source,oracle,options,work_options);
      report["sampling_completed"]=report["work_estimate"].at("all_paths_finished");
      report["completed"]=false; // No classifier was constructed by this operation.
      report["wall_seconds_before_report"]=seconds(start);
      std::filesystem::create_directories(output);
      dpnative::atomic_json(output/"result.json",report);
      std::cout<<Json{{"event","work_estimate_complete"},{"report",(output/"result.json").string()},
        {"all_paths_finished",report["sampling_completed"]},{"model_conversion_performed",false}}.dump()<<'\n';
      return 0;
    }
    phase="adaptive_conversion";auto construction_start=Clock::now();auto converted=class_study::convert_model(source,oracle,options);
    report["conversion_seconds"]=seconds(construction_start);report["conversion"]=converted.metrics;
    require(bool(converted.runtime),"converter returned no validated runtime");const auto& metadata=converted.runtime->metadata();
    report.update({{"canonical_sha256",metadata.canonical_sha256},{"compact_sha256",metadata.compact_sha256},
      {"canonical_bytes",converted.canonical_bytes.size()},{"compact_bytes",converted.compact_bytes.size()},
      {"nodes",metadata.nodes},{"resident_graph_device_bytes",metadata.device_bytes},
      {"canonical_resident",metadata.canonical_resident},{"compact_resident",metadata.compact_resident}});
    phase="final_export";std::filesystem::create_directories(output);
    dpnative::atomic_text(output/"model.canonical",converted.canonical_bytes);report["canonical_path"]=(output/"model.canonical").string();
    if(!converted.compact_bytes.empty()){dpnative::atomic_text(output/"model.compact",converted.compact_bytes);report["compact_path"]=(output/"model.compact").string();}
    report.update({{"completed",true},{"full_source_native_model_loads",1},{"source_file_reads",1},{"initial_binding_library_file_reads",1},
      {"trial_files_written",0},{"conversion_filesystem_access",!options.checkpoint_path.empty()||
        !options.resume_from.empty()||!options.proof_module_directory.empty()||!options.proof_module_request.empty()},
      {"source_or_library_end_rereads",false},
      {"timing_scope","host wall time; no matched speedup claim"},{"wall_seconds_before_report",seconds(start)}});
    dpnative::atomic_json(output/"result.json",report);
    std::cout<<Json{{"event","conversion_complete"},{"nodes",metadata.nodes},{"canonical_bytes",converted.canonical_bytes.size()},
      {"compact_bytes",converted.compact_bytes.size()},{"report",(output/"result.json").string()}}.dump()<<'\n';return 0;
  } catch(const std::exception& error) {
    report["failure"]={{"phase",phase},{"message",error.what()}};
    if(auto failure=dynamic_cast<const class_study::ConversionFailure*>(&error))report["failure"]["partial_statistics"]=failure->partial_statistics;
    report["wall_seconds_before_report"]=seconds(start);
    if(output_ready)try{std::filesystem::create_directories(output);dpnative::atomic_json(output/"result.json",report);}
      catch(const std::exception& export_error){report["failure"]["report_export_error"]=export_error.what();}
    std::cerr<<report.dump()<<'\n';return 1;
  }
}
