#pragma once
// Default-off, same-process qualification for the reviewed seven-class softprob
// gap theorem, restricted to the previously qualified source identity.
// Source-agnostic transform algebra does not establish generic native-predictor
// correspondence; metadata-only eligibility cannot widen this authorization.
// This file launches no CUDA and never substitutes for sound source
// interval construction. Keep the live Oracle/context immutable during its use.
#include "class_io.hpp"
#include "class_native_softprob_eligibility.hpp"
#include <cfenv>
#include <dlfcn.h>
#include <link.h>
#include <map>
#include <set>
#include <unistd.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__FAST_MATH__)
#error "Native softprob gap qualification requires strict floating-point semantics"
#endif
namespace native_softprob_gap {
using json=nlohmann::json;namespace fs=std::filesystem;using Bounds=std::array<float,7>;
// Retained qualified source scope, pending a generic native-predictor contract.
inline constexpr const char* source_sha="6a85cff77f850ab1e8679845f56b86f335998f3e7348808f3a64ab108c84f9c3";
inline constexpr const char* library_sha="462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4";
inline constexpr const char* helper_sha="a37832de8eaac932fe2261a419ce11147dd73b627d380adc6457d7eb29053a74";
inline constexpr const char* cubin_sha="162bff5b9fd9fa3d3f7a49378f011371fd965d676c85f0af7ef23fed12ed0153";
inline constexpr const char* configuration_sha="c07eba51392be3db4257b89703e94d6eb7765093e0e46614d2fba3f3a74b3155";
inline constexpr const char* build_sha="66af94fc63f2d242b37516a7b27ee0c8270cdd8de874dfc9c0315e9d3775379d";
inline constexpr const char* qualification_sha="ad405318a316a23285287b40b65180db78974bac3d5ad10149a01d040adef3c5";
inline constexpr const char* mutable_symbol="_ZN7xgboost6common6detail16LaunchCUDAKernelIZNKS_3obj20SoftmaxMultiClassObj9TransformEPNS_16HostDeviceVectorIfEEbEUlmNS0_4SpanIfLm18446744073709551615EEEE_JS9_EEEvT_NS0_5RangeEDpT0_";
inline constexpr const char* readonly_symbol="_ZN7xgboost6common6detail16LaunchCUDAKernelIZNKS_3obj20SoftmaxMultiClassObj9TransformEPNS_16HostDeviceVectorIfEEbEUlmNS0_4SpanIKfLm18446744073709551615EEENS8_IfLm18446744073709551615EEEE_JSB_SB_EEEvT_NS0_5RangeEDpT0_";
namespace detail {
inline void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
inline uint64_t unsigned_integer(const json& value){require(value.is_number_unsigned()||(value.is_number_integer()&&value.get<int64_t>()>=0),"capture integer is negative or noninteger");return value.get<uint64_t>();}
inline bool environment_ok()noexcept{
  if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__)
  if(_mm_getcsr()&((1u<<15)|(1u<<6)))return false;
#endif
  return true;
}
// A numerical sufficient condition only. It does not create or enable a gate.
inline int strict_interval_class(const Bounds& lower,const Bounds& upper)noexcept{
  static_assert(sizeof(float)==4&&sizeof(double)==8&&std::numeric_limits<float>::is_iec559&&std::numeric_limits<float>::digits==24&&std::numeric_limits<double>::is_iec559&&std::numeric_limits<double>::digits==53);
  if(!environment_ok())return -1;
  for(int c=0;c<7;++c)if(!std::isfinite(lower[c])||!std::isfinite(upper[c])||lower[c]>upper[c]||lower[c]<-10.0f||upper[c]>10.0f)return -1;
  for(int winner=0;winner<7;++winner){bool strict=true;for(int c=0;c<7;++c)if(c!=winner){volatile double gap=double(lower[winner])-double(upper[c]);if(gap<0x1p-10){strict=false;break;}}if(strict)return winner;}
  return -1;
}
inline json runtime_identity(const std::string& source,const json& evidence){
  require(source==source_sha,"gap source has no qualified native-margin correspondence contract");
  require(valid_source_sha256(source)&&evidence.at("source_sha256")==source,"gap active source identity differs");
  if(evidence.contains("model_classes"))require(unsigned_integer(evidence.at("model_classes"))==reviewed_classes,"gap active model class count differs");
  if(evidence.contains("model_objective"))require(evidence.at("model_objective")=="multi:softprob","gap active model objective differs");
  require(evidence.at("format")=="source-live-native-transform-1"&&evidence.at("library_sha256")==library_sha,"gap live native library/format differs");
  require(evidence.at("xgboost_version")==json::array({3,4,1})&&evidence.at("qualified_zero_tree_receipt_sha256")==qualification_sha,"gap native version/qualification differs");
  require(evidence.at("gpu")=="NVIDIA RTX A5000 Laptop GPU"&&evidence.at("compute_capability")==json::array({8,6})&&evidence.at("cuda_driver_version")==13040&&evidence.at("cuda_runtime_version")==13040,"gap device/runtime pin differs");
  require(dpnative::sha256(evidence.at("native_configuration").dump())==configuration_sha&&dpnative::sha256(evidence.at("native_build_info").dump())==build_sha,"gap native configuration/build pin differs");
  require(evidence.at("training_performed")==false&&evidence.at("dataset_records_read")==false,"gap oracle scope differs");
  return {{"source_sha256",source},{"classes",reviewed_classes},{"objective","multi:softprob"},{"reviewed_transform_contract",reviewed_contract},{"library_sha256",library_sha},{"qualified_zero_tree_receipt_sha256",qualification_sha},{"gpu",evidence.at("gpu")},{"compute_capability",evidence.at("compute_capability")},{"cuda_driver_version",13040},{"cuda_runtime_version",13040},{"native_configuration_sha256",configuration_sha},{"native_build_info_sha256",build_sha},{"xgboost_version",json::array({3,4,1})}};
}
inline std::string read_prefix(const fs::path& path,uint64_t length){
  require(length>0&&length<=uint64_t(SIZE_MAX)&&length<=uint64_t(std::numeric_limits<std::streamsize>::max()),"capture prefix size unsupported");require(fs::file_size(path)>=length,"capture ledger prefix is truncated");std::ifstream in(path,std::ios::binary);require(bool(in),"capture ledger unavailable");std::string out(std::size_t(length),'\0');in.read(out.data(),std::streamsize(length));require(uint64_t(in.gcount())==length,"capture ledger prefix read incomplete");return out;
}
// This validator is public only for CPU corruption checks. It cannot manufacture
// RuntimeGate: only a successful call through the pinned loaded helper can do so.
inline json validate_capture(const json& snapshot,const std::string& prefix,const fs::path& capture_directory,uint64_t process){
  require(snapshot.at("schema")=="softprob-inprocess-capture-snapshot-1"&&snapshot.at("complete")==true&&snapshot.at("flushed")==true&&snapshot.at("subscription_retained")==true,"capture snapshot incomplete");
  require(unsigned_integer(snapshot.at("process_id"))==process,"capture PID differs");require(snapshot.at("errors")==0&&snapshot.at("dropped_activity_records")==0,"capture errors or dropped activity");
  require(unsigned_integer(snapshot.at("capture_ledger_prefix_bytes"))==prefix.size()&&snapshot.at("capture_ledger_prefix_sha256")==dpnative::sha256(prefix),"capture prefix binding differs");require(!prefix.empty()&&prefix.back()=='\n',"capture prefix ends inside a record");
  require(fs::weakly_canonical(snapshot.at("capture_ledger_path").get<std::string>())==fs::weakly_canonical(capture_directory/"capture.jsonl"),"capture ledger path differs");
  std::map<uint64_t,json> modules;std::map<std::pair<uint64_t,std::string>,json> functions;std::set<uint64_t> function_ids;std::map<std::pair<uint64_t,uint64_t>,json> entered;json completed=json::array();uint64_t function_count=0,exits=0,initializing=0,initialized=0;
  std::istringstream input(prefix);std::string line;while(std::getline(input,line)){require(!line.empty(),"empty capture record");const auto row=json::parse(line);const auto kind=row.at("kind").get<std::string>();
    if(kind=="initializing"){require(row.at("capture_only")==true&&++initializing==1&&initialized==0&&modules.empty()&&entered.empty(),"duplicate/late capture initialization");}
    else if(kind=="initialized"){require(row.at("capture_only")==true&&initializing==1&&++initialized==1,"capture initialization state differs");}
    else if(kind=="matching_module_loaded"){require(initialized==1&&row.at("sha256")==cubin_sha&&row.at("bytes")==794352&&row.at("copy_complete")==true,"captured module identity/copy differs");auto id=unsigned_integer(row.at("module_id"));require(modules.emplace(id,row).second,"ambiguous captured module ID");const auto name="module-"+std::to_string(id)+"-"+cubin_sha+".cubin";require(row.at("file")==name,"capture module filename differs");const auto bytes=dpnative::read_text(capture_directory/name);require(bytes.size()==794352&&dpnative::sha256(bytes)==cubin_sha,"captured module bytes differ");}
    else if(kind=="matching_function"){require(initialized==1,"function before capture initialization");const auto symbol=row.at("symbol").get<std::string>();require((symbol==mutable_symbol&&row.at("function_index")==169)||(symbol==readonly_symbol&&row.at("function_index")==168),"capture function symbol/index differs");const auto context=unsigned_integer(row.at("context_id"));require(functions.emplace(std::make_pair(context,symbol),row).second&&function_ids.insert(unsigned_integer(row.at("function_id"))).second,"ambiguous captured function");++function_count;}
    else if(kind=="matching_driver_launch"){require(initialized==1,"launch before capture initialization");const auto symbol=row.at("symbol").get<std::string>();require(symbol==mutable_symbol||symbol==readonly_symbol,"launch symbol differs");const auto context=unsigned_integer(row.at("context_id")),correlation=unsigned_integer(row.at("correlation_id"));const auto key=std::make_pair(context,correlation);const auto site=row.at("site").get<std::string>();if(site=="enter"){require(entered.emplace(key,row).second,"duplicate launch enter");}else{require(site=="exit"&&row.at("cuda_result")==0,"native launch failed or has unknown site");const auto found=entered.find(key);require(found!=entered.end()&&found->second.at("symbol")==row.at("symbol")&&found->second.at("api")==row.at("api")&&found->second.at("callback_id")==row.at("callback_id"),"launch enter/exit binding differs");completed.push_back(row);entered.erase(found);++exits;}}
    else throw std::runtime_error("capture ledger contains error, finished subscription or unknown record");
  }
  require(initializing==1&&initialized==1&&!modules.empty()&&!functions.empty()&&entered.empty()&&exits>0,"capture lacks complete module/function/launch sequence");
  require(unsigned_integer(snapshot.at("matching_modules"))==modules.size()&&unsigned_integer(snapshot.at("matching_functions"))==function_count&&unsigned_integer(snapshot.at("matching_launches"))==exits,"capture summary counts differ");
  for(const auto& [key,row]:functions){(void)key;require(modules.contains(unsigned_integer(row.at("module_id"))),"captured function lacks exact module binding");}
  uint64_t mutable_launches=0;std::set<uint64_t> mutable_modules,contexts;for(const auto& row:completed){auto context=unsigned_integer(row.at("context_id"));const auto symbol=row.at("symbol").get<std::string>();const auto found=functions.find({context,symbol});require(found!=functions.end(),"launch lacks same-context function/module binding");if(symbol==mutable_symbol){++mutable_launches;mutable_modules.insert(unsigned_integer(found->second.at("module_id")));contexts.insert(context);}}
  require(mutable_launches>0,"no successfully launched audited mutable-span kernel");return {{"same_process",true},{"process_id",process},{"captured_cubin_sha256",cubin_sha},{"mutable_function_symbol",mutable_symbol},{"mutable_function_index",169},{"mutable_module_ids",mutable_modules},{"mutable_context_ids",contexts},{"successful_mutable_launches",mutable_launches},{"matching_modules",modules.size()},{"matching_functions",function_count},{"matching_launches",exits},{"ledger_prefix_bytes",prefix.size()},{"ledger_prefix_sha256",dpnative::sha256(prefix)},{"errors",0},{"dropped_activity_records",0}};
}
inline std::string loaded_library(){
  std::vector<std::string> paths;auto visitor=[](dl_phdr_info* info,std::size_t,void* raw){if(info->dlpi_name&&*info->dlpi_name&&fs::path(info->dlpi_name).filename()=="libxgboost.so")static_cast<std::vector<std::string>*>(raw)->push_back(info->dlpi_name);return 0;};dl_iterate_phdr(visitor,&paths);require(paths.size()==1,"pinned XGBoost library is not uniquely loaded in this process");const auto path=fs::canonical(paths.front());require(dpnative::sha256(dpnative::read_text(path))==library_sha,"loaded XGBoost library bytes differ");return path.string();
}
} // namespace detail
class RuntimeGate {
  bool qualified_=false;uint64_t process_=0;std::shared_ptr<void> helper_;json runtime_=json::object(),receipt_={{"enabled",false},{"reason","not qualified"}};std::string binding_;
 public:
  RuntimeGate()=default; // Copy retains immutable qualification and loader handle.
  static RuntimeGate qualify(const std::string& source,const json& live_oracle_evidence,const fs::path& loaded_helper,const fs::path& fresh_snapshot){
    RuntimeGate gate;try{
      detail::require(detail::environment_ok(),"gap host floating-point environment differs");gate.runtime_=detail::runtime_identity(source,live_oracle_evidence);const auto helper_path=fs::canonical(loaded_helper);detail::require(dpnative::sha256(dpnative::read_text(helper_path))==helper_sha,"capture helper binary is not pinned");
      void* handle=dlopen(helper_path.c_str(),RTLD_NOW|RTLD_NOLOAD|RTLD_LOCAL);detail::require(handle!=nullptr,"capture helper is not loaded in this process");gate.helper_=std::shared_ptr<void>(handle,[](void* p){if(p)dlclose(p);});dlerror();void* symbol=dlsym(handle,"dp_softprob_capture_snapshot");const char* error=dlerror();detail::require(!error&&symbol,"capture helper snapshot symbol unavailable");Dl_info info{};detail::require(dladdr(symbol,&info)!=0&&info.dli_fname&&fs::canonical(info.dli_fname)==helper_path,"snapshot symbol originates in another object");
      const auto library_path=detail::loaded_library();detail::require(!fs::exists(fresh_snapshot)&&fs::is_directory(fresh_snapshot.parent_path()),"capture snapshot path must be fresh in an existing directory");using Snapshot=int(*)(const char*);auto snapshot_fn=reinterpret_cast<Snapshot>(symbol);const auto snapshot_path=fs::absolute(fresh_snapshot);detail::require(snapshot_fn(snapshot_path.c_str())==1,"same-process capture helper is unresolved or incomplete");
      const auto snapshot_bytes=dpnative::read_text(snapshot_path);const auto snapshot=json::parse(snapshot_bytes);const fs::path ledger=snapshot.at("capture_ledger_path").get<std::string>();const auto prefix=detail::read_prefix(ledger,detail::unsigned_integer(snapshot.at("capture_ledger_prefix_bytes")));const auto process=uint64_t(getpid());auto capture=detail::validate_capture(snapshot,prefix,ledger.parent_path(),process);
      detail::require(dpnative::sha256(dpnative::read_text(helper_path))==helper_sha&&dpnative::sha256(dpnative::read_text(library_path))==library_sha&&dpnative::read_text(snapshot_path)==snapshot_bytes,"qualification input changed during capture validation");
      gate.receipt_={{"schema","native-softprob-gap-runtime-gate-1"},{"enabled",true},{"process_id",process},{"runtime",gate.runtime_},{"helper_path",helper_path.string()},{"helper_sha256",helper_sha},{"loaded_library_path",library_path},{"snapshot_path",snapshot_path.string()},{"snapshot_sha256",dpnative::sha256(snapshot_bytes)},{"capture",capture},{"computed_fp64_gap_minimum",0x1p-10},{"all_endpoints_absolute_maximum",10},{"fixed_tie_shortcut",false},{"theorem","symmetric_exp_relative_2^-16_div_relative_2^-20_true_gap_2^-11"},{"ordinary_nvidia_instruction_contracts_trusted",true},{"future_process_reuse_permitted",false}};
      gate.binding_=dpnative::sha256(gate.receipt_.dump());gate.receipt_["binding_sha256"]=gate.binding_;gate.process_=process;gate.qualified_=true;
    }catch(const std::exception& e){gate.qualified_=false;gate.process_=0;gate.binding_.clear();gate.helper_.reset();gate.receipt_={{"schema","native-softprob-gap-runtime-gate-1"},{"enabled",false},{"reason",e.what()},{"exact_fallback_required",true}};}
    return gate;
  }
  static RuntimeGate qualify(const dpnative::SourceData& source,const json& evidence,const fs::path& helper,const fs::path& snapshot){
    if(source.outputs!=std::int32_t(reviewed_classes)){
      RuntimeGate refused;refused.receipt_={{"schema","native-softprob-gap-runtime-gate-1"},{"enabled",false},
        {"reason","active source class count has no reviewed native transform contract"},{"exact_fallback_required",true}};return refused;
    }
    return qualify(source.identity,evidence,helper,snapshot);
  }
  bool enabled()const noexcept{return qualified_&&process_==uint64_t(getpid())&&bool(helper_);}
  bool matches_source(const std::string& source,std::uint32_t classes,
      const std::string& objective,const std::string& library)const noexcept {
    try{return enabled()&&source==source_sha&&classes==reviewed_classes&&objective=="multi:softprob"&&
      runtime_.at("source_sha256")==source&&runtime_.at("classes")==classes&&
      runtime_.at("objective")==objective&&runtime_.at("library_sha256")==library;}
    catch(...){return false;}
  }
  int classify(const Bounds& lower,const Bounds& upper)const noexcept{return enabled()?detail::strict_interval_class(lower,upper):-1;}
  const std::string& binding_sha256()const noexcept{return binding_;}
  bool matches_runtime(const std::string& source,const json& evidence)const noexcept{try{return enabled()&&runtime_==detail::runtime_identity(source,evidence);}catch(...){return false;}}
  json evidence()const{auto out=receipt_;out["enabled"]=enabled();return out;}
};
} // namespace native_softprob_gap
