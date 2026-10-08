#include "class_study_train.hpp"
#include "class_study_data.hpp"
#include "class_study_group_loss_cuda.cuh"
// System headers are global; legacy transport/helper declarations are private.
#include <cuda.h>
#include <cuda_runtime.h>
#include <openssl/evp.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace class_study::train_detail {
#include "class_model_dataset.cuh"
#include "class_native_training.cuh"
}

namespace class_study {
namespace d=train_detail;
using json=nlohmann::json;
json ResidentTrainer::validate_metadata(const json&p){
  const auto shape=d::class_model_contract::dense(p.at("dataset"),true);
  d::need(p.at("TEST_read").is_boolean()&&!p.at("TEST_read").get<bool>()&&p.at("VALID_read").is_boolean()&&!p.at("VALID_read").get<bool>(),"resident training requires FIT-only role");
  d::need(shape.classes<=d::class_native_training::maximum_exact_native_classes,"native FP32 class-ID capacity");
  d::need(!d::native_class_reference::library_identity(p).empty(),"resident pinned native library metadata");
  return {{"features",shape.features},{"classes",shape.classes},{"FIT_rows",shape.rows},{"row_stride",shape.stride},{"FIT_only",true},{"VALID_read",false},{"TEST_read",false},{"native_objective","multi:softmax"},{"native_library_sha256",p.at("native_library_sha256").get<std::string>()}};
}
json ResidentTrainer::validate_metadata(const json&p,const ResidentDataView&v){
  d::need(p.at("TEST_read").is_boolean()&&!p.at("TEST_read").get<bool>()&&p.at("VALID_read").is_boolean()&&!p.at("VALID_read").get<bool>(),"resident training requires FIT-only role");
  d::need(v.features>0&&v.features<=std::uint32_t(INT32_MAX)&&v.classes>=2&&v.classes<=d::class_native_training::maximum_exact_native_classes,"resident FIT feature/class capacity");
  d::need(v.rows>0&&v.row_stride>=v.features&&v.fit_rows==v.rows&&v.valid_rows==0,"resident training view must contain only FIT rows");
  d::need(v.rows<=UINT64_MAX/v.row_stride&&v.rows*v.row_stride<=SIZE_MAX/sizeof(float)&&v.rows<=SIZE_MAX/sizeof(std::uint32_t),"resident FIT storage extent overflow");
  d::need(v.values&&v.labels&&reinterpret_cast<std::uintptr_t>(v.values)%alignof(float)==0&&reinterpret_cast<std::uintptr_t>(v.labels)%alignof(std::uint32_t)==0,"resident FIT pointers missing or misaligned");
  d::need(bool(v.owner)&&v.binding.is_object(),"resident FIT source owner/binding missing");
  d::need(!v.sample_weights||reinterpret_cast<std::uintptr_t>(v.sample_weights)%alignof(float)==0,"resident FIT weight pointer misaligned");
  d::need(!d::native_class_reference::library_identity(p).empty(),"resident pinned native library metadata");
  return {{"features",v.features},{"classes",v.classes},{"FIT_rows",v.rows},{"row_stride",v.row_stride},{"FIT_only",true},{"VALID_read",false},{"TEST_read",false},{"native_objective","multi:softmax"},{"native_library_sha256",p.at("native_library_sha256").get<std::string>()}};
}
namespace {
using NativeHandle=d::native_class_reference::Api::H;
struct OwnedNativeHandle {
  NativeHandle handle=nullptr;
  int(*free)(NativeHandle)=nullptr;
  ~OwnedNativeHandle(){if(handle&&free)free(handle);}
  OwnedNativeHandle()=default;
  explicit OwnedNativeHandle(int(*f)(NativeHandle)):free(f){}
  OwnedNativeHandle(const OwnedNativeHandle&)=delete;
  OwnedNativeHandle&operator=(const OwnedNativeHandle&)=delete;
};
void resident_allocation(const void*p,std::uint64_t bytes,const char*message){
  d::need(bytes&&bytes<=UINTPTR_MAX-reinterpret_cast<std::uintptr_t>(p),message);
  cudaPointerAttributes attributes{};d::ck(cudaPointerGetAttributes(&attributes,p),message);
  d::need(attributes.type==cudaMemoryTypeDevice&&attributes.device==0,message);
  CUdeviceptr base=0;std::size_t size=0;const auto address=reinterpret_cast<CUdeviceptr>(p);
  d::need(cuMemGetAddressRange(&base,&size,address)==CUDA_SUCCESS&&address>=base&&bytes<=size&&address-base<=size-bytes,message);
}
using NativeApi=d::native_class_reference::Api;
struct ProbabilityState {
  OwnedNativeHandle owner;
  std::string source_pin;
  std::uint64_t clones=0,predictions=0;
  json view_metadata;
};
void verify_native_tree(NativeApi&api,NativeHandle source,std::uint32_t F,std::uint32_t K,const char*objective="multi:softmax"){
  d::U size=0;const char*buffer=nullptr;
  api.check(api.symbol<int(*)(NativeHandle,d::U*,const char**)>("XGBoosterSaveJsonConfig")(source,&size,&buffer),"native model configuration");
  d::need(buffer&&size&&size<=SIZE_MAX,"native configuration missing/extent");
  const auto config=json::parse(std::string(buffer,size));const auto&learner=config.at("learner");const auto&shape=learner.at("learner_model_param");
  d::need(learner.at("gradient_booster").at("name")=="gbtree"&&learner.at("objective").at("name")==objective&&shape.at("num_feature").get<std::string>()==std::to_string(F)&&shape.at("num_class").get<std::string>()==std::to_string(K),"native gbtree/objective/shape differs");
}
// Caller has checked the opaque buffer identity. Native loading handles native
// Infinity sentinels; only native configuration is parsed as strict JSON.
void load_native_model(NativeApi&api,const std::string&bytes,std::uint32_t F,std::uint32_t K,
                       OwnedNativeHandle&candidate,NativeHandle other,NativeHandle probability,
                       bool positive_rounds,bool(*aliases)(const void*,NativeHandle)=nullptr,
                       const void*alias_context=nullptr){
  d::need(!bytes.empty()&&!candidate.handle,"native buffer/temporary owner missing");d::gpu_sync();using H=NativeHandle;
  const int created=api.symbol<int(*)(const H*,d::U,H*)>("XGBoosterCreate")(nullptr,0,&candidate.handle);
  if(candidate.handle&&(candidate.handle==api.model||candidate.handle==other||candidate.handle==probability||(aliases&&aliases(alias_context,candidate.handle)))){candidate.handle=nullptr;api.check(created,"create native buffer booster");d::need(false,"native buffer booster aliases an existing owner");}
  api.check(created,"create native buffer booster");d::need(candidate.handle,"native buffer handle missing");
  api.check(api.symbol<int(*)(H,const void*,d::U)>("XGBoosterLoadModelFromBuffer")(candidate.handle,bytes.data(),bytes.size()),"load verified native model buffer");
  api.check(api.symbol<int(*)(H,const char*,const char*)>("XGBoosterSetParam")(candidate.handle,"device","cuda:0"),"native buffer CUDA device");
  verify_native_tree(api,candidate.handle,F,K);int rounds=-1;
  api.check(api.symbol<int(*)(H,int*)>("XGBoosterBoostedRounds")(candidate.handle,&rounds),"native buffer boosted rounds");
  d::need(positive_rounds?rounds>0:rounds>=0,"native buffer round extent");d::gpu_sync();
}
void clear_probability(NativeApi&api,ProbabilityState&state,json&session){
  if(state.owner.handle){d::gpu_sync();api.check(api.free_booster(state.owner.handle),"free native probability view");state.owner.handle=nullptr;}
  state.source_pin.clear();
  if(session.contains("probability_response_view"))session["probability_response_view"]["current_source_bound"]=false;
}
void install_native_model(NativeApi&api,ProbabilityState&state,json&session,std::string&current_pin,
                          OwnedNativeHandle&candidate,const std::string&pin){
  auto next_pin=pin;clear_probability(api,state,session);
  if(api.model)api.check(api.free_booster(api.model),"free previous native current model");
  api.model=std::exchange(candidate.handle,nullptr);current_pin.swap(next_pin);
}
NativeHandle probability_model(NativeApi&api,std::uint32_t F,std::uint32_t K,const std::string&pin,
                               ProbabilityState&state,json&session,NativeHandle other,
                               std::optional<std::uint64_t> completed_fit={},
                               bool(*aliases)(const void*,NativeHandle)=nullptr,const void*alias_context=nullptr,
                               NativeHandle source=nullptr){
  if(!source)source=api.model;
  if(state.owner.handle&&state.source_pin==pin)return state.owner.handle;
  d::need(source&&!pin.empty(),"probability view requires a completed native model");
  verify_native_tree(api,source,F,K);auto next_pin=pin;using H=NativeHandle;
  int rounds=-1;api.check(api.symbol<int(*)(H,int*)>("XGBoosterBoostedRounds")(source,&rounds),"probability view source rounds");
  d::need(rounds>0&&state.clones<UINT64_MAX,"probability view requires positive completed rounds/clone capacity");
  d::gpu_sync();OwnedNativeHandle candidate(api.free_booster);
  const int status=api.symbol<int(*)(H,int,int,int,H*)>("XGBoosterSlice")(source,0,0,1,&candidate.handle);
  if(candidate.handle&&(candidate.handle==source||candidate.handle==api.model||candidate.handle==other||candidate.handle==state.owner.handle||(aliases&&aliases(alias_context,candidate.handle)))){candidate.handle=nullptr;api.check(status,"clone native probability view");d::need(false,"probability view aliases an existing owner");}
  api.check(status,"clone native probability view");d::need(candidate.handle,"native probability clone missing");
  const auto set=api.symbol<int(*)(H,const char*,const char*)>("XGBoosterSetParam");
  api.check(set(candidate.handle,"device","cuda:0"),"probability view CUDA device");
  api.check(set(candidate.handle,"objective","multi:softprob"),"probability view objective");
  verify_native_tree(api,candidate.handle,F,K,"multi:softprob");int selected=-1;
  api.check(api.symbol<int(*)(H,int*)>("XGBoosterBoostedRounds")(candidate.handle,&selected),"probability view clone rounds");
  d::need(selected==rounds,"probability view clone loses rounds");verify_native_tree(api,source,F,K);d::gpu_sync();
  json metadata={{"source_model_sha256",next_pin},{"source_objective","multi:softmax"},{"derived_objective","multi:softprob"},{"operation","native_full_round_probability_view"},{"native_clone_method","XGBoosterSlice"},{"rounds",rounds},{"features",F},{"classes",K},{"native_library_sha256",session.at("native_library_sha256")},{"FIT_performed",false},{"current_source_bound",true},{"source_objective_unchanged",true},{"softprob_argmax_hard_class_equivalence_claim",false},{"native_view_clones",state.clones+1},{"prediction_calls",state.predictions},{"trial_file_reads",0},{"trial_file_writes",0}};
  if(completed_fit)metadata["completed_FIT_calls"]=*completed_fit;
  auto stored_metadata=metadata;
  auto next_session=session;next_session["probability_response_view"]=std::move(metadata);
  if(state.owner.handle)api.check(api.free_booster(state.owner.handle),"free previous native probability view");
  state.owner.handle=std::exchange(candidate.handle,nullptr);state.owner.free=api.free_booster;
  state.source_pin.swap(next_pin);state.view_metadata=std::move(stored_metadata);++state.clones;session.swap(next_session);return state.owner.handle;
}
void native_input(const float*x,std::uint64_t rows,std::uint32_t F,std::uint32_t K){
  d::need(x&&rows&&reinterpret_cast<std::uintptr_t>(x)%alignof(float)==0&&rows<=SIZE_MAX/sizeof(float)/F&&rows<=SIZE_MAX/sizeof(float)/K,"native input/output shape or alignment");
  resident_allocation(x,rows*std::uint64_t(F)*sizeof(float),"native input exceeds CUDA0 allocation");
}
const float*native_prediction(NativeApi&api,NativeHandle model,const float*x,std::uint64_t rows,
                             std::uint32_t F,std::uint32_t K,bool margin,bool probabilities,
                             std::uint32_t iteration_end=0,std::uint32_t iteration_begin=0,NativeHandle proxy=nullptr){
  d::need(model,"native prediction requires a loaded model");native_input(x,rows,F,K);d::gpu_sync();
  const d::U*shape=nullptr;d::U dims=0;const float*out=nullptr;const auto data=api.array(x,rows);
  auto options=json{{"type",margin?1:0},{"training",false},{"iteration_begin",iteration_begin},{"iteration_end",iteration_end},{"strict_shape",true},{"cache_id",0}}.dump();
  auto config=options.substr(0,options.size()-1)+",\"missing\":NaN}";
  api.check(api.symbol<int(*)(NativeHandle,const char*,const char*,NativeHandle,const d::U**,d::U*,const float**)>("XGBoosterPredictFromCudaArray")(model,data.c_str(),config.c_str(),proxy,&shape,&dims,&out),"native CUDA prediction");
  const std::uint32_t width=margin||probabilities?K:1;
  d::need(shape&&dims==2&&shape[0]==rows&&shape[1]==width&&out&&reinterpret_cast<std::uintptr_t>(out)%alignof(float)==0,"native output aligned shape differs");
  resident_allocation(out,rows*std::uint64_t(width)*sizeof(float),"native output exceeds CUDA0 allocation");d::gpu_sync();return out;
}
}
json ResidentTrainer::validate_hyperparameters(const json&hp,std::uint32_t K){
  d::need(K>=2&&K<=d::class_native_training::maximum_exact_native_classes,"native class-ID capacity");
  return d::class_native_training::parameters(json{{"hyperparameters",hp}},K);
}
json ResidentTrainer::validate_loss(const json&parameters,const TrainingLossSpec&loss,
    std::uint64_t rows,std::uint32_t groups,std::uint32_t classes){
  const auto hp=validate_hyperparameters(parameters,classes);
  d::need(loss.version==1&&classes==2,"group loss version/binary classes differ");
  d::need(loss.margin_mode==TrainingMarginMode::full_ensemble||loss.margin_mode==TrainingMarginMode::incremental_native_proxy,"unknown training margin mode");
  const bool incremental=loss.margin_mode==TrainingMarginMode::incremental_native_proxy;
  d::need(!loss.verify_incremental_margins||incremental,"incremental verification requires incremental margin mode");
  d::need(hp.contains("max_delta_step")&&hp.at("max_delta_step").get<double>()>0,
          "group loss requires explicit positive max_delta_step");
  d::need(hp.value("subsample",1.0)==1.0,"group loss requires subsample=1");
  // Native tree parameters are FP32. The bound must use the values the native
  // updater actually applies, not a more optimistic unrounded declaration.
  group_loss::Config c;c.beta=loss.beta;c.loss_scale=loss.loss_scale;
  c.curvature_policy=loss.curvature_policy;
  c.eta=double(float(hp.at("eta").get<double>()));
  c.max_delta_step=double(float(hp.at("max_delta_step").get<double>()));
  c.require_all_positives=loss.require_all_positives;
  d::need(groups<=rows&&!group_loss::validate_config(c,rows,groups),"invalid group loss shape/configuration");
  d::need(rows<=UINT64_MAX/hp.at("rounds").get<std::uint64_t>()/(loss.verify_incremental_margins?2:1),"group loss row-round extent overflow");
  const std::uint64_t owned_per_row=loss.verify_incremental_margins?32:24;
  const std::uint64_t known_native_per_row=incremental?32:24;
  d::need(rows<=(UINT64_MAX-36)/owned_per_row&&rows<=UINT64_MAX/known_native_per_row&&
    std::uint64_t(groups)<=(UINT64_MAX-36-owned_per_row*rows)/sizeof(group_loss::GroupStats)&&
    std::uint64_t(groups)<=(UINT64_MAX-4*rows)/sizeof(FitGroupSpec),"group loss workspace extent overflow");
  return {{"format","group-mean-ce-dice-loss-1"},{"version",loss.version},{"beta",loss.beta},
    {"loss_scale",loss.loss_scale},{"require_all_positives",loss.require_all_positives},
    {"classes",classes},{"FIT_rows",rows},{"FIT_groups",groups},{"effective_eta",c.eta},
    {"effective_max_delta_step",c.max_delta_step},{"logit_step_radius",group_loss::radius(c)},
    {"initial_margin",json::array({0.0,0.0})},{"boost_from_average",false},
    {"curvature_policy","diagonal positive bounded-step majorizer; ordinary FP32 export"},
    {"curvature_policy_version",loss.curvature_policy==TrainingCurvaturePolicy::uniform_v1?"uniform_v1":"class_row_sum_v2"},
    {"moment_reduction_policy",group_loss::moment_reduction_policy},
    {"zero_inside_positive_policy","normalized background cross entropy"},
    {"data_cut_weight_policy","retain the unchanged existing FIT DMatrix weights"},
    {"objective_weight_policy","FIT group full/retained class masses; supplied g/h applied once"},
    {"common_group_scale_policy","validated then analytically cancelled"},
    {"owned_objective_CUDA_bytes",owned_per_row*rows+std::uint64_t(groups)*sizeof(group_loss::GroupStats)+36},
    {"borrowed_group_CUDA_bytes",4*rows+std::uint64_t(groups)*sizeof(FitGroupSpec)},
    {"known_native_margin_and_gradient_pair_bytes",24*rows},
    {"known_native_prediction_proxy_bytes",incremental?8*rows:0},
    {"known_native_total_additional_bytes",known_native_per_row*rows},
    {"margin_prediction_mode",incremental?"incremental_native_proxy":"full_ensemble"},
    {"default_margin_prediction_mode","incremental_native_proxy"},
    {"verify_incremental_margins",loss.verify_incremental_margins},
    {"step_audit_policy","each class inside RN(before +/- RN(eta*cap)); finite postupdate margins; no FP32 descent certificate"},
    {"opaque_native_workspace_additional",true},{"VALID_read",false},{"TEST_read",false}};
}
namespace {
__global__ void audit_zero_margins(const float* margins,std::uint64_t words,std::uint32_t*bad){
  for(std::uint64_t j=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<words;j+=std::uint64_t(blockDim.x)*gridDim.x)
    if(__float_as_uint(margins[j])!=0u)atomicOr(bad,1u);
}
__global__ void compare_margin_words(const float*a,const float*b,std::uint64_t words,std::uint32_t*bad){
  for(std::uint64_t j=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<words;j+=std::uint64_t(blockDim.x)*gridDim.x)
    if(__float_as_uint(a[j])!=__float_as_uint(b[j]))atomicOr(bad,4096u);
}
std::string binary_margin_array(const float*data,std::uint64_t rows){
  return json{{"data",json::array({reinterpret_cast<std::uintptr_t>(data),true})},
    {"shape",json::array({rows,2})},{"strides",nullptr},{"typestr","<f4"},{"version",3},{"stream",1}}.dump();
}
void verify_external_configuration(const json&config,const json&loss){
  const auto&l=config.at("learner");const auto&b=l.at("gradient_booster");
  const auto&t=b.at("tree_train_param");const auto&m=l.at("learner_model_param");
  d::need(l.at("generic_param").at("device")=="cuda:0"&&b.at("name")=="gbtree"&&
    l.at("objective").at("name")=="multi:softmax"&&m.at("num_class")=="2",
    "external objective native shape/objective/device differs");
  d::need(l.at("learner_train_param").at("multi_strategy")=="one_output_per_tree"&&
    b.at("gbtree_model_param").at("num_parallel_tree")=="1",
    "external objective requires one tree per class per round");
  auto number=[](const json&v){return std::stod(v.get<std::string>());};
  d::need(float(number(t.at("eta")))==float(loss.at("effective_eta").get<double>())&&
    float(number(t.at("max_delta_step")))==float(loss.at("effective_max_delta_step").get<double>())&&
    number(t.at("subsample"))==1.0,"external objective effective step/sampling differs");
  const auto base=json::parse(m.at("base_score").get<std::string>());
  d::need(base.is_array()&&base.size()==2&&base[0]==0&&base[1]==0&&m.at("boost_from_average")=="0",
    "external objective requires fixed explicit zero base margins");
}
}
struct ResidentTrainer::Impl {
  json session;
  d::Dev<float> values,labels,weights;
  std::unique_ptr<d::class_native_training::TrainingApi> native;
  std::uint32_t F=0,K=0;
  std::uint64_t rows=0,trial=0;
  std::string model_pin;
  std::string library_path;
  std::shared_ptr<void> source_owner;
  // Declared after native so this handle is released before the library closes.
  OwnedNativeHandle prefix_parent;
  std::unique_ptr<ProbabilityState> prefix_parent_probability;
  std::string prefix_parent_pin;
  std::uint64_t prefix_parent_json_bytes=0,prefix_parent_loads=0;
  bool prefix_parent_load_pending=false;
  double prefix_parent_load_seconds=0;
  struct CachedPrefixParent {
    OwnedNativeHandle owner;
    std::unique_ptr<ProbabilityState> probability;
    std::string pin;
    std::uint64_t bytes;
    bool load_pending;
    double load_seconds;
    CachedPrefixParent(int(*free)(NativeHandle),const std::string& p,std::uint64_t n,bool pending,double seconds)
      :owner(free),pin(p),bytes(n),load_pending(pending),load_seconds(seconds){}
  };
  // Inactive entries, most recently selected first. No JSON buffer copy. All
  // owners are declared after native and therefore freed before dlclose.
  std::list<CachedPrefixParent> prefix_parents;
  std::uint64_t prefix_cache_budget=1ULL<<30,prefix_cache_bytes=0;
  std::uint64_t prefix_cache_hits=0,prefix_cache_evictions=0;
  bool prefix_cache_hit_pending=false;
  ProbabilityState probability;
  static bool aliases_prefix(const void*,NativeHandle);
  bool active_cache_eligible()const;
  void publish_prefix_cache();
  void trim_prefix_cache();
  void park_prefix_parent();
  void free_parent_probability(std::unique_ptr<ProbabilityState>&);
  void verify_tree(NativeHandle source,const char* objective="multi:softmax");
  void invalidate_probability_view();
  NativeHandle ensure_probability_view();
  TrainingResult make_prefix(NativeHandle source,const std::string& source_pin,
                             std::uint32_t rounds,OwnedNativeHandle& sliced,
                             bool source_prevalidated=false);
  void install_prefix(OwnedNativeHandle& sliced,const std::string& pin);
  explicit Impl(const json&p):session(ResidentTrainer::validate_metadata(p)){
    const auto start=std::chrono::steady_clock::now();
    const auto library=p.at("native_library_path").get<std::string>();
    library_path=library;
    d::need(d::sha256(d::read_text(library))==p.at("native_library_sha256").get<std::string>(),"resident native library bytes differ");
    d::ck(cudaSetDevice(0),"resident training device");
    auto data=d::load_model_dataset(p,{},true);F=data.F;K=data.K;rows=data.rows;
    values=d::Dev<float>(rows*F);labels=d::Dev<float>(rows);d::Dev<d::u32>bad(1);bad.zero();
    d::class_native_training::prepare_values<<<d::blocks(values.n),256>>>(data.x.p,values.p,rows,data.stride,F,bad.p);
    d::class_native_training::prepare_labels<<<d::blocks(rows),256>>>(data.labels.p,labels.p,rows,K,bad.p);d::done();
    d::need(!bad.at(0),"FIT features must be finite/NaN and labels exact native class IDs");
    native=std::make_unique<d::class_native_training::TrainingApi>(library,F);native->training_matrix(values.p,labels.p,rows);
    session["dataset"]=data.binding;session["dataset_loads"]=1;session["data_loaded_from_host"]=true;session["data_source"]="pinned FIT files";session["matrix_creations"]=1;session["values_retained_CUDA_bytes"]=values.n*4;session["labels_retained_CUDA_bytes"]=labels.n*4;
    session["initialization_wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    session["scope"]="FIT byte pins and native library checked once at session creation; same CUDA values/labels/DMatrix retained; no trial filesystem reads/writes or validation access";
  }
  Impl(const json&p,const ResidentDataView&v,const Impl*verified=nullptr):session(ResidentTrainer::validate_metadata(p,v)),source_owner(v.owner){
    const auto start=std::chrono::steady_clock::now();
    const auto library=p.at("native_library_path").get<std::string>();
    library_path=library;
    if(verified){
      d::need(verified->native&&verified->native->library&&verified->library_path==library&&verified->session.at("native_library_sha256")==session.at("native_library_sha256"),"verified resident library path/pin differs");
      native=std::make_unique<d::class_native_training::TrainingApi>(library,v.features);
      d::need(native->library==verified->native->library,"verified resident library loaded handle differs");
      session["native_library_content_reads"]=0;session["native_library_content_reads_skipped"]=1;
      session["native_library_authorization"]="same already-loaded library handle as live verified ResidentTrainer";
    }else d::need(d::sha256(d::read_text(library))==p.at("native_library_sha256").get<std::string>(),"resident native library bytes differ");
    d::ck(cudaSetDevice(0),"resident training device");
    F=v.features;K=v.classes;rows=v.rows;
    resident_allocation(v.values,((rows-1)*v.row_stride+F)*sizeof(float),"resident FIT values exceed CUDA0 allocation");
    resident_allocation(v.labels,rows*sizeof(std::uint32_t),"resident FIT labels exceed CUDA0 allocation");
    values=d::Dev<float>(rows*F);labels=d::Dev<float>(rows);d::Dev<d::u32>bad(1);bad.zero();
    d::class_native_training::prepare_values<<<d::blocks(values.n),256>>>(v.values,values.p,rows,v.row_stride,F,bad.p);
    d::class_native_training::prepare_labels<<<d::blocks(rows),256>>>(v.labels,labels.p,rows,K,bad.p);d::done();
    d::need(!bad.at(0),"FIT features must be finite/NaN and labels exact native class IDs");
    if(v.sample_weights){
      resident_allocation(v.sample_weights,rows*sizeof(float),"resident FIT weights exceed CUDA0 allocation");
      weights=d::Dev<float>(rows);d::Dev<d::u32>weight_status(1);weight_status.zero();
      d::class_native_training::prepare_weights<<<d::blocks(rows),256>>>(v.sample_weights,weights.p,rows,weight_status.p);d::done();
      const auto status=weight_status.at(0);
      d::need(!(status&1u)&&(status&2u),"FIT weights must be finite, nonnegative, and include a positive value");
    }
    if(!native)native=std::make_unique<d::class_native_training::TrainingApi>(library,F);native->training_matrix(values.p,labels.p,rows,weights.p);
    session["sample_weights_present"]=bool(v.sample_weights);session["weights_retained_CUDA_bytes"]=weights.n*4;
    session["dataset"]=v.binding;session["dataset_loads"]=0;session["data_loaded_from_host"]=false;session["data_source"]="shared resident CUDA FIT view";session["matrix_creations"]=1;session["values_retained_CUDA_bytes"]=values.n*4;session["labels_retained_CUDA_bytes"]=labels.n*4;
    session["initialization_wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    session["scope"]=verified?"Borrowed FIT view retained; existing CUDA preparation and independent DMatrix used; library authorized by same loaded handle as live verified trainer; no library/dataset file reads or VALID rows consumed":"Borrowed FIT view and its source owner retained; existing CUDA value/label preparation and native DMatrix used; native library checked once; no dataset files or VALID rows consumed by trainer";
  }
};
bool ResidentTrainer::Impl::aliases_prefix(const void*context,NativeHandle handle){
  const auto&i=*static_cast<const Impl*>(context);
  if(handle==i.prefix_parent.handle||(i.prefix_parent_probability&&handle==i.prefix_parent_probability->owner.handle))return true;
  for(const auto& parent:i.prefix_parents)
    if(handle==parent.owner.handle||(parent.probability&&handle==parent.probability->owner.handle))return true;
  return false;
}
void ResidentTrainer::Impl::free_parent_probability(std::unique_ptr<ProbabilityState>&p){
  if(!p)return;
  if(p->owner.handle){native->check(native->free_booster(p->owner.handle),"free retained parent probability view");p->owner.handle=nullptr;}
  p.reset();
}
bool ResidentTrainer::Impl::active_cache_eligible()const{
  return prefix_parent.handle&&prefix_cache_budget&&prefix_parent_json_bytes<=prefix_cache_budget;
}
void ResidentTrainer::Impl::publish_prefix_cache(){
  const bool eligible=active_cache_eligible();
  std::uint64_t probability_views=prefix_parent_probability&&prefix_parent_probability->owner.handle;
  for(const auto&p:prefix_parents)probability_views+=bool(p.probability&&p.probability->owner.handle);
  d::need(prefix_cache_bytes<=UINT64_MAX-prefix_parent_json_bytes,"native parent resident proxy overflow");
  session["retained_prefix_parent_cache"]={{"serialized_parent_byte_budget",prefix_cache_budget},
    {"serialized_cached_bytes",prefix_cache_bytes+(eligible?prefix_parent_json_bytes:0)},
    {"serialized_resident_parent_bytes",prefix_cache_bytes+prefix_parent_json_bytes},
    {"inactive_parent_models",prefix_parents.size()},{"resident_parent_models",prefix_parents.size()+bool(prefix_parent.handle)},
    {"uncached_active_parent_bytes",prefix_parent.handle&&!eligible?prefix_parent_json_bytes:0},
    {"cache_hits",prefix_cache_hits},{"LRU_evictions",prefix_cache_evictions},{"native_parent_loads",prefix_parent_loads},
    {"policy",prefix_cache_budget?"serialized-byte-bounded LRU; oversized active parent uncached":"legacy one active parent; no inactive retention"},
    {"additional_parent_JSON_buffer_copies",0},{"native_opaque_peak_bounded",false},
    {"transactional_replacement_may_coexist_with_old_handles",true},
    {"checkpointed_native_handles",false},{"budget_scope","serialized parent extents are a resource proxy; native decoded models, device caches and replacement transients are not bounded by it"}};
  session["retained_prefix_parent_cache"]["retained_parent_probability_views"]=probability_views;
  session["retained_prefix_parent_cache"]["probability_view_opaque_bytes_bounded"]=false;
}
void ResidentTrainer::Impl::trim_prefix_cache(){
  // An oversized required active parent uses the legacy one-parent fallback;
  // do not retain inactive owners alongside that exception to the proxy cap.
  const auto allowed=active_cache_eligible()?prefix_cache_budget-prefix_parent_json_bytes:
    prefix_parent.handle?0:prefix_cache_budget;
  while(prefix_cache_bytes>allowed){
    d::need(!prefix_parents.empty(),"native prefix cache byte accounting differs");
    auto last=std::prev(prefix_parents.end());
    free_parent_probability(last->probability);
    native->check(native->free_booster(last->owner.handle),"evict inactive native parent");
    last->owner.handle=nullptr;prefix_cache_bytes-=last->bytes;prefix_parents.erase(last);++prefix_cache_evictions;
  }
}
void ResidentTrainer::Impl::park_prefix_parent(){
  if(!prefix_parent.handle)return;
  if(active_cache_eligible()){
    d::need(prefix_parent_json_bytes<=UINT64_MAX-prefix_cache_bytes,"native parent cache extent overflow");
    // Allocate/copy metadata before taking ownership from the active entry.
    prefix_parents.emplace_front(native->free_booster,prefix_parent_pin,prefix_parent_json_bytes,
                                 prefix_parent_load_pending,prefix_parent_load_seconds);
    prefix_parents.front().owner.handle=std::exchange(prefix_parent.handle,nullptr);
    prefix_parents.front().probability=std::move(prefix_parent_probability);
    prefix_cache_bytes+=prefix_parent_json_bytes;
  }else{
    free_parent_probability(prefix_parent_probability);
    native->check(native->free_booster(prefix_parent.handle),"release uncached native parent");prefix_parent.handle=nullptr;
  }
}
ResidentTrainer::ResidentTrainer(const json&p):impl_(std::make_unique<Impl>(p)){}
ResidentTrainer::ResidentTrainer(const json&p,const ResidentDataView&v):impl_(std::make_unique<Impl>(p,v)){}
ResidentTrainer::ResidentTrainer(const json&p,const ResidentDataView&v,const ResidentTrainer&verified){
  d::need(bool(verified.impl_),"moved-from verified resident trainer");impl_=std::make_unique<Impl>(p,v,verified.impl_.get());
}
ResidentTrainer::~ResidentTrainer()=default;
ResidentTrainer::ResidentTrainer(ResidentTrainer&&)noexcept=default;
ResidentTrainer&ResidentTrainer::operator=(ResidentTrainer&&)noexcept=default;
TrainingResult ResidentTrainer::train(const json&parameters){
  d::need(bool(impl_),"moved-from resident trainer");auto&i=*impl_;const auto hp=validate_hyperparameters(parameters,i.K);
  const auto start=std::chrono::steady_clock::now();i.invalidate_probability_view();i.model_pin.clear();i.native->configure(hp,i.K);i.native->fit(hp.at("rounds").get<d::U>());
  auto config=json::parse(i.native->configuration());d::need(config.at("learner").at("generic_param").at("device")=="cuda:0"&&config.at("learner").at("gradient_booster").at("updater").size()==1&&config.at("learner").at("gradient_booster").at("updater")[0].at("name")=="grow_gpu_hist","resident native updater is not the required CUDA updater");
  int rounds=-1;i.native->check(i.native->symbol<int(*)(d::native_class_reference::Api::H,int*)>("XGBoosterBoostedRounds")(i.native->model,&rounds),"resident completed rounds");d::need(rounds==hp.at("rounds").get<int>(),"resident round count differs");
  d::U size=0;const char*buffer=nullptr;i.native->check(i.native->symbol<int(*)(d::native_class_reference::Api::H,const char*,d::U*,const char**)>("XGBoosterSaveModelToBuffer")(i.native->model,"{\"format\":\"json\"}",&size,&buffer),"resident in-memory model export");
  d::need(buffer&&size&&size<=SIZE_MAX,"resident model buffer missing/extent");TrainingResult out;out.model_json.assign(buffer,size);out.model_sha256=d::sha256(out.model_json);i.model_pin=out.model_sha256;++i.trial;
  out.metrics={{"trial",i.trial},{"hyperparameters",hp},{"features",i.F},{"classes",i.K},{"FIT_rows",i.rows},{"native_objective","multi:softmax"},{"CUDA_training",true},{"VALID_read",false},{"TEST_read",false},{"selection_performed",false},{"CPU_model_predictions",false},{"trial_file_reads",0},{"trial_file_writes",0},{"dataset_loads",i.session.at("dataset_loads")},{"data_loaded_from_host",i.session.at("data_loaded_from_host")},{"data_source",i.session.at("data_source")},{"matrix_creations",1},{"native_model_loads",0},{"model_bytes",out.model_json.size()},{"model_sha256",out.model_sha256},{"model_buffer_hashes",1},{"seconds_including_buffer_export",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};return out;
}
TrainingResult ResidentTrainer::train(const json&parameters,const TrainingLossSpec&loss,
    const FitGroupView&groups){
  d::need(bool(impl_),"moved-from resident trainer");auto&i=*impl_;
  const auto hp=validate_hyperparameters(parameters,i.K);
  auto provenance=validate_loss(hp,loss,i.rows,groups.groups,i.K);
  d::need(groups.rows==i.rows&&groups.owner&&groups.fit_owner&&i.source_owner&&
    !groups.fit_owner.owner_before(i.source_owner)&&!i.source_owner.owner_before(groups.fit_owner)&&
    groups.fit_binding==i.session.at("dataset"),"group loss FIT owner/ordered binding differs");
  d::need(groups.group_ids&&groups.group_specs&&
    reinterpret_cast<std::uintptr_t>(groups.group_ids)%alignof(std::uint32_t)==0&&
    reinterpret_cast<std::uintptr_t>(groups.group_specs)%alignof(FitGroupSpec)==0,
    "group loss CUDA pointers missing/misaligned");
  resident_allocation(groups.group_ids,i.rows*sizeof(std::uint32_t),"group IDs exceed CUDA0 allocation");
  resident_allocation(groups.group_specs,std::uint64_t(groups.groups)*sizeof(FitGroupSpec),"group counts exceed CUDA0 allocation");
  // Resolve the supported native API before invalidating a completed model.
  i.native->symbol<int(*)(NativeHandle,NativeHandle,int,const char*,const char*)>("XGBoosterTrainOneIter");
  d::need(i.trial<UINT64_MAX,"completed FIT counter overflow");
  const auto start=std::chrono::steady_clock::now();
  d::Dev<float>gradients(i.rows*2),hessians(i.rows*2),previous_margins(i.rows*2);
  d::Dev<group_loss::StepAudit>step_audit(1);step_audit.zero();
  d::Dev<float>incremental_check;
  if(loss.verify_incremental_margins)incremental_check=d::Dev<float>(i.rows*2);
  const bool incremental=loss.margin_mode==TrainingMarginMode::incremental_native_proxy;
  // Independent prediction-only proxy. The FIT DMatrix labels, weights and
  // base-margin metadata are never changed. Destruction precedes native dlclose.
  OwnedNativeHandle prediction_proxy(i.native->free_matrix);
  if(incremental){
    i.native->check(i.native->symbol<int(*)(NativeHandle*)>("XGProxyDMatrixCreate")(&prediction_proxy.handle),"create prediction-only native proxy");
    if(prediction_proxy.handle==i.native->matrix||prediction_proxy.handle==i.native->model){prediction_proxy.handle=nullptr;d::need(false,"prediction proxy aliases training owner");}
    d::need(prediction_proxy.handle,"prediction proxy owner missing");
    const auto input=i.native->array(i.values.p,i.rows);
    i.native->check(i.native->symbol<int(*)(NativeHandle,const char*)>("XGProxyDMatrixSetDataCudaArrayInterface")(prediction_proxy.handle,input.c_str()),"bind CUDA prediction proxy features");d::gpu_sync();
  }
  d::Dev<group_loss::GroupStats>stats(groups.groups);d::Dev<std::uint32_t>status(1);
  group_loss::Config cfg;cfg.beta=loss.beta;cfg.loss_scale=loss.loss_scale;
  cfg.curvature_policy=loss.curvature_policy;
  cfg.eta=provenance.at("effective_eta");cfg.max_delta_step=provenance.at("effective_max_delta_step");
  cfg.require_all_positives=loss.require_all_positives;
  i.invalidate_probability_view();i.model_pin.clear();i.native->configure(hp,i.K);
  i.native->set("multi_strategy","one_output_per_tree");i.native->set("num_parallel_tree","1");
  verify_external_configuration(json::parse(i.native->configuration()),provenance);
  const auto rounds=hp.at("rounds").get<std::uint32_t>();
  d::need(i.rows<=UINT64_MAX/rounds,"step audit row-round count overflow");
  const float*margin=native_prediction(*i.native,i.native->model,i.values.p,i.rows,i.F,i.K,true,false);
  status.zero();audit_zero_margins<<<d::blocks(i.rows*2),256>>>(margin,i.rows*2,status.p);d::done();
  d::need(!status.at(0),"native initial margin words are not explicit positive zero");
  for(std::uint32_t round=0;round<rounds;++round){
    d::ck(cudaMemcpy(previous_margins.p,margin,i.rows*2*sizeof(float),cudaMemcpyDeviceToDevice),"save preupdate native margins");
    group_loss::DeviceInputs input{margin,i.labels.p,groups.group_ids,groups.group_specs,i.rows,groups.groups};
    group_loss::DeviceOutputs output{gradients.p,hessians.p,stats.p,status.p};
    d::ck(group_loss::enqueue_objective(input,output,cfg,nullptr),"group loss CUDA launches");d::done();
    const auto error=status.at(0);
    d::need(error==group_loss::ok,"group loss rejected numerical/support contract, status="+std::to_string(error));
    // The objective has consumed the borrowed margins completely. No DMatrix
    // weights are applied to g/h here; those remain fixed for native cuts.
    i.native->external_update(round,gradients.p,hessians.p,i.rows,i.K);
    if(incremental){
      const auto base=binary_margin_array(previous_margins.p,i.rows);
      i.native->check(i.native->symbol<int(*)(NativeHandle,const char*,const char*)>("XGDMatrixSetInfoFromInterface")(
        prediction_proxy.handle,"base_margin",base.c_str()),"copy previous margins into prediction-only proxy");d::gpu_sync();
      // round is the newly completed boosting layer. end=0 means all trees in
      // the native API, so the initial empty model always uses the full path.
      margin=native_prediction(*i.native,i.native->model,i.values.p,i.rows,i.F,i.K,true,false,
        round+1,round,prediction_proxy.handle);
      if(loss.verify_incremental_margins){
        d::ck(cudaMemcpy(incremental_check.p,margin,i.rows*2*sizeof(float),cudaMemcpyDeviceToDevice),"save incremental differential margins");
        const auto*full=native_prediction(*i.native,i.native->model,i.values.p,i.rows,i.F,i.K,true,false);
        compare_margin_words<<<d::blocks(i.rows*2),256>>>(incremental_check.p,full,i.rows*2,status.p);d::done();
        d::need(!status.at(0),"incremental native margins differ from full null-proxy prediction");
        margin=incremental_check.p;
      }
    }else margin=native_prediction(*i.native,i.native->model,i.values.p,i.rows,i.F,i.K,true,false);
    d::ck(group_loss::enqueue_step_audit(previous_margins.p,margin,i.rows,
      float(cfg.eta),float(cfg.max_delta_step),group_loss::radius(cfg),step_audit.p,status.p,nullptr),"native step audit launch");d::done();
    d::need(!status.at(0),"native postupdate margins exceed the finite FP32 step envelope");
  }
  const auto observed=step_audit.at(0);
  d::need(observed.rows_checked==i.rows*rounds,"native step audit coverage differs");
  provenance["step_audit"]={{"rows_checked",observed.rows_checked},{"updates_checked",rounds},
    {"maximum_abs_logit_step",observed.maximum_abs_logit_step},
    {"maximum_FP32_rounding_envelope",observed.maximum_rounding_envelope},
    {"steps_exceeding_real_radius",observed.steps_exceeding_real_radius},
    {"real_radius",group_loss::radius(cfg)},{"FP32_descent_certificate",false}};
  const auto config=json::parse(i.native->configuration());verify_external_configuration(config,provenance);
  const auto&booster=config.at("learner").at("gradient_booster");
  d::need(booster.at("updater").size()==1&&booster.at("updater")[0].at("name")=="grow_gpu_hist",
          "external objective native updater is not required CUDA updater");
  int completed=-1;i.native->check(i.native->symbol<int(*)(NativeHandle,int*)>("XGBoosterBoostedRounds")(i.native->model,&completed),"external objective completed rounds");
  d::need(completed==int(rounds),"external objective native round count differs");
  d::U size=0;const char*buffer=nullptr;
  i.native->check(i.native->symbol<int(*)(NativeHandle,const char*,d::U*,const char**)>("XGBoosterSaveModelToBuffer")(i.native->model,"{\"format\":\"json\"}",&size,&buffer),"external objective in-memory model export");
  d::need(buffer&&size&&size<=SIZE_MAX,"external objective model buffer missing/extent");
  TrainingResult out;out.model_json.assign(buffer,size);
  // SaveJsonConfig exposes the actual model tree count without parsing native
  // model JSON, whose Infinity sentinels are not strict JSON numbers.
  const auto tree_count=std::stoull(booster.at("gbtree_model_param").at("num_trees").get<std::string>());
  d::need(tree_count==std::uint64_t(rounds)*2,"external objective native tree count differs");
  out.model_sha256=d::sha256(out.model_json);
  provenance["fit_binding"]=groups.fit_binding;provenance["initial_margin_words_verified_zero"]=true;
  provenance["data_cut_weights_present"]=bool(i.weights.p);provenance["native_external_updates"]=rounds;
  provenance["native_tree_count"]=std::uint64_t(rounds)*2;
  provenance["native_full_margin_predictions"]=incremental?(loss.verify_incremental_margins?rounds+1:1):rounds+1;
  provenance["native_incremental_margin_predictions"]=incremental?rounds:0;
  provenance["incremental_margin_words_compared"]=loss.verify_incremental_margins?i.rows*2*rounds:0;
  provenance["training_DMatrix_base_margin_modified"]=false;
  out.metrics={{"trial",i.trial+1},{"hyperparameters",hp},{"training_loss",provenance},
    {"features",i.F},{"classes",i.K},{"FIT_rows",i.rows},{"native_objective","multi:softmax"},
    {"CUDA_training",true},{"VALID_read",false},{"TEST_read",false},{"selection_performed",false},
    {"CPU_model_predictions",false},{"trial_file_reads",0},{"trial_file_writes",0},
    {"dataset_loads",i.session.at("dataset_loads")},{"data_loaded_from_host",i.session.at("data_loaded_from_host")},
    {"data_source",i.session.at("data_source")},{"matrix_creations",1},{"native_model_loads",0},
    {"model_bytes",out.model_json.size()},{"model_sha256",out.model_sha256},{"model_buffer_hashes",1},
    {"seconds_including_buffer_export",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
  i.session["last_completed_custom_training_loss"]=provenance;i.model_pin=out.model_sha256;++i.trial;
  return out;
}
void ResidentTrainer::Impl::verify_tree(NativeHandle source,const char*objective){
  verify_native_tree(*native,source,F,K,objective);
}
void ResidentTrainer::Impl::invalidate_probability_view(){
  clear_probability(*native,probability,session);
}
NativeHandle ResidentTrainer::Impl::ensure_probability_view(){
  return probability_model(*native,F,K,model_pin,probability,session,prefix_parent.handle,trial,aliases_prefix,this);
}
TrainingResult ResidentTrainer::Impl::make_prefix(NativeHandle source,const std::string&source_pin,std::uint32_t rounds,OwnedNativeHandle&sliced,bool source_prevalidated){
  d::need(source&&!source_pin.empty()&&!sliced.handle,"round prefix requires a completed native source and empty temporary owner");
  d::need(rounds>0&&rounds<=std::uint32_t(INT_MAX),"positive native prefix rounds within int capacity required");
  auto&i=*this;auto&api=*i.native;using H=NativeHandle;
  const auto start=std::chrono::steady_clock::now();const auto parent_pin=source_pin;
  d::gpu_sync();
  const auto boosted_rounds=api.symbol<int(*)(H,int*)>("XGBoosterBoostedRounds");
  int original_rounds=-1;api.check(boosted_rounds(source,&original_rounds),"native prefix original rounds");
  d::need(original_rounds>0&&rounds<=std::uint32_t(original_rounds),"native prefix exceeds current completed rounds");
  if(source_prevalidated)d::need(source==prefix_parent.handle&&source_pin==prefix_parent_pin,"native retained source binding differs");
  else verify_tree(source);
  // C API slice indices count boosting rounds, including every class tree
  // in each selected round. Copy every borrowed API buffer before another call.
  const int slice_status=api.symbol<int(*)(H,int,int,int,H*)>("XGBoosterSlice")(source,0,int(rounds),1,&sliced.handle);
  if(sliced.handle==source||sliced.handle==api.model||aliases_prefix(this,sliced.handle)||sliced.handle==probability.owner.handle){sliced.handle=nullptr;api.check(slice_status,"native round-prefix slice");d::need(false,"native prefix slice handle aliases an existing owner");}
  api.check(slice_status,"native round-prefix slice");
  d::need(sliced.handle,"native prefix slice handle missing");
  api.check(api.symbol<int(*)(H,const char*,const char*)>("XGBoosterSetParam")(sliced.handle,"device","cuda:0"),"native prefix CUDA device");
  int selected_rounds=-1;api.check(boosted_rounds(sliced.handle,&selected_rounds),"native prefix selected rounds");
  d::need(selected_rounds==int(rounds),"native prefix round count differs");verify_tree(sliced.handle);
  d::U size=0;const char*buffer=nullptr;
  api.check(api.symbol<int(*)(H,const char*,d::U*,const char**)>("XGBoosterSaveModelToBuffer")(sliced.handle,"{\"format\":\"json\"}",&size,&buffer),"native prefix in-memory export");
  d::need(buffer&&size&&size<=SIZE_MAX,"native prefix model buffer missing/extent");
  TrainingResult out;out.model_json.assign(buffer,size);out.model_sha256=d::sha256(out.model_json);
  d::gpu_sync();
  out.metrics={{"trial",i.trial},{"completed_FIT_calls",i.trial},{"operation","native_existing_round_prefix"},{"FIT_performed",false},{"CUDA_training",false},{"features",i.F},{"classes",i.K},{"FIT_rows",i.rows},{"native_objective","multi:softmax"},{"native_library_sha256",i.session.at("native_library_sha256")},{"source_parent_sha256",parent_pin},{"original_rounds",original_rounds},{"selected_rounds",rounds},{"model_bytes",out.model_json.size()},{"model_sha256",out.model_sha256},{"model_buffer_hashes",1},{"VALID_read",false},{"TEST_read",false},{"selection_performed",false},{"CPU_model_predictions",false},{"trial_file_reads",0},{"trial_file_writes",0},{"dataset_loads",0},{"matrix_creations",0},{"native_model_loads",0},{"native_model_slice_calls",1},{"prefix_slice_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},{"seconds_including_buffer_export",0.0}};
  return out;
}
void ResidentTrainer::Impl::install_prefix(OwnedNativeHandle&sliced,const std::string&pin){
  install_native_model(*native,probability,session,model_pin,sliced,pin);
}
TrainingResult ResidentTrainer::slice_prefix(std::uint32_t rounds){
  d::need(bool(impl_)&&impl_->native&&impl_->native->model&&!impl_->model_pin.empty(),"round prefix requires a current completed native model");
  auto&i=*impl_;OwnedNativeHandle sliced(i.native->free_booster);
  auto out=i.make_prefix(i.native->model,i.model_pin,rounds,sliced);i.install_prefix(sliced,out.model_sha256);return out;
}
bool ResidentTrainer::retain_prefix_source(const std::string&bytes,const std::string&expected_sha){
  d::need(bool(impl_)&&impl_->native,"moved-from resident trainer");auto&i=*impl_;auto&api=*i.native;
  // A repeated explicit bind still verifies its supplied bytes. Slice calls take
  // no source buffer, so the established immutable handle needs no trial rehash.
  d::need(!bytes.empty()&&d::sha256(bytes)==expected_sha,"retained prefix source buffer identity differs");
  if(i.prefix_parent.handle&&i.prefix_parent_pin==expected_sha)return false;
  if(select_retained_prefix_source(expected_sha))return false;
  const auto started=std::chrono::steady_clock::now();
  auto next_pin=expected_sha;OwnedNativeHandle candidate(api.free_booster);
  load_native_model(api,bytes,i.F,i.K,candidate,i.prefix_parent.handle,i.probability.owner.handle,true,Impl::aliases_prefix,&i);
  d::need(i.prefix_parent_loads<UINT64_MAX,"retained prefix source load extent");
  i.park_prefix_parent();
  i.prefix_parent.handle=std::exchange(candidate.handle,nullptr);i.prefix_parent.free=api.free_booster;i.prefix_parent_pin.swap(next_pin);
  i.prefix_parent_json_bytes=bytes.size();++i.prefix_parent_loads;i.prefix_parent_load_pending=true;
  i.prefix_parent_load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  i.prefix_cache_hit_pending=false;i.trim_prefix_cache();i.publish_prefix_cache();return true;
}
const std::string&ResidentTrainer::retained_prefix_source_sha256()const{
  d::need(bool(impl_),"moved-from resident trainer");return impl_->prefix_parent_pin;
}
bool ResidentTrainer::select_retained_prefix_source(const std::string&pin){
  d::need(bool(impl_)&&impl_->native,"moved-from resident trainer");auto&i=*impl_;
  if(i.prefix_parent.handle&&i.prefix_parent_pin==pin)return true;
  auto found=std::find_if(i.prefix_parents.begin(),i.prefix_parents.end(),[&](const auto&p){return p.pin==pin;});
  if(found==i.prefix_parents.end())return false;
  auto next_pin=found->pin;d::need(i.prefix_cache_hits<UINT64_MAX,"native parent cache hit extent");d::gpu_sync();
  i.park_prefix_parent();
  i.prefix_parent.handle=std::exchange(found->owner.handle,nullptr);i.prefix_parent.free=i.native->free_booster;
  i.prefix_parent_probability=std::move(found->probability);
  i.prefix_parent_pin.swap(next_pin);i.prefix_parent_json_bytes=found->bytes;
  i.prefix_parent_load_pending=found->load_pending;i.prefix_parent_load_seconds=found->load_seconds;
  i.prefix_cache_bytes-=found->bytes;i.prefix_parents.erase(found);++i.prefix_cache_hits;
  i.prefix_cache_hit_pending=true;i.trim_prefix_cache();i.publish_prefix_cache();return true;
}
void ResidentTrainer::set_retained_prefix_cache_byte_budget(std::uint64_t bytes){
  d::need(bool(impl_)&&impl_->native,"moved-from resident trainer");auto&i=*impl_;d::gpu_sync();
  i.prefix_cache_budget=bytes;i.trim_prefix_cache();i.publish_prefix_cache();
}
void ResidentTrainer::clear_retained_prefix_sources(){
  d::need(bool(impl_)&&impl_->native,"moved-from resident trainer");auto&i=*impl_;d::gpu_sync();
  while(!i.prefix_parents.empty()){
    auto parent=i.prefix_parents.begin();
    i.free_parent_probability(parent->probability);
    i.native->check(i.native->free_booster(parent->owner.handle),"clear inactive native parent");
    parent->owner.handle=nullptr;i.prefix_cache_bytes-=parent->bytes;i.prefix_parents.erase(parent);
  }
  i.free_parent_probability(i.prefix_parent_probability);
  if(i.prefix_parent.handle){i.native->check(i.native->free_booster(i.prefix_parent.handle),"clear selected native parent");i.prefix_parent.handle=nullptr;}
  i.prefix_parent_pin.clear();i.prefix_parent_json_bytes=0;i.prefix_parent_load_pending=false;
  i.prefix_parent_load_seconds=0;i.prefix_cache_hit_pending=false;i.publish_prefix_cache();
}
TrainingResult ResidentTrainer::slice_retained_prefix(std::uint32_t rounds){
  d::need(bool(impl_)&&impl_->prefix_parent.handle&&!impl_->prefix_parent_pin.empty(),"no retained native prefix source");
  auto&i=*impl_;OwnedNativeHandle sliced(i.native->free_booster);
  auto out=i.make_prefix(i.prefix_parent.handle,i.prefix_parent_pin,rounds,sliced,true);
  out.metrics["retained_prefix_source"]=true;out.metrics["native_parent_model_loads"]=i.prefix_parent_load_pending?1:0;
  out.metrics["native_model_loads"]=i.prefix_parent_load_pending?1:0;out.metrics["native_parent_model_loads_total"]=i.prefix_parent_loads;
  out.metrics["native_parent_load_seconds"]=i.prefix_parent_load_pending?i.prefix_parent_load_seconds:0.0;
  out.metrics["native_parent_json_bytes"]=i.prefix_parent_json_bytes;
  out.metrics["native_parent_cache_hit"]=i.prefix_cache_hit_pending;
  out.metrics["native_parent_cache_hits_total"]=i.prefix_cache_hits;
  out.metrics["native_parent_cache_evictions_total"]=i.prefix_cache_evictions;
  i.install_prefix(sliced,out.model_sha256);i.prefix_parent_load_pending=false;i.prefix_cache_hit_pending=false;return out;
}
const float*ResidentTrainer::predict_retained_prefix_probabilities(const float*x,std::uint64_t rows,std::uint32_t rounds){
  d::need(bool(impl_)&&impl_->native&&impl_->prefix_parent.handle&&!impl_->prefix_parent_pin.empty(),"no retained native response source");
  auto&i=*impl_;auto&api=*i.native;native_input(x,rows,i.F,i.K);
  int maximum=-1;api.check(api.symbol<int(*)(NativeHandle,int*)>("XGBoosterBoostedRounds")(i.prefix_parent.handle,&maximum),"retained response source rounds");
  d::need(rounds&&maximum>0&&rounds<=std::uint32_t(maximum),"native response prefix rounds outside source");
  if(!i.prefix_parent_probability)i.prefix_parent_probability=std::make_unique<ProbabilityState>();
  auto&state=*i.prefix_parent_probability;json view_session={{"native_library_sha256",i.session.at("native_library_sha256")}};
  const auto before=state.clones;
  const auto handle=probability_model(api,i.F,i.K,i.prefix_parent_pin,state,view_session,
      i.probability.owner.handle,i.trial,Impl::aliases_prefix,&i,i.prefix_parent.handle);
  const auto started=std::chrono::steady_clock::now();
  const float*out=native_prediction(api,handle,x,rows,i.F,i.K,false,true,rounds);
  d::need(state.predictions<UINT64_MAX,"retained response prediction count overflow");++state.predictions;
  auto receipt=state.view_metadata;
  receipt["operation"]="native_iteration_range_probability_view";
  receipt["iteration_begin"]=0;receipt["iteration_end"]=rounds;
  receipt["source_parent_rounds"]=maximum;receipt["rounds"]=rounds;
  receipt["prediction_calls"]=state.predictions;receipt["native_view_clones_this_call"]=state.clones-before;
  receipt["completed_FIT_calls"]=i.trial;
  receipt["native_parent_model_loads"]=i.prefix_parent_load_pending?1:0;
  receipt["native_parent_model_loads_total"]=i.prefix_parent_loads;
  receipt["native_parent_cache_hit"]=i.prefix_cache_hit_pending;
  receipt["native_model_exports_this_call"]=0;receipt["native_prefix_slice_calls_this_call"]=0;
  receipt["prefix_prediction_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  receipt["response_source"]={{"format","native-round-prefix-response-1"},{"parent_model_sha256",i.prefix_parent_pin},
      {"iteration_begin",0},{"iteration_end",rounds},{"source_objective","multi:softmax"},
      {"derived_objective","multi:softprob"},{"native_library_sha256",i.session.at("native_library_sha256")}};
  i.session["retained_prefix_probability_response"]=std::move(receipt);
  i.prefix_parent_load_pending=false;i.prefix_cache_hit_pending=false;i.publish_prefix_cache();return out;
}
void ResidentTrainer::restore_model(const std::string& bytes,const std::string& expected_sha){
  d::need(bool(impl_)&&impl_->native,"moved-from resident trainer");auto&i=*impl_;auto&api=*i.native;
  d::need(!bytes.empty()&&d::sha256(bytes)==expected_sha,"retained model buffer identity differs");
  auto next_pin=expected_sha;OwnedNativeHandle candidate(api.free_booster);
  load_native_model(api,bytes,i.F,i.K,candidate,i.prefix_parent.handle,i.probability.owner.handle,false,Impl::aliases_prefix,&i);
  i.install_prefix(candidate,next_pin);
}
void ResidentTrainer::restore_completed_trial_count(std::uint64_t completed){
  d::need(bool(impl_)&&completed<UINT64_MAX,"resident restored trial counter extent");impl_->trial=completed;
}
const float*ResidentTrainer::predict(const float*x,std::uint64_t rows,bool margin){
  d::need(bool(impl_)&&!impl_->model_pin.empty(),"resident current-model prediction requires a completed model");auto&i=*impl_;
  return native_prediction(*i.native,i.native->model,x,rows,i.F,i.K,margin,false);
}
const float*ResidentTrainer::predict_probabilities(const float*x,std::uint64_t rows){
  d::need(bool(impl_)&&impl_->native&&!impl_->model_pin.empty(),"probability response requires a current completed native model");auto&i=*impl_;
  native_input(x,rows,i.F,i.K);auto handle=i.ensure_probability_view();
  const float*out=native_prediction(*i.native,handle,x,rows,i.F,i.K,false,true);
  d::need(i.probability.predictions<UINT64_MAX,"native probability prediction counter overflow");++i.probability.predictions;
  i.session["probability_response_view"]["prediction_calls"]=i.probability.predictions;
  return out;
}
std::uint32_t ResidentTrainer::features()const{d::need(bool(impl_),"moved-from resident trainer");return impl_->F;}
std::uint32_t ResidentTrainer::classes()const{d::need(bool(impl_),"moved-from resident trainer");return impl_->K;}
const std::string&ResidentTrainer::current_model_sha256()const{d::need(bool(impl_)&&!impl_->model_pin.empty(),"resident trainer has no current completed model");return impl_->model_pin;}
const json&ResidentTrainer::metadata()const{d::need(bool(impl_),"moved-from resident trainer");return impl_->session;}
struct ResidentPredictor::Impl {
  json session;
  std::uint32_t F=0,K=0;
  std::string library_path,model_pin;
  std::unique_ptr<NativeApi> native;
  // Declaration order releases the derived view before closing the library.
  ProbabilityState probability;
  std::uint64_t model_loads=0;
  Impl(const json&p,std::uint32_t features,std::uint32_t classes,
       const std::string*verified_path=nullptr,void*verified_library=nullptr):F(features),K(classes){
    d::need(F>0&&F<=std::uint32_t(INT32_MAX)&&K>=2&&K<=d::class_native_training::maximum_exact_native_classes,"native predictor feature/class capacity");
    d::need(!d::native_class_reference::library_identity(p).empty(),"native predictor pinned library metadata");
    library_path=p.at("native_library_path").get<std::string>();
    if(verified_path)d::need(verified_library&&*verified_path==library_path,"verified predictor library path differs");
    else d::need(d::sha256(d::read_text(library_path))==p.at("native_library_sha256").get<std::string>(),"native predictor library bytes differ");
    native=std::make_unique<NativeApi>(library_path,F);
    if(verified_path)d::need(native->library==verified_library,"verified predictor loaded library handle differs");
    d::ck(cudaSetDevice(0),"native predictor device");
    session={{"features",F},{"classes",K},{"inference_only",true},{"training_supported",false},{"matrix_creations",0},{"training_values_bytes",0},{"training_labels_bytes",0},{"native_objective","multi:softmax"},{"native_library_path",library_path},{"native_library_sha256",p.at("native_library_sha256").get<std::string>()},{"native_library_content_reads",verified_path?0:1},{"native_library_content_reads_skipped",verified_path?1:0},{"model_loads",0},{"prediction_file_reads",0},{"prediction_file_writes",0},{"native_probability_view_memory_bytes_measured",false}};
  }
};
ResidentPredictor::ResidentPredictor(const json&p,std::uint32_t F,std::uint32_t K):impl_(std::make_unique<Impl>(p,F,K)){}
ResidentPredictor::ResidentPredictor(const json&p,std::uint32_t F,std::uint32_t K,const ResidentTrainer&verified){
  d::need(bool(verified.impl_)&&verified.impl_->native&&verified.impl_->native->library&&verified.impl_->session.at("native_library_sha256")==p.at("native_library_sha256"),"moved-from/unverified predictor training authorization");
  impl_=std::make_unique<Impl>(p,F,K,&verified.impl_->library_path,verified.impl_->native->library);
}
ResidentPredictor::ResidentPredictor(const json&p,std::uint32_t F,std::uint32_t K,const ResidentPredictor&verified){
  d::need(bool(verified.impl_)&&verified.impl_->native&&verified.impl_->native->library&&verified.impl_->session.at("native_library_sha256")==p.at("native_library_sha256"),"moved-from/unverified predictor authorization");
  impl_=std::make_unique<Impl>(p,F,K,&verified.impl_->library_path,verified.impl_->native->library);
}
ResidentPredictor::~ResidentPredictor()=default;
ResidentPredictor::ResidentPredictor(ResidentPredictor&&)noexcept=default;
ResidentPredictor&ResidentPredictor::operator=(ResidentPredictor&&)noexcept=default;
void ResidentPredictor::restore_model(const std::string&bytes,const std::string&expected_sha){
  d::need(bool(impl_)&&impl_->native,"moved-from native predictor");auto&i=*impl_;
  d::need(!bytes.empty()&&d::sha256(bytes)==expected_sha,"native predictor model identity differs");
  d::need(i.model_loads<UINT64_MAX,"native predictor load counter overflow");
  OwnedNativeHandle candidate(i.native->free_booster);
  load_native_model(*i.native,bytes,i.F,i.K,candidate,nullptr,i.probability.owner.handle,false);
  auto next_session=i.session;next_session["model_loads"]=i.model_loads+1;
  next_session["source_model_sha256"]=expected_sha;
  install_native_model(*i.native,i.probability,next_session,i.model_pin,candidate,expected_sha);
  ++i.model_loads;i.session.swap(next_session);
}
const float*ResidentPredictor::predict(const float*x,std::uint64_t rows,bool margin){
  d::need(bool(impl_)&&!impl_->model_pin.empty(),"native predictor requires a loaded model");auto&i=*impl_;
  return native_prediction(*i.native,i.native->model,x,rows,i.F,i.K,margin,false);
}
const float*ResidentPredictor::predict_probabilities(const float*x,std::uint64_t rows){
  d::need(bool(impl_)&&!impl_->model_pin.empty(),"native probability predictor requires a loaded model");auto&i=*impl_;
  native_input(x,rows,i.F,i.K);
  auto handle=probability_model(*i.native,i.F,i.K,i.model_pin,i.probability,i.session,nullptr);
  auto out=native_prediction(*i.native,handle,x,rows,i.F,i.K,false,true);
  d::need(i.probability.predictions<UINT64_MAX,"native predictor probability counter overflow");++i.probability.predictions;
  i.session["probability_response_view"]["prediction_calls"]=i.probability.predictions;return out;
}
std::uint32_t ResidentPredictor::features()const{d::need(bool(impl_),"moved-from native predictor");return impl_->F;}
std::uint32_t ResidentPredictor::classes()const{d::need(bool(impl_),"moved-from native predictor");return impl_->K;}
const std::string&ResidentPredictor::current_model_sha256()const{d::need(bool(impl_)&&!impl_->model_pin.empty(),"native predictor has no loaded model");return impl_->model_pin;}
const json&ResidentPredictor::metadata()const{d::need(bool(impl_),"moved-from native predictor");return impl_->session;}
} // namespace class_study
