#pragma once
// Shared existing native training staging and C API sequence; import after dataset helpers.
namespace class_native_training {
constexpr u64 maximum_exact_native_classes = 16777217ULL;
u64 integer_value(const J& value, const char* name, u64 maximum) {
  need(value.is_number_integer(),std::string(name)+" must be an integer");
  u64 out=0;
  if(value.is_number_unsigned())out=value.get<u64>();
  else { auto v=value.get<std::int64_t>();need(v>=0,std::string(name)+" must be nonnegative");out=u64(v); }
  need(out<=maximum,std::string(name)+" exceeds native metadata capacity");return out;
}
double numeric_value(const J& value,const char* name,double low,double high,bool strict_low=false) {
  need(value.is_number(),std::string(name)+" must be numeric");double out=value.get<double>();
  need(std::isfinite(out)&&(strict_low?out>low:out>=low)&&out<=high,std::string(name)+" is outside the supported native parameter range");return out;
}
J parameters(const J& p,u32 K) {
  auto in=p.at("hyperparameters");need(in.is_object(),"hyperparameters must be an object");
  const std::set<std::string> allowed={"rounds","max_depth","eta","lambda","alpha","gamma","min_child_weight","max_bin","seed","subsample","colsample_bytree","colsample_bylevel","colsample_bynode","max_delta_step"};
  for(auto i=in.begin();i!=in.end();++i)need(allowed.contains(i.key()),"unsupported training hyperparameter: "+i.key());
  J hp;
  hp["rounds"]=integer_value(in.at("rounds"),"rounds",u64(INT_MAX)/K);
  hp["max_depth"]=integer_value(in.at("max_depth"),"max_depth",INT_MAX);
  need(hp.at("rounds").get<u64>()>0&&hp.at("max_depth").get<u64>()>0,"positive rounds and max_depth required");
  hp["max_bin"]=integer_value(in.value("max_bin",J(256)),"max_bin",INT_MAX);need(hp.at("max_bin").get<u64>()>=2,"max_bin must be at least 2");
  hp["seed"]=integer_value(in.value("seed",J(0)),"seed",INT_MAX);
  hp["eta"]=numeric_value(in.value("eta",J(0.3)),"eta",0,1,true);
  for(const char* key:{"lambda","alpha","gamma","min_child_weight"})
    hp[key]=numeric_value(in.value(key,J(std::string(key)=="lambda"||std::string(key)=="min_child_weight"?1.0:0.0)),key,0,std::numeric_limits<double>::max());
  // Omitted sampling controls retain the original normalized checkpoint object.
  for(const char* key:{"subsample","colsample_bytree","colsample_bylevel","colsample_bynode"})
    if(in.contains(key))hp[key]=numeric_value(in.at(key),key,0,1,true);
  if(in.contains("max_delta_step"))hp["max_delta_step"]=numeric_value(in.at("max_delta_step"),"max_delta_step",0,std::numeric_limits<float>::max());
  return hp;
}
// Labels and active feature words are checked and staged on CUDA, including
// exact conversion from the common u32 labels to the native FP32 label API.
__global__ void prepare_labels(const u32* input,float* output,u64 rows,u32 K,u32* bad) {
  for(u64 r=u64(blockIdx.x)*blockDim.x+threadIdx.x;r<rows;r+=u64(blockDim.x)*gridDim.x) {
    u32 y=input[r];float value=float(y);
    if(y>=K||double(value)!=double(y)){atomicOr(bad,1u);continue;}output[r]=value;
  }
}
__global__ void prepare_values(const float* input,float* output,u64 rows,u64 stride,u32 F,u32* bad) {
  for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*u64(F);i+=u64(blockDim.x)*gridDim.x) {
    float value=input[(i/F)*stride+i%F];if(isinf(value)){atomicOr(bad,2u);continue;}output[i]=value;
  }
}
__global__ void prepare_weights(const float* input,float* output,u64 rows,u32* status) {
  for(u64 r=u64(blockIdx.x)*blockDim.x+threadIdx.x;r<rows;r+=u64(blockDim.x)*gridDim.x) {
    const float value=input[r];
    if(!isfinite(value)||value<0){atomicOr(status,1u);continue;}
    output[r]=value;if(value>0)atomicOr(status,2u);
  }
}
struct TrainingApi final : native_class_reference::Api {
  using native_class_reference::Api::Api;
  void training_matrix(const float* x,const float* y,U rows,const float* weights=nullptr) {
    need(!matrix,"training matrix already exists");auto a=array(x,rows),b=array(y,rows,true);
    check(symbol<int(*)(const char*,const char*,H*)>("XGDMatrixCreateFromCudaArrayInterface")(a.c_str(),"{\"missing\":NaN,\"nthread\":2}",&matrix),"create native CUDA FIT DMatrix");
    check(symbol<int(*)(H,const char*,const char*)>("XGDMatrixSetInfoFromInterface")(matrix,"label",b.c_str()),"assign CUDA FIT labels");
    if(weights){auto w=array(weights,rows,true);check(symbol<int(*)(H,const char*,const char*)>("XGDMatrixSetInfoFromInterface")(matrix,"weight",w.c_str()),"assign CUDA FIT weights");}
  }
  void configure(const J& hp,u32 K) {
    if(model){check(free_booster(model),"free booster");model=nullptr;}
    check(symbol<int(*)(const H*,U,H*)>("XGBoosterCreate")(&matrix,1,&model),"create training booster");
    auto token=[](const J& v){return v.dump();};
    J zero=J::array();for(u32 k=0;k<K;++k)zero.push_back(0);
    for(const auto&[key,value]:std::map<std::string,std::string>{
        {"objective","multi:softmax"},{"num_class",std::to_string(K)},{"tree_method","hist"},{"device","cuda:0"},{"booster","gbtree"},
        {"max_bin",token(hp.at("max_bin"))},{"max_depth",token(hp.at("max_depth"))},{"eta",token(hp.at("eta"))},
        {"lambda",token(hp.at("lambda"))},{"alpha",token(hp.at("alpha"))},{"gamma",token(hp.at("gamma"))},{"min_child_weight",token(hp.at("min_child_weight"))},
        {"subsample",token(hp.value("subsample",J(1)))},{"colsample_bytree",token(hp.value("colsample_bytree",J(1)))},{"colsample_bylevel",token(hp.value("colsample_bylevel",J(1)))},{"colsample_bynode",token(hp.value("colsample_bynode",J(1)))},{"base_score",zero.dump()},
        {"boost_from_average","0"},{"seed",token(hp.at("seed"))},{"seed_per_iteration","0"},{"nthread","2"},{"verbosity","0"},{"validate_parameters","1"}})
      set(key.c_str(),value);
    if(hp.contains("max_delta_step"))set("max_delta_step",token(hp.at("max_delta_step")));
  }
  void fit(U rounds) {for(U i=0;i<rounds;++i){check(symbol<int(*)(H,int,H)>("XGBoosterUpdateOneIter")(model,int(i),matrix),"native CUDA FIT update");gpu_sync();}}
  // Gradients have K columns, independent of input feature width D. The caller
  // validates CUDA ownership, finite values, positive Hessians and lifecycle.
  void external_update(U iteration,const float* gradients,const float* hessians,U rows,u32 K) {
    need(model&&matrix&&gradients&&hessians&&rows&&K>=2&&iteration<=INT_MAX,"external native update shape/owner");
    auto descriptor=[&](const float* p){return J{{"data",J::array({reinterpret_cast<std::uintptr_t>(p),true})},{"shape",J::array({rows,U(K)})},{"strides",nullptr},{"typestr","<f4"},{"version",3},{"stream",1}}.dump();};
    const auto g=descriptor(gradients),h=descriptor(hessians);
    check(symbol<int(*)(H,H,int,const char*,const char*)>("XGBoosterTrainOneIter")(model,matrix,int(iteration),g.c_str(),h.c_str()),"native CUDA external-gradient FIT update");gpu_sync();
  }
  std::string configuration() {U n=0;const char* p=nullptr;check(symbol<int(*)(H,U*,const char**)>("XGBoosterSaveJsonConfig")(model,&n,&p),"save native configuration");need(p&&n,"native configuration missing");return std::string(p,n);}
  void save(const fs::path& path) {check(symbol<int(*)(H,const char*)>("XGBoosterSaveModel")(model,path.c_str()),"save native source model");}
};
} // namespace class_native_training
