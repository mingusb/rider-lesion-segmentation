#include "class_native_model_json.hpp"
// One configured FIT-only native CUDA XGBoost fit on the common dense descriptor.
// The native C API sequence is inherited from the qualified MNIST trainer;
// dataset shape, labels, class count and parameters are caller declarations.
#include "class_model_dataset.cuh"
#include <chrono>
#ifdef DP_TRAIN_SOURCE_MANIFEST
#include "dp_train_source_manifest.hpp"
#endif
#include <set>
#include "class_native_training.cuh"

namespace {
using namespace class_native_training;
J source_manifest() {
#ifdef DP_TRAIN_SOURCE_MANIFEST
  return J::parse(DP_TRAIN_SOURCE_MANIFEST_JSON);
#else
  fs::path root=fs::path(__FILE__).parent_path();J files=J::object();
  for(const char* name:{"class_model_train.cu","class_model_dataset.cuh","class_model_dataset_contract.hpp","class_native_training.cuh","class_model_io.cuh","class_model_mnist_adapter.cuh","class_model_mnist_dataset.cuh","class_model_native_adapter.cuh","class_io.hpp","class_cuda.hpp","Makefile.class-pipeline"})
    files[name]=sha256(read_text(root/name));
  return {{"files",files},{"sha256",sha256(files.dump())},{"role","maintained compilation-source metadata; executable pin is separate"}};
#endif
}
J check_plan(const fs::path& path,const std::string& pin,const fs::path& exe) {
  Cpath(path);auto bytes=mn_read(path);need(sha256(bytes)==pin,"training plan SHA differs");auto p=J::parse(bytes);
  need(p.at("format")=="class-model-training-plan-1"&&p.at("TEST_read")==false&&p.at("VALID_read")==false&&p.at("selection_performed")==false&&p.at("training_performed")==true,"FIT-only training protocol");
  auto d=p.at("dataset");auto contract=class_model_contract::dense(d,true);auto K=contract.classes;
  need(K<=maximum_exact_native_classes,"class IDs exceed the exact native FP32 label capacity");
  Cpath(d.at("values_path").get<std::string>());Cpath(d.at("labels_path").get<std::string>());
  auto lib=fs::path(p.at("native_library_path").get<std::string>());
  need(sha256(read_text(lib))==native_class_reference::library_identity(p),"pinned native CUDA library differs");
  need(sha256(read_text(exe))==p.at("executable_sha256").get<std::string>(),"training executable differs");
  auto manifest=source_manifest();need(manifest.at("sha256")==p.at("source_manifest_sha256"),"training source manifest differs");
  p["hyperparameters"]=parameters(p,u32(K));p["checked_source_manifest"]=manifest;return p;
}

J train(const fs::path& planpath,const std::string& pin,const fs::path& out,const fs::path& exe,bool& output_owned) {
  auto p=check_plan(planpath,pin,exe);Cpath(out);clean_output(out);
  output_owned=true;
  atomic_json(out/"plan.json",J::parse(mn_read(planpath)));atomic_json(out/"source-manifest.json",p.at("checked_source_manifest"));
  ck(cudaSetDevice(0),"training device");auto start=std::chrono::steady_clock::now();
  auto data=load_model_dataset(p,out,true);Dev<float>x(data.rows*data.F),labels(data.rows);Dev<u32>bad(1);bad.zero();
  prepare_values<<<blocks(x.n),256>>>(data.x.p,x.p,data.rows,data.stride,data.F,bad.p);
  prepare_labels<<<blocks(data.rows),256>>>(data.labels.p,labels.p,data.rows,data.K,bad.p);done();need(!bad.at(0),"FIT features must be finite/NaN and labels exact native class IDs");
  TrainingApi api(p.at("native_library_path").get<std::string>(),data.F);api.training_matrix(x.p,labels.p,data.rows);
  const auto& hp=p.at("hyperparameters");api.configure(hp,data.K);api.fit(hp.at("rounds").get<U>());
  auto config=J::parse(api.configuration());need(config.at("learner").at("generic_param").at("device")=="cuda:0"&&config.at("learner").at("gradient_booster").at("updater").size()==1&&config.at("learner").at("gradient_booster").at("updater")[0].at("name")=="grow_gpu_hist","native training updater differs from the required CUDA updater");
  int rounds=-1;api.check(api.symbol<int(*)(TrainingApi::H,int*)>("XGBoosterBoostedRounds")(api.model,&rounds),"native completed rounds");need(rounds==hp.at("rounds").get<int>(),"native completed round count differs");
  api.save(out/"model.json");auto model_bytes=read_text(out/"model.json");auto model=dp_native_json::parse(model_bytes);
  need(model.at("learner").at("objective").at("name")=="multi:softmax"&&model.at("learner").at("learner_model_param").at("num_feature").get<std::string>()==std::to_string(data.F)&&model.at("learner").at("learner_model_param").at("num_class").get<std::string>()==std::to_string(data.K),"saved native objective/shape differs");
  atomic_json(out/"configuration.json",config);recheck_model_dataset(data);
  need(source_manifest()==p.at("checked_source_manifest")&&sha256(mn_read(planpath))==pin&&sha256(read_text(exe))==p.at("executable_sha256").get<std::string>()&&sha256(read_text(p.at("native_library_path").get<std::string>()))==p.at("native_library_sha256").get<std::string>(),"final training pins differ");
  J result={{"format","class-model-training-result-1"},{"complete",true},{"CUDA_executed",true},{"CPU_model_numerical_work",false},
    {"FIT_only",true},{"FIT_rows",data.rows},{"VALID_read",false},{"TEST_read",false},{"selection_performed",false},{"evaluation_performed",false},
    {"features",data.F},{"classes",data.K},{"dataset",data.binding},{"hyperparameters",hp},{"native_objective","multi:softmax"},
    {"model_path",(out/"model.json").string()},{"model_sha256",sha256(model_bytes)},{"model_bytes",model_bytes.size()},
    {"configuration_sha256",sha256(read_text(out/"configuration.json"))},{"plan_sha256",pin},{"executable_sha256",p.at("executable_sha256")},
    {"source_manifest_sha256",p.at("source_manifest_sha256")},{"native_library_sha256",p.at("native_library_sha256").get<std::string>()},
    {"native_class_ID_precision_limit",maximum_exact_native_classes},{"caller_coordinate_contract",data.binding.at("preprocessing")},
    {"consumer_peak_owned_CUDA_bytes_excluding_native_XGBoost",peak_owned},{"seconds_before_receipt",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
    {"scope","one user-configured FIT-only native GPU classification fit; no competitive-quality, conversion-completion or general-scaling claim"}};
  atomic_json(out/"result.json",result);atomic_json(out/"result-seal.json",{{"sealed",true},{"result_sha256",sha256(read_text(out/"result.json"))},{"model_sha256",sha256(model_bytes)},{"FIT_only",true},{"VALID_read",false},{"TEST_read",false}});return result;
}
} // namespace
int main(int argc,char** argv) {
  fs::path output;bool output_owned=false;
  try {
    static_assert(sizeof(float)==4&&sizeof(u32)==4&&std::endian::native==std::endian::little);
    if(argc==2&&std::string(argv[1])=="source-manifest"){std::cout<<source_manifest().dump(2)<<'\n';return 0;}
    if(argc==4&&std::string(argv[1])=="--check-plan"){auto p=check_plan(argv[2],argv[3],fs::canonical("/proc/self/exe"));std::cout<<J{{"checked",true},{"CUDA_executed",false},{"dataset",p.at("dataset")},{"hyperparameters",p.at("hyperparameters")}}.dump(2)<<'\n';return 0;}
    need(argc==5&&std::string(argv[1])=="fit","usage: class_model_train source-manifest | --check-plan PLAN SHA | fit PLAN SHA NEW_OUT");
    output=fs::absolute(argv[4]);auto result=train(argv[2],argv[3],output,fs::canonical("/proc/self/exe"),output_owned);std::cout<<result.dump(2)<<'\n';return 0;
  } catch(const std::exception& e) {
    if(output_owned&&fs::exists(output))try{atomic_json(output/"failure.json",{{"complete",false},{"error",e.what()},{"model_may_be_prepared",fs::exists(output/"model.json")},{"FIT_only",true},{"VALID_read",false},{"TEST_read",false}});}catch(...){}
    std::cerr<<e.what()<<'\n';return 1;
  }
}
