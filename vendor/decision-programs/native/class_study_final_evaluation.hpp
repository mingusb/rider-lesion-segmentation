#pragma once
// Reserved final scoring. No Trainer, OOF owner, search or selection is created.
#include "class_study_checkpoint.hpp"
#include "class_study_composition.hpp"
#include "class_study_data.hpp"
#include "class_study_process.hpp"
#include "class_model_dataset_contract.hpp"
#include "class_io.hpp"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <optional>
#include <unistd.h>
#include <sys/file.h>

namespace class_study::final_evaluation {
using J=nlohmann::json;
namespace cp=class_study::checkpoint;
inline void need(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
inline std::uint64_t number(const J&x,const char*m){return class_model_contract::unsigned_integer(x,m);}
inline void flag(const J&x,const char*field,bool expected){need(x.at(field).is_boolean()&&x.at(field).get<bool>()==expected,"final evaluation role/operation declaration");}
inline void absolute(const J&x,const char*field){
  need(x.at(field).is_string()&&!x.at(field).get_ref<const std::string&>().empty(),"final evaluation path type");
  const auto&p=x.at(field).get_ref<const std::string&>();
  need(p.find('\0')==std::string::npos&&std::filesystem::path(p).is_absolute(),"final evaluation path must be absolute without NUL");
}
inline CompositionOptions options(const J&plan){
  CompositionOptions out;
  if(!plan.contains("composition_options"))return out;
  const auto&o=plan.at("composition_options");need(o.is_object(),"final composition options object");
  for(const auto&[key,value]:o.items()){
    if(key=="response_gpu_byte_budget")out.response_gpu_byte_budget=number(value,"final response byte budget");
    else if(key=="teacher_model_host_byte_budget")out.teacher_model_host_byte_budget=number(value,"final teacher host byte budget");
    else if(key=="teacher_owner_window"){
      const auto width=number(value,"final teacher window");need(width<=UINT32_MAX,"final teacher window capacity");out.teacher_owner_window=std::uint32_t(width);
    }else need(false,"unknown final composition option");
  }
  need(out.response_gpu_byte_budget>0&&(out.teacher_owner_window==0||out.teacher_model_host_byte_budget>0),"final composition budgets");return out;
}
inline void preflight(const J&plan,const J&descriptor){
  need(plan.at("format")=="native-final-evaluation-plan-1"&&plan.at("workflow")=="native-final-evaluation-1"&&
       plan.at("authorization")=="score-frozen-finalist-once-1","final frozen scoring plan/authorization");
  flag(plan,"TEST_read",true);flag(plan,"training_allowed",false);flag(plan,"selection_allowed",false);
  for(const char*field:{"frozen_refit_result_path","selected_bundle_path","one_time_guard_path"})absolute(plan,field);
  for(const char*field:{"frozen_refit_result_sha256","selected_bundle_sha256"})need(class_model_contract::sha256(plan.at(field)),"final source byte pin");
  (void)validate_evaluation_dataset(descriptor);(void)options(plan);
}
// Pure metadata and opaque native bytes only. Caller reads pinned artifacts once.
inline J validate(const J&plan,const J&descriptor,const J&refit,const J&bundle,
                  const J*standalone=nullptr,std::uint64_t bundle_bytes=0){
  need(plan.at("format")=="native-final-evaluation-plan-1"&&plan.at("workflow")=="native-final-evaluation-1"&&
       plan.at("authorization")=="score-frozen-finalist-once-1","final frozen scoring plan/authorization");
  flag(plan,"TEST_read",true);flag(plan,"training_allowed",false);flag(plan,"selection_allowed",false);
  for(const char*field:{"frozen_refit_result_path","selected_bundle_path","one_time_guard_path"})absolute(plan,field);
  for(const char*field:{"frozen_refit_result_sha256","selected_bundle_sha256"})need(class_model_contract::sha256(plan.at(field)),"final source byte pin");
  const auto shape=validate_evaluation_dataset(descriptor);
  need(refit.at("format")=="native-frozen-refit-result-1"&&refit.at("complete")==true,"final scoring requires completed frozen refit");
  flag(refit,"TEST_read",false);flag(refit,"VALID_read",false);flag(refit,"selection_performed",false);flag(refit,"HPO_performed",false);flag(refit,"accuracy_evaluated",false);
  const auto&identity=refit.at("identity");const auto&recipe=identity.at("recipe");const auto&training=identity.at("dataset");
  need(recipe.at("workflow")=="native-frozen-refit-1"&&recipe.at("format")=="native-frozen-refit-recipe-1","final source recipe identity");
  flag(recipe,"TEST_read",false);flag(recipe,"VALID_read",false);flag(training,"TEST_read",false);
  const auto F=number(training.at("features"),"frozen training features"),K=number(training.at("classes"),"frozen training classes");
  need(number(training.at("FIT_rows"),"frozen training FIT rows")>0&&number(training.at("VALID_rows"),"frozen training VALID rows")==0&&
       shape.at("features")==F&&shape.at("classes")==K,"final shape/FIT-only frozen source differs");
  if(training.contains("rows"))need(number(training.at("rows"),"frozen training total rows")==number(training.at("FIT_rows"),"frozen training FIT rows"),"final frozen optional row extent differs");
  const auto&checkpoint=refit.at("experiment_checkpoint");
  need(checkpoint.at("committed")==true&&checkpoint.at("failed")==false&&
       plan.at("frozen_refit_generation")==checkpoint.at("generation"),"final frozen refit generation not committed/bound");
  const auto opts=options(plan);const auto meta=ResidentComposition::validate_bundle(bundle,opts);
  need(bundle.at("selected_kind")=="nonlinear_combination"&&bundle.at("raw_dataset_contract")==training&&
       bundle.at("native_library_path")==recipe.at("native_library_path")&&bundle.at("native_library_sha256")==recipe.at("native_library_sha256")&&
       bundle.at("meta_model").at("sha256")==refit.at("meta_model_sha256"),"final frozen composition source/library closure differs");
  if(bundle_bytes)need(refit.at("selected_bundle_bytes")==bundle_bytes,"final selected bundle byte extent differs");
  const auto&teachers=refit.at("teacher_models");
  need(teachers.is_array()&&!teachers.empty()&&teachers.size()==bundle.at("teacher_order").size(),"final ordered teacher extent");
  for(std::size_t i=0;i<teachers.size();++i)
    need(teachers.at(i).at("model_sha256")==bundle.at("teacher_order").at(i),"final teacher order/pin differs");
  const auto policy = recipe.value("refit_policy", std::string("full_fit_oof_meta"));
  need(policy == "full_fit_oof_meta" || policy == "teachers_only_frozen_meta", "final frozen refit policy unsupported");
  if (policy == "teachers_only_frozen_meta") {
    need(refit.at("refit_policy") == policy && refit.at("meta_FIT_calls_total") == 0 && refit.at("meta_FIT_calls_this_process") == 0 &&
         refit.at("OOF_FIT_calls_total") == 0 && refit.at("OOF_FIT_calls_this_process") == 0 && refit.at("OOF_response_teacher_fold_units") == 0 &&
         refit.at("OOF").is_null() && refit.at("response_binding").is_null() && refit.at("prior_VALID_accuracy_preserved") == false,
         "final teachers-only refit falsely claims OOF/meta FIT or preserved accuracy");
    const auto& origin = refit.at("frozen_meta_provenance"); const auto& offered = recipe.at("selected_prior_study");
    need(origin.at("format") == "frozen-development-meta-provenance-1" && bundle.at("refit_policy") == policy &&
         bundle.at("meta_training_provenance") == origin, "final frozen meta provenance format/bundle binding");
    for (const char* key : {"result_sha256", "generation", "selected_model_sha256", "bundle_sha256"})
      need(origin.at(key) == offered.at(key), "final frozen development meta source differs");
    const auto& development = origin.at("development_dataset");
    const auto development_fit = number(development.at("FIT_rows"), "final meta development FIT rows");
    const auto development_valid = number(development.at("VALID_rows"), "final meta development VALID rows");
    need(development_fit > 0 && development_valid > 0 && development_fit <= UINT64_MAX - development_valid &&
         development_fit + development_valid == number(training.at("FIT_rows"), "final refit rows") &&
         development.at("features") == F && development.at("classes") == K && development.at("TEST_read") == false,
         "final frozen meta development roles differ");
    auto old_contract = development, new_contract = training;
    for (const char* key : {"FIT_rows", "VALID_rows", "rows", "split"}) { old_contract.erase(key); new_contract.erase(key); }
    const auto& mt = refit.at("meta_training");
    need(old_contract == new_contract && origin.at("teacher_hyperparameters") == identity.at("teacher_hyperparameters") &&
         origin.at("meta_hyperparameters") == identity.at("meta_hyperparameters") &&
         origin.at("original_teacher_order").size() == teachers.size() &&
         origin.at("OOF_exclusion_verified_on_CUDA") == true && origin.at("exactly_once_coverage_verified_on_CUDA") == true &&
         origin.at("full_FIT_models_used_for_OOF") == false &&
         origin.at("folds") == recipe.at("oof").at("folds") && origin.at("seed") == recipe.at("oof").at("seed") &&
         refit.at("meta_model_sha256") == offered.at("selected_model_sha256") &&
         mt.at("model_sha256") == refit.at("meta_model_sha256") && mt.at("model_bytes") == meta.at("selected_model").at("bytes") &&
         mt.at("FIT_rows") == development_fit && mt.at("features") == teachers.size() * K && mt.at("classes") == K &&
         mt.at("hyperparameters") == identity.at("meta_hyperparameters") && mt.at("native_objective") == "multi:softmax" &&
         mt.at("native_library_sha256") == recipe.at("native_library_sha256") && mt.at("training_performed_this_workflow") == false &&
         mt.at("TEST_read") == false && mt.at("VALID_read") == false,
         "final frozen meta bytes/HP/OOF origin closure differs");
    need(teachers.size() == identity.at("teacher_hyperparameters").size(), "final fixed ordered teacher recipe extent");
    for (std::size_t i = 0; i < teachers.size(); ++i)
      need(teachers.at(i).at("hyperparameters") == identity.at("teacher_hyperparameters").at(i) &&
           teachers.at(i).at("training").at("FIT_rows") == training.at("FIT_rows"), "final all-FIT teacher recipe/training rows differ");
  } else need(!refit.contains("refit_policy") || refit.at("refit_policy") == "full_fit_oof_meta", "final source recipe/report policy differs");
  const bool declared_standalone=!refit.at("standalone_baseline_artifact").is_null();
  const bool offered=plan.contains("standalone_bundle_path")||plan.contains("standalone_bundle_sha256");
  need(declared_standalone==offered&&declared_standalone==bool(standalone),"final standalone comparison declaration differs");
  J raw_metadata=nullptr;
  if(declared_standalone){
    absolute(plan,"standalone_bundle_path");need(class_model_contract::sha256(plan.at("standalone_bundle_sha256")),"final standalone pin");
    raw_metadata=ResidentComposition::validate_bundle(*standalone,opts);
    const auto&artifact=refit.at("standalone_baseline_artifact");const auto index=number(artifact.at("teacher_index"),"final raw teacher index");
    need(index<teachers.size()&&standalone->at("selected_kind")=="native_teacher"&&standalone->at("raw_dataset_contract")==training&&
         standalone->at("native_library_path")==recipe.at("native_library_path")&&standalone->at("native_library_sha256")==recipe.at("native_library_sha256")&&
         standalone->at("native_model").at("sha256")==artifact.at("model_sha256")&&artifact.at("model_sha256")==teachers.at(std::size_t(index)).at("model_sha256"),
         "final standalone fixed teacher/library closure differs");
  }
  const bool official=descriptor.at("format")=="mnist-official-test-idx-1"&&training.at("format")=="mnist-idx-permutation-1"&&
       F==784&&K==10&&number(training.at("FIT_rows"),"official training rows")==60000;
  J out = {{"format","native-final-evaluation-identity-1"},{"frozen_refit_result_sha256",plan.at("frozen_refit_result_sha256")},
          {"frozen_refit_generation",plan.at("frozen_refit_generation")},{"frozen_recipe_identity_sha256",dpnative::sha256(identity.dump())},
          {"selected_bundle_sha256",plan.at("selected_bundle_sha256")},{"standalone_bundle_sha256",offered?plan.at("standalone_bundle_sha256"):J(nullptr)},
          {"evaluation_descriptor",descriptor},{"composition_options",plan.value("composition_options",J::object())},
          {"composition_metadata",meta},{"standalone_metadata",raw_metadata},{"official_MNIST_geometry_and_raw_IDX_route",official},
          {"role","TEST"},{"training_allowed",false},{"selection_allowed",false}};
           if (policy == "teachers_only_frozen_meta") { out["final_training_policy"] = policy; out["development_meta_provenance"] = refit.at("frozen_meta_provenance"); }
  return out;
}
// One immutable caller-bound job, with a process-lifetime exclusive lock. A
// crash releases the lock; only the identical frozen job/output may recover.
class AttemptGuard {
  cp::detail::Fd lock_{-1};
 public:
  bool recovering=false;
  AttemptGuard(const std::filesystem::path&path,const J&identity,const std::filesystem::path&output) {
    need(path.is_absolute()&&output.is_absolute(),"final guard/output paths absolute");
    std::filesystem::create_directories(path.parent_path());
    const auto lockpath=path.string()+".lock";
    lock_.n=::open(lockpath.c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0600);
    need(lock_.n>=0,"final guard lock open failed");
    need(::flock(lock_.n,LOCK_EX|LOCK_NB)==0,"identical final evaluation already running");
    const J declaration={{"format","native-final-evaluation-attempt-1"},{"identity",identity},{"output",output.string()},
                         {"TEST_input_opened_before_claim",false}};
    if(std::filesystem::exists(path)) {
      need(cp::detail::read_manifest(path)==declaration,"final guard freezes a different model/data/options/output job");
      recovering=true; return;
    }
    need(!std::filesystem::exists(output),"fresh final evaluation output already exists");
    const auto temporary=path.string()+".pending-"+std::to_string(::getpid())+"-"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    bool renamed=false;
    try {
      const auto text=declaration.dump()+"\n";
      cp::detail::Fd file(::open(temporary.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600));
      need(file.n>=0,"final immutable guard temporary open failed");
      cp::detail::write_all(file.n,text.data(),text.size());cp::detail::sync(file.n);
      need(::rename(temporary.c_str(),path.c_str())==0,"final immutable guard rename failed");renamed=true;
      cp::detail::sync_dir(path.parent_path());
    } catch(...) {if(!renamed)::unlink(temporary.c_str());throw;}
  }
};
inline int run(const J&plan,const J&descriptor,const std::filesystem::path&output){
  need(output.is_absolute(),"final scoring output must be an absolute directory");
  preflight(plan,descriptor);
  const auto started=std::chrono::steady_clock::now();
  const auto refit_bytes=dpnative::read_text(plan.at("frozen_refit_result_path").get<std::string>());
  const auto bundle_bytes=dpnative::read_text(plan.at("selected_bundle_path").get<std::string>());
  need(dpnative::sha256(refit_bytes)==plan.at("frozen_refit_result_sha256").get<std::string>()&&
       dpnative::sha256(bundle_bytes)==plan.at("selected_bundle_sha256").get<std::string>(),"final pinned artifact bytes differ");
  const auto refit=J::parse(refit_bytes),bundle=J::from_cbor(bundle_bytes);
  std::optional<J> standalone;std::string raw_bytes;
  if(plan.contains("standalone_bundle_path")){
    raw_bytes=dpnative::read_text(plan.at("standalone_bundle_path").get<std::string>());
    need(dpnative::sha256(raw_bytes)==plan.at("standalone_bundle_sha256").get<std::string>(),"final standalone artifact byte pin differs");standalone=J::from_cbor(raw_bytes);
  }
  const auto identity=validate(plan,descriptor,refit,bundle,standalone?&*standalone:nullptr,bundle_bytes.size());
  AttemptGuard guard(plan.at("one_time_guard_path").get<std::string>(),identity,output);
  if(std::filesystem::exists(output/"result.json")) {
    const auto saved=J::parse(dpnative::read_text(output/"result.json"));
    need(saved.at("complete")==true&&saved.at("identity")==identity,"completed final scoring receipt identity differs");
    std::cout<<J{{"event","final_TEST_completed_receipt_replay"},{"output",output.string()},{"TEST_input_reads_this_process",0}}.dump()<<'\n'<<std::flush;return 0;
  }
  std::filesystem::create_directories(output);
  J results=J::array();bool test_read_started=false;
  try{
    test_read_started=true;auto data=stage_evaluation_dataset(descriptor);ResidentEvaluation evaluation(data);
    const auto opts=options(plan);
    auto score=[&](const J&model,const char*category,const std::string&pin,std::uint64_t bytes){
      ResidentComposition composition(model,opts);const auto metadata=composition.metadata();
      NativeOracle oracle;oracle.features=composition.features();oracle.classes=composition.classes();oracle.objective="multi:softmax";
      oracle.source_sha256=metadata.at("composition_identity_sha256").get<std::string>();oracle.library_sha256=model.at("native_library_sha256").get<std::string>();
      oracle.predict=[&composition](const float*x,std::uint64_t rows,bool margin){return composition.predict(x,rows,margin);};
      const auto measured=evaluation.evaluate_native(oracle);
      need(measured.at("evaluation_role")=="TEST"&&measured.at("selection_allowed")==false,"final evaluator role drift");
      results.push_back({{"category",category},{"bundle_sha256",pin},{"bundle_bytes",bytes},{"evaluation",measured},{"composition",composition.metadata()}});
      std::cout<<J{{"event","final_TEST_score"},{"category",category},{"TEST_errors",measured.at("evaluation_errors")},{"TEST_rows",measured.at("evaluation_rows")}}.dump()<<'\n'<<std::flush;
    };
    if(standalone)score(*standalone,"standalone_native_XGBoost",plan.at("standalone_bundle_sha256").get<std::string>(),raw_bytes.size());
    score(bundle,"learned_XGBoost_only_composition",plan.at("selected_bundle_sha256").get<std::string>(),bundle_bytes.size());
    J report={{"format","native-final-evaluation-result-1"},{"complete",true},{"identity",identity},{"evaluation_binding",data.binding},
      {"categories",results},{"FIT_calls",0},{"training_or_OOF_owners_created",false},{"HPO_performed",false},{"selection_performed",false},
      {"TEST_read",true},{"one_time_guard_path",plan.at("one_time_guard_path")},{"one_time_policy","immutable model/data/options/output job; exclusive scoring; identical recovery only; completed receipt replay reads no TEST"},
      {"official_protocol_scope","named IDX route verifies byte pins/header/geometry and exact raw pixels; source-download provenance remains in the explicit descriptor"},
      {"recovered_identical_job",guard.recovering},{"world_record_claim",false},{"process_wall_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
    dpnative::atomic_json(output/"result.json",report);return 0;
  }catch(const std::exception&e){
    dpnative::atomic_json(output/"failure.json",J{{"format","native-final-evaluation-failure-1"},{"complete",false},{"identity",identity},
      {"TEST_read_started",test_read_started},{"completed_categories",results},{"failure",e.what()},{"FIT_calls",0},{"selection_performed",false}});throw;
  }
}
} // namespace class_study::final_evaluation