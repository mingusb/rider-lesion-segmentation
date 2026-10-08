#pragma once
// Host orchestration only. Fold construction, native fits, probability transport,
// prediction and VALID ranking use the maintained resident CUDA components.
#include "class_study_oof.hpp"
#include "class_study_process.hpp"
#include "class_study_training_cache.hpp"
#include "class_study_composition.hpp"
#include "class_study_holdout.hpp"
#include "class_study_nested_holdout.hpp"
#include "class_io.hpp"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <csignal>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>

namespace class_study::nonlinear_search {
using J=nlohmann::json;
namespace cp=class_study::checkpoint;
inline void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
inline std::uint64_t number(const J& v,const char* why){
  need(v.is_number_integer(),why);
  if(v.is_number_unsigned())return v.get<std::uint64_t>();
  const auto n=v.get<std::int64_t>();need(n>=0,why);return std::uint64_t(n);
}
inline J pack(const TrainingResult& n){return {{"model",cp::binary(n.model_json)},{"sha256",n.model_sha256},{"metrics",n.metrics}};}
inline TrainingResult unpack(const J& n){
  TrainingResult r{cp::bytes(n.at("model")),n.at("sha256").get<std::string>(),n.at("metrics")};
  need(dpnative::sha256(r.model_json)==r.model_sha256,"nonlinear model buffer identity differs");return r;
}
inline J pack_oof(const OofCheckpoint& c){
  J out={{"state",c.state},{"responses",J::binary(c.response_bytes)},{"coverage",J::binary(c.coverage_bytes)}};
  if(!c.round_prefix_parent_bytes.empty()){
    out["round_prefix_parent_bytes"]=J::array();
    for(const auto& bytes:c.round_prefix_parent_bytes)out["round_prefix_parent_bytes"].push_back(J::binary(bytes));
  }
  return out;
}
inline OofCheckpoint unpack_oof(const J& c){
  need(c.at("responses").is_binary()&&c.at("coverage").is_binary(),"nonlinear OOF payload types");
  OofCheckpoint out;out.state=c.at("state");out.response_bytes=c.at("responses").get_binary();out.coverage_bytes=c.at("coverage").get_binary();
  if(c.contains("round_prefix_parent_bytes")){
    need(c.at("round_prefix_parent_bytes").is_array(),"nonlinear round-prefix parent payload list");
    for(const auto& bytes:c.at("round_prefix_parent_bytes")){
      need(bytes.is_binary(),"nonlinear round-prefix parent payload type");out.round_prefix_parent_bytes.push_back(bytes.get_binary());
    }
  }
  return out;
}
inline NativeOracle oracle(ResidentTrainer& t,const std::string& pin){
  NativeOracle o;o.features=t.features();o.classes=t.classes();o.objective="multi:softmax";
  o.source_sha256=pin;o.library_sha256=t.metadata().at("native_library_sha256").get<std::string>();
  o.predict=[&t](const float* x,std::uint64_t n,bool margin){return t.predict(x,n,margin);};return o;
}
inline void write(const std::filesystem::path& p,const std::string& s){
  std::ofstream f(p,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write(s.data(),s.size());f.flush();
}
inline std::vector<J> parameters(const J& list,std::uint32_t K){
  need(list.is_array()&&!list.empty(),"nonlinear parameter list must be nonempty");
  std::vector<J> out;std::set<std::string> unique;
  for(const auto& hp:list){auto n=ResidentTrainer::validate_hyperparameters(hp,K);need(unique.insert(n.dump()).second,"duplicate nonlinear declared parameters");out.push_back(std::move(n));}
  return out;
}
inline std::vector<std::uint32_t> meta_teacher_counts(const J& plan,std::size_t teachers){
  const auto& metas=plan.at("meta_hyperparameters");
  need(metas.is_array()&&!metas.empty()&&teachers&&teachers<=UINT32_MAX,"nonlinear meta/teacher list extent");
  std::vector<std::uint32_t> out(metas.size(),std::uint32_t(teachers));
  if(plan.contains("meta_teacher_counts")){
    const auto& counts=plan.at("meta_teacher_counts");
    need(counts.is_array()&&counts.size()==metas.size(),"nonlinear meta teacher counts must align with meta hyperparameters");
    for(std::size_t i=0;i<counts.size();++i){const auto n=number(counts[i],"nonlinear meta teacher count type");
      need(n>0&&n<=teachers,"nonlinear meta teacher count outside declared bank");out[i]=std::uint32_t(n);}
  }
  return out;
}
inline std::vector<J> meta_parameters(const J& list,std::uint32_t K,
                                    const std::vector<std::uint32_t>& counts,bool prefixes){
  if(!prefixes)return parameters(list,K);
  need(list.is_array()&&!list.empty()&&list.size()==counts.size(),"nonlinear aligned meta list extent");
  std::vector<J> out;std::set<std::string> unique;
  for(std::size_t i=0;i<list.size();++i){auto hp=ResidentTrainer::validate_hyperparameters(list[i],K);
    need(unique.insert(J::array({counts[i],hp}).dump()).second,"duplicate nonlinear meta teacher-count/parameter pair");out.push_back(std::move(hp));}
  return out;
}
inline ResidentDataView teacher_prefix(const ResidentDataView& all,std::uint32_t count){
  need(all.classes&&all.features%all.classes==0&&count>0&&count<=all.features/all.classes&&
       all.row_stride>=all.features&&all.binding.at("teacher_declarations").is_array()&&
       all.binding.at("teacher_declarations").size()==all.features/all.classes,"nonlinear response teacher prefix shape/binding");
  if(count==all.features/all.classes)return all;
  auto view=all;view.features=count*all.classes;
  J declarations=J::array();for(std::uint32_t i=0;i<count;++i)declarations.push_back(all.binding.at("teacher_declarations")[i]);
  view.binding["features"]=view.features;view.binding["row_stride"]=view.row_stride;
  view.binding["teacher_declarations"]=std::move(declarations);
  view.binding["teacher_prefix_count"]=count;view.binding["source_response_teacher_count"]=all.features/all.classes;
  view.binding["source_response_features"]=all.features;
  view.binding["feature_selection"]="ordered teacher prefix; active coordinates packed by existing CUDA stride-aware consumers";
  return view;
}
inline std::uint64_t teacher_closure_bytes(const std::vector<OofTeacher>& teachers,std::size_t count){
  need(count>0&&count<=teachers.size(),"nonlinear teacher byte closure extent");std::uint64_t bytes=0;std::set<std::string> pins;
  for(std::size_t i=0;i<count;++i){const auto& model=teachers[i].full_fit_model;
    if(pins.insert(model.model_sha256).second){need(model.model_json.size()<=UINT64_MAX-bytes,"nonlinear teacher closure byte overflow");bytes+=model.model_json.size();}}
  return bytes;
}
inline bool baseline_requested(const J& plan){
  bool requested=false;
  for(const char* key:{"baseline_composition_bundle","baseline_composition_bundle_sha256",
      "baseline_completed_result","baseline_completed_result_sha256","baseline_provenance"})requested|=plan.contains(key);
  if(requested){
    for(const char* key:{"baseline_composition_bundle","baseline_composition_bundle_sha256",
        "baseline_completed_result","baseline_completed_result_sha256"})
      need(plan.at(key).is_string()&&!plan.at(key).get_ref<const std::string&>().empty(),"required nonlinear baseline path/pin missing");
    need(plan.at("baseline_provenance").is_object()&&plan.at("baseline_provenance").at("generation").is_string(),
         "nonlinear baseline committed generation provenance missing");
  }
  return requested;
}
inline void baseline_path(const std::string& value,bool physical){
  std::filesystem::path path=value;
  need(value.find('\0')==std::string::npos&&path.is_absolute(),"nonlinear baseline absolute path required");
  if(physical)(void)std::filesystem::canonical(path);
}
struct RetainedComposition {
  std::string bundle_bytes,result_bytes;
  J saved,declaration,recipe;
};
// Opaque bytes and metadata only. Native model shape validation occurs when the
// inference-only replay owner loads these exact payloads; no CPU predictions.
inline RetainedComposition retained_composition(const J& plan,const J& descriptor,const J* snapshot=nullptr){
  need(baseline_requested(plan),"nonlinear retained composition not declared");
  RetainedComposition out;
  const auto bundle_path=plan.at("baseline_composition_bundle").get<std::string>();
  const auto result_path=plan.at("baseline_completed_result").get<std::string>();
  baseline_path(bundle_path,snapshot==nullptr);baseline_path(result_path,snapshot==nullptr);
  if(snapshot){
    need(snapshot->at("format")=="nonlinear-retained-composition-1"&&snapshot->at("bundle").is_binary()&&
         snapshot->at("completed_result").is_binary(),"nonlinear baseline checkpoint byte payload types");
    out.bundle_bytes=cp::bytes(snapshot->at("bundle"));out.result_bytes=cp::bytes(snapshot->at("completed_result"));
  }else{
    out.bundle_bytes=dpnative::read_text(bundle_path);out.result_bytes=dpnative::read_text(result_path);
  }
  need(!out.bundle_bytes.empty()&&!out.result_bytes.empty()&&
       dpnative::sha256(out.bundle_bytes)==plan.at("baseline_composition_bundle_sha256").get<std::string>()&&
       dpnative::sha256(out.result_bytes)==plan.at("baseline_completed_result_sha256").get<std::string>(),
       "nonlinear baseline captured byte pins differ");
  auto bundle=J::from_cbor(out.bundle_bytes);out.declaration=ResidentComposition::validate_bundle(bundle);
  need(bundle.at("raw_dataset_contract")==descriptor&&bundle.at("raw_features")==descriptor.at("features")&&
       bundle.at("classes")==descriptor.at("classes")&&bundle.at("native_library_path")==plan.at("native_library_path")&&
       bundle.at("native_library_sha256")==plan.at("native_library_sha256"),"nonlinear baseline dataset/library differs");
  const auto receipt=J::parse(out.result_bytes);const auto& prior=receipt.at("identity");const auto& source_plan=prior.at("plan");
  nested_holdout::reject_spent_donor(plan,receipt);
  need(receipt.at("format")=="native-nonlinear-combination-result-1"&&receipt.at("complete")==true&&
       receipt.at("TEST_read")==false&&receipt.at("VALID_used_for_selection")==true&&prior.at("dataset")==descriptor&&
       source_plan.at("workflow")=="native-nonlinear-combination-1"&&source_plan.at("TEST_read")==false&&
       source_plan.at("VALID_read")==false&&source_plan.at("native_library_path")==plan.at("native_library_path")&&
       source_plan.at("native_library_sha256")==plan.at("native_library_sha256"),
       "nonlinear baseline completed source receipt differs");
  const auto& selected=receipt.at("selected");const auto kind=bundle.at("selected_kind").get<std::string>();
  const auto pin=out.declaration.at("selected_model").at("sha256").get<std::string>();
  need(selected.at("kind")==kind&&selected.at("model_sha256")==pin&&
       selected.at("native_model_bytes")==out.declaration.at("selected_model").at("bytes")&&
       selected.at("deployment_native_model_bytes")==out.declaration.at("unique_native_model_bytes")&&
       selected.at("evaluation").at("source_sha256")==pin&&
       selected.at("evaluation").at("native_library_sha256")==plan.at("native_library_sha256")&&
       selected.at("evaluation").at("objective")=="multi:softmax"&&
       receipt.at("selected_bundle_bytes")==out.bundle_bytes.size(),"nonlinear baseline selected model/byte closure differs");
  const auto fit=number(descriptor.at("FIT_rows"),"nonlinear baseline FIT rows"),valid=number(descriptor.at("VALID_rows"),"nonlinear baseline VALID rows");
  need(fit&&valid&&fit<=UINT64_MAX-valid&&selected.at("evaluation").at("FIT_rows")==fit&&
       selected.at("evaluation").at("VALID_rows")==valid&&selected.at("evaluation").at("rows")==fit+valid&&
       number(selected.at("evaluation").at("VALID_errors"),"nonlinear baseline prior errors")<=valid,
       "nonlinear baseline prior evaluation role differs");
  const auto generation=plan.at("baseline_provenance").at("generation").get<std::string>();
  need(generation.starts_with("generation-")&&generation.find_first_of("/\\")==std::string::npos&&
       receipt.at("experiment_checkpoint").at("committed")==true&&receipt.at("experiment_checkpoint").at("failed")==false&&
       receipt.at("experiment_checkpoint").at("generation")==generation,"nonlinear baseline committed generation differs");
  const auto K=std::uint32_t(number(descriptor.at("classes"),"nonlinear baseline K"));
  const auto teachers=parameters(prior.at("teacher_hyperparameters"),K);
  const auto counts=meta_teacher_counts(source_plan,teachers.size());
  const auto metas=meta_parameters(prior.at("meta_hyperparameters"),K,counts,source_plan.contains("meta_teacher_counts"));
  const auto index=number(selected.at("index"),"nonlinear baseline prior selected index");
  J recipe_teachers=J::array(),recipe_metas=J::array();
  if(kind=="nonlinear_combination"){
    need(index<metas.size(),"nonlinear baseline prior meta index");const auto count=counts.at(std::size_t(index));
    need(count==bundle.at("teacher_order").size()&&(!source_plan.contains("meta_teacher_counts")||
         (prior.at("meta_teacher_counts")==counts&&selected.at("meta_teacher_count")==count)),"nonlinear baseline prior prefix recipe extent");
    // The original receipt binds each ordered teacher declaration to the actual
    // full-FIT model pin used by this bundle, even if that study had a baseline.
    need(receipt.at("candidates").is_array(),"nonlinear baseline candidate declarations missing");
    for(std::size_t i=0;i<count;++i){
      const J* candidate=nullptr;
      for(const auto& c:receipt.at("candidates"))if(c.at("kind")=="native_teacher"&&c.at("index")==i){
        need(candidate==nullptr,"nonlinear baseline duplicate teacher receipt");candidate=&c;
      }
      const auto& teacher_pin=bundle.at("teacher_order").at(i);
      need(candidate&&candidate->at("model_sha256")==teacher_pin&&
           candidate->at("native_model_bytes")==out.declaration.at("teacher_models").at(teacher_pin.get<std::string>()).at("bytes"),
           "nonlinear baseline ordered teacher recipe/model differs");
    }
    for(std::size_t i=0;i<count;++i)recipe_teachers.push_back(teachers[i]);recipe_metas.push_back(metas.at(std::size_t(index)));
  }else{need(index<teachers.size(),"nonlinear raw baseline recipe index");recipe_teachers.push_back(teachers.at(std::size_t(index)));}
  out.recipe={{"selected_prior_study",{{"result_path",result_path},{"result_sha256",plan.at("baseline_completed_result_sha256")},
      {"generation",generation},{"selected_model_sha256",pin}}},{"selected_kind",kind},
      {"teacher_hyperparameters",recipe_teachers},{"meta_hyperparameters",recipe_metas},{"oof",source_plan.at("oof")}};
  if(kind=="nonlinear_combination"&&source_plan.contains("meta_teacher_counts"))out.recipe["meta_teacher_count"]=counts.at(std::size_t(index));
  out.saved={{"format","nonlinear-retained-composition-1"},{"bundle",cp::binary(out.bundle_bytes)},
      {"completed_result",cp::binary(out.result_bytes)},{"bundle_path",bundle_path},{"bundle_sha256",plan.at("baseline_composition_bundle_sha256")},
      {"completed_result_path",result_path},{"completed_result_sha256",plan.at("baseline_completed_result_sha256")},
      {"provenance",plan.at("baseline_provenance")},{"declaration",out.declaration},{"recipe",out.recipe},
      {"prior_VALID_errors",selected.at("evaluation").at("VALID_errors")}};
  if(snapshot)need(out.saved==*snapshot,"nonlinear baseline saved declaration/provenance differs");
  return out;
}
inline J baseline_metadata(const RetainedComposition& baseline){
  auto metadata=baseline.saved;metadata.erase("bundle");metadata.erase("completed_result");
  metadata["bundle_bytes"]=baseline.bundle_bytes.size();metadata["completed_result_bytes"]=baseline.result_bytes.size();
  metadata["byte_claim"]="captured immutable baseline bundle/receipt used; original files not reread on resume";return metadata;
}
inline void response_routes(const J& plan){
  need(!(plan.contains("response_import_checkpoint")&&plan.contains("response_source_checkpoint")),
       "nonlinear completed-column import and identical-bank restore are mutually exclusive");
  if(plan.contains("response_reuse_completed_oof")){
    need(plan.at("response_reuse_completed_oof").is_boolean(),
         "nonlinear completed OOF reuse flag type");
    if(plan.at("response_reuse_completed_oof").get<bool>()){
      need(plan.contains("response_source_checkpoint")&&plan.at("response_source_checkpoint").is_string()&&
           plan.contains("response_source_generation")&&plan.at("response_source_generation").is_string(),
           "nonlinear completed OOF reuse requires checkpoint and explicit generation");
      baseline_path(plan.at("response_source_checkpoint").get<std::string>(),false);
      const auto generation=plan.at("response_source_generation").get<std::string>();
      need(generation.starts_with("generation-")&&generation.find_first_of("/\\")==std::string::npos,
           "nonlinear completed OOF reuse generation path/type");
    }
  }
  if(plan.contains("response_import_checkpoint")){
    need(plan.at("response_import_checkpoint").is_string(),"nonlinear import checkpoint path type");
    baseline_path(plan.at("response_import_checkpoint").get<std::string>(),false);
  }
  if(plan.contains("response_import_generation")){
    need(plan.contains("response_import_checkpoint")&&plan.at("response_import_generation").is_string(),
         "nonlinear import generation requires its checkpoint");
    const auto generation=plan.at("response_import_generation").get<std::string>();
    need(generation.starts_with("generation-")&&generation.find_first_of("/\\")==std::string::npos,
         "nonlinear import generation path/type");
  }
}
// Create a fresh experiment view of a captured completed response bank. The
// source state is passed by value and its files/generation remain unchanged.
// Existing resume validation below checks every adopted model and score again.
inline J completed_oof_fork(const J& plan,const J& descriptor,const J& identity,
                            J source,const std::string& generation){
  nested_holdout::reject_spent_donor(plan,source);
  const auto& prior_identity=source.at("identity");
  const auto& prior_plan=prior_identity.at("plan");
  const auto prior_phase=source.at("phase").get<std::string>();
  need(plan.at("response_source_generation")==generation,
       "nonlinear completed OOF fork generation differs");
  need(prior_phase=="meta_training"||prior_phase=="meta_scoring"||prior_phase=="complete",
       "nonlinear completed OOF fork requires a meta-stage boundary");
  need(prior_identity.at("dataset")==descriptor&&
       prior_identity.at("teacher_hyperparameters")==identity.at("teacher_hyperparameters")&&
       prior_identity.at("selection_rule")==identity.at("selection_rule")&&
       prior_plan.at("workflow")=="native-nonlinear-combination-1"&&
       prior_plan.at("TEST_read")==false&&prior_plan.at("VALID_read")==false&&
       prior_plan.at("native_library_path")==plan.at("native_library_path")&&
       prior_plan.at("native_library_sha256")==plan.at("native_library_sha256")&&
       prior_plan.at("oof")==plan.at("oof"),
       "nonlinear completed OOF fork data/teacher/fold/library identity differs");
  const bool baseline=baseline_requested(plan);
  need(baseline_requested(prior_plan)==baseline,
       "nonlinear completed OOF fork baseline eligibility differs");
  if(baseline){
    for(const char* key:{"baseline_composition_bundle","baseline_composition_bundle_sha256",
        "baseline_completed_result","baseline_completed_result_sha256","baseline_provenance"})
      need(prior_plan.at(key)==plan.at(key),
           "nonlinear completed OOF fork retained baseline binding differs");
  }
  const auto K=std::uint32_t(number(descriptor.at("classes"),"nonlinear fork K"));
  const auto& prior_teachers=prior_identity.at("teacher_hyperparameters");
  const auto teacher_count=prior_teachers.size();
  const auto baseline_count=baseline?1u:0u;
  const auto source_meta_index=number(source.at("meta_index"),"nonlinear fork source meta cursor");
  const bool pending=!source.at("current").is_null();
  const auto prior_counts=meta_teacher_counts(prior_plan,teacher_count);
  const auto prior_metas=meta_parameters(prior_identity.at("meta_hyperparameters"),K,
                                       prior_counts,prior_plan.contains("meta_teacher_counts"));
  need(source.at("teachers").is_array()&&source.at("teachers").size()==teacher_count&&
       number(source.at("teacher_scored"),"nonlinear fork teacher cursor")==teacher_count&&
       (baseline?number(source.at("baseline_scored"),"nonlinear fork baseline cursor")==1:
                 !source.contains("baseline_scored"))&&
       source.at("meta_models").is_array()&&source.at("meta_models").size()==source_meta_index&&
       source_meta_index<=prior_metas.size()&&
       number(source.at("meta_fits"),"nonlinear fork source FIT count")==source_meta_index+pending&&
       (pending==(prior_phase=="meta_scoring"))&&
       (prior_phase=="complete"?source_meta_index==prior_metas.size():source_meta_index<prior_metas.size()),
       "nonlinear completed OOF fork source phase/cursor extent differs");
  for(std::size_t i=0;i<source_meta_index;++i)
    need(source.at("meta_models").at(i).at("metrics").at("hyperparameters")==prior_metas.at(i)&&
         source.at("meta_models").at(i).at("metrics").at("features")==std::uint64_t(prior_counts.at(i))*K,
         "nonlinear completed OOF fork source meta declaration differs");
  if(pending)need(source.at("current").at("metrics").at("hyperparameters")==prior_metas.at(source_meta_index)&&
       source.at("current").at("metrics").at("features")==std::uint64_t(prior_counts.at(source_meta_index))*K,
       "nonlinear completed OOF fork pending meta declaration differs");
  need(!source.at("oof_checkpoint").is_null()&&
       source.at("oof_checkpoint").at("state").at("phase")=="complete",
       "nonlinear completed OOF fork responses are incomplete");
  auto& scores=source.at("results");
  const std::size_t adopted_count=teacher_count+baseline_count;
  need(scores.is_array()&&adopted_count&&scores.size()==adopted_count+source_meta_index,
       "nonlinear completed OOF fork score extent differs");
  const auto fit_rows=number(descriptor.at("FIT_rows"),"nonlinear fork FIT rows");
  const auto valid_rows=number(descriptor.at("VALID_rows"),"nonlinear fork VALID rows");
  need(fit_rows>0&&valid_rows>0&&fit_rows<=UINT64_MAX-valid_rows,
       "nonlinear completed OOF fork row extent differs");
  std::size_t prefix_best=0,whole_best=0;
  auto preferred=[&](std::size_t a,std::size_t b){
    const auto ae=number(scores.at(a).at("evaluation").at("VALID_errors"),"nonlinear fork VALID errors");
    const auto be=number(scores.at(b).at("evaluation").at("VALID_errors"),"nonlinear fork VALID errors");
    return ae<be||(ae==be&&number(scores.at(a).at("deployment_native_model_bytes"),"nonlinear fork bytes")<
                              number(scores.at(b).at("deployment_native_model_bytes"),"nonlinear fork bytes"));
  };
  for(std::size_t i=0;i<scores.size();++i){
    const auto& result=scores.at(i);const auto& evaluation=result.at("evaluation");
    need(evaluation.at("CUDA_computed")==true&&evaluation.at("objective")=="multi:softmax"&&
         evaluation.at("native_library_sha256")==plan.at("native_library_sha256")&&
         evaluation.at("FIT_rows")==fit_rows&&evaluation.at("VALID_rows")==valid_rows&&
         evaluation.at("rows")==fit_rows+valid_rows&&
         number(evaluation.at("FIT_errors"),"nonlinear fork FIT errors")<=fit_rows&&
         number(evaluation.at("VALID_errors"),"nonlinear fork VALID errors")<=valid_rows,
         "nonlinear completed OOF fork source score role/library differs");
    const bool becomes=i==0||(i&&preferred(i,whole_best));
    need(result.at("became_incumbent")==becomes,
         "nonlinear completed OOF fork source incumbent sequence differs");
    if(becomes)whole_best=i;
    if(i<adopted_count&&(i==0||preferred(i,prefix_best)))prefix_best=i;
  }
  need(number(source.at("best"),"nonlinear fork source winner")==whole_best,
       "nonlinear completed OOF fork source winner differs");
  need(whole_best<adopted_count,
       "nonlinear completed OOF fork would omit a winning source meta; retain that winner separately first");
  scores.erase(scores.begin()+std::ptrdiff_t(adopted_count),scores.end());
  source["identity"]=identity;source["phase"]="oof";source["current"]=nullptr;
  source.erase("nested_holdout"); // New outer data starts after the new inner search.
  source["meta_models"]=J::array();source["meta_index"]=0;source["meta_fits"]=0;source["best"]=prefix_best;
  source["response_reuse"]={{"mode","completed_oof_same_bank_fork"},
    {"checkpoint",plan.at("response_source_checkpoint")},{"generation",generation},
    {"source_phase",prior_phase},{"source_meta_candidates_not_adopted",source_meta_index},
    {"source_pending_meta_model_not_adopted",pending},{"source_scored_incumbent_preserved",true},
    {"teacher_models_adopted",teacher_count},{"teacher_scores_replayed",teacher_count},
    {"baseline_scores_replayed",baseline_count},{"new_teacher_score_calls",0},
    {"new_teacher_buffer_materializations",0},{"new_training_source_checkpoint_loads",0},
    {"new_fold_FIT_calls",0},{"coverage_revalidated_on_CUDA",false},
    {"score_scope","captured prior CUDA baseline/teacher scores; not rescored in this process"},
    {"source_checkpoint_changed",false}};
  return source;
}
inline J response_import_provenance(const J& plan,const J& descriptor,const J& donor,
                                    const std::string& generation){
  nested_holdout::reject_spent_donor(plan,donor);
  response_routes(plan);need(plan.contains("response_import_checkpoint"),"nonlinear import checkpoint undeclared");
  need(donor.at("phase")=="complete"&&donor.at("current").is_null(),
       "response import requires a complete nonlinear study checkpoint");
  const auto& source_identity=donor.at("identity");const auto& source_plan=source_identity.at("plan");
  need(source_identity.at("dataset")==descriptor&&source_plan.at("workflow")=="native-nonlinear-combination-1"&&
       source_plan.at("TEST_read")==false&&source_plan.at("VALID_read")==false&&
       source_plan.at("native_library_path")==plan.at("native_library_path")&&
       source_plan.at("native_library_sha256")==plan.at("native_library_sha256"),
       "response import dataset/library/training role differs");
  if(plan.contains("response_import_generation"))need(plan.at("response_import_generation")==generation,
       "response import committed generation differs");
  need(generation.starts_with("generation-")&&generation.find_first_of("/\\")==std::string::npos,
       "response import committed generation path/type");
  need(!donor.at("oof_checkpoint").is_null()&&donor.at("oof_checkpoint").at("state").at("phase")=="complete",
       "response import requires completed OOF and VALID responses");
  return {{"checkpoint",plan.at("response_import_checkpoint")},{"generation",generation}};
}
inline int run(const J& plan,const J& descriptor,const std::filesystem::path& output,
               const std::string& requested_resume,volatile std::sig_atomic_t& stopping,
               volatile std::sig_atomic_t& checkpoint_requested,bool stop_after_oof=false){
  const auto started=std::chrono::steady_clock::now();
  response_routes(plan);
  need(plan.at("workflow")=="native-nonlinear-combination-1","nonlinear workflow identifier");
  need(plan.at("TEST_read")==false&&plan.at("VALID_read")==false&&descriptor.at("TEST_read")==false,
       "nonlinear training must exclude VALID and TEST");
  if(plan.contains("dataset_source"))need(plan.at("dataset_source")==descriptor,"nonlinear declared dataset differs");
  const bool nested=plan.contains("nested_holdout");
  J holdout_config=nullptr,holdout_state=nullptr;
  if(nested){
    holdout_config=nested_holdout::configuration(plan.at("nested_holdout"),descriptor);
    (void)validate_holdout_dataset(holdout_config.at("dataset"));
  }
  const auto class_count=number(descriptor.at("classes"),"nonlinear class count");
  need(class_count>=2&&class_count<=UINT32_MAX,"nonlinear class capacity");const auto K=std::uint32_t(class_count);
  const auto teacher_hp=parameters(plan.at("teacher_hyperparameters"),K);
  const auto meta_counts=meta_teacher_counts(plan,teacher_hp.size());const bool prefix_counts=plan.contains("meta_teacher_counts");
  const auto meta_hp=meta_parameters(plan.at("meta_hyperparameters"),K,meta_counts,prefix_counts);
  const auto required_baseline=plan.at("required_baseline_sha256").get<std::string>();
  need(required_baseline.size()==64,"nonlinear required baseline identity missing");
  need(teacher_hp.size()<=UINT32_MAX/K,"nonlinear teacher-response width overflow");
  OofOptions options;const auto& opts=plan.at("oof");
  const auto folds=number(opts.at("folds"),"nonlinear fold count");need(folds>=2&&folds<=UINT32_MAX,"nonlinear fold capacity");
  options.folds=std::uint32_t(folds);options.seed=number(opts.value("seed",J(0)),"nonlinear fold seed");
  options.gpu_byte_budget=number(opts.at("gpu_byte_budget"),"nonlinear OOF GPU budget");
  if(opts.contains("group_round_prefixes")){need(opts.at("group_round_prefixes").is_boolean(),"nonlinear grouped-prefix option type");options.group_round_prefixes=opts.at("group_round_prefixes").get<bool>();}
  if(opts.contains("round_prefix_iteration_range")){
    need(opts.at("round_prefix_iteration_range").is_boolean(),"nonlinear iteration-range option type");
    options.round_prefix_iteration_range=opts.at("round_prefix_iteration_range").get<bool>();
  }
  need(!options.round_prefix_iteration_range||options.group_round_prefixes,
       "nonlinear iteration-range responses require grouped round prefixes");
  auto semantic=plan;for(const char* k:{"experiment_checkpoint_path","experiment_resume_from","experiment_checkpoint_host_byte_budget"})semantic.erase(k);
  J identity={{"plan",semantic},{"dataset",descriptor},{"teacher_hyperparameters",teacher_hp},
    {"meta_hyperparameters",meta_hp},{"selection_rule","VALID_errors_then_total_native_deployment_model_bytes_then_declared_order"}};
  if(prefix_counts)identity["meta_teacher_counts"]=meta_counts;
  constexpr const char* format="native-nonlinear-combination-checkpoint-1";
  const auto resume=requested_resume.empty()?plan.value("experiment_resume_from",std::string{}):requested_resume;
  const auto checkpoint_root=plan.value("experiment_checkpoint_path",resume);
  const auto budget=number(plan.value("experiment_checkpoint_host_byte_budget",J(0)),"nonlinear host budget");
  std::unique_ptr<cp::Coordinator> coordinator;if(!checkpoint_root.empty())coordinator=std::make_unique<cp::Coordinator>(checkpoint_root,budget,format);
  need(!stop_after_oof||bool(coordinator),"stop-after-OOF requires an experiment checkpoint path");
  std::vector<OofTeacher> teachers;std::vector<TrainingResult> meta_models;std::optional<TrainingResult> current;
  J results=J::array(),source_cache=nullptr,saved_oof=nullptr,oof_metrics=nullptr,response_binding=nullptr,response_reuse=nullptr;
  std::size_t teacher_scored=0,meta_index=0,best=0;std::uint64_t meta_fits=0,meta_fits_this_process=0;
  bool completed_oof_stop=false;
  const bool has_baseline=baseline_requested(plan);
  std::optional<RetainedComposition> baseline;std::size_t baseline_scored=0;
  std::string phase=has_baseline?"baseline_scoring":"teacher_scoring";bool resumed=false;
  const bool forking=resume.empty()&&plan.value("response_reuse_completed_oof",false);
  if(!resume.empty()||forking){
    const auto source=forking?plan.at("response_source_checkpoint").get<std::string>():resume;
    auto loaded=cp::Coordinator::load(source,budget,format);
    if(forking){
      need(loaded.engine_snapshot_path.empty(),"nonlinear completed OOF fork requires host-only state");
      loaded.state=completed_oof_fork(plan,descriptor,identity,std::move(loaded.state),loaded.generation);
    }
    const auto& s=loaded.state;
    need(loaded.engine_snapshot_path.empty()&&s.at("identity")==identity,"nonlinear checkpoint identity differs");
    phase=s.at("phase").get<std::string>();need(phase=="baseline_scoring"||phase=="teacher_scoring"||phase=="oof"||phase=="meta_training"||phase=="meta_scoring"||phase=="complete","nonlinear saved phase");
    if(has_baseline){
      baseline=retained_composition(plan,descriptor,&s.at("baseline_composition"));
      baseline_scored=std::size_t(number(s.at("baseline_scored"),"nonlinear saved baseline cursor"));
      need(baseline_scored<=1&&((phase=="baseline_scoring")==!baseline_scored),"nonlinear saved baseline phase/cursor");
    }else need(phase!="baseline_scoring"&&!s.contains("baseline_composition")&&!s.contains("baseline_scored"),"undeclared nonlinear saved baseline");
    for(const auto& t:s.at("teachers"))teachers.push_back({t.at("hyperparameters"),unpack(t.at("native")),t.at("FIT_binding")});
    for(const auto& m:s.at("meta_models"))meta_models.push_back(unpack(m));
    if(!s.at("current").is_null())current=unpack(s.at("current"));
    results=s.at("results");teacher_scored=number(s.at("teacher_scored"),"nonlinear teacher cursor");
    meta_index=number(s.at("meta_index"),"nonlinear meta cursor");best=number(s.at("best"),"nonlinear incumbent");
    meta_fits=number(s.at("meta_fits"),"nonlinear completed meta fits");source_cache=s.at("source_cache");
    saved_oof=s.at("oof_checkpoint");oof_metrics=s.at("oof_metrics");response_binding=s.at("response_binding");
    response_reuse=s.value("response_reuse",J(nullptr));
    holdout_state=s.value("nested_holdout",J(nullptr));
    need(holdout_state.is_null()||(nested&&phase=="complete"),"saved holdout decision precedes completed inner search");
    need(teachers.size()==teacher_hp.size()&&teacher_scored<=teachers.size()&&meta_index<=meta_hp.size()&&meta_models.size()==meta_index,"nonlinear saved model/cursor extent");
    need(results.is_array()&&results.size()==baseline_scored+teacher_scored+meta_index&&(results.empty()?best==0:best<results.size()),"nonlinear saved scoring extent");
    need((phase=="meta_scoring")==bool(current)&&meta_fits==meta_index+bool(current),"nonlinear pending FIT extent");
    need(!stop_after_oof||meta_fits==0,"stop-after-OOF boundary has already passed the first meta FIT");
    need((phase=="baseline_scoring"||phase=="teacher_scoring")?(meta_index==0&&teacher_scored<teachers.size()&&saved_oof.is_null()):teacher_scored==teachers.size(),"nonlinear teacher phase extent");
    need(phase!="baseline_scoring"||teacher_scored==0,"nonlinear baseline must score before teachers");
    need(phase=="complete"?meta_index==meta_hp.size():meta_index<meta_hp.size(),"nonlinear meta completion extent");
    need(phase=="oof"?meta_index==0:true,"nonlinear OOF phase extent");
    need((phase=="meta_training"||phase=="meta_scoring"||phase=="complete")?!saved_oof.is_null():true,"nonlinear OOF response checkpoint absent");
    for(std::size_t i=0;i<teachers.size();++i)need(teachers[i].hyperparameters==teacher_hp[i],"nonlinear restored teacher parameters differ");
    for(std::size_t i=0;i<meta_models.size();++i)need(meta_models[i].metrics.at("hyperparameters")==meta_hp[i]&&
        (!prefix_counts||meta_models[i].metrics.at("features")==std::uint64_t(meta_counts[i])*K),"nonlinear restored meta parameters/response width differ");
    if(current)need(current->metrics.at("hyperparameters")==meta_hp[meta_index]&&
        (!prefix_counts||current->metrics.at("features")==std::uint64_t(meta_counts[meta_index])*K),"nonlinear current meta parameters/response width differ");
    std::uint64_t source_bytes=0;std::set<std::string> pins;
    for(const auto& t:teachers)if(pins.insert(t.full_fit_model.model_sha256).second){
      need(t.full_fit_model.model_json.size()<=UINT64_MAX-source_bytes,"nonlinear saved source byte overflow");source_bytes+=t.full_fit_model.model_json.size();
    }
    const auto fit_rows=number(descriptor.at("FIT_rows"),"nonlinear FIT extent");
    const auto valid_rows=number(descriptor.at("VALID_rows"),"nonlinear VALID extent");
    need(fit_rows>0&&valid_rows>0&&fit_rows<=UINT64_MAX-valid_rows,"nonlinear role extent");
    std::size_t computed_best=0;
    for(std::size_t i=0;i<results.size();++i){
      const bool is_baseline=i<baseline_scored;
      const bool is_teacher=!is_baseline&&i-baseline_scored<teacher_scored;
      const auto index=is_baseline?0:is_teacher?i-baseline_scored:i-baseline_scored-teacher_scored;
      std::string pin,kind;std::uint64_t native_bytes=0,total=0;
      if(is_baseline){
        pin=baseline->declaration.at("composition_identity_sha256").get<std::string>();kind="retained_composition";
        native_bytes=number(baseline->declaration.at("selected_model").at("bytes"),"nonlinear baseline native bytes");
        total=number(baseline->declaration.at("unique_native_model_bytes"),"nonlinear baseline deployment bytes");
      }else{
        const auto& model=is_teacher?teachers.at(index).full_fit_model:meta_models.at(index);
        const auto required_bytes=is_teacher?0:teacher_closure_bytes(teachers,meta_counts.at(index));
        need(model.model_json.size()<=UINT64_MAX-required_bytes,"nonlinear saved composite byte overflow");
        pin=model.model_sha256;kind=is_teacher?"native_teacher":"nonlinear_combination";native_bytes=model.model_json.size();total=native_bytes+required_bytes;
      }
      const auto& r=results.at(i);const auto& e=r.at("evaluation");
      if(prefix_counts&&!is_baseline&&!is_teacher)need(r.at("meta_teacher_count")==meta_counts.at(index)&&
          r.at("response_features")==std::uint64_t(meta_counts.at(index))*K,"nonlinear saved meta prefix binding differs");
      need(r.at("kind")==kind&&r.at("index")==index&&r.at("model_sha256")==pin&&r.at("native_model_bytes")==native_bytes&&
           r.at("deployment_native_model_bytes")==total&&e.at("source_sha256")==pin&&
           e.at("native_library_sha256")==plan.at("native_library_sha256")&&e.at("objective")=="multi:softmax"&&
           e.at("FIT_rows")==fit_rows&&e.at("VALID_rows")==valid_rows&&e.at("rows")==fit_rows+valid_rows&&
           number(e.at("FIT_errors"),"nonlinear FIT errors")<=fit_rows&&number(e.at("VALID_errors"),"nonlinear VALID errors")<=valid_rows,
           "nonlinear saved result/model binding differs");
      if(is_baseline)need(r.at("selected_native_model_sha256")==baseline->declaration.at("selected_model").at("sha256")&&
          r.at("underlying_kind")==baseline->declaration.at("selected_kind")&&e.at("VALID_errors")==baseline->saved.at("prior_VALID_errors"),
          "nonlinear restored baseline native identity/score differs");
      bool preferred=i==0;
      if(i){const auto& b=results.at(computed_best);const auto err=number(e.at("VALID_errors"),"nonlinear VALID errors");
        const auto old=number(b.at("evaluation").at("VALID_errors"),"nonlinear saved incumbent errors");
        preferred=err<old||(err==old&&total<number(b.at("deployment_native_model_bytes"),"nonlinear saved incumbent bytes"));}
      need(r.at("became_incumbent")==preferred,"nonlinear saved winner sequence differs");if(preferred)computed_best=i;
    }
    need(computed_best==best,"nonlinear saved incumbent differs from retained results");
    resumed=!forking;
  }
  if(has_baseline&&!baseline)baseline=retained_composition(plan,descriptor);
  std::unique_ptr<ResidentOof> oof;
  auto snapshot=[&]{
    J ts=J::array(),ms=J::array();for(const auto& t:teachers)ts.push_back({{"hyperparameters",t.hyperparameters},{"native",pack(t.full_fit_model)},{"FIT_binding",t.full_fit_data_binding}});
    for(const auto& m:meta_models)ms.push_back(pack(m));
    J saved={{"format",format},{"identity",identity},{"phase",phase},{"teachers",std::move(ts)},{"meta_models",std::move(ms)},
      {"current",current?pack(*current):J(nullptr)},{"teacher_scored",teacher_scored},{"meta_index",meta_index},{"best",best},
      {"meta_fits",meta_fits},{"results",results},{"source_cache",source_cache},{"oof_checkpoint",oof?pack_oof(oof->checkpoint()):saved_oof},
      {"oof_metrics",oof?oof->metrics():oof_metrics},{"response_binding",response_binding},{"response_reuse",response_reuse},
      {"periodic_checkpoint_writes",false},{"mid_native_round_resume",false}};
    if(has_baseline){saved["baseline_composition"]=baseline->saved;saved["baseline_scored"]=baseline_scored;}
    if(nested)saved["nested_holdout"]=holdout_state;
    return saved;
  };
  auto boundary=[&](bool final=false){
    const bool stop_requested=bool(stopping)||completed_oof_stop;
    if(coordinator&&(final||stop_requested||checkpoint_requested)){
      coordinator->finish();need(!coordinator->poll().failed,"nonlinear prior checkpoint writer failed");checkpoint_requested=0;
      need(coordinator->save_host(snapshot()),"nonlinear checkpoint writer busy");
      if(final||stop_requested){coordinator->finish();need(!coordinator->poll().failed,"nonlinear checkpoint publication failed");}
    }
    J progress=nullptr;if(oof)progress=oof->progress();
    J event={{"event","nonlinear_search_boundary"},{"phase",phase},{"teachers_scored",teacher_scored},{"meta_candidates_scored",meta_index},
      {"meta_FIT_calls",meta_fits},{"OOF",progress},{"stop_requested",stop_requested}};
    if(completed_oof_stop)event["stop_reason"]="completed_OOF_before_first_meta_FIT";
    if(!holdout_state.is_null())event["nested_holdout"]={{"status",holdout_state.at("status")},{"next_gate",holdout_state.at("next_gate")}};
    std::cout<<event.dump()<<'\n'<<std::flush;
    return stop_requested;
  };
  std::uint64_t teacher_bytes=0;std::set<std::string> counted;
  auto count_teacher_bytes=[&]{teacher_bytes=0;counted.clear();for(const auto& t:teachers)if(counted.insert(t.full_fit_model.model_sha256).second){
    need(t.full_fit_model.model_json.size()<=UINT64_MAX-teacher_bytes,"nonlinear deployment byte extent overflow");teacher_bytes+=t.full_fit_model.model_json.size();}
    need(counted.contains(required_baseline),"nonlinear teacher bank must retain the explicitly required native baseline");};
  if(phase!="complete"){
    auto data=stage_dataset(descriptor);auto fit=fit_prefix(data);ResidentTrainer raw(plan,fit);
    if(teachers.empty()){
      if(budget)raw.set_retained_prefix_cache_byte_budget(std::min<std::uint64_t>(budget,1ull<<30));
      TrainingCache cache(plan,descriptor,budget);source_cache=cache.metadata();
      for(const auto& hp:teacher_hp){
        auto source=cache.find(hp);need(source.has_value(),"nonlinear teacher missing: complete its FIT-only native source study first");
        if(!raw.select_retained_prefix_source(source->model->model_sha256))
          raw.retain_prefix_source(source->model->model_json,source->model->model_sha256);
        auto n=raw.slice_retained_prefix(hp.at("rounds").get<std::uint32_t>());n.metrics["hyperparameters"]=hp;
        n.metrics["source_checkpoint"]=source->source_checkpoint;n.metrics["training_reused"]=true;
        teachers.push_back({hp,std::move(n),fit.binding});
      }
      raw.clear_retained_prefix_sources();
    }
    for(const auto& t:teachers)need(t.full_fit_data_binding==fit.binding,"nonlinear teacher FIT binding differs");
    count_teacher_bytes();
    auto score=[&](const NativeOracle& source,ResidentEvaluation& evaluator,const TrainingResult& model,
                   const char* kind,std::size_t index,std::uint64_t deployment_bytes){
      auto evaluation=evaluator.evaluate_native(source);
      evaluation["FIT_accuracy_scope"]=std::string(kind)=="nonlinear_combination"
        ?"meta training accuracy on OOF response features; not deployed-composition FIT accuracy"
        :"full-FIT source model on its own training rows";
      evaluation["VALID_accuracy_scope"]="deployment predictions on the unchanged held-out source VALID rows";
      const auto errors=results.empty()?UINT64_MAX:results.at(best).at("evaluation").at("VALID_errors").get<std::uint64_t>();
      const auto bytes=results.empty()?UINT64_MAX:results.at(best).at("deployment_native_model_bytes").get<std::uint64_t>();
      const bool prefer=evaluator.prefer_last_native_candidate(deployment_bytes,errors,bytes);
      J result={{"kind",kind},{"index",index},{"model_sha256",model.model_sha256},{"evaluation",evaluation},
        {"native_model_bytes",model.model_json.size()},{"deployment_native_model_bytes",deployment_bytes},{"became_incumbent",prefer}};
      if(prefix_counts&&std::string(kind)=="nonlinear_combination"){
        result["meta_teacher_count"]=meta_counts.at(index);result["response_features"]=std::uint64_t(meta_counts.at(index))*K;
      }
      if(prefer)best=results.size();results.push_back(result);
      std::cout<<J{{"event","nonlinear_candidate"},{"kind",kind},{"index",index},{"VALID_errors",evaluation.at("VALID_errors")},
        {"VALID_accuracy",evaluation.at("VALID_accuracy")},{"deployment_native_model_bytes",deployment_bytes},{"became_incumbent",prefer}}.dump()<<'\n'<<std::flush;
    };
    if(phase=="baseline_scoring"||phase=="teacher_scoring"){
      ResidentEvaluation evaluation(data);
      if(phase=="baseline_scoring"){
        if(boundary())return 2;
        auto bundle=J::from_cbor(baseline->bundle_bytes);ResidentComposition replay(bundle);
        NativeOracle source;source.features=replay.features();source.classes=replay.classes();source.objective="multi:softmax";
        source.library_sha256=plan.at("native_library_sha256").get<std::string>();
        source.source_sha256=baseline->declaration.at("composition_identity_sha256").get<std::string>();
        source.predict=[&replay](const float* x,std::uint64_t rows,bool margin){return replay.predict(x,rows,margin);};
        auto measured=evaluation.evaluate_native(source);
        need(measured.at("VALID_errors")==baseline->saved.at("prior_VALID_errors"),"required prior composition did not reproduce its VALID error count");
        measured["FIT_accuracy_scope"]="deployed retained composition on original FIT rows; not OOF meta-training accuracy";
        measured["VALID_accuracy_scope"]="retained deployment rescored on unchanged source VALID rows";
        const auto bytes=number(baseline->declaration.at("unique_native_model_bytes"),"nonlinear baseline bytes");
        need(results.empty()&&evaluation.prefer_last_native_candidate(bytes,UINT64_MAX,UINT64_MAX),"nonlinear initial baseline selection failed");
        results.push_back({{"kind","retained_composition"},{"index",0},{"model_sha256",source.source_sha256},
          {"selected_native_model_sha256",baseline->declaration.at("selected_model").at("sha256")},
          {"underlying_kind",baseline->declaration.at("selected_kind")},{"evaluation",measured},
          {"native_model_bytes",baseline->declaration.at("selected_model").at("bytes")},{"deployment_native_model_bytes",bytes},{"became_incumbent",true}});
        best=0;baseline_scored=1;phase="teacher_scoring";
        std::cout<<J{{"event","nonlinear_retained_baseline"},{"VALID_errors",measured.at("VALID_errors")},
          {"deployment_native_model_bytes",bytes},{"recipe_provenance",baseline->recipe.at("selected_prior_study")}}.dump()<<'\n'<<std::flush;
      }
      while(teacher_scored<teachers.size()){
        if(boundary())return 2;const auto& n=teachers[teacher_scored].full_fit_model;raw.restore_model(n.model_json,n.model_sha256);
        score(oracle(raw,n.model_sha256),evaluation,n,"native_teacher",teacher_scored,n.model_json.size());++teacher_scored;
      }
      phase="oof";
    }
    oof=std::make_unique<ResidentOof>(plan,data,teachers,options);
    if(!saved_oof.is_null()){
      oof->restore(unpack_oof(saved_oof));saved_oof=nullptr;
      if(!response_reuse.is_null()&&response_reuse.value("mode",std::string{})=="completed_oof_same_bank_fork"){
        need(oof->complete(),"nonlinear fork response restore did not complete");
        response_reuse["coverage_revalidated_on_CUDA"]=true;
      }
    }
    else if(plan.contains("response_source_checkpoint")){
      // Reuse completed fold predictions, never predictions from in-sample FIT
      // models. ResidentOof::restore authorizes the exact data/teacher order,
      // library, fold rule, response byte pins and CUDA coverage before use.
      const auto source=plan.at("response_source_checkpoint").get<std::string>();
      auto loaded=cp::Coordinator::load(source,budget,format);const auto& s=loaded.state;
      nested_holdout::reject_spent_donor(plan,s);
      need(loaded.engine_snapshot_path.empty()&&s.at("phase")=="complete"&&s.at("current").is_null(),
           "response reuse requires a complete nonlinear study checkpoint");
      const auto& source_identity=s.at("identity");const auto& source_plan=source_identity.at("plan");
      need(source_identity.at("dataset")==descriptor&&source_identity.at("teacher_hyperparameters")==J(teacher_hp)&&
           source_plan.at("workflow")=="native-nonlinear-combination-1"&&source_plan.at("TEST_read")==false&&
           source_plan.at("VALID_read")==false&&source_plan.at("native_library_path")==plan.at("native_library_path")&&
           source_plan.at("native_library_sha256")==plan.at("native_library_sha256"),
           "response reuse dataset/teacher/library/training role differs");
      if(plan.contains("response_source_generation"))need(plan.at("response_source_generation")==loaded.generation,
           "response reuse committed generation differs");
      need(!s.at("oof_checkpoint").is_null()&&s.at("oof_checkpoint").at("state").at("phase")=="complete",
           "response reuse requires completed OOF and VALID responses");
      oof->restore(unpack_oof(s.at("oof_checkpoint")));
      need(oof->complete(),"response reuse incomplete after GPU audit");
      response_reuse={{"checkpoint",source},{"generation",loaded.generation},{"completed_teacher_folds",oof->state().at("oof_cursor")},
                     {"new_fold_FIT_calls",0},{"coverage_revalidated_on_CUDA",true}};
      std::cout<<J{{"event","nonlinear_response_reuse"},{"source",response_reuse}}.dump()<<'\n'<<std::flush;
    }
    else if(plan.contains("response_import_checkpoint")){
      const auto source=plan.at("response_import_checkpoint").get<std::string>();
      // Fresh import reads the committed donor once. A restored experiment uses
      // its captured target OOF state above, without reopening the donor.
      baseline_path(source,true);
      auto loaded=cp::Coordinator::load(source,budget,format);
      need(loaded.engine_snapshot_path.empty(),"response import requires a host-only completed study");
      auto provenance=response_import_provenance(plan,descriptor,loaded.state,loaded.generation);
      auto imported=oof->import_completed_columns(unpack_oof(loaded.state.at("oof_checkpoint")),provenance);
      response_reuse={{"mode","completed_column_import"},{"checkpoint",source},{"generation",loaded.generation},
          {"import",imported.statistics},{"new_fold_FIT_calls_at_import",0},{"coverage_revalidated_on_CUDA",true}};
      std::cout<<J{{"event","nonlinear_response_import"},{"source",response_reuse}}.dump()<<'\n'<<std::flush;
    }
    if(phase=="oof"){
      while(!oof->complete()){if(boundary())return 2;need(oof->advance(),"nonlinear OOF advance failed");}
      phase="meta_training";
    }
    need(oof->complete(),"nonlinear meta FIT requires completed OOF responses");
    auto responses=oof->completed_data();response_binding=responses.binding;
    need(responses.binding.at("source_dataset_binding")==data.binding&&responses.rows==data.rows&&
         responses.fit_rows==data.fit_rows&&responses.valid_rows==data.valid_rows&&responses.classes==data.classes&&
         responses.features==teacher_hp.size()*K,"nonlinear response/source role binding differs");
    if(stop_after_oof){
      need(phase=="meta_training"&&meta_index==0&&meta_fits==0&&!current,
           "stop-after-OOF requires the boundary before the first meta FIT");
      completed_oof_stop=true;
      need(boundary(),"stop-after-OOF checkpoint boundary was not stopped");
      return 2;
    }
    std::unique_ptr<ResidentTrainer> meta;std::unique_ptr<ResidentEvaluation> evaluation;std::uint32_t active_count=0;
    auto activate_meta=[&](std::uint32_t count){
      if(active_count==count)return;
      evaluation.reset();meta.reset();auto input=teacher_prefix(responses,count);
      meta=std::make_unique<ResidentTrainer>(plan,fit_prefix(input),raw);meta->restore_completed_trial_count(meta_fits);
      evaluation=std::make_unique<ResidentEvaluation>(input);
      if(current)meta->restore_model(current->model_json,current->model_sha256);active_count=count;
    };
    while(meta_index<meta_hp.size()){
      if(boundary())return 2;
      activate_meta(meta_counts.at(meta_index));
      if(phase=="meta_training"){
        current=meta->train(meta_hp[meta_index]);++meta_fits;++meta_fits_this_process;phase="meta_scoring";
        if(boundary())return 2;
      }
      need(current.has_value(),"nonlinear meta candidate absent");
      const auto required_bytes=teacher_closure_bytes(teachers,meta_counts.at(meta_index));
      need(current->model_json.size()<=UINT64_MAX-required_bytes,"nonlinear composite byte extent overflow");
      score(oracle(*meta,current->model_sha256),*evaluation,*current,"nonlinear_combination",meta_index,required_bytes+current->model_json.size());
      meta_models.push_back(std::move(*current));current.reset();++meta_index;phase=meta_index==meta_hp.size()?"complete":"meta_training";
    }
    oof_metrics=oof->metrics();
  }else count_teacher_bytes();
  need(!results.empty()&&best<results.size(),"nonlinear selected candidate missing");
  if(has_baseline)need(number(results.at(best).at("evaluation").at("VALID_errors"),"nonlinear selected errors")<=
      number(baseline->saved.at("prior_VALID_errors"),"nonlinear baseline errors"),"nonlinear required baseline regressed");
  auto deployment_for=[&](const J& candidate){
    const auto index=candidate.at("index").get<std::size_t>();
    if(candidate.at("kind")=="retained_composition")return J::from_cbor(baseline->bundle_bytes);
    J deployment={{"format","native-nonlinear-deployment-bundle-1"},{"selected_kind",candidate.at("kind")},
      {"raw_features",descriptor.at("features")},{"classes",K},{"native_library_path",plan.at("native_library_path")},
      {"native_library_sha256",plan.at("native_library_sha256")},{"raw_dataset_contract",descriptor},
      {"teacher_probability_contract","native full-round gbtree clone; derived multi:softprob; FP32 teacher-major/class-minor; no CPU transform"},
      {"final_class_contract","selected native multi:softmax public class"},{"teacher_models",J::object()},{"teacher_order",J::array()},
      {"compiled_single_tree",false},{"deployment_replay_qualified",false}};
    if(candidate.at("kind")=="nonlinear_combination"){
      for(std::size_t i=0;i<meta_counts.at(index);++i){const auto& n=teachers[i].full_fit_model;
        deployment["teacher_order"].push_back(n.model_sha256);
        if(!deployment["teacher_models"].contains(n.model_sha256))deployment["teacher_models"][n.model_sha256]={{"model",cp::binary(n.model_json)},{"sha256",n.model_sha256}};
      }
      const auto& m=meta_models.at(index);deployment["meta_model"]={{"model",cp::binary(m.model_json)},{"sha256",m.model_sha256}};
    }else deployment["native_model"]=pack(teachers.at(index).full_fit_model);
    return deployment;
  };
  std::size_t final_best=best,holdout_gates_this_process=0;
  if(nested){
    // The comparator is declared before search, not picked using outer labels.
    std::size_t comparator=results.size();
    for(std::size_t i=0;i<results.size();++i)
      if(results.at(i).at("kind")=="native_teacher"&&results.at(i).at("model_sha256")==required_baseline){comparator=i;break;}
    need(comparator<results.size(),"nested holdout required baseline result missing");
    if(holdout_state.is_null())holdout_state=nested_holdout::frozen(holdout_config,best,results.at(best),comparator,results.at(comparator));
    else nested_holdout::validate_state(holdout_state,holdout_config,best,results.at(best),comparator,results.at(comparator));
    if(boundary())return 2;
    if(holdout_state.at("status")=="pending"){
      // Development owners have left scope. Stage independent outer rows once.
      auto heldout=stage_holdout_dataset(holdout_config.at("dataset"));
      ResidentComposition candidate_replay(deployment_for(results.at(best)));
      ResidentComposition baseline_replay(deployment_for(results.at(comparator)));
      auto replay_oracle=[&](ResidentComposition& replay,const J& result){
        NativeOracle source;source.features=replay.features();source.classes=replay.classes();source.objective="multi:softmax";
        source.library_sha256=plan.at("native_library_sha256").get<std::string>();
        source.source_sha256=result.at("model_sha256").get<std::string>();
        source.predict=[&replay](const float* x,std::uint64_t rows,bool margin){return replay.predict(x,rows,margin);};
        return source;
      };
      const auto challenger=replay_oracle(candidate_replay,results.at(best));
      const auto incumbent=replay_oracle(baseline_replay,results.at(comparator));
      while(holdout_state.at("status")=="pending"){
        if(boundary())return 2;
        const auto gate=holdout_state.at("next_gate").get<std::size_t>();
        const auto& part=holdout_config.at("partitions").at(gate);
        const auto score=evaluate_holdout_gate(heldout,part.at("offset").get<std::uint64_t>(),part.at("rows").get<std::uint64_t>(),challenger,incumbent);
        nested_holdout::record(holdout_state,score);++holdout_gates_this_process;
        std::cout<<J{{"event","nested_holdout_gate"},{"level",gate+1},{"score",score},{"status",holdout_state.at("status")}}.dump()<<'\n'<<std::flush;
      }
    }
    final_best=holdout_state.at("final_selected_index").get<std::size_t>();
  }
  if(boundary(true))return 2;
  const auto& selected=results.at(final_best);
  const auto selected_index=selected.at("index").get<std::size_t>();
  const bool retained=selected.at("kind")=="retained_composition";
  const bool combined=retained?baseline->declaration.at("selected_kind")=="nonlinear_combination"
                              :selected.at("kind")=="nonlinear_combination";
  const auto deployment=deployment_for(selected);
  std::filesystem::create_directories(output);
  if(retained)write(output/"selected-bundle.cbor",baseline->bundle_bytes);
  else{std::ofstream f(output/"selected-bundle.cbor",std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);
    J::to_cbor(deployment,nlohmann::detail::output_adapter<char>(f));f.flush();}
  if(!combined)write(output/"selected-model.json",retained?cp::bytes(deployment.at("native_model").at("model"))
                                                       :teachers.at(selected_index).full_fit_model.model_json);
  J report={{"format","native-nonlinear-combination-result-1"},{"complete",true},{"selected",selected},{"candidates",results},
    {"identity",identity},{"source_cache",source_cache},{"OOF",oof_metrics},{"response_binding",response_binding},{"response_reuse",response_reuse},
    {"meta_FIT_calls",meta_fits},{"meta_FIT_calls_this_process",meta_fits_this_process},{"new_full_FIT_teacher_calls",0},
    {"source_models_reused",teachers.size()},{"selected_combination",combined},{"selection_computed_on_CUDA",true},
    {"selection_rule",identity.at("selection_rule")},{"VALID_used_for_selection",true},{"TEST_read",false},
    {"RL_used",false},{"conversion_performed",false},{"deployment_replay_qualified",false},
    {"per_trial_file_writes",0},{"periodic_checkpoint_writes",false},{"resumed_experiment",resumed},
    {"completed_oof_fork_this_process",forking},
    {"selected_bundle_path",(output/"selected-bundle.cbor").string()},
    {"selected_bundle_bytes",std::filesystem::file_size(output/"selected-bundle.cbor")},
    {"deployment_model_byte_metric","native JSON model buffers including required full-FIT teachers; excludes envelope/runtime library; not compiled tree size"},
    {"process_wall_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
  if(nested){
    report["inner_selected"]=results.at(best);report["nested_holdout"]=holdout_state;
    report["holdout_gates_computed_this_process"]=holdout_gates_this_process;
    report["selection_rule"]="inner VALID errors/bytes/order, then frozen candidate must strictly improve on the required baseline at every outer holdout gate";
    report["holdout_score_scope"]="selection evidence, not an independent final TEST estimate";
    report["holdout_independence_scope"]="caller supplies fresh group/time-separated data; distinct files do not establish global independence";
  }
  if(!response_reuse.is_null()&&response_reuse.value("mode",std::string{})=="completed_oof_same_bank_fork"){
    report["selection_scope"]="baseline/teacher incumbent is exact replay of prior CUDA scores, revalidated on host; newly declared meta candidates use CUDA scoring and preference";
    report["baseline_teacher_scores_computed_this_process"]=false;
    report["inherited_incumbent_revalidated_on_host"]=true;
    report["new_meta_candidate_preference_engine"]="CUDA";
  }
  if(has_baseline){report["baseline_composition"]=baseline_metadata(*baseline);report["baseline_scored"]=baseline_scored;
    report["selected_retained_composition"]=retained;
    if(retained)report["selected_recipe"]=baseline->recipe;}
  if(!retained&&combined&&prefix_counts){J recipe_teachers=J::array();for(std::size_t i=0;i<meta_counts.at(selected_index);++i)recipe_teachers.push_back(teacher_hp[i]);
    report["selected_recipe"]={{"selected_kind","nonlinear_combination"},{"selected_candidate_index",selected_index},
      {"meta_teacher_count",meta_counts.at(selected_index)},{"teacher_hyperparameters",recipe_teachers},
      {"meta_hyperparameters",J::array({meta_hp.at(selected_index)})},{"oof",plan.at("oof")}};}
  if(coordinator){auto s=coordinator->poll();report["experiment_checkpoint"]={{"committed",s.committed},{"failed",s.failed},{"generation",s.generation},{"host_bytes",s.host_bytes}};}
  write(output/"result.json",report.dump(2)+"\n");
  std::cout<<J{{"event","nonlinear_search_complete"},{"selected_kind",selected.at("kind")},{"VALID_errors",selected.at("evaluation").at("VALID_errors")},
    {"output",output.string()},{"TEST_read",false},{"RL_used",false},{"conversion_performed",false}}.dump()<<'\n'<<std::flush;
  return 0;
}
} // namespace class_study::nonlinear_search
