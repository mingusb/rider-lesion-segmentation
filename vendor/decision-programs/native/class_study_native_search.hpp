#pragma once
#include "class_study_data.hpp"
#include "class_study_train.hpp"
#include "class_study_process.hpp"
#include "class_study_checkpoint.hpp"
#include "class_study_training_cache.hpp"
#include "class_io.hpp"
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace class_study::native_search {
using J=nlohmann::json;
namespace cp=class_study::checkpoint;
inline void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
inline J pack(const TrainingResult& n){return {{"model",cp::binary(n.model_json)},{"sha256",n.model_sha256},{"metrics",n.metrics}};}
inline TrainingResult unpack(const J& n){TrainingResult r{cp::bytes(n.at("model")),n.at("sha256").get<std::string>(),n.at("metrics")};need(dpnative::sha256(r.model_json)==r.model_sha256,"saved native candidate identity differs");return r;}
inline void write(const std::filesystem::path& p,const std::string& bytes){std::ofstream f(p,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write(bytes.data(),bytes.size());f.flush();}

inline int run(const J& plan,const J& descriptor,const std::filesystem::path& output,
               const std::string& requested_resume,volatile std::sig_atomic_t& stopping,
               volatile std::sig_atomic_t& checkpoint_requested){
  const auto started=std::chrono::steady_clock::now();
  need(plan.at("workflow")=="native-accuracy-search-1","native search workflow");
  need(plan.at("TEST_read")==false && plan.at("VALID_read")==false,
       "native optimizer plan must remain FIT-only");
  need(descriptor.at("TEST_read")==false,"native search cannot access TEST");
  if(plan.contains("dataset_source"))
    need(plan.at("dataset_source")==descriptor,"native search plan dataset differs from actual input");
  auto semantic=plan;
  for(const char* k:{"experiment_checkpoint_path","experiment_resume_from","experiment_checkpoint_host_byte_budget"})semantic.erase(k);
  const auto K=descriptor.at("classes").get<std::uint32_t>();
  std::vector<J> trials;
  const auto base=plan.at("hyperparameters");
  need(base.is_object(),"native search base parameters");
  if(plan.contains("trials")){
    need(plan.at("trials").is_array()&&!plan.at("trials").empty(),"native search nonempty trial list");
    for(const auto& over:plan.at("trials")){need(over.is_object(),"native trial override");J hp=base;hp.update(over);trials.push_back(ResidentTrainer::validate_hyperparameters(hp,K));}
  }else trials.push_back(ResidentTrainer::validate_hyperparameters(base,K));
  const J identity={{"plan",semantic},{"dataset",descriptor},{"trials",trials},{"selection_rule","VALID_errors_then_native_model_bytes_then_declared_order"}};
  constexpr const char* format="native-accuracy-search-checkpoint-1";
  const auto resume=requested_resume.empty()?plan.value("experiment_resume_from",std::string{}):requested_resume;
  const auto checkpoint_root=plan.value("experiment_checkpoint_path",resume);
  const auto budget=plan.value("experiment_checkpoint_host_byte_budget",std::uint64_t(0));
  std::unique_ptr<cp::Coordinator> coordinator;
  if(!checkpoint_root.empty())coordinator=std::make_unique<cp::Coordinator>(checkpoint_root,budget,format);
  std::vector<TrainingResult> models;
  J results=J::array();
  std::optional<TrainingResult> current;
  std::size_t index=0,best=0;
  std::string phase="training";
  J training_cache_sources=J::array(),training_cache_this_process=nullptr;
  bool resumed=false;
  std::uint64_t fit_calls=0,fit_calls_this_process=0,reused_calls=0,reused_calls_this_process=0;
  if(!resume.empty()){
    auto loaded=cp::Coordinator::load(resume,budget,format);const auto& s=loaded.state;
    need(loaded.engine_snapshot_path.empty()&&s.at("identity")==identity,"native search checkpoint identity differs");
    phase=s.at("phase").get<std::string>();need(phase=="training"||phase=="scoring"||phase=="complete","native search saved phase");
    index=s.at("index").get<std::size_t>();best=s.at("best").get<std::size_t>();fit_calls=s.at("fit_calls").get<std::uint64_t>();reused_calls=s.value("reused_calls",std::uint64_t(0));results=s.at("results");
    training_cache_sources=s.value("training_cache_sources",J::array());
    for(const auto& n:s.at("models"))models.push_back(unpack(n));
    if(!s.at("current").is_null())current=unpack(s.at("current"));
    need(index==models.size()&&index==results.size()&&index<=trials.size(),"native search completed prefix differs");
    const auto proposed=std::uint64_t(index)+(current?1:0);
    need((phase=="scoring")==bool(current)&&fit_calls<=proposed&&reused_calls==proposed-fit_calls,"native search completed FIT/reuse extent differs");
    need(models.empty()?best==0:best<models.size(),"native search incumbent extent");
    need(phase=="complete"?index==trials.size():index<trials.size(),"native search phase extent");
    resumed=true;
  }
  auto snapshot=[&]{J retained=J::array();for(const auto& n:models)retained.push_back(pack(n));return J{{"format",format},{"identity",identity},{"phase",phase},{"index",index},{"best",best},{"fit_calls",fit_calls},{"reused_calls",reused_calls},{"training_cache_sources",training_cache_sources},{"results",results},{"models",std::move(retained)},{"current",current?pack(*current):J(nullptr)},{"mid_native_round_resume",false},{"periodic_checkpoint_writes",false}};};
  auto save=[&]{need(bool(coordinator),"native checkpoint path not configured");coordinator->finish();auto status=coordinator->poll();need(!status.failed,"native search checkpoint writer failed");need(coordinator->save_host(snapshot()),"native search checkpoint writer busy");};
  auto boundary=[&](bool final=false){
    if(coordinator&&(final||stopping||checkpoint_requested)){checkpoint_requested=0;save();if(final||stopping){coordinator->finish();need(!coordinator->poll().failed,"native checkpoint publication failed");}}
    std::cout<<J{{"event","native_search_boundary"},{"phase",phase},{"completed_candidates",index},{"completed_fit_calls",fit_calls},{"fit_calls_this_process",fit_calls_this_process},{"reused_training_candidates",reused_calls},{"stop_requested",bool(stopping)}}.dump()<<'\n'<<std::flush;
    return bool(stopping);
  };
  if(phase!="complete"){
    TrainingCache cache(plan,descriptor,budget);
    training_cache_this_process=cache.metadata();
    if(resumed)need(training_cache_sources==cache.metadata().at("sources"),"resumed training cache source generations differ");
    training_cache_sources=cache.metadata().at("sources");
    auto all=stage_dataset(descriptor);
    auto fit=fit_prefix(all);
    ResidentTrainer trainer(plan,fit);trainer.restore_completed_trial_count(fit_calls);
    ResidentEvaluation evaluation(all);
    if(current)trainer.restore_model(current->model_json,current->model_sha256);
    while(index<trials.size()){
      if(boundary())return 2;
      if(phase=="training"){
        if(auto cached=cache.find(trials[index])){
          const auto reuse_started=std::chrono::steady_clock::now();
          // TrainingCache owns immutable source buffers. Bind and validate a
          // parent once, then select any requested prefix from that parent.
          if(trainer.retained_prefix_source_sha256()!=cached->model->model_sha256)
            trainer.retain_prefix_source(cached->model->model_json,cached->model->model_sha256);
          current=trainer.slice_retained_prefix(trials[index].at("rounds").get<std::uint32_t>());
          current->metrics["hyperparameters"]=trials[index];
          current->metrics["training_reused"]=true;
          current->metrics["source_checkpoint"]=cached->source_checkpoint;
          current->metrics["prefix_reuse_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-reuse_started).count();
          ++reused_calls;++reused_calls_this_process;
        }else{
          current=trainer.train(trials[index]);++fit_calls;++fit_calls_this_process;
        }
        phase="scoring";
        if(boundary())return 2;
      }
      need(current.has_value(),"native search scoring source absent");
      NativeOracle oracle;oracle.features=trainer.features();oracle.classes=trainer.classes();oracle.objective="multi:softmax";oracle.source_sha256=current->model_sha256;oracle.library_sha256=trainer.metadata().at("native_library_sha256").get<std::string>();oracle.predict=[&trainer](const float* x,std::uint64_t n,bool margin){return trainer.predict(x,n,margin);};
      auto score=evaluation.evaluate_native(oracle);
      const auto incumbent_errors=models.empty()?UINT64_MAX:results.at(best).at("evaluation").at("VALID_errors").get<std::uint64_t>();
      const auto incumbent_bytes=models.empty()?UINT64_MAX:std::uint64_t(models.at(best).model_json.size());
      const bool preferred=evaluation.prefer_last_native_candidate(current->model_json.size(),incumbent_errors,incumbent_bytes);
      J result={{"trial",index+1},{"hyperparameters",trials[index]},{"training",current->metrics},{"evaluation",score},{"native_model_bytes",current->model_json.size()},{"source_sha256",current->model_sha256},{"became_incumbent",preferred}};
      models.push_back(std::move(*current));current.reset();results.push_back(result);if(preferred)best=index;
      ++index;phase=index==trials.size()?"complete":"training";
      std::cout<<J{{"event","native_search_candidate"},{"trial",index},{"VALID_errors",score.at("VALID_errors")},{"VALID_accuracy",score.at("VALID_accuracy")},{"incumbent_trial",best+1},{"training_seconds",result.at("training").at("seconds_including_buffer_export")},{"training_reused",result.at("training").value("training_reused",false)},{"native_model_bytes",result.at("native_model_bytes")},{"conversion_performed",false}}.dump()<<'\n'<<std::flush;
    }
  }
  if(boundary(true))return 2;
  need(!models.empty()&&best<models.size(),"native search incumbent missing");
  J report={{"format","native-accuracy-search-result-1"},{"complete",true},{"dataset",descriptor},{"fixed_trials",trials},{"trials",results},{"selected_trial",best+1},{"selected",results.at(best)},{"selected_model_path",(output/"selected-model.json").string()},{"selection_rule",identity.at("selection_rule")},{"selection_performed",true},{"selection_computed_on_CUDA",true},{"VALID_used_for_selection",true},{"TEST_read",false},{"conversion_performed",false},{"resumed_experiment",resumed},{"completed_fit_calls",fit_calls},{"fit_calls_this_process",fit_calls_this_process},{"per_trial_file_writes",0},{"periodic_checkpoint_writes",false},{"process_wall_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
  report["reused_training_candidates"]=reused_calls;
  report["reused_training_candidates_this_process"]=reused_calls_this_process;
  report["training_cache_sources"]=training_cache_sources;
  report["training_cache_this_process"]=training_cache_this_process;
  if(coordinator){auto s=coordinator->poll();report["experiment_checkpoint"]={{"committed",s.committed},{"failed",s.failed},{"generation",s.generation},{"host_bytes",s.host_bytes}};}
  std::filesystem::create_directories(output);
  write(output/"selected-model.json",models.at(best).model_json);
  write(output/"result.json",report.dump(2)+"\n");
  std::cout<<J{{"event","native_search_complete"},{"selected_trial",best+1},{"VALID_errors",results.at(best).at("evaluation").at("VALID_errors")},{"output",output.string()},{"TEST_read",false}}.dump()<<'\n'<<std::flush;
  return 0;
}
} // namespace class_study::native_search
