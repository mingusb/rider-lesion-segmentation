#include "class_study_grouped_segmentation.hpp"
#include "class_study_train.hpp"
#include "class_study_process.hpp"
#include "class_study_options.hpp"
#include "class_study_checkpoint.hpp"
#include "class_study_native_search.hpp"
#include "class_study_nonlinear_search.hpp"
#include "class_study_bundle.hpp"
#include "class_study_frozen_refit.hpp"
#include "class_study_final_evaluation.hpp"
#include "class_model_dataset_contract.hpp"
#include "class_io.hpp"
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
using J=nlohmann::json;
namespace {
using Clock=std::chrono::steady_clock;namespace cp=class_study::checkpoint;
volatile std::sig_atomic_t interruption=0,checkpoint_request=0;
void record_interruption(int signal){interruption=signal;}
void record_checkpoint_request(int){checkpoint_request=1;}
void require(bool b,const std::string& message){if(!b)throw std::runtime_error(message);}
std::uint64_t integer(const J&v,const char*key,std::uint64_t max){require(v.is_number_integer(),std::string(key)+" must be integer");std::uint64_t n;if(v.is_number_unsigned())n=v.get<std::uint64_t>();else{auto s=v.get<std::int64_t>();require(s>=0,std::string(key)+" must be nonnegative");n=std::uint64_t(s);}require(n<=max,std::string(key)+" exceeds capacity");return n;}
std::vector<J> fixed_trials(const J&plan,std::uint32_t K){const auto&base=plan.at("hyperparameters");require(base.is_object(),"hyperparameters must be object");(void)class_study::ResidentTrainer::validate_hyperparameters(base,K);std::vector<J>trials;if(plan.contains("trials")){const auto&r=plan.at("trials");require(r.is_array()&&!r.empty(),"trials must be nonempty array");for(const auto&over:r){require(over.is_object(),"trial override must be object");J hp=base;hp.update(over);trials.push_back(class_study::ResidentTrainer::validate_hyperparameters(hp,K));}}else trials.push_back(class_study::ResidentTrainer::validate_hyperparameters(base,K));return trials;}
std::string trial_name(std::size_t n){std::ostringstream s;s<<"trial-"<<std::setw(4)<<std::setfill('0')<<n+1;return s.str();}
void write_bytes(const std::filesystem::path&p,const std::string&b){std::ofstream o(p,std::ios::binary);o.exceptions(std::ios::badbit|std::ios::failbit);o.write(b.data(),b.size());o.flush();}
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
class_study::NativeOracle oracle(class_study::ResidentTrainer&t,const std::string&source){class_study::NativeOracle o;o.features=t.features();o.classes=t.classes();o.source_sha256=source;o.library_sha256=t.metadata().at("native_library_sha256").get<std::string>();o.predict=[&t](const float*x,std::uint64_t n,bool margin){return t.predict(x,n,margin);};return o;}
struct Retained{class_study::TrainingResult native;class_study::ConvertedModel model;};
J packed_native(const class_study::TrainingResult&n){return{{"model_json",cp::binary(n.model_json)},{"model_sha256",n.model_sha256},{"metrics",n.metrics}};}
class_study::TrainingResult unpack_native(const J&j){return{cp::bytes(j.at("model_json")),j.at("model_sha256").get<std::string>(),j.at("metrics")};}
J packed_model(const class_study::ConvertedModel&m){const auto&v=m.runtime->metadata();return{{"canonical",cp::binary(m.canonical_bytes)},{"compact",cp::binary(m.compact_bytes)},{"source_sha256",v.source_sha256},{"canonical_sha256",v.canonical_sha256},{"compact_sha256",m.compact_bytes.empty()?std::string{}:v.compact_sha256},{"metrics",m.metrics}};}
class_study::ConvertedModel unpack_model(const J&j,class_runtime::Residency residency){class_study::ConvertedModel m;m.canonical_bytes=cp::bytes(j.at("canonical"));m.compact_bytes=cp::bytes(j.at("compact"));m.metrics=j.at("metrics");m.runtime=class_runtime::Runtime::load(m.canonical_bytes,j.at("canonical_sha256").get<std::string>(),j.at("source_sha256").get<std::string>(),m.compact_bytes,j.at("compact_sha256").get<std::string>(),residency);return m;}
J semantic_plan(J p){for(const char*k:{"experiment_checkpoint_path","experiment_resume_from","experiment_checkpoint_host_byte_budget"})p.erase(k);if(p.contains("conversion")&&p["conversion"].is_object())for(const char*k:{"checkpoint_path","resume_from","checkpoint_interval_seconds","checkpoint_host_byte_budget"})p["conversion"].erase(k);return p;}
}
int main(int argc,char**argv){try{
 if(argc==5&&std::string(argv[1])=="evaluate-bundle")
  return class_study::bundle_evaluation::run(argv[2],argv[3],argv[4]);
 require(argc>=4,"usage: class_study TRAIN_PLAN EVAL_DESCRIPTOR FRESH_FINAL_OUTPUT_DIRECTORY [--resume EXPERIMENT_CHECKPOINT_DIRECTORY] [--stop-after-oof]");
 std::string cli_resume;bool stop_after_oof=false;
 for(int i=4;i<argc;++i){const std::string option=argv[i];
  if(option=="--resume"){require(cli_resume.empty()&&i+1<argc,"--resume requires one checkpoint directory and may occur only once");cli_resume=argv[++i];require(!cli_resume.empty()&&cli_resume.rfind("--",0)!=0,"--resume requires a checkpoint directory");}
  else if(option=="--stop-after-oof"){require(!stop_after_oof,"--stop-after-oof may occur only once");stop_after_oof=true;}
  else require(false,"unknown study option: "+option);
 }
 auto start=Clock::now();std::filesystem::path output=argv[3];require(output.is_absolute(),"final output must be absolute directory");
 auto plan=J::parse(dpnative::read_text(argv[1]));auto data=J::parse(dpnative::read_text(argv[2]));
 require(!stop_after_oof||plan.value("workflow",std::string{})=="native-nonlinear-combination-1","--stop-after-oof is supported only by the nonlinear combination workflow");
 if(plan.value("workflow",std::string{})=="native-final-evaluation-1")
  return class_study::final_evaluation::run(plan,data,output);
 require(!std::filesystem::exists(output),"final output must be fresh absolute directory");
 if(plan.value("workflow",std::string{})=="native-grouped-segmentation-replay-1"){
  if(plan.value("validate_only",false)){std::cout<<class_study::grouped_segmentation::validate_replay(plan,data).dump(2)<<'\n';return 0;}
  std::signal(SIGINT,record_interruption);std::signal(SIGTERM,record_interruption);
  return class_study::grouped_segmentation::replay(plan,data,output,interruption);
 }
 if(plan.value("workflow",std::string{})=="native-grouped-segmentation-1"){
  if(plan.value("validate_only",false)){std::cout<<class_study::grouped_segmentation::validate(plan,data).dump(2)<<'\n';return 0;}
  std::signal(SIGINT,record_interruption);std::signal(SIGTERM,record_interruption);
  return class_study::grouped_segmentation::run(plan,data,output,interruption);
 }
 if(plan.value("workflow",std::string{})=="native-frozen-refit-1"){
  std::signal(SIGINT,record_interruption);std::signal(SIGTERM,record_interruption);std::signal(SIGUSR1,record_checkpoint_request);
  return class_study::frozen_refit::run(plan,data,output,cli_resume,interruption,checkpoint_request);
 }
 if(plan.value("workflow",std::string{})=="native-nonlinear-combination-1"){
  std::signal(SIGINT,record_interruption);std::signal(SIGTERM,record_interruption);std::signal(SIGUSR1,record_checkpoint_request);
  return class_study::nonlinear_search::run(plan,data,output,cli_resume,interruption,checkpoint_request,stop_after_oof);
 }
 if(plan.value("workflow",std::string{})=="native-accuracy-search-1"){
  std::signal(SIGINT,record_interruption);std::signal(SIGTERM,record_interruption);std::signal(SIGUSR1,record_checkpoint_request);
  return class_study::native_search::run(plan,data,output,cli_resume,interruption,checkpoint_request);
 }
 auto shape=class_study::ResidentTrainer::validate_metadata(plan);auto eval_shape=class_model_contract::dense(data,false);const auto F=shape.at("features").get<std::uint32_t>(),K=shape.at("classes").get<std::uint32_t>();require(F==eval_shape.features&&K==eval_shape.classes,"training/evaluation shape differs");
 std::signal(SIGINT,record_interruption);std::signal(SIGTERM,record_interruption);std::signal(SIGUSR1,record_checkpoint_request);
 auto options=class_study::parse_conversion_options(plan.value("conversion",J::object()),F,class_runtime::Residency::compact_only);auto passes=std::uint32_t(integer(plan.value("simplifier_passes",J(4)),"simplifier_passes",16));require(passes>0,"positive simplifier passes required");
 auto trials=fixed_trials(plan,K);const auto workflow_plan=semantic_plan(plan);const auto plan_pin=dpnative::sha256(workflow_plan.dump());
 std::string resume_root=cli_resume.empty()?plan.value("experiment_resume_from",std::string{}):cli_resume;std::string checkpoint_root=plan.value("experiment_checkpoint_path",resume_root);auto host_budget=integer(plan.value("experiment_checkpoint_host_byte_budget",J(0)),"experiment_checkpoint_host_byte_budget",UINT64_MAX);
 std::unique_ptr<cp::Coordinator> coordinator;if(!checkpoint_root.empty())coordinator=std::make_unique<cp::Coordinator>(checkpoint_root,host_budget);
 J results=J::array(),failure=nullptr;std::vector<Retained>retained;retained.reserve(trials.size());std::optional<class_study::TrainingResult>current;std::optional<class_study::ConvertedModel>current_model;
 std::size_t trial_index=0,evaluations=0;std::uint64_t fit_calls=0,fit_calls_this_process=0;std::string phase="training";bool resumed=false,evaluation_loading_attempted=false,evaluation_loaded=false;
 double staging_seconds=0,construction_seconds=0,evaluation_seconds=0;std::string resumed_engine;
 if(!resume_root.empty()){
  auto loaded=cp::Coordinator::load(resume_root,host_budget);auto&s=loaded.state;require(s.at("plan_sha256")==plan_pin&&s.at("workflow_plan")==workflow_plan,"experiment plan identity differs");require(s.at("evaluation_descriptor")==data,"experiment evaluation descriptor differs");require(s.at("fixed_trials")==J(trials),"experiment expanded trials differ");require(s.at("features")==F&&s.at("classes")==K&&s.at("selection_performed")==false&&s.at("controller_kind")=="fixed_trials","unsupported experiment controller/shape");
  trial_index=std::size_t(integer(s.at("current_trial"),"current_trial",trials.size()));evaluations=std::size_t(integer(s.at("evaluation_completed"),"evaluation_completed",trials.size()));fit_calls=integer(s.at("completed_fit_calls"),"completed_fit_calls",trials.size());phase=s.at("phase").get<std::string>();
  require(phase=="training"||phase=="conversion"||phase=="simplification"||phase=="evaluation"||phase=="complete","unsupported experiment saved phase");results=s.at("results");require(results.is_array(),"experiment results shape");
  for(const auto&r:s.at("retained")){auto n=unpack_native(r.at("native"));auto m=unpack_model(r.at("model"),options.residency);require(n.model_sha256==m.runtime->metadata().source_sha256,"retained source binding differs");retained.push_back({std::move(n),std::move(m)});}
  require(retained.size()<=trials.size()&&evaluations<=retained.size()&&results.size()==retained.size(),"experiment retained/evaluation extent");
  if(!s.at("current_native").is_null())current=unpack_native(s.at("current_native"));if(!s.at("current_model").is_null())current_model=unpack_model(s.at("current_model"),class_runtime::Residency::dual);
  const bool constructing=phase=="training"||phase=="conversion"||phase=="simplification";require(constructing?trial_index==retained.size():(retained.size()==trials.size()&&trial_index==trials.size()),"experiment trial-stage prefix differs");require(fit_calls==retained.size()+(current?1:0),"experiment completed fit count differs");require((phase=="conversion"||phase=="simplification")?bool(current):!current,"experiment current native phase differs");require((phase=="simplification")==bool(current_model),"experiment current compiled phase differs");
  resumed_engine=loaded.engine_snapshot_path;require(resumed_engine.empty()||phase=="conversion","engine snapshot at incompatible study phase");resumed=true;
 }
 auto snapshot_state=[&]{J kept=J::array();for(const auto&r:retained)kept.push_back({{"native",packed_native(r.native)},{"model",packed_model(r.model)}});return J{{"format","resident-study-checkpoint-1"},{"workflow_plan",workflow_plan},{"complete_plan",plan},{"plan_sha256",plan_pin},{"evaluation_descriptor",data},{"features",F},{"classes",K},{"fixed_trials",trials},{"declared_seeds",[&]{J seeds=J::array();for(const auto&hp:trials)seeds.push_back(hp.at("seed"));return seeds;}()},{"controller_kind","fixed_trials"},{"HPO_state",nullptr},{"RL_state",nullptr},{"nested_controller_frames",J::array()},{"current_trial",trial_index},{"phase",phase},{"completed_fit_calls",fit_calls},{"retained",std::move(kept)},{"results",results},{"current_native",current?packed_native(*current):J(nullptr)},{"current_model",current_model?packed_model(*current_model):J(nullptr)},{"evaluation_completed",evaluations},{"selection_performed",false},{"selected_candidate",nullptr},{"VALID_evaluation_after_all_fixed_construction",true},{"training_resume_scope","completed FIT calls only; no partial native optimizer/RNG snapshot"}};};
 auto check_writer=[&]{if(coordinator){coordinator->finish();auto s=coordinator->poll();require(!s.failed,"experiment checkpoint write failed: "+s.error);}};
 auto stage_boundary=[&](bool force=false){if(coordinator&&(force||checkpoint_request||interruption)){check_writer();checkpoint_request=0;require(coordinator->save_host(snapshot_state()),"experiment writer busy");if(force||interruption)check_writer();}if(interruption)throw std::runtime_error("study interrupted at completed stage boundary");};
 options.stop_requested=[] {return interruption!=0;};options.checkpoint_requested=[] {if(!checkpoint_request)return false;checkpoint_request=0;return true;};
 if(coordinator){options.checkpoint_path=(std::filesystem::path(checkpoint_root)/".dispatch-only-engine").string();options.checkpoint_publication=[&](std::uint64_t gpu_bytes){return coordinator->transaction(snapshot_state(),gpu_bytes);};options.checkpoint_on_completion=false;}
 options.resume_from=resumed_engine;
 std::unique_ptr<class_study::ResidentTrainer>trainer;try{auto t=Clock::now();trainer=std::make_unique<class_study::ResidentTrainer>(plan);trainer->restore_completed_trial_count(fit_calls);if(current)trainer->restore_model(current->model_json,current->model_sha256);staging_seconds=seconds(t);}catch(const std::exception&e){failure={{"phase","initial_training_staging"},{"message",e.what()}};}
 auto construction_start=Clock::now();
 if(trainer&&failure.is_null())try{
  for(;trial_index<trials.size();){auto trial_start=Clock::now();J item={{"trial",trial_index+1},{"hyperparameters",trials[trial_index]},{"construction_complete",false},{"evaluation_status","pending_after_fixed_construction"},{"intermediate_files_written",coordinator?J(nullptr):J(0)}};
   options.progress=[&,i=trial_index,throttle=class_study::ProgressThrottle{}](const J&p)mutable{if(interruption&&!coordinator&&options.checkpoint_path.empty())throw class_study::ConversionFailure("study conversion interrupted at progress boundary",p);if(throttle.emit(p))std::cout<<J{{"event","conversion_progress"},{"trial",i+1},{"statistics",p}}.dump()<<'\n'<<std::flush;};
   if(phase=="training"){stage_boundary();auto t=Clock::now();current=trainer->train(trials[trial_index]);++fit_calls;++fit_calls_this_process;phase="conversion";item["stage_seconds"]["training"]=seconds(t);stage_boundary();}
   if(phase=="conversion"){require(bool(current),"conversion source missing");auto t=Clock::now();auto native=oracle(*trainer,current->model_sha256);class_study::ConversionSource source{current->model_json,F,K,current->model_sha256};current_model=class_study::convert_model(source,native,options);options.resume_from.clear();phase="simplification";item["conversion"]=current_model->metrics;item["stage_seconds"]["conversion"]=seconds(t);stage_boundary();}
   require(phase=="simplification"&&current&&current_model,"study construction phase invalid");auto conversion=current_model->metrics;auto t=Clock::now();auto simplified=class_study::simplify_model(std::move(*current_model),passes);current_model.reset();double simplify_seconds=seconds(t);if(options.residency!=class_runtime::Residency::dual)simplified.runtime->retain(options.residency);const auto&m=simplified.runtime->metadata();
   item.update({{"construction_complete",true},{"training",current->metrics},{"conversion",conversion},{"simplification",simplified.metrics},{"source_sha256",current->model_sha256},{"canonical_sha256",m.canonical_sha256},{"compact_sha256",m.compact_sha256},{"retained_graph_device_bytes",m.device_bytes},{"canonical_resident",m.canonical_resident},{"compact_resident",m.compact_resident},{"construction_wall_seconds",seconds(trial_start)}});item["stage_seconds"]["simplification"]=simplify_seconds;retained.push_back({std::move(*current),std::move(simplified)});current.reset();results.push_back(std::move(item));++trial_index;phase=trial_index==trials.size()?"evaluation":"training";
   std::cout<<"trial "<<trial_index<<" construction complete; nodes="<<m.nodes<<" retained_graph_bytes="<<m.device_bytes<<'\n'<<std::flush;stage_boundary();
  }
 }catch(const std::exception&e){failure={{"phase",phase},{"trial",trial_index+1},{"message",e.what()}};if(auto*p=dynamic_cast<const class_study::ConversionFailure*>(&e))failure["partial_statistics"]=p->partial_statistics;std::cerr<<"study stopped during "<<phase<<": "<<e.what()<<'\n';}
 construction_seconds=seconds(construction_start);bool all_constructed=failure.is_null()&&retained.size()==trials.size();
 if(all_constructed&&evaluations<trials.size()){auto t=Clock::now();try{phase="evaluation";stage_boundary();evaluation_loading_attempted=true;class_study::ResidentEvaluation evaluator(data);evaluation_loaded=true;for(;evaluations<retained.size();){auto&r=retained[evaluations];trainer->restore_model(r.native.model_json,r.native.model_sha256);auto native=oracle(*trainer,r.native.model_sha256);auto metrics=evaluator.evaluate(r.model,native);results[evaluations]["evaluation"]=metrics;results[evaluations]["evaluation_status"]="complete";++evaluations;phase=evaluations==trials.size()?"complete":"evaluation";std::cout<<"trial "<<evaluations<<" evaluation complete; class_mismatches="<<metrics.at("class_mismatches")<<'\n'<<std::flush;stage_boundary();}}catch(const std::exception&e){failure={{"phase",phase},{"trial",evaluations+1},{"message",e.what()}};}evaluation_seconds=seconds(t);}
 bool completed=failure.is_null()&&all_constructed&&evaluations==trials.size();if(completed){phase="complete";if(coordinator)try{stage_boundary(true);}catch(const std::exception&e){failure={{"phase","final_experiment_checkpoint"},{"message",e.what()}};completed=false;}}
 check_writer();auto export_start=Clock::now();std::size_t exports=0;std::filesystem::create_directories(output);try{for(std::size_t i=0;i<retained.size();++i){auto base=output/trial_name(i);write_bytes(base.string()+".canonical",retained[i].model.canonical_bytes);results[i]["canonical_path"]=base.string()+".canonical";if(!retained[i].model.compact_bytes.empty()){write_bytes(base.string()+".compact",retained[i].model.compact_bytes);results[i]["compact_path"]=base.string()+".compact";}results[i]["final_export_complete"]=true;++exports;}}catch(const std::exception&e){failure={{"phase","final_export"},{"message",e.what()}};completed=false;}
 J checkpoint_status=nullptr;if(coordinator){auto s=coordinator->poll();checkpoint_status={{"committed",s.committed},{"failed",s.failed},{"generation",s.generation},{"host_bytes",s.host_bytes},{"error",s.error}};}
 J report={{"format","resident-study-result-3"},{"completed",completed},{"failure",failure},{"features",F},{"classes",K},{"fixed_trials",trials},{"construction_completed",retained.size()},{"evaluation_completed",evaluations},{"final_export_completed",exports},{"conversion_options",class_study::describe_conversion_options(options)},{"simplifier_passes",passes},{"trials",results},{"selection_performed",false},{"VALID_evaluation_after_all_fixed_construction",true},{"evaluation_loading_attempted",evaluation_loading_attempted},{"evaluation_dataset_loaded_once",evaluation_loaded},{"intermediate_files_written",coordinator||!options.checkpoint_path.empty()?J(nullptr):J(0)},{"experiment_checkpoint",checkpoint_status},{"resumed_experiment",resumed},{"checkpoint_scope",coordinator?"whole maintained fixed-trial study plus current converter; completed training calls; no active HPO/RL controller":"current conversion only when opted in"},{"native_model_restore_transport","retained JSON bytes in RAM"},{"completed_fit_calls",fit_calls},{"fit_calls_this_process",fit_calls_this_process},{"stage_timing_scope","current process unless included in retained training/conversion/simplification metrics"},{"phase_seconds",{{"training_staging",staging_seconds},{"fixed_construction",construction_seconds},{"deferred_evaluation",evaluation_seconds},{"final_artifact_export",seconds(export_start)}}},{"process_wall_seconds",seconds(start)}};
 write_bytes(output/"result.json",report.dump(2)+"\n");std::cout<<"study "<<(completed?"complete":"partial")<<"; constructed="<<retained.size()<<'/'<<trials.size()<<" evaluated="<<evaluations<<" report="<<output/"result.json"<<'\n';return completed?0:1;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
