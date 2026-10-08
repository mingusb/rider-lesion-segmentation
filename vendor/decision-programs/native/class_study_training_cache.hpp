#pragma once
// Host metadata and opaque completed native models only. Training, slicing and
// scoring remain in ResidentTrainer/the existing native search workflow.
#include "class_study_checkpoint.hpp"
#include "class_study_train.hpp"
#include "class_io.hpp"
#include <filesystem>
#include <optional>
#include <set>
#include <vector>

namespace class_study {
struct CachedTraining {
  const TrainingResult* model = nullptr;
  const nlohmann::json* hyperparameters = nullptr;
  std::string source_checkpoint;
  std::uint64_t available_rounds = 0;
};
class TrainingCache {
  using J=nlohmann::json; using u64=std::uint64_t;
  struct Entry { TrainingResult model; J hp,match;std::string source;u64 rounds=0; };
  std::vector<Entry> entries_;
  std::uint32_t classes_=0;
  J metadata_={{"format","native-training-checkpoint-cache-1"},{"checkpoint_loads",0},
               {"model_buffer_hashes",0},{"resident_model_bytes",0},
               {"sources",J::array()},{"matching","all normalized parameters except rounds; shortest sufficient, then loaded order"},
               {"sampling_omission_policy","exact normalized keys; omitted versus explicit 1 may conservatively miss"},
               {"CPU_model_predictions",false},{"GPU_operations",0}};
  static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
  static u64 number(const J&v,const char*m){need(v.is_number_integer(),m);if(v.is_number_unsigned())return v.get<u64>();const auto n=v.get<std::int64_t>();need(n>=0,m);return u64(n);}
  static void false_role(const J&j,const char*k){need(j.at(k).is_boolean()&&!j.at(k).get<bool>(),"training cache source FIT-only/TEST role differs");}
  static std::vector<J> declared_trials(const J&plan,std::uint32_t K){
    const auto&base=plan.at("hyperparameters");need(base.is_object(),"training cache source base parameters");std::vector<J>out;
    if(plan.contains("trials")){need(plan.at("trials").is_array()&&!plan.at("trials").empty(),"training cache source trial list");for(const auto&o:plan.at("trials")){need(o.is_object(),"training cache trial override");auto hp=base;hp.update(o);out.push_back(ResidentTrainer::validate_hyperparameters(hp,K));}}
    else out.push_back(ResidentTrainer::validate_hyperparameters(base,K));return out;
  }
 public:
  // Only versioned COMPLETE Coordinator checkpoints are accepted. Nothing is
  // loaded when the optional list is absent/empty. Returned model pointers stay
  // valid for this cache's lifetime; construction is its only mutation phase.
  TrainingCache(const J&plan,const J&descriptor,u64 budget=0){
    if(!plan.contains("training_source_checkpoints"))return;
    const auto&sources=plan.at("training_source_checkpoints");need(sources.is_array(),"training_source_checkpoints must be an array");if(sources.empty())return;
    const auto K=number(descriptor.at("classes"),"training cache classes integer");need(K>=2&&K<=UINT32_MAX,"training cache classes range");classes_=std::uint32_t(K);
    false_role(plan,"TEST_read");false_role(plan,"VALID_read");false_role(descriptor,"TEST_read");
    need(plan.at("native_library_path").is_string()&&plan.at("native_library_sha256").is_string(),"training cache library identity");
    std::vector<std::filesystem::path>paths;std::set<std::string>seen;
    for(const auto&value:sources){need(value.is_string(),"training cache checkpoint path type");auto s=value.get<std::string>();need(!s.empty()&&s.find('\0')==std::string::npos,"training cache checkpoint path");auto p=std::filesystem::path(s).lexically_normal();need(p.is_absolute()&&seen.insert(p.string()).second,"training cache absolute unique checkpoint roots");paths.push_back(std::move(p));}
    u64 retained=0,hashes=0;
    for(const auto&path:paths){
      auto loaded=checkpoint::Coordinator::load(path,budget,"native-accuracy-search-checkpoint-1");const auto&state=loaded.state;
      need(loaded.engine_snapshot_path.empty()&&state.at("phase")=="complete"&&state.at("current").is_null(),"training cache requires a complete native-only checkpoint");
      const auto&identity=state.at("identity");const auto&original=identity.at("plan");
      need(identity.at("dataset")==descriptor&&original.at("workflow")=="native-accuracy-search-1","training cache source dataset/workflow differs");
      need(original.at("native_library_path")==plan.at("native_library_path")&&original.at("native_library_sha256")==plan.at("native_library_sha256"),"training cache native library identity differs");
      false_role(original,"TEST_read");false_role(original,"VALID_read");
      if(original.contains("dataset_source"))need(original.at("dataset_source")==descriptor,"training cache declared dataset differs");
      const auto expected=declared_trials(original,classes_);const auto&trials=identity.at("trials");const auto&results=state.at("results");const auto&models=state.at("models");
      need(trials.is_array()&&results.is_array()&&models.is_array()&&!models.empty()&&trials.size()==expected.size()&&models.size()==trials.size()&&results.size()==models.size(),"training cache completed model/result/trial extents");
      const auto index=number(state.at("index"),"training cache index integer"),fits=number(state.at("fit_calls"),"training cache fit calls integer");const auto reused=state.contains("reused_calls")?number(state.at("reused_calls"),"training cache reuse calls integer"):0;
      need(index==models.size()&&fits<=index&&reused==index-fits&&number(state.at("best"),"training cache incumbent integer")<index,"training cache completed counters/incumbent differ");
      u64 source_bytes=0;
      for(std::size_t i=0;i<models.size();++i){
        const auto hp=ResidentTrainer::validate_hyperparameters(trials.at(i),classes_);
        need(hp==trials.at(i)&&hp==expected.at(i),"training cache normalized/declaration parameters differ");const auto&r=results.at(i);const auto&m=models.at(i);const auto&metrics=m.at("metrics");
        need(number(r.at("trial"),"training cache trial integer")==i+1&&r.at("hyperparameters")==hp&&metrics.is_object()&&metrics.at("hyperparameters")==hp&&r.at("training")==metrics,"training cache result/metrics parameter binding differs");
        need(m.at("model").is_binary()&&m.at("sha256").is_string(),"training cache opaque model buffer");const auto n=u64(m.at("model").get_binary().size());need(n>0&&n<=UINT64_MAX-retained&&(!budget||n<=budget-retained),"training cache retained model byte budget/extent");
        TrainingResult native{checkpoint::bytes(m.at("model")),m.at("sha256").get<std::string>(),metrics};
        need(dpnative::sha256(native.model_json)==native.model_sha256&&r.at("source_sha256")==native.model_sha256&&number(r.at("native_model_bytes"),"training cache model bytes integer")==n,"training cache native model identity/extent differs");++hashes;
        if(metrics.contains("model_sha256"))need(metrics.at("model_sha256")==native.model_sha256,"training cache metric model identity differs");
        if(metrics.contains("model_bytes"))need(number(metrics.at("model_bytes"),"training cache metric model bytes integer")==n,"training cache metric model extent differs");
        for(const char*role:{"VALID_read","TEST_read"})if(metrics.contains(role))false_role(metrics,role);
        auto match=hp;match.erase("rounds");const auto rounds=number(hp.at("rounds"),"training cache rounds integer");entries_.push_back({std::move(native),hp,std::move(match),path.string(),rounds});retained+=n;source_bytes+=n;
      }
      metadata_["sources"].push_back({{"checkpoint",path.string()},{"generation",loaded.generation},{"models",models.size()},{"model_bytes",source_bytes}});
    }
    metadata_["checkpoint_loads"]=paths.size();metadata_["model_buffer_hashes"]=hashes;metadata_["resident_model_bytes"]=retained;
  }
  TrainingCache(const TrainingCache&)=delete;TrainingCache&operator=(const TrainingCache&)=delete;
  TrainingCache(TrainingCache&&)noexcept=default;TrainingCache&operator=(TrainingCache&&)noexcept=default;
  std::optional<CachedTraining> find(const J&parameters)const{
    if(entries_.empty())return std::nullopt;
    auto hp=ResidentTrainer::validate_hyperparameters(parameters,classes_);const auto rounds=number(hp.at("rounds"),"training cache requested rounds");hp.erase("rounds");
    const Entry*best=nullptr;for(const auto&e:entries_)if(e.rounds>=rounds&&e.match==hp&&(!best||e.rounds<best->rounds))best=&e;
    if(!best)return std::nullopt;return CachedTraining{&best->model,&best->hp,best->source,best->rounds};
  }
  const J& metadata()const{return metadata_;}
};
} // namespace class_study