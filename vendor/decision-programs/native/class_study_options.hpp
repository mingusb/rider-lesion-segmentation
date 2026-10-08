#pragma once
#include "class_study_convert.hpp"
#include <limits>
#include <chrono>
#include <cmath>
#include <unordered_set>

// Metadata parsing only. Both maintained frontends call the same converter.
namespace class_study {
inline void option_require(bool condition,const std::string& message) {
  if(!condition) throw std::invalid_argument(message);
}
inline std::uint64_t option_integer(const nlohmann::json& value,const std::string& key,
                                    std::uint64_t maximum=UINT64_MAX) {
  option_require(value.is_number_integer(),key+" must be an integer");
  std::uint64_t result;
  if(value.is_number_unsigned()) result=value.get<std::uint64_t>();
  else {auto signed_value=value.get<std::int64_t>();
    option_require(signed_value>=0,key+" must be nonnegative");result=std::uint64_t(signed_value);}
  option_require(result<=maximum,key+" exceeds supported capacity");return result;
}
inline const char* residency_name(class_runtime::Residency residency) {
  switch(residency) {
    case class_runtime::Residency::dual:return "dual";
    case class_runtime::Residency::canonical_only:return "canonical_only";
    case class_runtime::Residency::compact_only:return "compact_only";
  }
  throw std::invalid_argument("unsupported runtime residency");
}
inline ConversionOptions parse_conversion_options(const nlohmann::json& input,
    std::uint32_t features,class_runtime::Residency default_residency=class_runtime::Residency::dual) {
  option_require(features>0,"conversion features must be positive");
  option_require(input.is_object(),"conversion options must be an object");
  ConversionOptions output;output.residency=default_residency;
  for(auto item=input.begin();item!=input.end();++item) {
    const auto& key=item.key();const auto& value=item.value();
    if(key=="initial_states") output.initial_states=option_integer(value,key,0x3fffffffu);
    else if(key=="max_states") output.max_states=option_integer(value,key,0x3fffffffu);
    else if(key=="initial_nodes") output.initial_nodes=option_integer(value,key,0x3fffffffu);
    else if(key=="max_nodes") output.max_nodes=option_integer(value,key,0x3fffffffu);
    else if(key=="gpu_byte_budget") output.gpu_byte_budget=option_integer(value,key);
    else if(key=="max_expansions") output.max_expansions=option_integer(value,key);
    else if(key=="batch_size") output.batch_size=std::uint32_t(option_integer(value,key,65536));
    else if(key=="max_batch_size") output.max_batch_size=std::uint32_t(option_integer(value,key,65536));
    else if(key=="admission_threads") output.admission_threads=std::uint32_t(option_integer(value,key,UINT32_MAX));
    else if(key=="draft_threads") output.draft_threads=std::uint32_t(option_integer(value,key,UINT32_MAX));
    else if(key=="split_policy") {option_require(value.is_string(),"split_policy must be a string");output.split_policy=value.get<std::string>();}
    else if(key=="oldest_ready_jobs") output.oldest_ready_jobs=std::uint32_t(option_integer(value,key,UINT32_MAX));
    else if(key=="completed_cache_limit") {if(value.is_null())output.completed_cache_limit.reset();else output.completed_cache_limit=std::uint32_t(option_integer(value,key,UINT32_MAX));}
    else if(key=="refinement_visit_budget") {if(value.is_null())output.refinement_visit_budget.reset();else output.refinement_visit_budget=std::uint32_t(option_integer(value,key,UINT32_MAX));}
    else if(key=="cover_visit_budget") {if(value.is_null())output.cover_visit_budget.reset();else output.cover_visit_budget=std::uint32_t(option_integer(value,key,UINT32_MAX));}
    else if(key=="completion_estimate_enabled") {
      option_require(value.is_boolean(),key+" must be Boolean");output.completion_estimate_enabled=value.get<bool>();
    }
    else if(key=="completion_estimate_start_seconds") {
      option_require(value.is_number(),key+" must be numeric");output.completion_estimate_start_seconds=value.get<double>();
      option_require(std::isfinite(output.completion_estimate_start_seconds)&&output.completion_estimate_start_seconds>=0,
                     key+" must be finite and nonnegative");
    }
    else if(key=="completion_estimate") {
      option_require(value.is_object(),key+" must be an object");
      for(auto field=value.begin();field!=value.end();++field) {
        const auto& name=field.key();const auto& setting=field.value();
        if(name=="paths")output.completion_estimate.paths=std::uint32_t(option_integer(setting,name,UINT32_MAX));
        else if(name=="maximum_decisions")output.completion_estimate.maximum_decisions=std::uint32_t(option_integer(setting,name,UINT32_MAX));
        else if(name=="decisions_per_chunk")output.completion_estimate.decisions_per_chunk=std::uint32_t(option_integer(setting,name,UINT32_MAX));
        else if(name=="refinement_visit_budget")output.completion_estimate.refinement_visit_budget=std::uint32_t(option_integer(setting,name,UINT32_MAX));
        else if(name=="seed")output.completion_estimate.seed=option_integer(setting,name);
        else if(name=="maximum_seconds") {
          option_require(setting.is_number(),"completion_estimate.maximum_seconds must be numeric");
          output.completion_estimate.maximum_seconds=setting.get<double>();
          option_require(std::isfinite(output.completion_estimate.maximum_seconds)&&output.completion_estimate.maximum_seconds>0,
                         "completion_estimate.maximum_seconds must be finite and positive");
        } else throw std::invalid_argument("unsupported completion_estimate option: "+name);
      }
      option_require(output.completion_estimate.paths&&output.completion_estimate.decisions_per_chunk,
                     "completion_estimate paths and chunk size must be positive");
    }
    else if(key=="checkpoint_path"||key=="resume_from"||key=="proof_module_directory"||key=="proof_module_request") {
      option_require(value.is_string(),key+" must be a string");auto path=value.get<std::string>();
      option_require(path.find('\0')==std::string::npos,key+" contains NUL");
      if(key=="checkpoint_path")output.checkpoint_path=std::move(path);
      else if(key=="resume_from")output.resume_from=std::move(path);
      else if(key=="proof_module_directory")output.proof_module_directory=std::move(path);
      else output.proof_module_request=std::move(path);
    }
    else if(key=="checkpoint_interval_seconds") {
      option_require(value.is_number(),key+" must be numeric");output.checkpoint_interval_seconds=value.get<double>();
      option_require(std::isfinite(output.checkpoint_interval_seconds)&&output.checkpoint_interval_seconds>=0,
                     key+" must be finite and nonnegative (zero disables periodic saving)");
    }
    else if(key=="checkpoint_host_byte_budget")output.checkpoint_host_byte_budget=option_integer(value,key);
    else if(key=="checkpoint_on_completion") {
      option_require(value.is_boolean(),key+" must be Boolean");output.checkpoint_on_completion=value.get<bool>();
    }
    else if(key=="runtime_residency") {
      option_require(value.is_string(),key+" must be a string");auto name=value.get<std::string>();
      if(name=="dual") output.residency=class_runtime::Residency::dual;
      else if(name=="canonical_only") output.residency=class_runtime::Residency::canonical_only;
      else if(name=="compact_only") output.residency=class_runtime::Residency::compact_only;
      else throw std::invalid_argument("unsupported runtime_residency: "+name);
    } else if(key=="domain") {
      option_require(value.is_object(),"domain must be an object");
      std::unordered_set<std::uint32_t> used;
      for(auto field=value.begin();field!=value.end();++field) {
        if(field.key()=="allow_nan") {
          option_require(field.value().is_boolean(),"domain.allow_nan must be Boolean");
          output.domain.allow_nan=field.value().get<bool>();
        } else if(field.key()=="one_hot_groups") {
          option_require(field.value().is_array(),"domain.one_hot_groups must be an array");
          for(const auto& group:field.value()) {
            option_require(group.is_array()&&!group.empty(),"each one_hot_group must be a nonempty array");
            std::vector<std::uint32_t> indices;indices.reserve(group.size());
            for(const auto& feature:group) {
              auto index=std::uint32_t(option_integer(feature,"one_hot feature",features-1));
              option_require(used.insert(index).second,"one_hot_groups must be disjoint with distinct features");
              indices.push_back(index);
            }
            output.domain.one_hot_groups.push_back(std::move(indices));
          }
        } else throw std::invalid_argument("unsupported conversion domain option: "+field.key());
      }
    } else if(key=="max_cells"||key=="max_terms"||key=="global_verify_cap"||
              key=="prune_margin_floor"||key=="parallel_priority_builder"||key=="parallel_margin_separation")
      throw std::invalid_argument("retired Cartesian conversion option: "+key);
    else throw std::invalid_argument("unsupported conversion option: "+key);
  }
  option_require(output.initial_states>0&&output.max_states>0&&output.initial_states<=output.max_states,
                 "state capacities must be positive with initial_states <= max_states");
  option_require(output.initial_nodes>0&&output.max_nodes>0&&output.initial_nodes<=output.max_nodes,
                 "node capacities must be positive with initial_nodes <= max_nodes");
  option_require(output.gpu_byte_budget>0,"gpu_byte_budget must be positive");
  option_require(!output.max_batch_size || output.batch_size<=output.max_batch_size,
                 "batch_size must not exceed max_batch_size");
  option_require(!output.admission_threads || !(output.admission_threads&(output.admission_threads-1)),
                 "admission_threads must be zero or a power of two; device limits are checked at launch setup");
  option_require(!output.draft_threads || !(output.draft_threads&(output.draft_threads-1)),
                 "draft_threads must be zero or a power of two; device limits are checked at launch setup");
  option_require(output.split_policy=="source_order"||output.split_policy=="widest_residual"||output.split_policy=="aggregate_residual"||output.split_policy=="contracting_residual",
                 "unsupported split_policy");
  return output;
}
// Host telemetry only; phase changes are always visible, ordinary updates are
// emitted at most once per second. Callers check interruption before filtering.
class ProgressThrottle {
 public:
  bool emit(const nlohmann::json& statistics) {
    auto now=std::chrono::steady_clock::now();
    auto phase=statistics.is_object()&&statistics.contains("phase")?statistics.at("phase"):nlohmann::json(nullptr);
    auto eta_status=statistics.is_object()&&statistics.contains("eta")&&statistics.at("eta").is_object()?
      statistics.at("eta").value("status",std::string{}):std::string{};
    bool final=statistics.is_object()&&statistics.contains("completed")&&statistics.at("completed").is_boolean()&&statistics.at("completed").get<bool>();
    if(!started_||phase!=phase_||eta_status!=eta_status_||final||now-last_>=std::chrono::seconds(1)) {
      started_=true;phase_=std::move(phase);eta_status_=std::move(eta_status);last_=now;return true;
    }
    return false;
  }
 private:
  bool started_=false;nlohmann::json phase_;std::string eta_status_;std::chrono::steady_clock::time_point last_{};
};
inline nlohmann::json describe_conversion_options(const ConversionOptions& options) {
  return {{"domain",{{"allow_nan",options.domain.allow_nan},{"one_hot_groups",options.domain.one_hot_groups}}},
    {"initial_states",options.initial_states},{"max_states",options.max_states},
    {"initial_nodes",options.initial_nodes},{"max_nodes",options.max_nodes},
    {"gpu_byte_budget",options.gpu_byte_budget},{"max_expansions",options.max_expansions},
    {"batch_size",options.batch_size},{"max_batch_size",options.max_batch_size},
    {"admission_threads",options.admission_threads},
    {"draft_threads",options.draft_threads},
    {"split_policy",options.split_policy},
    {"oldest_ready_jobs",options.oldest_ready_jobs},
    {"completed_cache_limit",options.completed_cache_limit?nlohmann::json(*options.completed_cache_limit):nlohmann::json(nullptr)},
    {"refinement_visit_budget",options.refinement_visit_budget?nlohmann::json(*options.refinement_visit_budget):nlohmann::json(nullptr)},
    {"cover_visit_budget",options.cover_visit_budget?nlohmann::json(*options.cover_visit_budget):nlohmann::json(nullptr)},
    {"checkpoint_path",options.checkpoint_path},{"resume_from",options.resume_from},
    {"checkpoint_interval_seconds",options.checkpoint_interval_seconds},
    {"checkpoint_on_completion",options.checkpoint_on_completion},
    {"checkpoint_host_byte_budget",options.checkpoint_host_byte_budget},
    {"proof_module_directory",options.proof_module_directory},{"proof_module_request",options.proof_module_request},
    {"completion_estimate_enabled",options.completion_estimate_enabled},
    {"completion_estimate_start_seconds",options.completion_estimate_start_seconds},
    {"completion_estimate",{{"paths",options.completion_estimate.paths},
      {"maximum_decisions",options.completion_estimate.maximum_decisions},
      {"decisions_per_chunk",options.completion_estimate.decisions_per_chunk},
      {"refinement_visit_budget",options.completion_estimate.refinement_visit_budget},
      {"seed",options.completion_estimate.seed},{"maximum_seconds",options.completion_estimate.maximum_seconds}}},
    {"runtime_residency",residency_name(options.residency)}};
}
} // namespace class_study
