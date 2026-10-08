#pragma once
#include "rl_category_policy.hpp"
#include "class_io.hpp"
#include <bit>
namespace session_policy_state {
using J=dpnative::json;using U=rl_category_policy::U;
struct Imported {U version=0;std::array<U,44>words{};};
inline Imported parse(const J&state){
  auto need=[](bool v){if(!v)throw std::invalid_argument("policy state schema/sampler/version/44 unsigned word metadata differs");};
  need(state.is_object()&&state.contains("schema")&&state.at("schema").is_number_unsigned()&&state.at("schema").get<U>()==1&&
       state.contains("sampler_schema_version")&&state.at("sampler_schema_version").is_number_unsigned()&&state.at("sampler_schema_version").get<U>()==rl_category_policy::sampler_schema_version&&
       state.contains("version")&&state.at("version").is_number_unsigned()&&state.at("version").get<U>()>0&&
       state.contains("logit_words")&&state.at("logit_words").is_array()&&state.at("logit_words").size()==44);
  Imported out;out.version=state.at("version").get<U>();
  for(unsigned k=0;k<44;++k){need(state.at("logit_words")[k].is_number_unsigned());out.words[k]=state.at("logit_words")[k].get<U>();}
  // Word finiteness is deliberately checked by Policy::upload on CUDA.
  return out;
}
inline J trajectory(const std::array<rl_category_policy::Trace,2>&traces){J out=J::array();
  for(const auto&t:traces){if(t.steps>40)throw std::runtime_error("trace metadata extent");
    J actions=J::array(),remaining=J::array(),probabilities=J::array(),gradients=J::array();
    for(unsigned k=0;k<t.steps;++k){actions.push_back(t.actions[k]);remaining.push_back(t.remaining[k]);probabilities.push_back(std::bit_cast<U>(t.log_probability[k]));}
    for(auto g:t.gradient)gradients.push_back(std::bit_cast<U>(g));
    out.push_back(J{{"version",t.version},{"seed",t.seed},{"episode",t.episode},{"allowed_mask",t.allowed_mask},{"fallback_mask",t.fallback_mask},{"group",t.group},{"steps",t.steps},{"actions",actions},{"remaining",remaining},{"log_probability_words",probabilities},{"gradient_words",gradients},{"total_log_probability_word",std::bit_cast<U>(t.total_log_probability)}});
  }return out;
}
inline J words(U version,const std::array<U,44>&value){return J{{"schema",U(1)},{"sampler_schema_version",U(rl_category_policy::sampler_schema_version)},{"version",version},{"logit_words",value}};}
}
