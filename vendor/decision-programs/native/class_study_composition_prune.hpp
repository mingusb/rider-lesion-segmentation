#pragma once
#include "class_study_train.hpp"
#include "class_conversion/class_io.hpp"
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace class_study::composition_prune {
using J=nlohmann::json;
struct Result {
  std::vector<TrainingResult> bases;
  TrainingResult meta;
  std::vector<std::uint32_t> retained_indices;
  J provenance;
};
inline void require(bool yes,const char* message){if(!yes)throw std::invalid_argument(message);}
inline bool same_words(const std::vector<float>& a,const std::vector<float>& b){
  if(a.size()!=b.size())return false;
  for(std::size_t i=0;i<a.size();++i)if(std::bit_cast<std::uint32_t>(a[i])!=std::bit_cast<std::uint32_t>(b[i]))return false;
  return true;
}
// Remove only complete teacher probability groups that no internal meta node
// reads. Retained teachers and their class channels keep their original order.
// This is an exact feature-index substitution, not fitting or approximation.
inline Result unused_bases(const std::vector<TrainingResult>& bases,const TrainingResult& meta,
                           std::span<const std::uint32_t> raw_indices = {}){
  using namespace class_conversion_native;
  require(!bases.empty()&&bases.size()<=std::numeric_limits<std::uint32_t>::max(),"composition pruning needs a bounded nonempty base bank");
  auto source=read_source_bytes(meta.model_json);require(source.identity==meta.model_sha256,"composition pruning meta SHA256 differs");
  auto classes=std::uint32_t(source.outputs);require(bases.size()<=std::size_t(INT32_MAX)/classes,"composition pruning probability width overflow");
  const auto probability_columns=std::uint32_t(bases.size()*classes);
  require(raw_indices.size()<=std::size_t(INT32_MAX)-probability_columns&&std::size_t(source.features)==probability_columns+raw_indices.size(),"composition pruning response feature shape differs");
  std::vector<bool> used(bases.size(),false);
  bool constant=true;
  for(auto feature:source.feature)if(feature>=0){constant=false;if(std::uint32_t(feature)<probability_columns)used[std::size_t(feature)/classes]=true;}
  // The maintained native model/converter require a nonempty feature space.
  // Constant and raw-only meta forests retain one genuine unchanged teacher
  // as the domain carrier. This is exact, but is not a minimal source claim.
  const bool carrier=std::none_of(used.begin(),used.end(),[](bool x){return x;});if(carrier)used[0]=true;
  Result result;result.meta=meta;std::vector<std::int32_t> mapping(source.features,-1);J original_pins=J::array(),retained_pins=J::array(),removed=J::array();
  std::int32_t raw_features=0;
  for(std::size_t i=0;i<bases.size();++i){auto base=read_source_bytes(bases[i].model_json);require(base.identity==bases[i].model_sha256&&base.outputs==source.outputs,"composition pruning base identity/class shape differs");
    if(!i)raw_features=base.features;require(base.features==raw_features,"composition pruning base raw feature shape differs");original_pins.push_back(base.identity);
    if(used[i]){auto next=std::uint32_t(result.bases.size());result.retained_indices.push_back(std::uint32_t(i));result.bases.push_back(bases[i]);retained_pins.push_back(base.identity);
      for(std::uint32_t c=0;c<classes;++c)mapping[i*classes+c]=std::int32_t(next*classes+c);
    }else removed.push_back(i);
  }
  mapped_meta_raw_predicates(source,probability_columns,raw_indices,std::uint32_t(raw_features));
  const auto new_probability_columns=std::uint32_t(result.bases.size()*classes);
  for(std::size_t i=0;i<raw_indices.size();++i)mapping[probability_columns+i]=std::int32_t(new_probability_columns+i);
  auto width=std::uint32_t(new_probability_columns+raw_indices.size());
  if(result.bases.size()!=bases.size()){
    auto document=dp_native_json::parse(meta.model_json);auto& learner=document.at("learner");learner.at("learner_model_param").at("num_feature")=std::to_string(width);
    for(auto& tree:learner.at("gradient_booster").at("model").at("trees")){
      tree.at("tree_param").at("num_feature")=std::to_string(width);
      const auto& left=tree.at("left_children");const auto& right=tree.at("right_children");auto& indices=tree.at("split_indices");
      for(std::size_t node=0;node<left.size();++node)if(left[node]!=-1||right[node]!=-1){auto old=indices[node].get<std::uint32_t>();require(old<mapping.size()&&mapping[old]>=0,"composition pruning split has no retained response");indices[node]=mapping[old];}
    }
    for(const char* key:{"feature_names","feature_types"})if(learner.contains(key)){
      auto& metadata=learner.at(key);require(metadata.is_array()&&(metadata.empty()||metadata.size()==mapping.size()),"composition pruning feature metadata extent differs");
      if(!metadata.empty()){J kept=J::array();for(std::size_t i=0;i<mapping.size();++i)if(mapping[i]>=0)kept.push_back(metadata[i]);metadata=std::move(kept);}
    }
    result.meta.model_json=dp_native_json::dump(document);result.meta.model_sha256=sha256(result.meta.model_json);
    auto transformed=read_source_bytes(result.meta.model_json);auto expected_features=source.feature;for(auto& f:expected_features)if(f>=0)f=mapping[f];
    require(transformed.features==std::int32_t(width)&&transformed.outputs==source.outputs&&transformed.objective==source.objective&&transformed.depth==source.depth&&transformed.version==source.version&&
            transformed.feature==expected_features&&transformed.left==source.left&&transformed.right==source.right&&transformed.roots==source.roots&&transformed.channels==source.channels&&transformed.missing_left==source.missing_left&&
            same_words(transformed.cut,source.cut)&&same_words(transformed.value,source.value)&&same_words(transformed.bias,source.bias),"composition pruning changed native tree semantics beyond feature remapping");
  }
  J feature_map=J::array();for(auto f:mapping)feature_map.push_back(f<0?J(nullptr):J(f));
  result.provenance={{"format","exact-composition-source-pruning-1"},{"applied",result.bases.size()!=bases.size()},{"original_base_count",bases.size()},{"retained_base_count",result.bases.size()},{"retained_base_indices",result.retained_indices},{"removed_base_indices",removed},{"original_base_sha256",original_pins},{"retained_base_sha256",retained_pins},{"original_meta_sha256",meta.model_sha256},{"pruned_meta_sha256",result.meta.model_sha256},{"original_response_features",source.features},{"response_features",width},{"old_to_new_response_feature",feature_map},{"constant_meta_keeps_one_base",constant},{"retained_base_bytes_unchanged",true},{"meta_tree_order_and_numeric_words_unchanged",true},{"FIT_performed",false},{"proof","Every internal meta predicate reads a retained response; only its feature index changes, with full native probability class groups preserved in original order"}};
  if(!raw_indices.empty()){
    result.provenance["meta_input_mapping"]={{"version",1},{"raw_feature_indices",std::vector<std::uint32_t>(raw_indices.begin(),raw_indices.end())}};
    result.provenance["original_probability_columns"]=probability_columns;
    result.provenance["probability_columns"]=new_probability_columns;
    result.provenance["raw_only_meta_keeps_one_base"]=carrier&&!constant;
    result.provenance["domain_carrier_base_count"]=carrier?1:0;
    result.provenance["minimal_source_representation_claim"]=false;
    result.provenance["proof"]="Internal meta probability predicates retain complete native class groups in order; raw predicates retain their declared original feature mapping; only meta feature indices change";
  }
  if(result.bases.size()!=bases.size()){result.meta.metrics={{"operation","exact_unused_composition_base_removal"},{"FIT_performed",false},{"features",width},{"classes",classes},{"model_sha256",result.meta.model_sha256},{"source_training_metrics",meta.metrics},{"provenance",result.provenance}};}
  return result;
}
} // namespace class_study::composition_prune
