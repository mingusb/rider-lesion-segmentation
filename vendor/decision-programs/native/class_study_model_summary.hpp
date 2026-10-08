#pragma once
#include "class_conversion/class_io.hpp"
#include <set>
#include <map>
#include <string_view>

namespace class_study::model_summary {
using J=nlohmann::json;
using U=std::uint32_t;
using L=std::uint64_t;
inline void require(bool yes,const char* why){if(!yes)throw std::invalid_argument(why);}
// Structural FIT statistics only. No data rows, predictions or learner calls.
inline J summarize(const std::string& bytes,const std::string& expected_sha){
  const auto source=class_conversion_native::read_source_bytes(bytes);
  require(source.identity==expected_sha,"model summary source SHA differs");
  const auto document=dp_native_json::parse(bytes);
  struct Feature {L count=0,left=0;double gain=0,cover=0;std::set<std::size_t> trees;std::map<std::pair<U,bool>,L> cuts;};
  std::map<U,Feature> features;L internal=0;std::size_t offset=0,tree_index=0;
  for(const auto& tree:document.at("learner").at("gradient_booster").at("model").at("trees")){
    const auto n=tree.at("left_children").size();const auto& gains=tree.at("loss_changes");const auto& covers=tree.at("sum_hessian");
    require(gains.is_array()&&covers.is_array()&&gains.size()==n&&covers.size()==n,"model summary statistic array extent differs");
    for(std::size_t i=0;i<n;++i){if(source.feature[offset+i]<0)continue;
      require(gains[i].is_number()&&covers[i].is_number(),"model summary statistics must be numeric");
      const float gain=gains[i].get<float>(),cover=covers[i].get<float>();
      require(std::isfinite(gain)&&std::isfinite(cover),"model summary statistics must be finite");
      auto& feature=features[U(source.feature[offset+i])];++feature.count;++internal;feature.trees.insert(tree_index);
      feature.gain+=double(gain);feature.cover+=double(cover);const bool left=source.missing_left[offset+i];feature.left+=left;
      ++feature.cuts[{std::bit_cast<U>(source.cut[offset+i]),left}];
    }
    offset+=n;++tree_index;
  }
  J rows=J::array();for(const auto& [index,f]:features){J cuts=J::array();std::set<U> finite;L positive=0,negative=0;
    for(const auto& [key,count]:f.cuts){const auto [bits,left]=key;cuts.push_back(J::array({bits,left,count}));const float value=std::bit_cast<float>(bits);if(std::isfinite(value))finite.insert(bits);else if(std::signbit(value))negative+=count;else positive+=count;}
    rows.push_back({{"feature_index",index},{"split_count",f.count},{"trees_using_feature",f.trees.size()},{"stored_loss_change_sum",f.gain},{"stored_loss_change_mean",f.gain/double(f.count)},
      {"stored_split_hessian_sum",f.cover},{"stored_split_hessian_mean",f.cover/double(f.count)},{"missing_left_count",f.left},{"missing_right_count",f.count-f.left},
      {"finite_distinct_threshold_words",finite.size()},{"positive_infinity_split_count",positive},{"negative_infinity_split_count",negative},{"threshold_counts",cuts}});
  }
  return {{"model_sha256",expected_sha},{"model_bytes",bytes.size()},{"feature_count",source.features},{"classes",source.outputs},{"trees",source.roots.size()},{"internal_nodes",internal},
    {"unlisted_features_have_zero_splits",true},{"aggregation","stored native FP32 statistics, deterministic tree/node-order FP64 sums"},{"threshold_count_columns",{"threshold_FP32_bits","missing_left","split_count"}},{"features",rows}};
}
inline J raw_mapping(const J& descriptor,U columns){
  J mapping=J::array();for(U f=0;f<columns;++f){J entry={{"kind","raw_feature"},{"feature_index",f}};if(descriptor.contains("feature_names")&&descriptor.at("feature_names").size()==columns)entry["feature_name"]=descriptor.at("feature_names")[f];mapping.push_back(entry);}return mapping;
}
inline J meta_mapping(const J& raw,const std::vector<std::string>& teacher_pins,U classes,const std::vector<U>& suffix){
  J mapping=J::array();for(U b=0;b<teacher_pins.size();++b)for(U c=0;c<classes;++c)mapping.push_back({{"kind","base_probability"},{"base_hyperparameter_index",b},{"base_model_sha256",teacher_pins[b]},{"class_index",c}});
  for(U f:suffix){require(f<raw.size(),"model summary raw mapping index out of range");mapping.push_back(raw.at(f));}return mapping;
}
// Same bytes can have different input meanings. Deduplicate only intrinsic
// numeric-index statistics; every use retains its own role/mapping reference.
struct Collector {
  J models=J::object(),uses=J::array();std::map<std::string,std::string_view> originals;
  void add(const std::string& bytes,const std::string& sha,J use){
    auto prior=originals.find(sha);if(prior==originals.end()){models[sha]=summarize(bytes,sha);originals.emplace(sha,bytes);}else require(prior->second==bytes,"model summary duplicate SHA has different bytes");
    use["model_sha256"]=sha;uses.push_back(std::move(use));
  }
  J finish(J mappings)const{return {{"version",1},{"scope","outer-refit winners and their structurally used teacher dependencies"},{"statistics_kind","training split statistics; not heldout contribution, SHAP or causal importance"},{"models",models},{"uses",uses},{"input_mappings",std::move(mappings)},{"mapping_scope","outer-refit models used for outer scoring; meta was trained on FIT-only OOF teacher responses, not these refit predictions"}};}
};
} // namespace class_study::model_summary
