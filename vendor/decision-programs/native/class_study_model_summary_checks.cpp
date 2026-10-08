#include "class_study_model_summary.hpp"
#include <iostream>
using J=nlohmann::json;
using namespace class_study::model_summary;
J tree(unsigned id,int feature,float cut,bool left,float gain,float cover){
  bool split=feature>=0;unsigned n=split?3:1;
  return {{"id",id},{"tree_param",{{"num_feature","3"},{"num_nodes",std::to_string(n)},{"size_leaf_vector","1"}}},{"categories",J::array()},
    {"left_children",split?J::array({1,-1,-1}):J::array({-1})},{"right_children",split?J::array({2,-1,-1}):J::array({-1})},
    {"split_indices",split?J::array({feature,0,0}):J::array({0})},{"split_conditions",split?J::array({cut,0.f,1.f}):J::array({0.f})},
    {"default_left",split?J::array({left,0,0}):J::array({0})},{"split_type",std::vector<int>(n,0)},
    {"loss_changes",split?J::array({gain,0.f,0.f}):J::array({0.f})},{"sum_hessian",split?J::array({cover,0.f,0.f}):J::array({0.f})}};
}
int main(){try{
  const float inf=std::numeric_limits<float>::infinity();
  J model;model["version"]={3,4,1};auto& learner=model["learner"];
  learner["learner_model_param"]={{"num_feature","3"},{"num_class","2"},{"num_target","1"},{"base_score","[0,0]"}};learner["objective"]={{"name","multi:softmax"}};
  learner["gradient_booster"]={{"name","gbtree"}};learner["gradient_booster"]["model"]={{"gbtree_model_param",{{"num_trees","4"}}},{"tree_info",{0,1,0,1}},{"iteration_indptr",{0,2,4}},
    {"trees",J::array({tree(0,0,-inf,true,2.5f,8),tree(1,0,inf,false,-.5f,6),tree(2,1,-0.f,false,3,4),tree(3,-1,0,false,0,0)})}};

  auto run=[](const J& value){auto bytes=dp_native_json::dump(value);return summarize(bytes,class_conversion_native::sha256(bytes));};
  auto summary=run(model);require(summary.at("internal_nodes")==3&&summary.at("trees")==4&&summary.at("features").size()==2,"summary split totals differ");
  const auto& a=summary.at("features")[0];require(a.at("feature_index")==0&&a.at("split_count")==2&&a.at("trees_using_feature")==2&&a.at("stored_loss_change_sum")==2.0&&a.at("stored_loss_change_mean")==1.0&&a.at("stored_split_hessian_sum")==14.0&&a.at("stored_split_hessian_mean")==7.0&&a.at("missing_left_count")==1&&a.at("missing_right_count")==1&&a.at("positive_infinity_split_count")==1&&a.at("negative_infinity_split_count")==1,"hand-counted feature statistics differ");
  require(summary.at("features")[1].at("threshold_counts")[0][0]==0x80000000u,"threshold signed-zero word changed");
  require(J::parse(summary.dump())==summary,"summary JSON contains nonfinite values");
  auto raw=raw_mapping(J{{"feature_names",{"a","b","c"}}},3);auto meta=meta_mapping(raw,{"teacher0","teacher1"},2,{2,0});
  require(meta.size()==6&&meta[0].at("class_index")==0&&meta[1].at("class_index")==1&&meta[2].at("base_hyperparameter_index")==1&&meta[2].at("base_model_sha256")=="teacher1"&&meta[4].at("feature_index")==2&&meta[5].at("feature_index")==0,"teacher/class/reordered raw mapping differs");
  auto bytes=dp_native_json::dump(model);auto pin=class_conversion_native::sha256(bytes);Collector c;c.add(bytes,pin,{{"role","baseline_winner"},{"input_mapping","raw"}});c.add(bytes,pin,{{"role","combiner_winner"},{"input_mapping","meta"}});auto all=c.finish({{"raw",raw},{"meta",meta_mapping(raw,{"teacher0"},2,{2})}});
  require(all.at("models").size()==1&&all.at("uses").size()==2&&all.at("uses")[0].at("input_mapping")!=all.at("uses")[1].at("input_mapping"),"SHA dedup lost per-use interpretation");
  auto constant=model;for(unsigned i=0;i<4;++i)constant["learner"]["gradient_booster"]["model"]["trees"][i]=tree(i,-1,0,false,0,0);auto zero=run(constant);require(zero.at("internal_nodes")==0&&zero.at("features").empty(),"constant source reports feature use");
  unsigned refused=0;auto deny=[&](auto operation){bool failed=false;try{operation();}catch(const std::exception&){failed=true;}require(failed,"malformed model statistics accepted");++refused;};
  for(const char* field:{"loss_changes","sum_hessian"}){auto invalid=model;invalid["learner"]["gradient_booster"]["model"]["trees"][0][field]=J::array();deny([&]{run(invalid);});invalid=model;invalid["learner"]["gradient_booster"]["model"]["trees"][0][field][0]=inf;deny([&]{run(invalid);});}
  deny([&]{summarize(bytes,std::string(64,'0'));});deny([&]{meta_mapping(raw,{"t"},2,{3});});deny([&]{c.add(bytes+" ",pin,J::object());});
  std::cout<<J{{"complete",true},{"host_only",true},{"negative_gain_preserved",true},{"threshold_words_and_missing_directions",true},{"constant_zero_usage",true},{"teacher_class_and_raw_order",true},{"SHA_dedup_separate_use_mappings",true},{"refusals",refused}}.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
