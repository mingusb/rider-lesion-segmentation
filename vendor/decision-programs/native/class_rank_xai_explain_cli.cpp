#include "class_rank_xai_cuda.hpp"
#include "class_io.hpp"
#include <iostream>
int main(int argc,char**argv){namespace fs=std::filesystem;namespace codec=rank_regional_model_export;using J=dpnative::json;
 try{
  if(argc!=5)throw std::runtime_error("usage: xai_explain RUNTIME EXPECTED_SHA256 INPUT_JSON FRESH_OUTPUT_DIRECTORY");
  fs::path out=argv[4];if(fs::exists(out))throw std::runtime_error("xAI output must be fresh");
  auto runtime=codec::read_model(argv[1],argv[2]);auto model=rank_xai_export::lower(runtime);
  const auto input_bytes=dpnative::read_text(argv[3]);auto input=J::parse(input_bytes);
  if(!input.at("rows").is_array()||input.at("rows").empty()||input.at("rows").size()>256)throw std::runtime_error("xAI requires1..256 input rows");
  for(const auto&row:input.at("rows"))if(!row.is_array()||row.size()!=54)throw std::runtime_error("xAI each input row requires exactly54 features");
  if(!input.at("positive_feature_temperatures").is_array()||input.at("positive_feature_temperatures").size()!=10)throw std::runtime_error("xAI requires exactly10 positive feature temperatures");
  auto rows=input.at("rows").get<std::vector<std::array<float,54>>>();
  auto temperatures=input.at("positive_feature_temperatures").get<std::array<float,10>>();
  auto result=rank_xai_cuda::explain(model,rows,temperatures);
  J answers=J::array();for(std::size_t i=0;i<result.rows.size();++i){const auto&a=result.rows[i];J path=J::array();
   for(const auto&step:a.path)path.push_back({{"node",step.node},{"feature",step.feature},{"input_value",rows[i][step.feature]},{"raw_threshold_bits",step.threshold_bits},{"stored_rank_or_category_cut_bits",step.stored_cut_bits},{"gate_kind",static_cast<unsigned>(step.gate)},{"followed",step.left?"left":"right"}});
   answers.push_back({{"hard_class",a.hard_class},{"hard_decision_path",path},{"smooth_class",a.smooth_class},{"smooth_membership_scores",a.smooth_scores},{"smooth_continuous_derivatives_class_major",a.smooth_derivatives}});
  }
  if(dpnative::read_text(argv[3])!=input_bytes||dpnative::sha256(dpnative::read_text(argv[1]))!=model.runtime_sha256)throw std::runtime_error("xAI input changed during evaluation");
  J report={{"format","xai-GPU-explanation-1"},{"runtime_sha256",model.runtime_sha256},{"source_sha256",model.source_sha256},{"input_sha256",dpnative::sha256(input_bytes)},
   {"CUDA_executed",result.CUDA_executed},{"raw_and_original_rank_routes_equal",result.raw_and_rank_routes_equal},{"owned_device_peak_bytes",result.owned_device_peak_bytes},
   {"positive_feature_temperatures",temperatures},{"derivative_features","continuous input columns0..9; no derivative claimed for discrete one-hot categories"},
   {"smooth_scores_are_native_XGBoost_probabilities",false},{"smooth_accuracy_against_labels","not evaluated"},{"scores_use_FP64_arithmetic",true},{"answers",answers}};
  fs::create_directories(out);dpnative::atomic_text(out/"input.json",input_bytes);dpnative::atomic_json(out/"result.json",report);std::cout<<report.dump(2)<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
