#pragma once
#include "class_study_convert.hpp"
namespace class_study {
struct ResidentDataView;
struct ResidentEvaluationDataView;
ConvertedModel simplify_model(ConvertedModel input, std::uint32_t passes=4);
class ResidentEvaluation {
 public:
 explicit ResidentEvaluation(const nlohmann::json& dataset);
 // Copies active CUDA coordinates and labels once; padded rows are packed.
 explicit ResidentEvaluation(const ResidentDataView& dataset);
 // Named final holdout: no FIT/VALID bucket, candidate ranking is disabled.
 explicit ResidentEvaluation(const ResidentEvaluationDataView& dataset);
 ~ResidentEvaluation();
 nlohmann::json evaluate(const ConvertedModel&, const NativeOracle&);
 // Native public classes/probabilities only; no compiled-model comparison.
 nlohmann::json evaluate_native(const NativeOracle&);
 // CUDA lexicographic VALID errors/model bytes; incumbent wins an exact tie.
 bool prefer_last_native_candidate(std::uint64_t candidate_bytes,
                                   std::uint64_t incumbent_VALID_errors,
                                   std::uint64_t incumbent_bytes);
 private:
 struct Impl; std::unique_ptr<Impl> p_;
};
}