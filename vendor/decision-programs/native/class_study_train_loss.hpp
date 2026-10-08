#pragma once
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>

namespace class_study {
// Versioned group-mean CE/Dice objective. The native predictor remains binary
// multi:softmax; this training contract is not encoded by native model bytes.
enum class TrainingMarginMode { full_ensemble, incremental_native_proxy };
enum class TrainingCurvaturePolicy { uniform_v1, class_row_sum_v2 };
struct TrainingLossSpec {
  std::uint32_t version=1;
  double beta=0.5;
  double loss_scale=0; // explicit positive C, commonly the retained FIT row count
  bool require_all_positives=true;
  // Qualified against full native margins on the pinned backend; this is not
  // a universal bit-identity guarantee for other native binaries/configurations.
  TrainingMarginMode margin_mode=TrainingMarginMode::incremental_native_proxy;
  TrainingCurvaturePolicy curvature_policy=TrainingCurvaturePolicy::uniform_v1;
  // Qualification only: compare incremental vs full null-proxy margins after
  // every round on the same booster. Adds one full prediction and 8N scratch.
  bool verify_incremental_margins=false;
};
struct FitGroupSpec {
  unsigned long long full_negative=0, full_positive=0, outside_positive=0;
  // Validated and analytically cancelled; moments use normalized scale=1.
  double common_scale=1;
};
// Borrowed CUDA metadata for exactly the trainer's ordered FIT rows. The caller
// constructs counts exclusively from these FIT groups, including outside-ROI
// positives. This type cannot replace the trainer's trusted labels or margins.
struct FitGroupView {
  const std::uint32_t* group_ids=nullptr; // dense [0,groups), one per FIT row
  const FitGroupSpec* group_specs=nullptr;
  std::uint64_t rows=0;
  std::uint32_t groups=0;
  nlohmann::json fit_binding; // must equal the ResidentDataView binding
  std::shared_ptr<void> fit_owner; // exact source owner passed to the trainer
  std::shared_ptr<void> owner; // retains immutable IDs/counts for this call
};
} // namespace class_study
