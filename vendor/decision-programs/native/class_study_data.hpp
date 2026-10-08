#pragma once
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>

namespace class_study {
// Typed borrowed CUDA input. The owner keeps source allocations alive; consumers
// validate their role/extent and never serialize device addresses in checkpoints.
struct ResidentDataView {
  const float* values = nullptr;
  const std::uint32_t* labels = nullptr;
  std::uint64_t rows = 0, row_stride = 0, fit_rows = 0, valid_rows = 0;
  std::uint32_t features = 0, classes = 0;
  nlohmann::json binding;
  std::shared_ptr<void> owner;
  // Optional nonnegative finite CUDA weights for FIT rows; owner retains them.
  // Appended to preserve existing unweighted aggregate initializers.
  const float* sample_weights = nullptr;
};
// A final held-out input type, intentionally not accepted by Trainer/fit_prefix.
// It exposes no FIT/VALID role and no conversion to ResidentDataView.
struct ResidentEvaluationDataView {
  const float* values = nullptr;
  const std::uint32_t* labels = nullptr;
  std::uint64_t rows = 0, row_stride = 0;
  std::uint32_t features = 0, classes = 0;
  nlohmann::json binding;
  std::shared_ptr<void> owner;
};
// Metadata preflight only; never opens inputs or initializes CUDA.
nlohmann::json validate_evaluation_dataset(const nlohmann::json& descriptor);
// Explicit TEST-only transport. Caller must bind a frozen model before calling.
ResidentEvaluationDataView stage_evaluation_dataset(const nlohmann::json& descriptor);
// One source read and CUDA staging, including the declared FIT/VALID split.
// Existing dense descriptors and the existing raw-pixel IDX adapter are supported.
ResidentDataView stage_dataset(const nlohmann::json& descriptor);
// Final refit staging: every declared row is FIT and VALID_rows is zero.
// Supports generic dense FIT-only data and all 60,000 official MNIST training
// IDX rows. TEST remains excluded. Existing study split staging is unchanged.
ResidentDataView stage_fit_dataset(const nlohmann::json& descriptor);
ResidentDataView fit_prefix(const ResidentDataView&);
} // namespace class_study
