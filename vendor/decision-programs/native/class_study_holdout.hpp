#pragma once
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>

namespace class_study {
struct NativeOracle;
// Confirmation-only CUDA input. No implicit conversion to FIT/VALID or TEST
// views exists. The owner retains the complete, once-staged HOLDOUT allocation.
struct ResidentHoldoutDataView {
  const float* values = nullptr;
  const std::uint32_t* labels = nullptr;
  std::uint64_t rows = 0, row_stride = 0;
  std::uint32_t features = 0, classes = 0;
  nlohmann::json binding;
  std::shared_ptr<void> owner;
};
// Pure descriptor validation: opens no files and initializes no CUDA state.
nlohmann::json validate_holdout_dataset(const nlohmann::json& descriptor);
// Reads the two pinned dense files once, stages all rows on CUDA0, and validates
// every label on CUDA. This input never becomes eligible for native training.
ResidentHoldoutDataView stage_holdout_dataset(const nlohmann::json& descriptor);
// One caller-frozen candidate versus one caller-frozen baseline on the exact
// same nonempty interval. Both oracles must return native float class IDs.
// Each borrowed prediction is consumed before the other callback is invoked;
// CUDA computes error counts and strict fewer-errors acceptance. Ties fail.
// No candidate collection/ranking API is provided. The caller owns allocation
// and nonreuse of confirmation intervals across gates and checkpoint recovery.
nlohmann::json evaluate_holdout_gate(const ResidentHoldoutDataView&,
                                    std::uint64_t offset, std::uint64_t count,
                                    const NativeOracle& candidate,
                                    const NativeOracle& baseline);
} // namespace class_study
