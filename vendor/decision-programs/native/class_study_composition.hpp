#pragma once
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>

namespace class_study {
struct CompositionOptions {
  // Explicit resident teacher-response buffer only; native model/probability
  // working allocations are separate opaque allocations, not bounded by this.
  std::uint64_t response_gpu_byte_budget = 8ULL << 30;
  // Zero preserves eager ownership of every distinct teacher. Positive bounds
  // live native teacher owners; immutable opaque models remain in host RAM.
  std::uint32_t teacher_owner_window = 0;
  // Enforced only for combined mode with a positive owner window. Covers our
  // retained teacher buffers, not decoded caller bundles/native host objects.
  std::uint64_t teacher_model_host_byte_budget = 8ULL << 30;
};
// Replays native-nonlinear-deployment-bundle-1 independently of all training and
// OOF owners. The caller reads/decodes CBOR once; this owner does no file I/O.
class ResidentComposition {
 public:
  ResidentComposition(const nlohmann::json& decoded_bundle,
                      const CompositionOptions& = {});
  ~ResidentComposition();
  ResidentComposition(ResidentComposition&&) noexcept;
  ResidentComposition& operator=(ResidentComposition&&) noexcept;
  ResidentComposition(const ResidentComposition&) = delete;
  ResidentComposition& operator=(const ResidentComposition&) = delete;
  // Input is contiguous CUDA0 FP32 [rows,features], rows must be nonzero.
  // Source models retain their native multi:softmax public predictions.
  // Combined mode uses their explicit native derived softprob views, copies
  // exact FP32 words teacher-major, then appends optional version-1
  // meta_input_mapping.raw_feature_indices in their declared original input
  // order and evaluates the native meta model. Missing/zero words are copied.
  // Borrowed result [rows,1] or margin [rows,K] expires on next predict/free.
  const float* predict(const float* raw_values, std::uint64_t rows,
                       bool margin = false);
  std::uint32_t features() const;
  std::uint32_t classes() const;
  // Zero for a selected single teacher; M*K+P for combined mode, with P
  // optional mapped raw inputs (zero preserves the legacy response shape).
  std::uint32_t response_features() const;
  nlohmann::json metadata() const;
  // Host schema/provenance/opaque-byte checks only, no model predictions.
  static nlohmann::json validate_bundle(const nlohmann::json&,
                                      const CompositionOptions& = {});
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
} // namespace class_study
