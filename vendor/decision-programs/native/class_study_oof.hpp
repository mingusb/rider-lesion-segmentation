#pragma once
#include "class_study_data.hpp"
#include "class_study_train.hpp"
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace class_study {
// The supplied deployment model was trained on exactly full_fit_data_binding.
// Its probabilities are used only for VALID; new OOF units perform FIT-
// complement training with these same normalized hyperparameters. Imported
// columns retain completed matching complement-training evidence.
struct OofTeacher {
  nlohmann::json hyperparameters;
  TrainingResult full_fit_model;
  nlohmann::json full_fit_data_binding;
};
struct OofOptions {
  std::uint32_t folds = 5;
  std::uint64_t seed = 0;
  // Bounds explicit buffers plus known trainer value/label copies. Native
  // DMatrix/booster/probability allocations remain separately reported opaque.
  std::uint64_t gpu_byte_budget = 8ULL << 30;
  // Exact normalized configurations differing only in rounds share one maximum
  // complement FIT per fold; requested responses use native round slices.
  // Disabled preserves the original independent teacher-fold route.
  bool group_round_prefixes = false;
  // Opt-in response-only native [0,rounds) prediction from retained fold
  // parents. No sliced model buffer/hash is created for a response unit.
  // Requires grouping; disabled keeps legacy sliced responses/checkpoints.
  bool round_prefix_iteration_range = false;
};
struct OofCheckpoint {
  nlohmann::json state;
  // Opaque little-endian device payloads, copied only on checkpoint request.
  // No device pointer or native handle is serialized. Caller's coordinator
  // owns optional disk publication, at completed teacher-fold boundaries.
  std::vector<std::uint8_t> response_bytes, coverage_bytes;
  // Version-3/4 current-fold parent models. Aligned with state.round_prefix_parents;
  // copied only on checkpoint(), never on ordinary state()/metrics() calls.
  std::vector<std::vector<std::uint8_t>> round_prefix_parent_bytes;
};
struct OofImportReport {
  nlohmann::json statistics;
};
struct OofFailure : std::runtime_error {
  nlohmann::json partial_statistics;
  OofFailure(const std::string&, nlohmann::json);
};
class ResidentOof {
 public:
  ResidentOof(const nlohmann::json& trainer_plan, const ResidentDataView&,
              std::vector<OofTeacher>, const OofOptions& = {});
  ~ResidentOof();
  ResidentOof(ResidentOof&&) noexcept;
  ResidentOof& operator=(ResidentOof&&) noexcept;
  ResidentOof(const ResidentOof&) = delete;
  ResidentOof& operator=(const ResidentOof&) = delete;
  // Completes exactly one teacher-fold, or one teacher's VALID deployment
  // response. Returns false only if already complete. Native fits finish before
  // the boundary; interruption during a fit resumes from the prior checkpoint.
  bool advance();
  bool complete() const;
  // Small completed-boundary counters only. No layout/receipt/model-byte copies,
  // device transfers or filesystem operations. Grouped FITs and response units
  // are reported separately; imported evidence is not a new FIT call.
  nlohmann::json progress() const;
  nlohmann::json state() const;
  nlohmann::json metrics() const;
  OofCheckpoint checkpoint() const;
  // Only a fresh session can restore; identities, cursors, byte pins, exact-once
  // coverage and completed/uncomputed probability cells are all checked.
  void restore(const OofCheckpoint&);
  // One complete donor into a fresh target. Exact source/fold/library and
  // teacher declarations are checked; matched teacher columns keep target
  // order, while unmatched teachers still perform their own fold FIT calls.
  // Provenance names the caller-loaded committed checkpoint and generation.
  // No donor file is opened here. No-match or ambiguous declarations refuse.
  OofImportReport import_completed_columns(const OofCheckpoint&,
                                          const nlohmann::json& donor_provenance);
  // Exposes nothing partial. FIT rows contain OOF probabilities, VALID rows
  // contain full-FIT deployment probabilities; columns are teacher-major/K.
  // Labels and source row order are unchanged and kept alive by this owner.
  ResidentDataView completed_data() const;
  // Host metadata/opaque bytes only, suitable for capacity/refusal fixtures.
  static nlohmann::json validate_metadata(const nlohmann::json& trainer_plan,
      const ResidentDataView&, const std::vector<OofTeacher>&, const OofOptions&);
 private:
  struct Impl;
  std::shared_ptr<Impl> p_;
};
} // namespace class_study
