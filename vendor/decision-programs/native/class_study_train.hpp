#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "class_study_train_loss.hpp"

namespace class_study {
struct ResidentDataView;
struct TrainingResult {
  std::string model_json;
  std::string model_sha256;
  nlohmann::json metrics;
};
// One FIT-only dataset and native CUDA DMatrix per owner. No trial file I/O.
class ResidentTrainer {
 public:
  explicit ResidentTrainer(const nlohmann::json& plan);
  // FIT-only borrowed CUDA data from the shared source factory. Source owner
  // remains retained; no dataset files are opened by this overload.
  ResidentTrainer(const nlohmann::json& plan,const ResidentDataView&);
  // A live verified owner authorizes the exact same already-loaded library.
  // Owns a separate dlopen reference and FIT DMatrix; no library file reread.
  ResidentTrainer(const nlohmann::json& plan,const ResidentDataView&,
                  const ResidentTrainer& verified_session);
  ~ResidentTrainer();
  ResidentTrainer(ResidentTrainer&&) noexcept;
  ResidentTrainer& operator=(ResidentTrainer&&) noexcept;
  ResidentTrainer(const ResidentTrainer&)=delete;
  ResidentTrainer& operator=(const ResidentTrainer&)=delete;
  TrainingResult train(const nlohmann::json& hyperparameters);
  // One completed FIT using externally computed group-mean CE/Dice gradients.
  // Requires binary classes, explicit positive max_delta_step and subsample=1.
  // Existing fixed DMatrix weights affect cuts, never reweight supplied g/h.
  // No public predictions are available until every round and export succeed.
  TrainingResult train(const nlohmann::json& hyperparameters,
                       const TrainingLossSpec&,const FitGroupView&);
  // Host-only contract validation; no CUDA access or native library load.
  static nlohmann::json validate_loss(const nlohmann::json& hyperparameters,
      const TrainingLossSpec&,std::uint64_t rows,std::uint32_t groups,
      std::uint32_t classes);
  // Take [0,rounds) from the current completed gbtree booster. Installs the
  // validated sliced handle and exports RAM bytes without a FIT update/file.
  // The completed FIT-call counter is unchanged; prediction buffers expire.
  TrainingResult slice_prefix(std::uint32_t rounds);
  // Explicitly bind one immutable parent. Supplied bytes are verified on every
  // bind call; same verified identity avoids another native load. The getter
  // lets a caller holding immutable cache bytes bind once, then pass only rounds.
  bool retain_prefix_source(const std::string& model_json,const std::string& expected_sha);
  const std::string& retained_prefix_source_sha256() const;
  // Select only an already validated immutable parent owned by this session.
  // A miss returns false without changing the selected parent. No buffer hash
  // or native load is performed. Cached handles are ephemeral across resume.
  bool select_retained_prefix_source(const std::string& verified_sha);
  // Default 1GiB of serialized parent extents, an explicitly reported proxy
  // rather than a bound on opaque native allocations. Zero restores one-parent
  // behavior. An oversized required parent remains usable as uncached active.
  void set_retained_prefix_cache_byte_budget(std::uint64_t bytes);
  void clear_retained_prefix_sources();
  TrainingResult slice_retained_prefix(std::uint32_t rounds);
  // Opt-in response-only prefix: use [0,rounds) of an authenticated immutable
  // parent through the native CUDA iteration-range API. A separately owned
  // full-parent softprob view is cached with that parent. No prefix model bytes
  // or hash are invented, exported, or installed as the current model. Borrowed
  // [rows,K] output expires before another native call/source change.
  const float* predict_retained_prefix_probabilities(const float* values,
      std::uint64_t rows,std::uint32_t rounds);
  // Restore a retained model buffer for deferred comparison; FIT stays resident.
  // SHA validation is in RAM. No dataset, library or model file is reread.
  void restore_model(const std::string& model_json,const std::string& expected_sha);
  // Host bookkeeping for an enclosing study checkpoint. This does not claim
  // to restore an in-flight XGBoost update or a native random-number generator.
  void restore_completed_trial_count(std::uint64_t completed);
  // Borrowed native CUDA output: [rows,K] margins or [rows,1] public IDs.
  // Contiguous input has F columns. Output expires at next predict/train/free.
  const float* predict(const float* values,std::uint64_t rows,bool margin);
  // Explicit derived response view: full-round native clone of current gbtree,
  // with only its objective changed to multi:softprob. Original hard-class
  // objective, model identity and FIT counter remain unchanged. Borrowed CUDA
  // [rows,K] output must be copied/consumed before another native API call or
  // source change. No universal softprob-argmax/hard-class equivalence claim.
  const float* predict_probabilities(const float* values,std::uint64_t rows);
  std::uint32_t features() const;
  std::uint32_t classes() const;
  const std::string& current_model_sha256() const;
  const nlohmann::json& metadata() const;
  static nlohmann::json validate_metadata(const nlohmann::json& plan);
  static nlohmann::json validate_metadata(const nlohmann::json& plan,const ResidentDataView&);
  static nlohmann::json validate_hyperparameters(const nlohmann::json&,std::uint32_t classes);
 private:
  friend class ResidentPredictor;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// Inference-only native model owner. No training rows, labels, CUDA DMatrix or
// completed FIT-call counter. The plan needs only library path/pinned identity.
class ResidentPredictor {
 public:
  ResidentPredictor(const nlohmann::json& plan,std::uint32_t features,
                    std::uint32_t classes);
  ResidentPredictor(const nlohmann::json& plan,std::uint32_t features,
                    std::uint32_t classes,const ResidentTrainer& verified_session);
  ResidentPredictor(const nlohmann::json& plan,std::uint32_t features,
                    std::uint32_t classes,const ResidentPredictor& verified_session);
  ~ResidentPredictor();
  ResidentPredictor(ResidentPredictor&&) noexcept;
  ResidentPredictor& operator=(ResidentPredictor&&) noexcept;
  ResidentPredictor(const ResidentPredictor&)=delete;
  ResidentPredictor& operator=(const ResidentPredictor&)=delete;
  // Byte identity, native gbtree/softmax shape and rounds are checked before
  // installation. Invalid input preserves the existing model/probability view.
  void restore_model(const std::string& model_json,const std::string& expected_sha);
  // Both methods reject before a model is loaded. Contiguous CUDA input is
  // [rows,features]. Borrowed output expires before another native API call,
  // source change or destruction; copy/consume it first.
  const float* predict(const float*,std::uint64_t rows,bool margin=false);
  // Explicit private multi:softprob view; original source semantics unchanged.
  const float* predict_probabilities(const float*,std::uint64_t rows);
  std::uint32_t features() const;
  std::uint32_t classes() const;
  const std::string& current_model_sha256() const;
  const nlohmann::json& metadata() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace class_study
