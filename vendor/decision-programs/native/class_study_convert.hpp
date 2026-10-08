#pragma once
#include "class_runtime.hpp"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

namespace native_softprob_gap { class RuntimeGate; }

namespace class_study {
struct ConversionSource {
  std::string_view model_json;
  std::uint32_t features=0, classes=0;
  // Optional precomputed identity. The exact supplied JSON is hashed once and
  // bound in the returned model header; an offered identity must match.
  std::string expected_source_sha256;
};
struct NativeOracle {
  // Resident source model must correspond to model_json for this call. The
  // callback synchronizes input consumers/producers before prediction and its
  // output producer afterward. Output is borrowed until the next callback,
  // training operation or destruction; conversion finishes consumers first.
  std::function<const float*(const float*,std::uint64_t,bool)> predict;
  std::uint32_t features=0, classes=0;
  std::string objective="multi:softmax", library_sha256, source_sha256;
  // Optional authority from the existing, same-process qualified native rule.
  // Absence means interval class pruning is unavailable, not approximate.
  std::shared_ptr<const native_softprob_gap::RuntimeGate> softprob_gap_gate;
  // Optional calibrated binary decision rule. When present, predict(...,false)
  // returns direct CUDA FP32 class IDs from the full-round native probability
  // transform: p(class 1) >= this exact FP32 threshold. predict(...,true) still
  // returns the source's unmodified native margins. Only two-class sources are
  // admitted. The converter binds this rule in checkpoints and uses native
  // source-constant terminal qualification; ordinary argmax/gap pruning is off.
  std::optional<float> positive_probability_threshold;
};
// A native probability stack: meta features are all base probability columns
// concatenated in the supplied order, optionally followed by declared raw
// feature columns. Class-only base substitution is not admitted. Every trained
// source remains a validated native model; internal raw guards are structural.
struct ComposedConversionSource {
  std::vector<ConversionSource> bases;
  ConversionSource meta;
  std::string expected_source_sha256;
  // Ordered, unique original feature indices appended after all probabilities.
  // Empty preserves the legacy probability-only input and descriptor identity.
  std::vector<std::uint32_t> meta_input_feature_indices;
};
struct ComposedNativeOracle {
  // Exact native BASE margins, CUDA FP32 [rows,sum(base.classes)], teacher-major.
  // This is deliberately separate from the final meta model's margin output.
  std::function<const float*(const float*,std::uint64_t)> base_margins;
  // Direct native final composition class IDs, CUDA FP32 [rows,1].
  // Both outputs are borrowed; the converter consumes each before another call.
  std::function<const float*(const float*,std::uint64_t)> predict_classes;
  std::string library_sha256, source_sha256;
};
struct ConversionDomain {
  bool allow_nan=true;
  // Each listed feature must be exactly 0 or 1; exactly one in each group is 1.
  // Groups are disjoint. Unlisted features cover all finite FP32 values.
  std::vector<std::vector<std::uint32_t>> one_hot_groups;
};
// Optional enclosing experiment transaction. Capture this immutable host state
// at the same idle boundary as the GPU snapshot. Storage preparation and final
// publication execute on the checkpoint writer thread, never the search thread.
struct CheckpointPublication {
  std::string snapshot_path;
  std::function<void()> prepare_storage, publish;
  // Called if preparing, writing, publishing, or starting the writer fails.
  // Enclosing coordinators use this to release their in-flight transaction.
  std::function<void(const std::string&)> abort;
};
struct WorkEstimateOptions {
  std::uint32_t paths=256, maximum_decisions=0, decisions_per_chunk=32;
  std::uint32_t refinement_visit_budget=0;
  std::uint64_t seed=1;
  double maximum_seconds=10;
};
struct ConversionOptions {
  ConversionDomain domain;
  std::uint64_t initial_states=1024, initial_nodes=1024;
  // Index representability ceilings, not allocations. Arenas grow on demand
  // subject to gpu_byte_budget, including transient replacement allocations.
  std::uint64_t max_states=0x3fffffffu, max_nodes=0x3fffffffu;
  std::uint64_t gpu_byte_budget=8ull*1024*1024*1024;
  std::uint64_t max_expansions=0; // zero means no diagnostic expansion limit
  // Zero selects throughput/memory-guided scheduling. A positive size fixes
  // the launch batch for reproducible comparisons; it never limits total work.
  std::uint32_t batch_size=0, max_batch_size=0;
  // Zero tunes a legal power-of-two copy width from model shape and throughput.
  std::uint32_t admission_threads=0;
  // Zero tunes preparation launch grouping without changing per-region math.
  std::uint32_t draft_threads=0;
  // Split order changes construction only; original source additions and the
  // exact class acceptance rule remain unchanged. Policies are compared before
  // changing the default on existing callers.
  std::string split_policy="source_order";
  // Bounded scheduling experiment: oldest ready jobs selected per batch;
  // remaining jobs use the established newest-first order. Zero preserves it.
  std::uint32_t oldest_ready_jobs=0;
  // Absent selects automatic policy; zero disables; positive fixes the limit.
  // Cache value is an idle retention target; within-batch/final counts can
  // exceed it. Eviction recycles slots without shrinking arena allocations.
  // Refinement is bounded per region and requires the existing native authority.
  // Cover effort separately budgets both proof-only predicate branches and
  // reuses the same private traversal scratch. It grants no extra authority.
  std::optional<std::uint32_t> completed_cache_limit, refinement_visit_budget, cover_visit_budget;
  // Long-running callers explicitly opt into persistence. Ordinary study trials
  // remain entirely in memory. Boundaries are safe points, never the cadence.
  std::string checkpoint_path, resume_from;
  double checkpoint_interval_seconds=0; // periodic writes disabled by default
  bool checkpoint_on_completion=true; // study coordinator disables per-trial saves
  std::uint64_t checkpoint_host_byte_budget=0; // zero uses available host RAM
  std::function<bool()> stop_requested;
  std::function<bool()> checkpoint_requested; // consumes an explicit request
  std::function<CheckpointPublication(std::uint64_t gpu_captured_bytes)> checkpoint_publication;
  std::string proof_module_directory, proof_module_request;
  // Reporting only. Lazily sample after construction has lasted this long;
  // short completed calls need no forecast. Sampling has a bounded elapsed
  // budget checked between CUDA chunks, not a hard kernel interruption limit.
  bool completion_estimate_enabled=true;
  double completion_estimate_start_seconds=1;
  WorkEstimateOptions completion_estimate{256,0,32,0,1,0.25};
  class_runtime::Residency residency=class_runtime::Residency::dual;
  std::function<void(const nlohmann::json&)> progress;
};
class ConversionFailure : public std::runtime_error {
 public:
  nlohmann::json partial_statistics;
  ConversionFailure(const std::string& message, nlohmann::json statistics)
      : std::runtime_error(message), partial_statistics(std::move(statistics)) {}
};
struct ConvertedModel {
  std::string canonical_bytes, compact_bytes;
  std::unique_ptr<class_runtime::Runtime> runtime;
  nlohmann::json metrics;
};
// Synchronous, one CUDA device/default stream, non-reentrant. Source parsing,
// immutable identity, shapes and limits are checked once. Filesystem access is
// opt-in for checkpoints and proof modules; default trials have none. No native
// model loading, per-trial receipts or end-trial file/hash rereads occur.
// Numerical construction/reference work is CUDA. The returned Runtime remains
// ready for resident evaluation through the existing shared Runtime API.
ConvertedModel convert_model(const ConversionSource&,const NativeOracle&,const ConversionOptions& = {});
// Uses the same adaptive constructor, state/cache machinery and Runtime export.
// Independent base score channels are retained; source-class pruning is disabled.
// Initial composed support is in memory: checkpoint and proof-module options
// are rejected until their composition authority/transport is implemented.
ConvertedModel convert_composed_model(const ComposedConversionSource&,
    const ComposedNativeOracle&,const ConversionOptions& = {});
// Reporting-only Monte Carlo sampling of the same constructor's source-order
// search tree. Does not construct, export, accept or modify a classifier.
// Numerical path sampling and aggregation run on CUDA; no live arena is read.
nlohmann::json estimate_model_work(const ConversionSource&,const NativeOracle&,
                                  const ConversionOptions&,const WorkEstimateOptions&);
} // namespace class_study
