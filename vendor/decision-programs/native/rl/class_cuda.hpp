#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

namespace native_cuda {

// All arrays below are host storage. The constructor copies them to CUDA.
// Node IDs are global; leaf_nodes is depth-first within each source tree.
// The ancestor CSR rows contain internal root-to-leaf nodes and branch bits.
struct SourceHost {
  std::int32_t features = 0;
  std::int32_t outputs = 0;
  std::vector<std::int32_t> feature, left, right, roots, channels;
  std::vector<float> cut, value, bias;
  std::vector<std::uint8_t> missing_left;
  std::vector<std::int32_t> leaf_nodes, leaf_tree;
  std::vector<std::int32_t> ancestor_offsets, ancestor_nodes;
  std::vector<std::uint8_t> ancestor_right;
};

struct Predicate {
  std::int32_t feature = -1;
  float cut = 0;
  std::uint8_t missing_left = 0;
};

struct Region {
  std::vector<float> lower, upper;
  std::vector<std::uint8_t> missing, mask;
  bool nonempty = true;  // CUDA geometry result; CPU treats this as metadata.
};

enum class Protocol : std::int32_t { raw_argmax = 0, cuda_softprob_argmax = 1 };
enum class Status : std::int32_t { split = 0, dominant = 1, constant = 2 };
enum class SplitPolicy : std::int32_t {
  widest_tree = 0, competitive_channels = 1, aggregate_predicates = 2
};
enum class Domain : std::int32_t { dense_fp32_nan = 0, forest_valid = 1 };

struct Evaluation {
  Status status = Status::split;
  std::int32_t label = -1;
  bool paired_certified = false;
  Predicate predicate;
  std::array<float, 7> lower{}, upper{};
  std::vector<float> witness;
};

// Upper 32 bits are a generation; stale handles stay invalid after slot reuse.
using RegionSlot = std::uint64_t;
struct SlotChildren {
  RegionSlot left = 0, right = 0;
  bool left_nonempty = true, right_nonempty = true;
};

// This is a CUDA implementation with host transport/structural control only.
// All public calls complete their transfers before returning. No Torch/Python.
class Engine {
 public:
  explicit Engine(const SourceHost&, std::size_t batch_capacity = 128,
                  int device = 0, Domain domain = Domain::dense_fp32_nan);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) noexcept;
  Engine& operator=(Engine&&) noexcept;

  // Split priority only: bounds, class certificates and native-oracle fallback
  // retain the same full-source semantics. The default is widest_tree.
  // Aggregate priority requires extra shared memory and rejects unsupported
  // source sizes before changing the active policy.
  void set_split_policy(SplitPolicy);

  // Optional aggregate-only cooperation for independent source LCAs and scratch
  // initialization. FP64 score additions and tie scans retain source order.
  // Unsupported shared-memory requirements reject before changing the flag.
  void enable_cooperative_aggregate(bool enabled = true);

  // Optional correlation certificate for forest_valid only. The original
  // interval certificate and source-order FP32 bounds remain unchanged.
  // Unsupported/dense regions and inconclusive paired bounds fall back.
  void enable_paired_certificate(bool enabled = true);

  // Optional evaluator with one ordered FP32 accumulator per output channel.
  // Source ordering and all decision/tie rules are identical to the default.
  void enable_parallel_evaluation(bool enabled = true);

  // Optional arena-only path: kernels read/write their reserved slots directly
  // instead of gathering/scattering whole regions. Other experimental evaluator
  // modes currently retain the staging path. Host-region APIs are unchanged.
  void enable_direct_arena(bool enabled = true);

  Region initial_region();
  std::vector<Evaluation> evaluate(
      const std::vector<Region>&,
      Protocol protocol = Protocol::cuda_softprob_argmax);
  std::vector<std::pair<Region, Region>> split(
      const std::vector<Region>&, const std::vector<Predicate>&);

  // Exact source-order FP32 additions. Returned raw rows are host transport.
  std::vector<float> source_raw(const std::vector<float>& input);
  // One CUDA-constructed witness per source leaf; invalid leaf boxes produce
  // a zero probe row. This is independent of the region batch capacity.
  std::vector<float> leaf_witnesses();

  // Optional persistent CUDA frontier. Slot IDs and free-list bookkeeping are
  // host metadata; boxes and masks remain on the device between calls.
  // Configuration requires an empty arena. Exhaustion throws before a split.
  void configure_region_arena(std::size_t capacity);
  std::size_t region_arena_capacity() const noexcept;
  std::size_t region_arena_available() const noexcept;
  std::vector<RegionSlot> put_regions(const std::vector<Region>&);
  std::vector<Region> get_regions(const std::vector<RegionSlot>&);
  void release_regions(const std::vector<RegionSlot>&);
  std::vector<Evaluation> evaluate_slots(
      const std::vector<RegionSlot>&,
      Protocol protocol = Protocol::cuda_softprob_argmax);
  // Parents remain live until the caller commits and releases them. Reserve
  // two spare slots per split so failed construction preserves its frontier.
  std::vector<SlotChildren> split_slots(
      const std::vector<RegionSlot>&, const std::vector<Predicate>&);
  std::int32_t features() const noexcept;
  std::int32_t outputs() const noexcept;
  std::size_t leaves() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Strict > updates preserve the first class on a tie. Invalid/nonfinite input
// produces label -1; the synchronous host helper rejects any such output.
void first_argmax(const float* device_values, int rows, int classes,
                  std::int32_t* device_labels, cudaStream_t stream = nullptr);
std::vector<std::int32_t> first_argmax_host(const float* device_values,
                                           int rows, int classes);

struct TreeHost {
  std::int32_t features = 0, outputs = 0, max_depth = 0;
  std::vector<std::int32_t> feature, left, right, label;
  std::vector<float> cut;
  std::vector<std::uint8_t> missing_left;
};
std::vector<std::int32_t> predict_tree(const TreeHost&,
                                      const std::vector<float>& input,
                                      int device = 0);

}  // namespace native_cuda
