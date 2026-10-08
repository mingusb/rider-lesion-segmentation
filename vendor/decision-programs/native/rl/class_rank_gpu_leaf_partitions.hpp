#pragma once
#include "class_rank_gpu_score_factors.hpp"
#include <memory>

namespace rank_gpu_leaf_partitions {
using U = std::uint64_t;
using Stop = rank_gpu_score_factors::Stop;
namespace b = rank_gpu_bounds;
struct Options {
    U maximum_source_nodes = 4096, maximum_source_trees = 64;
    U maximum_source_depth = 128, maximum_native_rows = 262144;
    U native_batch_rows = 1024, maximum_device_bytes = 128ull*1024*1024;
};
struct Audit {
    bool complete = false, CUDA_executed = false;
    U rows = 0, native_margin_words = 0, factor_margin_words = 0;
    U owned_device_peak_bytes = 0;
    std::string reason;
};
struct Result {
    bool complete = false, CUDA_executed = false;
    rank_gpu_score_factors::Source source;
    std::array<std::vector<std::uint32_t>,10> rank_cut_bits;
    b::Box domain;
    U original_leaves = 0, feasible_leaves = 0, empty_leaves = 0;
    U disjoint_pairs_checked = 0, complete_tree_volumes_checked = 0;
    U native_witness_rows = 0, native_margin_words = 0;
    U owned_device_peak_bytes = 0, owned_device_resident_bytes = 0;
    std::string source_sha256, library_sha256, rank_sha256, reason;
};
// Immutable serialized source + pinned native library, finite-FP32 valid Forest
// domain only. GPU builds rank/category leaf boxes in original tree/leaf order.
// Direct ancestor-rank intersections are independently checked against the raw
// fmax/nextafter geometry of native/class_cuda.cu::initialize_source. GPU checks
// pairwise disjointness and exact 256-bit cardinality coverage of each tree.
// GPU witnesses compare source traversal, unique partition selection, ordered
// __fadd_rn words and pinned native margins; they do not assert class authority.
// Rank representatives cover the numeric source-predicate quotient, with -0/+0
// comparison-equivalent only. Leaf/bias score words preserve their exact bits.
// Host: parsing/topology, allocation, hashing and I/O only. No CPU geometry or
// inference. Internal CUDA/native storage is owned until this object is freed;
// returned leaf/factor data are owned host transport. Private native workspace
// and implicit thread stacks are outside the explicit device-byte accounting.
// Caps/cancellation publish no source payload. Numerical qualification is a
// separate root-run action; construction here is not a completed class compiler.
class Bridge {
    struct Impl; std::unique_ptr<Impl> p_;
public:
    Bridge(const std::string& library, const std::string& model,
           const std::string& expected_source_sha256, Options = {}, int device = 0);
    ~Bridge();
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;
    Result extract(const Stop& = {});
    // Witness audit of trusted same-process qualified prepare output; this is
    // not exhaustive factor authentication or authority for imported payloads.
    Audit audit_factors(const rank_gpu_score_factors::Result&, const Stop& = {});
};
}
