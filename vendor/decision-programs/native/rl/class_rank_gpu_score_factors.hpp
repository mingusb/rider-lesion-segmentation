#pragma once
#include "class_rank_gpu_bounds.hpp"
#include <functional>
#include <string>

namespace rank_gpu_score_factors {
namespace b = rank_gpu_bounds;
using U = std::uint64_t;
using Stop = std::function<bool()>;

struct Source { b::Source value; std::string binding; };
struct Options {
    U maximum_combinations = 262144;
    U maximum_device_bytes = 128ull * 1024 * 1024;
    unsigned maximum_trees_per_class = 8;
};
struct Factor {
    b::Box box;
    std::uint32_t score_bits = 0, channel = 0;
    U combination = 0;
};
struct Result {
    bool complete = false, CUDA_executed = false;
    std::string source_binding, reason;
    std::array<U, 8> cartesian_offsets{}, factor_offsets{};
    // Peak of allocations actually reached, including incomplete/cancelled returns.
    U combinations = 0, compatible = 0, owned_device_peak_bytes = 0;
    std::vector<Factor> factors;
};

// Bounded construction of COMPLETE per-class score factors from trusted,
// same-process source leaf-box partitions. For every compatible tuple, the
// score starts at that class's exact bias and performs __fadd_rn in original
// source-tree order. Boxes retain all ten ranks and both category groups.
// The module validates layout/finite values, not partition coverage or source
// authentication. Binding is identity transport, not proof authentication.
// It performs no classification and grants no native softprob/class authority.
// Caps/cancellation return incomplete with no factor payload. All geometry,
// source selection and numerical work are CUDA; CPU is I/O/allocation only.
// A fixed-order cofactor compiler is a separate, currently unimplemented step.
Result prepare(Source, Options = {}, const Stop& = {}, int device = 0);
}
