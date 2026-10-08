#pragma once
#include "class_rank_gpu_score_factors.hpp"
#include <limits>

namespace rank_gpu_score_diagram {
using U = std::uint64_t;
using Stop = rank_gpu_score_factors::Stop;
namespace b = rank_gpu_bounds;
inline constexpr U none = std::numeric_limits<U>::max();
struct Input {
    rank_gpu_score_factors::Result factors;
    b::Box domain;
    std::string domain_binding;
};
struct Options {
    U maximum_factors = 16384, maximum_factors_per_class = 4096;
    U maximum_states = 32768, maximum_rows = 262144;
    U maximum_edges = 131072, maximum_nodes = 32768, maximum_arcs = 131072;
    U state_buckets = 65537, node_buckets = 65537;
    U maximum_device_bytes = 128ull*1024*1024, states_per_launch = 16;
};
struct Row { b::Box box; std::uint32_t score_bits=0, reserved=0; };
struct Arc { std::int32_t lo=0, hi=0; U allowed=0, child=none; };
struct Node {
    U first_arc=0, arc_count=0, hash_next=none;
    std::uint32_t kind=0, dimension=12, score_bits=0, reserved=0;
};
struct State {
    U first_row=0, row_count=0, first_edge=0, edge_count=0;
    U result=none, hash_next=none;
    std::uint32_t level=0, status=0; // 0 pending, 1 expanded, 2 terminal, 3 reduced
};
struct Counters {
    U states=0, rows=0, edges=0, nodes=0, arcs=0, expanded=0, reduced=0;
    U state_attempts=0, state_hits=0, state_collisions=0, row_comparisons=0;
    U node_attempts=0, node_hits=0, node_collisions=0;
    U terminal_states=0, equal_child_reductions=0, adjacent_arc_merges=0;
    U partition_pairs=0, partition_volumes=0;
};
struct Result {
    bool complete=false, CUDA_executed=false;
    std::string source_binding, domain_binding, reason;
    b::Box domain;
    std::array<U,7> roots{none,none,none,none,none,none,none};
    Counters counters;
    U owned_device_peak_bytes=0;
    std::vector<Node> nodes;
    std::vector<Arc> arcs;
    // Retained on incomplete construction only, for explicit diagnostics.
    // Not an accepted model, and automatic cross-process resume is unsupported.
    std::vector<State> frontier_states;
    std::vector<Row> frontier_rows;
    std::vector<Arc> frontier_edges;
};
struct Audit {
    bool complete=false, CUDA_executed=false;
    U quotient_cells=0, score_words=0, owned_device_peak_bytes=0;
    std::string reason;
};
// Host-only metadata overflow/layout checks; no geometry, scores or inference.
U planned_device_bytes(U factors, const Options&);
void validate_transport(const Result&);
// Same-process qualified factor input; binding strings namespace immutable
// source/domain/order state but do not authenticate arbitrary archives.
// Fixed numeric0..9,wilderness,soil order. All partition validation, restriction,
// canonical sorting, memoization, reduction and interning execute on CUDA.
// A sequential device worker is intentional for this bounded first primitive;
// no speedup, polynomial bound, native class authority or joint compiler claim.
// Caps/cancellation return complete=false and all roots=none, plus the explicit
// partial construction frontier. Source/factor arithmetic is not recomputed.
// Explicit byte accounting excludes CUDA driver/runtime allocations and implicit
// thread stacks; compiler resources are reported with each frozen build.
Result construct(const Input&, Options = {}, const Stop& = {}, int device=0);
// Independent bounded exhaustive quotient audit, explicitly separate from
// construction. GPU compares diagram words against direct unique factor lookup.
// It enumerates only when the entire input quotient fits maximum_cells.
Audit audit_quotient(const Input&, const Result&, U maximum_cells=262144, int device=0);
}
