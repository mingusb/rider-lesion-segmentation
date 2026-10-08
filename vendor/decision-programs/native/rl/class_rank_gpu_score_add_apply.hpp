#pragma once
#include "class_rank_gpu_score_diagram.hpp"
namespace rank_gpu_score_add_apply {
namespace sd=rank_gpu_score_diagram;
namespace b=rank_gpu_bounds;
using U=sd::U; using Stop=sd::Stop; using Node=sd::Node; using Arc=sd::Arc;
inline constexpr U none=sd::none;
struct Graph {
 bool complete=false;
 std::string source_binding,domain_binding;
 b::Box domain{};
 U root=none;
 std::vector<Node> nodes;
 std::vector<Arc> arcs;
};
struct Options {
 U maximum_input_nodes=65536,maximum_input_arcs=262144;
 U maximum_pairs=32768,maximum_edges=131072,maximum_nodes=32768,maximum_arcs=131072;
 U pair_buckets=65537,node_buckets=65537,items_per_launch=16;
 U maximum_device_bytes=128ull*1024*1024;
};
struct Pair {
 U left=none,right=none,first_edge=0,edge_count=0,result=none,hash_next=none;
 std::uint32_t dimension=12,status=0; // pending0,expanded1,terminal2,resolved3
};
struct Counters {
 U pairs=0,edges=0,nodes=0,arcs=0,expanded=0,reduced=0;
 U pair_attempts=0,pair_hits=0,pair_collisions=0,open_hits=0,unexpanded_hits=0,earlier_child_hits=0;
 U node_attempts=0,node_hits=0,node_collisions=0,terminal_additions=0;
 U equal_child_reductions=0,adjacent_arc_merges=0;
};
struct Result {
 bool complete=false,CUDA_executed=false;
 Graph graph;
 std::string operation_binding,reason;
 Counters counters;
 U owned_device_peak_bytes=0;
 std::vector<Pair> frontier_pairs;
 std::vector<Arc> frontier_edges;
};
struct AuditOptions {
 U maximum_triples=262144,maximum_transitions=1048576,items_per_launch=16;
 U maximum_device_bytes=128ull*1024*1024;
};
struct Audit {
 bool complete=false,CUDA_executed=false;
 U triples=0,transitions=0,terminal_word_checks=0,checked_nodes=0,owned_device_peak_bytes=0;
 std::string reason;
};
// Host metadata only. No score calculation, predicate geometry or inference.
Graph select(const sd::Result&,U root);
void validate_transport(const Graph&);
U planned_device_bytes(const Graph&,const Graph&,const Options&);
// Same-process trusted immutable diagrams on the identical domain/rank/source.
// operation_binding identifies this ordered binary step; string equality is not
// authentication for imported archives. One device __fadd_rn per terminal pair.
// No reassociation, no add-zero shortcut, no native class/probability authority.
// Pending exact operand pairs are interned before child expansion. Reduction is
// by decreasing dimension, not state ID. First bounded worker is intentionally
// serial on device. All caps/cancel publish no accepted root; partial work is
// diagnostic only. Explicit bytes exclude native/runtime/driver and thread stack.
Result apply(const Graph&,const Graph&,const std::string& operation_binding,
             Options={},const Stop& = {},int device=0);
// Independent sparse universal triple-cofactor traversal, without producer
// pairs/edges. Every graph's full-domain arc cover/layout is checked on device.
// A capped/cancelled audit is incomplete, never sampling-based acceptance.
Audit audit(const Graph&,const Graph&,const Result&,AuditOptions={},
            const Stop& = {},int device=0);
}
