#pragma once
#include "class_rank_gpu_score_diagram_factor_audit.hpp"
#include "class_rank_gpu_leaf_partitions.hpp"
#include "class_rank_gpu_online_cache.hpp"
namespace rank_gpu_score_source_apply {class AcceptedScores;}
namespace rank_gpu_class_apply {
namespace sd=rank_gpu_score_diagram;namespace lp=rank_gpu_leaf_partitions;
using U=sd::U;using Stop=sd::Stop;using Arc=sd::Arc;using Node=sd::Node;
inline constexpr U none=sd::none;
struct Options {
 U maximum_states=262144,maximum_edges=1048576,maximum_nodes=131072,maximum_arcs=524288;
 U state_buckets=524287,node_buckets=262147,states_per_launch=128,native_batch_rows=1024;
 U maximum_device_bytes=128ull*1024*1024;
 bool state_cache=true;
};
struct Point {std::array<std::int32_t,10>rank{};U categories=0;};
struct State {
 std::array<U,7>tuple{};Point witness;
 U first_edge=0,edge_count=0,result=none,hash_next=none;
 std::array<std::uint32_t,7>probability_bits{};
 std::uint32_t level=12,status=0;std::int32_t label=-1;
};
struct Counters {
 U states=0,edges=0,nodes=0,arcs=0,expanded=0,reduced=0,terminal_states=0;
 U state_attempts=0,state_hits=0,open_state_hits=0,unexpanded_state_hits=0,state_collisions=0,tuple_comparisons=0;
 U node_attempts=0,node_hits=0,node_collisions=0,equal_child_reductions=0,arc_merges=0;
 U native_margin_calls=0,native_probability_calls=0,native_rows=0,native_margin_words=0,native_probability_words=0;
 U audited_states=0,audited_transitions=0,audited_nodes=0;
};
struct Result {
 bool complete=false,CUDA_executed=false,local_semantics_audited=false;
 U root=none,owned_device_peak_bytes=0;Counters counters;
 std::string reason,source_binding,domain_binding,score_pool_sha256,native_contract_json,score_input_binding,score_input_scheme;
 sd::b::Box domain;
 std::vector<Node>nodes;std::vector<Arc>arcs;
 // Evidence/diagnostics: terminal probabilities and first-path witnesses remain
 // bound to this in-process native oracle. Imported arrays are not authority.
 std::vector<State>states;std::vector<Arc>edges;
};
struct Lowered {bool complete=false,CUDA_executed=false;U root=none,owned_device_peak_bytes=0,audited_nodes=0,audited_numeric_segments=0,audited_category_values=0;std::string reason;std::vector<rank_gpu_online_cache::Draft>drafts;};
struct QuotientAudit {bool complete=false,CUDA_executed=false;U cells=0,native_rows=0,owned_device_peak_bytes=0;std::string reason;};
U planned_device_bytes(const sd::Result&,U cut_words,const Options&);
// SAME-PROCESS qualified source/factors/score roots only. CUDA validates score
// congruence, memoizes seven-score states before expansion, derives witnesses,
// checks native type=1 words, and labels with native type=0 first-argmax.
// Native row-local tuple congruence is an explicit pinned implementation premise;
// it is not the gap theorem, margin argmax, or a Lean/CUDA refinement proof.
// Caps/cancellation retain diagnostics but publish root=none and complete=false.
Result construct(const sd::Input&,const sd::Result&,const lp::Result&,
                 const std::string&library,const std::string&model,
                 Options={},const Stop& = {},int device=0);
// Opaque same-process sequential source-tree authority. No factor-product
// reconstruction and no caller-supplied acceptance flag. Original bias/tree
// order and universal per-prefix audits are supplied by AcceptedScores.
Result construct(const rank_gpu_score_source_apply::AcceptedScores&,const lp::Result&,
                 const std::string&library,const std::string&model,
                 Options={},const Stop& = {},int device=0);
// Independent CUDA local transition/reduction audit. Terminal native provenance
// is a premise from construct; caller-edited probability words are not authority.
void audit_local(const sd::Result&,const Result&,U maximum_device_bytes=128ull*1024*1024,int device=0);
// Numeric rank cuts and exactly-one category membership become binary GPU
// drafts. Qualified arena/collector subsequently intern and collect the root.
Lowered lower_binary(const Result&,U maximum_drafts=1048576,U maximum_device_bytes=128ull*1024*1024,const Stop& = {},int device=0);
// Independent bounded complete raw quotient class audit for synthetic fixtures.
// Calls native public type=0 on each point; no sample or raw-margin shortcut.
QuotientAudit audit_quotient(const Result&,const lp::Result&,const std::string&library,const std::string&model,U maximum_cells=262144,U batch_rows=1024,int device=0);
QuotientAudit audit_binary_quotient(const Result&,const lp::Result&,const std::vector<rank_disk_ledger::Node>&,U root,const std::string&library,const std::string&model,U maximum_cells=262144,U batch_rows=1024,int device=0);
// CUDA exact word comparison for all matching terminal tuples across separately
// constructed cache/batch configurations; qualification evidence, not a proof
// that an arbitrary native implementation is batch independent.
U audit_terminal_probabilities(const Result&,const Result&,U maximum_device_bytes=128ull*1024*1024,int device=0);
}
