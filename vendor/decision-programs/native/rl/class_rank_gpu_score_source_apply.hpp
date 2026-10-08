#pragma once
#include "class_rank_gpu_score_add_apply.hpp"
#include "class_rank_gpu_leaf_partitions.hpp"
#include "class_rank_gpu_score_diagram_factor_audit.hpp"
#include <memory>
namespace rank_gpu_score_source_apply {
namespace aa=rank_gpu_score_add_apply;namespace sd=rank_gpu_score_diagram;namespace lp=rank_gpu_leaf_partitions;
using U=aa::U;using Stop=aa::Stop;
struct Options {
 U maximum_source_trees=64,maximum_source_leaves=4096,maximum_steps=64;
 U maximum_packed_nodes=262144,maximum_packed_arcs=1048576;
 U maximum_device_bytes=128ull*1024*1024;
 sd::Options tree;
 sd::FactorAuditOptions tree_audit;
 aa::Options addition;
 aa::AuditOptions addition_audit;
};
struct TreeResult {
 bool complete=false,CUDA_executed=false;std::string reason;
 U source_tree=aa::none,leaf_count=0,transport_rows=0,owned_device_peak_bytes=0;
 aa::Graph graph;sd::Counters counters;sd::FactorAudit audit;
};
struct Step {
 U source_tree=aa::none,channel=0,class_prefix=0,tree_transport_rows=0;
 std::string operation_binding,tree_graph_digest,before_graph_digest,after_graph_digest;
 sd::Counters tree_counters;sd::FactorAudit tree_audit;
 aa::Counters add_counters;aa::Audit add_audit;
 U owned_device_peak_bytes=0;
 bool complete=false;std::string reason;
};
struct Result;
// Same-process token: no public constructor or archive import path. This is an
// API ownership boundary, not a cryptographic claim about arbitrary callers.
class AcceptedScores {
 sd::Result scores_;std::string source_digest_,sequence_digest_,graph_digest_;
 AcceptedScores(sd::Result,std::string,std::string);
 friend Result fold(const lp::Result&,Options,const Stop&,int);
public:
 AcceptedScores(const AcceptedScores&)=delete;AcceptedScores&operator=(const AcceptedScores&)=delete;
 const sd::Result& scores()const{return scores_;}
 const std::string& source_digest()const{return source_digest_;}
 const std::string& sequence_digest()const{return sequence_digest_;}
 const std::string& graph_digest()const{return graph_digest_;}
 bool matches(const lp::Result&)const;
};
struct Result {
 bool complete=false,CUDA_executed=false;std::string reason,source_digest,sequence_digest;
 U source_trees=0,completed_trees=0,bias_transport_rows=0,owned_device_peak_bytes=0;
 U retained_graph_logical_bytes_highwater=0,packing_audited_nodes=0,packing_audited_arcs=0;
 std::array<U,7> completed_class_prefixes{};
 sd::Counters bias_counters;sd::FactorAudit bias_audit;
 std::vector<Step> steps;
 std::unique_ptr<AcceptedScores> accepted;
};
// Host layout/count/binding checks and exact opaque-word streaming hashes only.
void validate_source_transport(const lp::Result&);
std::string source_digest(const lp::Result&);
std::string graph_digest(const aa::Graph&);
// Raw contribution of ONE authenticated source tree. No bias or FP addition.
// CUDA copies its qualified leaf rectangles/words into a single-channel table,
// then frozen ordered-DD construction and universal factor congruence run.
TreeResult tree(const lp::Result&,U source_tree,Options={},const Stop& = {},int device=0);
// GPU creates each exact bias once, then follows original global source ordinal,
// updating only that channel with one audited ordered RN32 Apply per tree.
// GPU packs7 final pools and independently checks every remapped semantic word.
// Caps/cancel never return AcceptedScores. Native margin witnesses are supplied
// by the qualified bridge, not established anew by symbolic Add congruence.
Result fold(const lp::Result&,Options={},const Stop& = {},int device=0);
struct QuotientAudit {bool complete=false,CUDA_executed=false;std::string reason;U cells=0,score_words=0,owned_device_peak_bytes=0;};
// Independent bounded ALL-cell direct source-leaf lookup and original-order
// RN32 fold versus final diagrams on GPU. Not native-library/class authority.
QuotientAudit audit_quotient(const lp::Result&,const sd::Result&,U maximum_cells=262144,U batch_rows=1024,U maximum_device_bytes=128ull*1024*1024,const Stop& = {},int device=0);
namespace detail {
struct PartitionResult {bool complete=false,CUDA_executed=false;std::string reason;U leaf_count=0,transport_rows=0,owned_device_peak_bytes=0;sd::Result diagrams;sd::FactorAudit audit;};
PartitionResult partition(const lp::Result&,U source_tree,bool bias,Options,const Stop&,int);
struct Packed {bool complete=false,CUDA_executed=false;std::string reason;U owned_device_peak_bytes=0,audited_nodes=0,audited_arcs=0;sd::Result scores;};
Packed pack(const std::array<aa::Graph,7>&,Options,const Stop&,int);
}
}