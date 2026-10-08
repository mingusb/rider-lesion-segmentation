#include "class_rank_gpu_score_source_apply.hpp"
#include "class_rank_gpu_score_pool_identity.hpp"
#include <algorithm>
#include <bit>
#include <stdexcept>
namespace rank_gpu_score_source_apply {namespace {
void need(bool b,const char*s){if(!b)throw std::invalid_argument(s);}
U plus(U a,U b){need(b<=aa::none-a,"source Apply count overflow");return a+b;}
void box(rank_gpu_score_identity::Hash&h,const aa::b::Box&q){for(auto x:q.lo)h.word(std::uint32_t(x),4);for(auto x:q.hi)h.word(std::uint32_t(x),4);h.word(q.allowed,8);}
void options(const Options&o){need(o.maximum_source_trees&&o.maximum_source_trees<=64&&o.maximum_source_leaves&&o.maximum_source_leaves<=4096&&o.maximum_steps&&o.maximum_packed_nodes&&o.maximum_packed_arcs&&o.maximum_device_bytes,"source Apply options invalid or beyond qualified source caps");}
U graph_bytes(const aa::Graph&g){need(g.nodes.size()<=aa::none/sizeof(aa::Node)&&g.arcs.size()<=aa::none/sizeof(aa::Arc),"source Apply host transport size");return plus(g.nodes.size()*sizeof(aa::Node),g.arcs.size()*sizeof(aa::Arc));}
U graphs_bytes(const std::array<aa::Graph,7>&g){U n=0;for(const auto&a:g)n=plus(n,graph_bytes(a));return n;}
}
void validate_source_transport(const lp::Result&p){
 const auto&s=p.source.value;
 need(p.complete&&!p.source.binding.empty()&&!p.source_sha256.empty()&&!p.library_sha256.empty()&&!p.rank_sha256.empty(),"source Apply requires same-process qualified complete bridge");
 need(s.offsets.size()==s.channels.size()+1&&!s.offsets.empty()&&s.offsets.front()==0&&s.offsets.back()>=0&&U(s.offsets.back())==s.leaves.size(),"source Apply offsets/counts");
 need(p.complete_tree_volumes_checked==s.channels.size()&&p.feasible_leaves==s.leaves.size(),"source Apply source coverage metadata");
 for(U t=0;t<s.channels.size();++t)need(s.offsets[t]>=0&&s.offsets[t]<s.offsets[t+1]&&s.channels[t]>=0&&s.channels[t]<7,"source Apply tree/channel layout");
 for(const auto&l:s.leaves)need(l.ordinal>=0,"source Apply leaf ordinal");
}
std::string source_digest(const lp::Result&p){
 validate_source_transport(p);rank_gpu_score_identity::Hash h;h.text("qualified-source-leaf-sequence-exact-words-v1");h.text(p.source_sha256);h.text(p.library_sha256);h.text(p.rank_sha256);h.text(p.source.binding);box(h,p.domain);
 for(const auto&a:p.rank_cut_bits){h.word(a.size(),8);for(auto w:a)h.word(w,4);}const auto&s=p.source.value;
 h.word(s.channels.size(),8);for(auto c:s.channels)h.word(std::uint32_t(c),4);for(auto x:s.offsets)h.word(std::uint32_t(x),4);for(float x:s.bias)h.word(std::bit_cast<std::uint32_t>(x),4);
 h.word(s.leaves.size(),8);for(const auto&l:s.leaves){box(h,l.box);h.word(std::uint32_t(l.ordinal),4);h.word(std::bit_cast<std::uint32_t>(l.value),4);}return h.finish();
}
std::string graph_digest(const aa::Graph&g){
 aa::validate_transport(g);rank_gpu_score_identity::Hash h;h.text("single-score-graph-exact-words-v1");h.text(g.source_binding);h.text(g.domain_binding);box(h,g.domain);h.word(g.root,8);h.word(g.nodes.size(),8);h.word(g.arcs.size(),8);
 for(const auto&n:g.nodes){h.word(n.kind,4);h.word(n.dimension,4);h.word(n.score_bits,4);h.word(n.reserved,4);h.word(n.first_arc,8);h.word(n.arc_count,8);}
 for(const auto&a:g.arcs){h.word(std::uint32_t(a.lo),4);h.word(std::uint32_t(a.hi),4);h.word(a.allowed,8);h.word(a.child,8);}return h.finish();
}
AcceptedScores::AcceptedScores(sd::Result s,std::string source,std::string sequence):scores_(std::move(s)),source_digest_(std::move(source)),sequence_digest_(std::move(sequence)),graph_digest_(rank_gpu_score_identity::digest(scores_)){}
bool AcceptedScores::matches(const lp::Result&p)const{
 validate_source_transport(p);return scores_.complete&&scores_.source_binding==p.source.binding&&scores_.domain_binding==p.rank_sha256&&source_digest_==rank_gpu_score_source_apply::source_digest(p)&&graph_digest_==rank_gpu_score_identity::digest(scores_);
}
TreeResult tree(const lp::Result&p,U t,Options o,const Stop&stop,int device){
 validate_source_transport(p);options(o);need(t<p.source.value.channels.size(),"source Apply tree ordinal out of range");TreeResult out;out.source_tree=t;
 if(p.source.value.channels.size()>o.maximum_source_trees||p.source.value.leaves.size()>o.maximum_source_leaves){out.reason="source_capacity";return out;}
 auto part=detail::partition(p,t,false,o,stop,device);out.CUDA_executed=part.CUDA_executed;out.owned_device_peak_bytes=part.owned_device_peak_bytes;out.leaf_count=part.leaf_count;out.transport_rows=part.transport_rows;out.counters=part.diagrams.counters;out.audit=part.audit;out.reason=part.reason;
 if(!part.complete)return out;
 if(stop&&stop()){out.reason="cancelled_before_tree_publication";return out;}
 out.graph=aa::select(part.diagrams,part.diagrams.roots[0]);out.complete=true;out.reason="complete_single_source_tree_universal_leaf_word_congruence";return out;
}
Result fold(const lp::Result&p,Options o,const Stop&stop,int device){
 validate_source_transport(p);options(o);Result out;out.source_trees=p.source.value.channels.size();out.source_digest=source_digest(p);
 auto halt=[&]{return stop&&stop();};auto fail=[&](const std::string&why){out.reason=why;return std::move(out);};
 if(out.source_trees>o.maximum_source_trees||p.source.value.leaves.size()>o.maximum_source_leaves)return fail("source_capacity");
 if(halt())return fail("cancelled_before_bias");
 auto initial=detail::partition(p,aa::none,true,o,stop,device);out.CUDA_executed=initial.CUDA_executed;out.owned_device_peak_bytes=initial.owned_device_peak_bytes;out.bias_transport_rows=initial.transport_rows;out.bias_counters=initial.diagrams.counters;out.bias_audit=initial.audit;
 if(!initial.complete)return fail("bias_"+initial.reason);
 std::array<aa::Graph,7>acc;
 for(int c=0;c<7;++c){acc[c]=aa::select(initial.diagrams,initial.diagrams.roots[c]);}
 initial.diagrams={};
 out.retained_graph_logical_bytes_highwater=graphs_bytes(acc);
 rank_gpu_score_identity::Hash sequence;sequence.text("original-source-order-bias-once-RN32-no-reassociation-v1");sequence.text(out.source_digest);
 for(int c=0;c<7;++c){sequence.word(c,4);sequence.text(graph_digest(acc[c]));}
 for(U t=0;t<out.source_trees;++t){
  if(t==o.maximum_steps)return fail("ordered_step_cap");
  if(halt())return fail("cancelled_before_source_tree");
  Step step;step.source_tree=t;step.channel=U(p.source.value.channels[t]);step.class_prefix=out.completed_class_prefixes[step.channel]+1;
  auto tr=tree(p,t,o,stop,device);step.tree_transport_rows=tr.transport_rows;step.tree_counters=tr.counters;step.tree_audit=tr.audit;step.owned_device_peak_bytes=tr.owned_device_peak_bytes;out.CUDA_executed|=tr.CUDA_executed;out.owned_device_peak_bytes=std::max(out.owned_device_peak_bytes,tr.owned_device_peak_bytes);
  if(!tr.complete){step.reason="tree_"+tr.reason;out.steps.push_back(step);return fail(step.reason);}
  step.tree_graph_digest=graph_digest(tr.graph);step.before_graph_digest=graph_digest(acc[step.channel]);
  rank_gpu_score_identity::Hash op;op.text("ordered-source-tree-RN32-step-v1");op.text(out.source_digest);op.word(t,8);op.word(step.channel,4);op.word(step.class_prefix,8);op.text(step.before_graph_digest);op.text(step.tree_graph_digest);step.operation_binding=op.finish();
  auto addition=o.addition;addition.maximum_device_bytes=std::min(addition.maximum_device_bytes,o.maximum_device_bytes);
  auto added=aa::apply(acc[step.channel],tr.graph,step.operation_binding,addition,stop,device);step.add_counters=added.counters;step.owned_device_peak_bytes=std::max(step.owned_device_peak_bytes,added.owned_device_peak_bytes);out.CUDA_executed|=added.CUDA_executed;out.owned_device_peak_bytes=std::max(out.owned_device_peak_bytes,added.owned_device_peak_bytes);
  out.retained_graph_logical_bytes_highwater=std::max(out.retained_graph_logical_bytes_highwater,plus(graphs_bytes(acc),plus(graph_bytes(tr.graph),graph_bytes(added.graph))));
  if(!added.complete){step.reason="add_"+added.reason;out.steps.push_back(step);return fail(step.reason);}
  auto audit_options=o.addition_audit;audit_options.maximum_device_bytes=std::min(audit_options.maximum_device_bytes,o.maximum_device_bytes);
  step.add_audit=aa::audit(acc[step.channel],tr.graph,added,audit_options,stop,device);out.CUDA_executed|=step.add_audit.CUDA_executed;step.owned_device_peak_bytes=std::max(step.owned_device_peak_bytes,step.add_audit.owned_device_peak_bytes);out.owned_device_peak_bytes=std::max(out.owned_device_peak_bytes,step.add_audit.owned_device_peak_bytes);
  if(!step.add_audit.complete){step.reason="add_audit_"+step.add_audit.reason;out.steps.push_back(step);return fail(step.reason);}
  if(halt()){step.reason="cancelled_before_prefix_commit";out.steps.push_back(step);return fail(step.reason);}
  step.after_graph_digest=graph_digest(added.graph);sequence.word(t,8);sequence.word(step.channel,4);sequence.word(step.class_prefix,8);sequence.text(step.operation_binding);sequence.text(step.after_graph_digest);
  acc[step.channel]=std::move(added.graph);++out.completed_trees;++out.completed_class_prefixes[step.channel];step.complete=true;step.reason="ordered_prefix_universal_RN32_audit_complete";out.steps.push_back(std::move(step));
 }
 out.sequence_digest=sequence.finish();if(halt())return fail("cancelled_before_packing");auto packed=detail::pack(acc,o,stop,device);out.CUDA_executed|=packed.CUDA_executed;out.owned_device_peak_bytes=std::max(out.owned_device_peak_bytes,packed.owned_device_peak_bytes);out.packing_audited_nodes=packed.audited_nodes;out.packing_audited_arcs=packed.audited_arcs;
 if(!packed.complete)return fail("packing_"+packed.reason);
 if(halt())return fail("cancelled_before_source_publication");need(source_digest(p)==out.source_digest,"source changed during ordered fold");
 auto token=std::unique_ptr<AcceptedScores>(new AcceptedScores(std::move(packed.scores),out.source_digest,out.sequence_digest));
 if(halt())return fail("cancelled_before_accepted_score_transfer");
 out.accepted=std::move(token);out.complete=true;out.reason="complete_seven_source_ordered_score_functions_universal_prefix_audits";return out;
}
}