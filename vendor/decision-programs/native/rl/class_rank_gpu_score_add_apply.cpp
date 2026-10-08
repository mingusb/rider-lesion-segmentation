#include "class_rank_gpu_score_add_apply.hpp"
#include <stdexcept>
namespace rank_gpu_score_add_apply {namespace {
void need(bool v,const char*s){if(!v)throw std::invalid_argument(s);}
U add(U a,U z){need(z<=none-a,"score-add byte addition overflow");return a+z;}
U mul(U a,U z){need(!z||a<=none/z,"score-add byte product overflow");return a*z;}
}
Graph select(const sd::Result&r,U root){
 need(r.complete&&root<r.nodes.size(),"score-add selected source incomplete/root");
 Graph g;g.complete=true;g.source_binding=r.source_binding;g.domain_binding=r.domain_binding;
 g.domain=r.domain;g.root=root;g.nodes=r.nodes;g.arcs=r.arcs;validate_transport(g);return g;
}
void validate_transport(const Graph&g){
 need(g.complete&&!g.source_binding.empty()&&!g.domain_binding.empty(),"score-add graph binding/completion");
 need(!g.nodes.empty()&&g.root<g.nodes.size(),"score-add graph root");
 for(U id=0;id<g.nodes.size();++id){const auto&n=g.nodes[id];
  need(n.kind<=1&&!n.reserved&&n.first_arc<=g.arcs.size()&&n.arc_count<=g.arcs.size()-n.first_arc,"score-add node transport");
  need(n.kind?(n.dimension<12&&n.arc_count>=2&&!n.score_bits):(n.dimension==12&&!n.arc_count),"score-add node kind");
  for(U j=0;j<n.arc_count;++j){const auto&a=g.arcs[n.first_arc+j];
   need(a.child<id,"score-add child ordering");
   need(!g.nodes[a.child].kind||g.nodes[a.child].dimension>n.dimension,"score-add dimension ordering");
  }
 }
}
U planned_device_bytes(const Graph&a,const Graph&z,const Options&o){
 need(o.maximum_input_nodes&&o.maximum_input_arcs&&o.maximum_pairs&&o.maximum_edges&&o.maximum_nodes&&o.maximum_arcs&&o.pair_buckets&&o.node_buckets&&o.items_per_launch&&o.maximum_device_bytes,"score-add options must be positive");
 U n=512;
 n=add(n,mul(add(a.nodes.size(),z.nodes.size()),sizeof(Node)));
 n=add(n,mul(add(a.arcs.size(),z.arcs.size()),sizeof(Arc)));
 n=add(n,mul(o.maximum_pairs,sizeof(Pair)));
 n=add(n,mul(o.maximum_edges,sizeof(Arc)));
 n=add(n,mul(o.maximum_nodes,sizeof(Node)));
 n=add(n,mul(o.maximum_arcs,sizeof(Arc)));
 n=add(n,mul(o.maximum_edges,sizeof(Arc))); // reduction scratch, never inferred from final output
 n=add(n,mul(add(o.pair_buckets,o.node_buckets),sizeof(U)));
 return n;
}
}
