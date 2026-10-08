#include "class_rank_gpu_score_diagram.hpp"
#include <stdexcept>
namespace rank_gpu_score_diagram {
namespace {
void need(bool x,const char* s){if(!x)throw std::invalid_argument(s);}
U add(U a,U c){need(c<=none-a,"score-diagram byte sum overflow");return a+c;}
U mul(U a,U c){need(!c||a<=none/c,"score-diagram byte product overflow");return a*c;}
}
U planned_device_bytes(U n,const Options& o){
    need(n&&o.maximum_factors&&o.maximum_factors_per_class&&o.maximum_states&&
         o.maximum_rows&&o.maximum_edges&&o.maximum_nodes&&o.maximum_arcs&&
         o.state_buckets&&o.node_buckets&&o.states_per_launch&&o.states_per_launch<=4096,
         "score-diagram zero or unsupported option");
    U bytes=4096; // Conservative allowance for device metadata and eight offsets.
    bytes=add(bytes,mul(n,sizeof(rank_gpu_score_factors::Factor)+sizeof(Row)));
    bytes=add(bytes,mul(o.maximum_states,sizeof(State)));
    bytes=add(bytes,mul(o.maximum_rows,sizeof(Row)));
    bytes=add(bytes,mul(o.maximum_edges,sizeof(Arc)));
    bytes=add(bytes,mul(o.maximum_nodes,sizeof(Node)));
    bytes=add(bytes,mul(o.maximum_arcs,sizeof(Arc)));
    bytes=add(bytes,mul(add(o.state_buckets,o.node_buckets),sizeof(U)));
    bytes=add(bytes,mul(add(mul(n,2),44),sizeof(Arc)));
    return bytes;
}
void validate_transport(const Result& r){
    need(r.complete,"score-diagram transport incomplete");
    need(!r.source_binding.empty()&&!r.domain_binding.empty(),"score-diagram binding absent");
    need(!r.nodes.empty()&&r.nodes.size()==r.counters.nodes&&r.arcs.size()==r.counters.arcs,
         "score-diagram transport counts differ");
    for(U root:r.roots)need(root<r.nodes.size(),"score-diagram root out of range");
    for(U i=0;i<r.nodes.size();++i){const auto& n=r.nodes[i];
        need(n.kind<=1&&n.reserved==0,"score-diagram node tag/padding invalid");
        need(n.first_arc<=r.arcs.size()&&n.arc_count<=r.arcs.size()-n.first_arc,
             "score-diagram arc range invalid");
        if(!n.kind)need(n.dimension==12&&!n.arc_count,"score-diagram terminal shape");
        else{
            need(n.dimension<12&&n.arc_count>1,"score-diagram branch shape");
            for(U k=n.first_arc;k<n.first_arc+n.arc_count;++k)
                need(r.arcs[k].child<i,"score-diagram child must precede parent");
        }
    }
}
}
