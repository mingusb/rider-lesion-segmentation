#include "class_rank_gpu_score_add_apply.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
namespace rank_gpu_score_add_apply {namespace {
void need(bool v,const char*s){if(!v)throw std::invalid_argument(s);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Budget{U limit,used=0,peak=0;void add(U n){need(n<=limit-used,"score-add device budget");used+=n;peak=std::max(peak,used);}};
template<class T>struct Buf{
 Budget&b;U n,bytes;T*p=nullptr;
 Buf(Budget&v,U count):b(v),n(count),bytes(0){need(count<=none/sizeof(T),"score-add allocation overflow");bytes=count*sizeof(T);b.add(bytes);if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){b.used-=bytes;cu(e);}e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);p=nullptr;b.used-=bytes;cu(e);}}}
 ~Buf(){if(p)cudaFree(p);b.used-=bytes;}Buf(const Buf&)=delete;
 void put(const T*x){if(bytes)cu(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}
 std::vector<T>get(U k){need(k<=n,"score-add download range");std::vector<T>v(k);if(k)cu(cudaMemcpy(v.data(),p,k*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct View{const Node*nodes;const Arc*arcs;U nn,na,root;};
struct Meta{Counters c;U cursor=0,reduce_cursor=0,root_pair=none,accepted=none;int bad=0,cap=0,reduce_dimension=12;};
static_assert(sizeof(Meta)<=512);
struct Work{View left,right;b::Box domain;Options o;Meta*m;Pair*pairs;Arc*edges;Node*nodes;Arc*arcs;Arc*scratch;U*pair_heads;U*node_heads;};
__device__ U mix(U h,U x){return (h^x)*1099511628211ull;}
__device__ unsigned dimension(View v,U id){return v.nodes[id].kind?v.nodes[id].dimension:12;}
__device__ bool same_domain(const b::Box&a,const b::Box&z){if(a.allowed!=z.allowed)return false;for(int d=0;d<10;++d)if(a.lo[d]!=z.lo[d]||a.hi[d]!=z.hi[d])return false;return true;}
__device__ bool valid_domain(const b::Box&q){
 if((q.allowed&~((U(1)<<44)-1))||!(q.allowed&b::wilderness)||!(q.allowed&b::soil))return false;
 for(int d=0;d<10;++d)if(q.lo[d]<0||q.lo[d]>q.hi[d]||q.hi[d]>16777216)return false;return true;
}
__device__ bool layout(View v,b::Box q){
 if(v.root>=v.nn)return false;
 for(U id=0;id<v.nn;++id){const auto&n=v.nodes[id];
  if(n.kind>1||n.reserved||n.first_arc>v.na||n.arc_count>v.na-n.first_arc)return false;
  if(!n.kind){if(n.dimension!=12||n.arc_count||!isfinite(__uint_as_float(n.score_bits)))return false;continue;}
  if(n.dimension>=12||n.arc_count<2||n.score_bits)return false;
  int next=n.dimension<10?q.lo[n.dimension]:0;U seen=0;int last=0;
  for(U j=0;j<n.arc_count;++j){const auto&a=v.arcs[n.first_arc+j];if(a.child>=id||dimension(v,a.child)<=n.dimension)return false;
   if(n.dimension<10){if(a.allowed||a.lo!=next||a.hi<a.lo||a.hi>q.hi[n.dimension]||(j&&v.arcs[n.first_arc+j-1].child==a.child))return false;next=a.hi+1;}
   else{U group=q.allowed&(n.dimension==10?b::wilderness:b::soil);int first=__ffsll(a.allowed);
    if(a.lo||a.hi||!a.allowed||(a.allowed&~group)||(seen&a.allowed)||first<=last)return false;
    for(U k=0;k<j;++k)if(v.arcs[n.first_arc+k].child==a.child)return false;
    seen|=a.allowed;last=first;
   }
  }
  if(n.dimension<10){if(next!=q.hi[n.dimension]+1)return false;}
  else if(seen!=(q.allowed&(n.dimension==10?b::wilderness:b::soil)))return false;
 }
 return true;
}
__global__ void validate(Work w,b::Box other){if(blockIdx.x||threadIdx.x)return;
 if(!valid_domain(w.domain)||!same_domain(w.domain,other)){w.m->bad=1;return;}
 if(!layout(w.left,w.domain)||!layout(w.right,w.domain))w.m->bad=2;
}
__device__ U pair(Work w,U a,U z,U parent=none){
 auto&m=*w.m;++m.c.pair_attempts;U h=mix(mix(1469598103934665603ull,a),z),bucket=h%w.o.pair_buckets;
 for(U id=w.pair_heads[bucket];id!=none;id=w.pairs[id].hash_next){auto&p=w.pairs[id];
  if(p.left==a&&p.right==z){++m.c.pair_hits;if(p.result==none)++m.c.open_hits;if(!p.status)++m.c.unexpanded_hits;if(parent!=none&&id<parent)++m.c.earlier_child_hits;return id;}
  ++m.c.pair_collisions;
 }
 if(m.c.pairs==w.o.maximum_pairs){m.cap=1;return none;}
 U id=m.c.pairs++;Pair p{};p.left=a;p.right=z;p.dimension=min(dimension(w.left,a),dimension(w.right,z));p.hash_next=w.pair_heads[bucket];
 // The pending key is published before any of its children are discovered.
 w.pairs[id]=p;w.pair_heads[bucket]=id;return id;
}
__global__ void initialize(Work w){if(blockIdx.x||threadIdx.x)return;
 for(U i=0;i<w.o.pair_buckets;++i)w.pair_heads[i]=none;
 for(U i=0;i<w.o.node_buckets;++i)w.node_heads[i]=none;
 w.m->root_pair=pair(w,w.left.root,w.right.root);
}
__device__ U at_rank(View v,U id,unsigned d,int rank,int&until){
 if(dimension(v,id)!=d)return id;
 const auto&n=v.nodes[id];
 for(U j=0;j<n.arc_count;++j){const auto&a=v.arcs[n.first_arc+j];if(a.lo<=rank&&rank<=a.hi){until=min(until,a.hi);return a.child;}}
 return none;
}
__device__ U at_category(View v,U id,unsigned d,U bit){
 if(dimension(v,id)!=d)return id;const auto&n=v.nodes[id];
 for(U j=0;j<n.arc_count;++j){const auto&a=v.arcs[n.first_arc+j];if(a.allowed&bit)return a.child;}
 return none;
}
__device__ bool edge(Work w,Pair&p,Arc a){
 auto&m=*w.m;
 if(p.dimension<10&&p.edge_count){auto&last=w.edges[p.first_edge+p.edge_count-1];if(last.child==a.child&&last.hi+1==a.lo){last.hi=a.hi;++m.c.adjacent_arc_merges;return true;}}
 if(p.dimension>=10)for(U j=0;j<p.edge_count;++j){auto&old=w.edges[p.first_edge+j];if(old.child==a.child){old.allowed|=a.allowed;return true;}}
 if(m.c.edges==w.o.maximum_edges){m.cap=2;return false;}w.edges[m.c.edges++]=a;++p.edge_count;return true;
}
__global__ void expand(Work w,U limit){if(blockIdx.x||threadIdx.x)return;auto&m=*w.m;
 for(U done=0;done<limit&&m.cursor<m.c.pairs;++done){U id=m.cursor;auto&p=w.pairs[id];
  if(p.dimension==12)p.status=2;
  else{p.first_edge=m.c.edges;
   if(p.dimension<10){int rank=w.domain.lo[p.dimension],last=w.domain.hi[p.dimension];
    while(rank<=last){int until=last;U a=at_rank(w.left,p.left,p.dimension,rank,until),z=at_rank(w.right,p.right,p.dimension,rank,until);
     if(a==none||z==none||until<rank){m.bad=3;return;}U child=pair(w,a,z,id);if(m.cap)return;
     Arc e{};e.lo=rank;e.hi=until;e.child=child;if(!edge(w,p,e))return;rank=until+1;
    }
   }else{U mask=w.domain.allowed&(p.dimension==10?b::wilderness:b::soil);
    while(mask){U bit=mask&(~mask+1);mask^=bit;U a=at_category(w.left,p.left,p.dimension,bit),z=at_category(w.right,p.right,p.dimension,bit);
     if(a==none||z==none){m.bad=4;return;}U child=pair(w,a,z,id);if(m.cap)return;Arc e{};e.allowed=bit;e.child=child;if(!edge(w,p,e))return;
    }
   }p.status=1;
  }++m.cursor;++m.c.expanded;
 }
}
__device__ bool equal_arc(const Arc&a,const Arc&z){return a.lo==z.lo&&a.hi==z.hi&&a.allowed==z.allowed&&a.child==z.child;}
__device__ U node(Work w,unsigned kind,unsigned d,unsigned bits,U count){
 auto&m=*w.m;++m.c.node_attempts;U h=mix(mix(mix(1469598103934665603ull,kind),d),bits);
 for(U i=0;i<count;++i){const auto&a=w.scratch[i];h=mix(mix(mix(mix(h,unsigned(a.lo)),unsigned(a.hi)),a.allowed),a.child);}U bucket=h%w.o.node_buckets;
 for(U id=w.node_heads[bucket];id!=none;id=w.nodes[id].hash_next){const auto&n=w.nodes[id];bool same=n.kind==kind&&n.dimension==d&&n.score_bits==bits&&n.arc_count==count;
  for(U i=0;same&&i<count;++i)same=equal_arc(w.arcs[n.first_arc+i],w.scratch[i]);
  if(same){++m.c.node_hits;return id;}++m.c.node_collisions;
 }
 if(m.c.nodes==w.o.maximum_nodes){m.cap=3;return none;}
 if(count>w.o.maximum_arcs-m.c.arcs){m.cap=4;return none;}
 U id=m.c.nodes++;Node n{};n.kind=kind;n.dimension=d;n.score_bits=bits;n.first_arc=m.c.arcs;n.arc_count=count;n.hash_next=w.node_heads[bucket];
 for(U i=0;i<count;++i)w.arcs[m.c.arcs++]=w.scratch[i];w.nodes[id]=n;w.node_heads[bucket]=id;return id;
}
__global__ void reduce(Work w,U limit){if(blockIdx.x||threadIdx.x)return;auto&m=*w.m;U done=0;
 while(m.reduce_dimension>=0&&done<limit){
  if(m.reduce_cursor==m.c.pairs){--m.reduce_dimension;m.reduce_cursor=0;continue;}
  U id=m.reduce_cursor++;auto&p=w.pairs[id];if(p.dimension!=unsigned(m.reduce_dimension))continue;++done;
  if(p.status==2){float value=__fadd_rn(__uint_as_float(w.left.nodes[p.left].score_bits),__uint_as_float(w.right.nodes[p.right].score_bits));++m.c.terminal_additions;
   if(!isfinite(value)){m.bad=5;return;}p.result=node(w,0,12,__float_as_uint(value),0);
  }else{if(p.status!=1||!p.edge_count){m.bad=6;return;}U count=0;
   for(U j=0;j<p.edge_count;++j){Arc e=w.edges[p.first_edge+j];
    if(e.child>=m.c.pairs||w.pairs[e.child].dimension<=p.dimension||w.pairs[e.child].status!=3){m.bad=7;return;}
    e.child=w.pairs[e.child].result;
    if(p.dimension<10&&count&&w.scratch[count-1].child==e.child){w.scratch[count-1].hi=e.hi;++m.c.adjacent_arc_merges;continue;}
    bool found=false;if(p.dimension>=10)for(U k=0;k<count;++k)if(w.scratch[k].child==e.child){w.scratch[k].allowed|=e.allowed;found=true;break;}
    if(!found)w.scratch[count++]=e;
   }
   if(count==1){p.result=w.scratch[0].child;++m.c.equal_child_reductions;}
   else p.result=node(w,1,p.dimension,0,count);
  }
  if(m.cap||m.bad)return;p.status=3;++m.c.reduced;
 }
}
__global__ void publish(Work w){if(blockIdx.x||threadIdx.x)return;auto&m=*w.m;
 if(m.bad||m.cap||m.c.reduced!=m.c.pairs||m.root_pair>=m.c.pairs||w.pairs[m.root_pair].result>=m.c.nodes){m.bad=8;return;}
 m.accepted=w.pairs[m.root_pair].result;
}
const char*cap_reason(int c){return c==1?"pair_cap":c==2?"edge_cap":c==3?"node_cap":c==4?"arc_cap":"unknown_cap";}
}
Result apply(const Graph&a,const Graph&z,const std::string&binding,Options o,const Stop&stop,int device){
 validate_transport(a);validate_transport(z);need(!binding.empty()&&a.source_binding==z.source_binding&&a.domain_binding==z.domain_binding,"score-add step/source/domain binding");
 Result out;out.operation_binding=binding;out.graph.source_binding=a.source_binding;out.graph.domain_binding=a.domain_binding;out.graph.domain=a.domain;
 U planned=planned_device_bytes(a,z,o);
 if(a.nodes.size()>o.maximum_input_nodes||z.nodes.size()>o.maximum_input_nodes||a.arcs.size()>o.maximum_input_arcs||z.arcs.size()>o.maximum_input_arcs){out.reason="input_cap";return out;}
 if(planned>o.maximum_device_bytes){out.reason="device_budget";return out;}
 auto halted=[&]{return stop&&stop();};if(halted()){out.reason="cancelled_before_allocation";return out;}
 cu(cudaSetDevice(device));Budget budget{o.maximum_device_bytes};
 Buf<Node>an(budget,a.nodes.size()),zn(budget,z.nodes.size()),nodes(budget,o.maximum_nodes);
 Buf<Arc>aa(budget,a.arcs.size()),za(budget,z.arcs.size()),edges(budget,o.maximum_edges),arcs(budget,o.maximum_arcs),scratch(budget,o.maximum_edges);
 Buf<Pair>pairs(budget,o.maximum_pairs);Buf<U>ph(budget,o.pair_buckets),nh(budget,o.node_buckets);Buf<unsigned char>control(budget,512);
 an.put(a.nodes.data());zn.put(z.nodes.data());aa.put(a.arcs.data());za.put(z.arcs.data());
 // cudaMemset zero is not C++ default initialization: copy sentinel-bearing metadata.
 Meta m{};cu(cudaMemcpy(control.p,&m,sizeof(m),cudaMemcpyHostToDevice));
 Work w{{an.p,aa.p,a.nodes.size(),a.arcs.size(),a.root},{zn.p,za.p,z.nodes.size(),z.arcs.size(),z.root},a.domain,o,reinterpret_cast<Meta*>(control.p),pairs.p,edges.p,nodes.p,arcs.p,scratch.p,ph.p,nh.p};
 auto checked=[&]{cu(cudaGetLastError());cu(cudaMemcpy(&m,control.p,sizeof(m),cudaMemcpyDeviceToHost));if(m.bad)throw std::runtime_error("CUDA score-add failed: "+std::to_string(m.bad));};
 auto finish=[&](bool complete,const char*reason){out.complete=out.graph.complete=complete;out.reason=reason;out.counters=m.c;out.owned_device_peak_bytes=budget.peak;out.graph.nodes=nodes.get(m.c.nodes);out.graph.arcs=arcs.get(m.c.arcs);
  if(complete)out.graph.root=m.accepted;else{out.frontier_pairs=pairs.get(m.c.pairs);out.frontier_edges=edges.get(m.c.edges);}return out;};
 validate<<<1,1>>>(w,z.domain);out.CUDA_executed=true;checked();if(halted())return finish(false,"cancelled_after_layout");
 initialize<<<1,1>>>(w);checked();if(m.cap)return finish(false,cap_reason(m.cap));
 while(m.cursor<m.c.pairs){if(halted())return finish(false,"cancelled_before_expansion");expand<<<1,1>>>(w,o.items_per_launch);checked();if(m.cap)return finish(false,cap_reason(m.cap));}
 while(m.c.reduced<m.c.pairs){if(halted())return finish(false,"cancelled_before_reduction");reduce<<<1,1>>>(w,o.items_per_launch);checked();if(m.cap)return finish(false,cap_reason(m.cap));}
 if(halted())return finish(false,"cancelled_before_publication");publish<<<1,1>>>(w);checked();
 if(halted())return finish(false,"cancelled_after_publication");auto r=finish(true,"CUDA_exact_ordered_binary_RN32_add_no_class_authority");validate_transport(r.graph);return r;
}
}
