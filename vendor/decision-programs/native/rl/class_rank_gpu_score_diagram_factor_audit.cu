#include "class_rank_gpu_score_diagram_factor_audit.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
namespace rank_gpu_score_diagram {namespace {
using Factor=rank_gpu_score_factors::Factor;
void need(bool x,const char*s){if(!x)throw std::invalid_argument(s);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U mul(U a,U z){need(!z||a<=none/z,"factor-congruence byte overflow");return a*z;}
U add(U a,U z){need(z<=none-a,"factor-congruence byte overflow");return a+z;}
struct Budget {U limit,used=0,peak=0;void add(U n){need(n<=limit-used,"factor-congruence device cap");used+=n;peak=std::max(peak,used);}};
template<class T>struct Buffer {Budget&b;U n,bytes;T*p=nullptr;
 Buffer(Budget&owner,U count):b(owner),n(count),bytes(mul(count,sizeof(T))){b.add(bytes);if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){b.used-=bytes;cu(e);}e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);p=nullptr;b.used-=bytes;cu(e);}}}
 ~Buffer(){if(p)cudaFree(p);b.used-=bytes;}Buffer(const Buffer&)=delete;
 void put(const T*x){if(bytes)cu(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}
 T first(){T x{};cu(cudaMemcpy(&x,p,sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
struct Metadata {U visited=0,edges=0,terminals=0,nodes=0;int bad=0;};
__device__ void bad(Metadata*m,int code){atomicCAS(&m->bad,0,code);}
__device__ bool same(const b::Box&a,const b::Box&z){if(a.allowed!=z.allowed)return false;for(int d=0;d<10;++d)if(a.lo[d]!=z.lo[d]||a.hi[d]!=z.hi[d])return false;return true;}
__global__ void domains(b::Box a,b::Box z,const U*offsets,U nf,const U*roots,U nn,Metadata*m){if(blockIdx.x||threadIdx.x)return;
 if(!same(a,z)||(a.allowed&~((U(1)<<44)-1))||!(a.allowed&b::wilderness)||!(a.allowed&b::soil)||offsets[0]||offsets[7]!=nf){bad(m,1);return;}
 for(int d=0;d<10;++d)if(a.lo[d]<0||a.hi[d]>16777216||a.lo[d]>a.hi[d]){bad(m,2);return;}
 for(int c=0;c<7;++c)if(offsets[c]>=offsets[c+1]||offsets[c+1]>nf||roots[c]>=nn){bad(m,3);return;}
}
__global__ void layout(b::Box domain,const Node*nodes,U nn,const Arc*arcs,U na,Metadata*m){
 U id=U(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=nn)return;const auto&n=nodes[id];
 if(n.kind>1||n.reserved||n.first_arc>na||n.arc_count>na-n.first_arc){bad(m,4);return;}
 if(!n.kind){if(n.dimension!=12||n.arc_count||!isfinite(__uint_as_float(n.score_bits))){bad(m,5);return;}}
 else{if(n.dimension>=12||n.arc_count<2||n.score_bits){bad(m,6);return;}int next=n.dimension<10?domain.lo[n.dimension]:0;U seen=0;int last_bit=0;
  for(U i=0;i<n.arc_count;++i){const auto&a=arcs[n.first_arc+i];if(a.child>=id||(nodes[a.child].kind&&nodes[a.child].dimension<=n.dimension)){bad(m,7);return;}
   if(n.dimension<10){if(a.allowed||a.lo!=next||a.hi<a.lo||a.hi>domain.hi[n.dimension]||(i&&arcs[n.first_arc+i-1].child==a.child)){bad(m,8);return;}next=a.hi+1;}
   else{U allowed=domain.allowed&(n.dimension==10?b::wilderness:b::soil);int first=__ffsll(a.allowed);
    if(a.lo||a.hi||!a.allowed||(a.allowed&~allowed)||(seen&a.allowed)||first<=last_bit){bad(m,9);return;}seen|=a.allowed;last_bit=first;
    for(U j=0;j<i;++j)if(arcs[n.first_arc+j].child==a.child){bad(m,10);return;}
   }
  }
  if(n.dimension<10){if(next!=domain.hi[n.dimension]+1){bad(m,11);return;}}
  else if(seen!=(domain.allowed&(n.dimension==10?b::wilderness:b::soil))){bad(m,12);return;}
 }atomicAdd(reinterpret_cast<unsigned long long*>(&m->nodes),1ull);
}
__global__ void congruence(b::Box domain,const Factor*f,U nf,const U*offsets,const Node*nodes,U nn,const Arc*arcs,const U*roots,unsigned char*reachable,Metadata*m){
 U id=U(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=nf)return;const auto&factor=f[id];int c=0;while(id>=offsets[c+1])++c;
 if(factor.channel!=unsigned(c)||!isfinite(__uint_as_float(factor.score_bits))||(factor.box.allowed&~domain.allowed)||!(factor.box.allowed&b::wilderness)||!(factor.box.allowed&b::soil)){bad(m,13);return;}
 for(int d=0;d<10;++d)if(factor.box.lo[d]<domain.lo[d]||factor.box.hi[d]>domain.hi[d]||factor.box.lo[d]>factor.box.hi[d]){bad(m,14);return;}
 unsigned char*marks=reachable+id*nn;marks[roots[c]]=1;U visited=0,edges=0,terminals=0;
 for(U at=nn;at-->0;){if(!marks[at])continue;++visited;const auto&n=nodes[at];
  if(!n.kind){if(n.score_bits!=factor.score_bits){bad(m,15);return;}++terminals;continue;}
  U hit=0;for(U i=0;i<n.arc_count;++i){const auto&a=arcs[n.first_arc+i];bool compatible=n.dimension<10?max(a.lo,factor.box.lo[n.dimension])<=min(a.hi,factor.box.hi[n.dimension]):bool(a.allowed&factor.box.allowed);
   if(compatible){marks[a.child]=1;++hit;++edges;}}
  if(!hit){bad(m,16);return;}
 }
 if(!terminals){bad(m,17);return;}
 atomicAdd(reinterpret_cast<unsigned long long*>(&m->visited),visited);atomicAdd(reinterpret_cast<unsigned long long*>(&m->edges),edges);atomicAdd(reinterpret_cast<unsigned long long*>(&m->terminals),terminals);
}
}
FactorAudit audit_factors(const Input&in,const Result&r,FactorAuditOptions options,const Stop&stop,int device){
 validate_transport(r);need(in.factors.complete&&in.factors.factors.size()==in.factors.compatible&&in.factors.source_binding==r.source_binding&&in.domain_binding==r.domain_binding,"factor-congruence input binding differs");
 U nf=in.factors.factors.size(),nn=r.nodes.size(),na=r.arcs.size();need(nf&&nn&&nf<=U(INT_MAX)*128&&nn<=U(INT_MAX)*128,"factor-congruence shape unsupported");
 FactorAudit out;out.factors=nf;out.nodes=nn;out.possible_factor_node_pairs=mul(nf,nn);
 if(out.possible_factor_node_pairs>options.maximum_factor_node_pairs){out.reason="factor_node_pair_cap";return out;}
 U bytes=add(add(mul(nf,sizeof(Factor)),mul(nn,sizeof(Node))),add(mul(na,sizeof(Arc)),out.possible_factor_node_pairs));bytes=add(bytes,15*sizeof(U)+sizeof(Metadata));
 if(bytes>options.maximum_device_bytes){out.reason="factor_audit_device_cap";return out;}
 auto halt=[&](const char*reason){if(stop&&stop()){out.reason=reason;return true;}return false;};if(halt("cancelled_before_factor_audit"))return out;
 cu(cudaSetDevice(device));Budget budget{options.maximum_device_bytes};Buffer<Factor> f(budget,nf);Buffer<Node> nodes(budget,nn);Buffer<Arc> arcs(budget,na);Buffer<U> offsets(budget,8),roots(budget,7);Buffer<unsigned char> marks(budget,out.possible_factor_node_pairs);Buffer<Metadata> meta(budget,1);
 out.owned_device_peak_bytes=budget.peak;f.put(in.factors.factors.data());nodes.put(r.nodes.data());arcs.put(r.arcs.data());offsets.put(in.factors.factor_offsets.data());roots.put(r.roots.data());
 auto checked=[&]{cu(cudaGetLastError());auto m=meta.first();if(m.bad)throw std::runtime_error("CUDA factor-congruence audit failed: "+std::to_string(m.bad));return m;};
 domains<<<1,1>>>(in.domain,r.domain,offsets.p,nf,roots.p,nn,meta.p);out.CUDA_executed=true;checked();
 layout<<<unsigned((nn+127)/128),128>>>(in.domain,nodes.p,nn,arcs.p,na,meta.p);auto m=checked();out.checked_nodes=m.nodes;
 if(halt("cancelled_after_diagram_layout_audit"))return out;
 congruence<<<unsigned((nf+127)/128),128>>>(in.domain,f.p,nf,offsets.p,nodes.p,nn,arcs.p,roots.p,marks.p,meta.p);m=checked();out.visited_factor_node_pairs=m.visited;out.intersecting_arcs=m.edges;out.terminal_score_word_checks=m.terminals;
 if(halt("cancelled_after_factor_congruence"))return out;
 need(m.nodes==nn&&m.terminals>=nf,"factor-congruence incomplete counters");out.complete=true;out.reason="GPU_universal_factor_box_score_diagram_congruence";return out;
}
}
