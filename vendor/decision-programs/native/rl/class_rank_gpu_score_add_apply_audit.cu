#include "class_rank_gpu_score_add_apply.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
namespace rank_gpu_score_add_apply {namespace {
void require(bool x,const char*s){if(!x)throw std::invalid_argument(s);}
void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U checked_add(U a,U z){require(z<=none-a,"score-add audit byte overflow");return a+z;}
U checked_mul(U a,U z){require(!z||a<=none/z,"score-add audit product overflow");return a*z;}
struct Memory{U limit,used=0,peak=0;void claim(U bytes){require(bytes<=limit-used,"score-add audit device budget");used+=bytes;peak=std::max(peak,used);}};
template<class T>struct Storage{
 Memory&m;U size,bytes;T*p=nullptr;
 Storage(Memory&owner,U n):m(owner),size(n),bytes(checked_mul(n,sizeof(T))){m.claim(bytes);if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){m.used-=bytes;cuda_check(e);}e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);p=nullptr;m.used-=bytes;cuda_check(e);}}}
 ~Storage(){if(p)cudaFree(p);m.used-=bytes;}Storage(const Storage&)=delete;
 void upload(const T*x){if(bytes)cuda_check(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}
};
struct Diagram{const Node*n;const Arc*a;U nodes,arcs,root;};
struct Triple{U left=none,right=none,result=none,next=none;};
struct State{U triples=0,transitions=0,terminals=0,nodes=0,cursor=0;int error=0,cap=0;};
static_assert(sizeof(State)<=512);
struct AuditWork{Diagram left,right,result;b::Box domain;Triple*triples;U*heads;U buckets;State*s;AuditOptions options;};
__device__ unsigned level(Diagram d,U id){return d.n[id].kind?d.n[id].dimension:12;}
__device__ bool valid_box(const b::Box&q){
 for(int i=0;i<10;++i)if(q.lo[i]<0||q.hi[i]<q.lo[i]||q.hi[i]>16777216)return false;
 return !(q.allowed&~((U(1)<<44)-1))&&(q.allowed&b::wilderness)&&(q.allowed&b::soil);
}
__device__ bool identical_box(const b::Box&a,const b::Box&z){for(int i=0;i<10;++i)if(a.lo[i]!=z.lo[i]||a.hi[i]!=z.hi[i])return false;return a.allowed==z.allowed;}
// Independent layout walker. It does not read the constructor's pair records,
// hash chains, reduced-state markers or edge correspondence.
__device__ bool check_diagram(Diagram g,b::Box box,State*s){
 if(g.root>=g.nodes)return false;
 for(U id=0;id<g.nodes;++id){Node n=g.n[id];if(n.kind>1||n.reserved||n.first_arc>g.arcs||n.arc_count>g.arcs-n.first_arc)return false;
  if(n.kind==0){if(n.dimension!=12||n.arc_count||!isfinite(__uint_as_float(n.score_bits)))return false;++s->nodes;continue;}
  if(n.dimension>=12||n.arc_count<2||n.score_bits)return false;
  U mask=0,span=0;int previous=0;
  for(U j=0;j<n.arc_count;++j){Arc a=g.a[n.first_arc+j];
   if(a.child>=id||level(g,a.child)<=n.dimension)return false;
   if(n.dimension<10){
    if(a.allowed||a.lo<box.lo[n.dimension]||a.hi>box.hi[n.dimension]||a.lo>a.hi)return false;
    if(!j){if(a.lo!=box.lo[n.dimension])return false;}else if(a.lo!=previous+1)return false;
    previous=a.hi;span+=U(a.hi-a.lo)+1;
   }else{
    U group=box.allowed&(n.dimension==10?b::wilderness:b::soil);
    if(a.lo||a.hi||!a.allowed||(a.allowed&~group)||(a.allowed&mask))return false;
    mask|=a.allowed;
   }
  }
  if(n.dimension<10){if(span!=U(box.hi[n.dimension]-box.lo[n.dimension])+1||previous!=box.hi[n.dimension])return false;}
  else if(mask!=(box.allowed&(n.dimension==10?b::wilderness:b::soil)))return false;
  ++s->nodes;
 }
 return true;
}
__device__ U hash_triple(U a,U z,U r){U h=(a+0x9e3779b97f4a7c15ull)*0xbf58476d1ce4e5b9ull;h^=(z+0x94d049bb133111ebull)*0x9e3779b97f4a7c15ull;h^=(r+0xbf58476d1ce4e5b9ull)*0x94d049bb133111ebull;return h;}
__device__ bool insert(AuditWork w,U a,U z,U r){
 U bucket=hash_triple(a,z,r)%w.buckets;
 for(U id=w.heads[bucket];id!=none;id=w.triples[id].next){const auto&t=w.triples[id];if(t.left==a&&t.right==z&&t.result==r)return true;}
 if(w.s->triples==w.options.maximum_triples){w.s->cap=1;return false;}
 U id=w.s->triples++;Triple t{};t.left=a;t.right=z;t.result=r;t.next=w.heads[bucket];w.triples[id]=t;w.heads[bucket]=id;return true;
}
__global__ void audit_layout(AuditWork w,b::Box right,b::Box result){if(blockIdx.x||threadIdx.x)return;
 if(!valid_box(w.domain)||!identical_box(w.domain,right)||!identical_box(w.domain,result)){w.s->error=1;return;}
 if(!check_diagram(w.left,w.domain,w.s)||!check_diagram(w.right,w.domain,w.s)||!check_diagram(w.result,w.domain,w.s)){w.s->error=2;return;}
 for(U i=0;i<w.buckets;++i)w.heads[i]=none;
 insert(w,w.left.root,w.right.root,w.result.root);
}
__device__ U count(Diagram d,U id,unsigned dim){return level(d,id)==dim?d.n[id].arc_count:1;}
__device__ Arc cofactor_arc(Diagram g,U id,unsigned dim,U which,b::Box domain){
 if(level(g,id)==dim)return g.a[g.n[id].first_arc+which];
 Arc a{};a.child=id;if(dim<10){a.lo=domain.lo[dim];a.hi=domain.hi[dim];}else a.allowed=domain.allowed&(dim==10?b::wilderness:b::soil);return a;
}
__global__ void audit_triples(AuditWork w,U limit){if(blockIdx.x||threadIdx.x)return;auto&s=*w.s;
 for(U processed=0;processed<limit&&s.cursor<s.triples;++processed){Triple t=w.triples[s.cursor];
  unsigned dim=min(min(level(w.left,t.left),level(w.right,t.right)),level(w.result,t.result));
  if(dim==12){float expected=__fadd_rn(__uint_as_float(w.left.n[t.left].score_bits),__uint_as_float(w.right.n[t.right].score_bits));
   if(!isfinite(expected)||__float_as_uint(expected)!=w.result.n[t.result].score_bits){s.error=3;return;}++s.terminals;
  }else{
   // Deliberately different from producer representative lookup: enumerate and
   // intersect all three outgoing arc sets (identity arcs for skipped dimensions).
   U hits=0,na=count(w.left,t.left,dim),nb=count(w.right,t.right,dim),nr=count(w.result,t.result,dim);
   for(U a=0;a<na;++a)for(U b0=0;b0<nb;++b0)for(U r=0;r<nr;++r){
    // Bound attempted combinations, including empty intersections, not only hits.
    if(s.transitions==w.options.maximum_transitions){s.cap=2;return;}++s.transitions;
    Arc aa=cofactor_arc(w.left,t.left,dim,a,w.domain),bb=cofactor_arc(w.right,t.right,dim,b0,w.domain),rr=cofactor_arc(w.result,t.result,dim,r,w.domain);
    bool nonempty=dim<10?max(max(aa.lo,bb.lo),rr.lo)<=min(min(aa.hi,bb.hi),rr.hi):bool(aa.allowed&bb.allowed&rr.allowed);
    if(nonempty){++hits;if(!insert(w,aa.child,bb.child,rr.child))return;}
   }
   if(!hits){s.error=4;return;}
  }
  ++s.cursor;
 }
}
}
Audit audit(const Graph&a,const Graph&z,const Result&r,AuditOptions o,const Stop&stop,int device){
 validate_transport(a);validate_transport(z);validate_transport(r.graph);
 require(r.complete&&!r.operation_binding.empty()&&a.source_binding==z.source_binding&&a.source_binding==r.graph.source_binding&&a.domain_binding==z.domain_binding&&a.domain_binding==r.graph.domain_binding,"score-add audit source/domain/step binding");
 require(o.maximum_triples&&o.maximum_transitions&&o.items_per_launch&&o.maximum_device_bytes,"score-add audit options positive");
 U buckets=checked_add(checked_mul(o.maximum_triples,2),1),bytes=512;
 for(const Graph*g:{&a,&z,&r.graph}){bytes=checked_add(bytes,checked_mul(g->nodes.size(),sizeof(Node)));bytes=checked_add(bytes,checked_mul(g->arcs.size(),sizeof(Arc)));}
 bytes=checked_add(bytes,checked_mul(o.maximum_triples,sizeof(Triple)));bytes=checked_add(bytes,checked_mul(buckets,sizeof(U)));
 Audit out;if(bytes>o.maximum_device_bytes){out.reason="audit_device_budget";return out;}
 auto halted=[&]{return stop&&stop();};if(halted()){out.reason="cancelled_before_audit_allocation";return out;}
 cuda_check(cudaSetDevice(device));Memory memory{o.maximum_device_bytes};
 Storage<Node>an(memory,a.nodes.size()),zn(memory,z.nodes.size()),rn(memory,r.graph.nodes.size());
 Storage<Arc>aa(memory,a.arcs.size()),za(memory,z.arcs.size()),ra(memory,r.graph.arcs.size());
 Storage<Triple>triples(memory,o.maximum_triples);Storage<U>heads(memory,buckets);Storage<unsigned char>control(memory,512);
 an.upload(a.nodes.data());zn.upload(z.nodes.data());rn.upload(r.graph.nodes.data());aa.upload(a.arcs.data());za.upload(z.arcs.data());ra.upload(r.graph.arcs.data());
 AuditWork w{{an.p,aa.p,a.nodes.size(),a.arcs.size(),a.root},{zn.p,za.p,z.nodes.size(),z.arcs.size(),z.root},{rn.p,ra.p,r.graph.nodes.size(),r.graph.arcs.size(),r.graph.root},a.domain,triples.p,heads.p,buckets,reinterpret_cast<State*>(control.p),o};
 State s{};auto checked=[&]{cuda_check(cudaGetLastError());cuda_check(cudaMemcpy(&s,control.p,sizeof(s),cudaMemcpyDeviceToHost));if(s.error)throw std::runtime_error("CUDA independent score-add audit failed: "+std::to_string(s.error));};
 auto finish=[&](bool complete,const char*why){out.complete=complete;out.reason=why;out.triples=s.triples;out.transitions=s.transitions;out.terminal_word_checks=s.terminals;out.checked_nodes=s.nodes;out.owned_device_peak_bytes=memory.peak;return out;};
 audit_layout<<<1,1>>>(w,z.domain,r.graph.domain);out.CUDA_executed=true;checked();
 if(s.cap)return finish(false,"triple_cap");
 if(halted())return finish(false,"cancelled_after_audit_layout");
 while(s.cursor<s.triples){if(halted())return finish(false,"cancelled_before_triple_expansion");audit_triples<<<1,1>>>(w,o.items_per_launch);checked();
  if(s.cap)return finish(false,s.cap==1?"triple_cap":"transition_cap");}
 if(halted())return finish(false,"cancelled_before_audit_acceptance");
 require(s.terminals&&s.nodes==a.nodes.size()+z.nodes.size()+r.graph.nodes.size(),"score-add audit incomplete counters");
 return finish(true,"GPU_universal_triple_cofactor_exact_RN32_words_no_class_authority");
}
}
