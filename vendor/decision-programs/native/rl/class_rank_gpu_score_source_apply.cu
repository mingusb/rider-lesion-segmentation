#include "class_rank_gpu_score_source_apply.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
namespace rank_gpu_score_source_apply {namespace {
using Factor=rank_gpu_score_factors::Factor;using Box=aa::b::Box;using Node=aa::Node;using Arc=aa::Arc;
void need(bool b,const char*s){if(!b)throw std::invalid_argument(s);}void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U add(U a,U b){need(b<=aa::none-a,"source Apply bytes overflow");return a+b;}U mul(U a,U b){need(!b||a<=aa::none/b,"source Apply bytes product overflow");return a*b;}
struct Budget{U cap,used=0,peak=0;void take(U n){need(n<=cap-used,"source Apply device budget");used+=n;peak=std::max(peak,used);}};
template<class T>struct Buf{Budget&b;U n,bytes;T*p=nullptr;Buf(Budget&a,U size):b(a),n(size),bytes(mul(size,sizeof(T))){b.take(bytes);if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){b.used-=bytes;cu(e);}e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);p=nullptr;b.used-=bytes;cu(e);}}}~Buf(){if(p)cudaFree(p);b.used-=bytes;}Buf(const Buf&)=delete;void put(const T*x){if(bytes)cu(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>x(n);if(bytes)cu(cudaMemcpy(x.data(),p,bytes,cudaMemcpyDeviceToHost));return x;}};
unsigned blocks(U n){need(n&&n<=U(INT_MAX)*128,"source Apply launch count");return unsigned((n+127)/128);}
struct Status{U nodes=0,arcs=0,cells=0,words=0;int bad=0,cap=0;};
__global__ void source_partitions(const aa::b::Leaf*leaves,U n,const float*bias,Box domain,bool is_bias,Factor*out,Status*s){
 if(blockIdx.x||threadIdx.x)return;
 if(is_bias){for(unsigned c=0;c<7;++c){if(!isfinite(bias[c])){s->bad=1;return;}Factor f{};f.box=domain;f.channel=c;f.score_bits=__float_as_uint(bias[c]);out[c]=f;}}
 else{for(U i=0;i<n;++i){if(!isfinite(leaves[i].value)||leaves[i].ordinal<0){s->bad=2;return;}Factor f{};f.box=leaves[i].box;f.score_bits=__float_as_uint(leaves[i].value);f.channel=0;f.combination=U(leaves[i].ordinal);out[i]=f;}
  for(unsigned c=1;c<7;++c){Factor f{};f.box=domain;f.channel=c;f.score_bits=0;out[n+c-1]=f;}}
}
// Independent row-by-row source-word/rectangle transport check before the
// ordered-DD engine sees these factors; no trust in producer copy counters.
__global__ void partition_transport_check(const aa::b::Leaf*leaves,U n,const float*bias,Box domain,bool is_bias,const Factor*f,U count,Status*s){U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;Box expected;unsigned word=0,channel=0;U ordinal=0;
 if(is_bias){expected=domain;channel=unsigned(i);word=__float_as_uint(bias[i]);}
 else if(i<n){expected=leaves[i].box;word=__float_as_uint(leaves[i].value);ordinal=U(leaves[i].ordinal);}
 else{expected=domain;channel=unsigned(i-n+1);}
 const auto&a=f[i];bool equal=a.score_bits==word&&a.channel==channel&&a.combination==ordinal&&a.box.allowed==expected.allowed;for(int d=0;d<10;++d)equal=equal&&a.box.lo[d]==expected.lo[d]&&a.box.hi[d]==expected.hi[d];if(!equal){atomicCAS(&s->bad,0,3);return;}atomicAdd(reinterpret_cast<unsigned long long*>(&s->nodes),1ull);
}
struct Descriptor{U node_base=0,nodes=0,arc_base=0,arcs=0,root=0;};
__device__ unsigned owner_node(const Descriptor*d,U i){for(unsigned c=0;c<7;++c)if(i>=d[c].node_base&&i-d[c].node_base<d[c].nodes)return c;return 7;}
__device__ unsigned owner_arc(const Descriptor*d,U i){for(unsigned c=0;c<7;++c)if(i>=d[c].arc_base&&i-d[c].arc_base<d[c].arcs)return c;return 7;}
__global__ void pack_copy(const Descriptor*d,const Node*in,const Arc*ia,U nn,U na,Node*out,Arc*oa,U*roots,Status*s){U i=U(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<7)roots[i]=d[i].node_base+d[i].root;
 if(i<nn){unsigned c=owner_node(d,i);if(c==7){atomicCAS(&s->bad,0,1);return;}Node n=in[i];n.first_arc+=d[c].arc_base;n.hash_next=aa::none;out[i]=n;}
 if(i<na){unsigned c=owner_arc(d,i);if(c==7){atomicCAS(&s->bad,0,2);return;}Arc a=ia[i];a.child+=d[c].node_base;oa[i]=a;}}
// Independent exact transport audit against the pre-remap pools, not the output
// generator's counters. Unused retained nodes/arcs are checked as well as roots.
__global__ void pack_check(const Descriptor*d,const Node*in,const Arc*ia,U nn,U na,const Node*out,const Arc*oa,const U*roots,Status*s){U i=U(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<7&&(d[i].root>=d[i].nodes||roots[i]<d[i].node_base||roots[i]-d[i].node_base!=d[i].root))atomicCAS(&s->bad,0,3);
 if(i<nn){unsigned c=owner_node(d,i);if(c==7){atomicCAS(&s->bad,0,4);return;}const auto&a=in[i];const auto&b=out[i];
  if(a.kind!=b.kind||a.dimension!=b.dimension||a.score_bits!=b.score_bits||a.reserved!=b.reserved||a.arc_count!=b.arc_count||b.first_arc<d[c].arc_base||b.first_arc-d[c].arc_base!=a.first_arc||b.hash_next!=aa::none||b.first_arc>na||b.arc_count>na-b.first_arc){atomicCAS(&s->bad,0,5);return;}
  for(U k=0;k<b.arc_count;++k){U child=oa[b.first_arc+k].child;if(child>=i||(out[child].kind&&out[child].dimension<=b.dimension)){atomicCAS(&s->bad,0,6);return;}}
  atomicAdd(reinterpret_cast<unsigned long long*>(&s->nodes),1ull);}
 if(i<na){unsigned c=owner_arc(d,i);if(c==7){atomicCAS(&s->bad,0,7);return;}const auto&a=ia[i];const auto&b=oa[i];if(a.lo!=b.lo||a.hi!=b.hi||a.allowed!=b.allowed||b.child<d[c].node_base||b.child-d[c].node_base!=a.child||a.child>=d[c].nodes){atomicCAS(&s->bad,0,8);return;}atomicAdd(reinterpret_cast<unsigned long long*>(&s->arcs),1ull);}}
__device__ bool same_box(const Box&a,const Box&b){if(a.allowed!=b.allowed)return false;for(int d=0;d<10;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])return false;return true;}
__global__ void quotient_count(Box a,Box b,U cap,Status*s){if(blockIdx.x||threadIdx.x)return;if(!same_box(a,b)||(a.allowed&~((U(1)<<44)-1))||!(a.allowed&aa::b::wilderness)||!(a.allowed&aa::b::soil)){s->bad=1;return;}U n=1;
 for(int d=0;d<12;++d){if(d<10&&(a.lo[d]<0||a.hi[d]<a.lo[d]||a.hi[d]>16777216)){s->bad=2;return;}U k=d<10?U(a.hi[d]-a.lo[d])+1:U(__popcll(a.allowed&(d==10?aa::b::wilderness:aa::b::soil)));if(n>cap/k){s->cap=1;return;}n*=k;}s->cells=n;}
__device__ U category(U mask,U at){while(at--)mask&=mask-1;return mask&(~mask+1);}
__global__ void quotient_words(Box domain,const aa::b::Leaf*leaves,const int*offsets,const int*channels,U nt,const float*bias,const Node*nodes,U nn,const Arc*arcs,U na,const U*roots,U base,U count,Status*s){U thread=U(blockIdx.x)*blockDim.x+threadIdx.x;if(thread>=count*7)return;unsigned channel=unsigned(thread%7);U id=base+thread/7;int rank[10];U cats=0;
 for(int d=11;d>=0;--d){U k=d<10?U(domain.hi[d]-domain.lo[d])+1:U(__popcll(domain.allowed&(d==10?aa::b::wilderness:aa::b::soil))),v=id%k;id/=k;if(d<10)rank[d]=domain.lo[d]+int(v);else cats|=category(domain.allowed&(d==10?aa::b::wilderness:aa::b::soil),v);}
 float expected=bias[channel];if(!isfinite(expected)){atomicCAS(&s->bad,0,3);return;}
 for(U t=0;t<nt;++t)if(channels[t]==int(channel)){U hits=0;float value=0;for(int k=offsets[t];k<offsets[t+1];++k){const auto&l=leaves[k];bool hit=(l.box.allowed&cats)==cats;for(int d=0;d<10;++d)hit=hit&&rank[d]>=l.box.lo[d]&&rank[d]<=l.box.hi[d];if(hit){++hits;value=l.value;}}if(hits!=1){atomicCAS(&s->bad,0,4);return;}expected=__fadd_rn(expected,value);if(!isfinite(expected)){atomicCAS(&s->bad,0,5);return;}}
 U at=roots[channel];int previous=-1;for(unsigned step=0;step<14;++step){if(at>=nn){atomicCAS(&s->bad,0,6);return;}const auto&n=nodes[at];if(!n.kind){if(n.dimension!=12||n.arc_count||n.score_bits!=__float_as_uint(expected)){atomicCAS(&s->bad,0,7);return;}atomicAdd(reinterpret_cast<unsigned long long*>(&s->words),1ull);return;}
  if(n.kind!=1||n.dimension>=12||int(n.dimension)<=previous||n.first_arc>na||n.arc_count>na-n.first_arc){atomicCAS(&s->bad,0,8);return;}U next=aa::none,hits=0;for(U k=0;k<n.arc_count;++k){const auto&a=arcs[n.first_arc+k];bool hit=n.dimension<10?rank[n.dimension]>=a.lo&&rank[n.dimension]<=a.hi:bool(cats&a.allowed);if(hit){next=a.child;++hits;}}if(hits!=1||next>=at){atomicCAS(&s->bad,0,9);return;}previous=int(n.dimension);at=next;}
 atomicCAS(&s->bad,0,10);
}
}
namespace detail {
PartitionResult partition(const lp::Result&p,U t,bool bias,Options o,const Stop&stop,int device){
 PartitionResult out;const auto&source=p.source.value;U begin=bias?0:U(source.offsets[t]),n=bias?0:U(source.offsets[t+1])-begin,nf=bias?7:add(n,6);out.leaf_count=n;
 U plan=add(add(mul(n,sizeof(aa::b::Leaf)),mul(nf,sizeof(Factor))),add(7*sizeof(float),sizeof(Status)));
 if(plan>o.maximum_device_bytes){out.reason="partition_transport_device_cap";return out;}auto halt=[&]{return stop&&stop();};if(halt()){out.reason="cancelled_before_partition_transport";return out;}
 sd::Input input;input.domain=p.domain;input.domain_binding=p.rank_sha256;input.factors.source_binding=p.source.binding;input.factors.complete=true;input.factors.compatible=nf;input.factors.combinations=nf;input.factors.factor_offsets[0]=0;
 for(int c=0;c<7;++c)input.factors.factor_offsets[c+1]=bias?U(c+1):(c?add(n,U(c)):n);
 {cu(cudaSetDevice(device));Budget budget{o.maximum_device_bytes};Buf<aa::b::Leaf>leaves(budget,n);Buf<float>values(budget,7);Buf<Factor>factors(budget,nf);Buf<Status>state(budget,1);if(n)leaves.put(source.leaves.data()+begin);values.put(source.bias.data());source_partitions<<<1,1>>>(leaves.p,n,values.p,p.domain,bias,factors.p,state.p);out.CUDA_executed=true;cu(cudaGetLastError());auto status=state.get()[0];need(!status.bad,"source Apply nonfinite leaf/bias");partition_transport_check<<<blocks(nf),128>>>(leaves.p,n,values.p,p.domain,bias,factors.p,nf,state.p);cu(cudaGetLastError());status=state.get()[0];need(!status.bad&&status.nodes==nf,"independent source partition transport audit failed");out.transport_rows=status.nodes;input.factors.factors=factors.get();out.owned_device_peak_bytes=budget.peak;}
 if(halt()){out.reason="cancelled_after_partition_transport";return out;}auto tree_options=o.tree;tree_options.maximum_device_bytes=std::min(tree_options.maximum_device_bytes,o.maximum_device_bytes);out.diagrams=sd::construct(input,tree_options,stop,device);out.CUDA_executed|=out.diagrams.CUDA_executed;out.owned_device_peak_bytes=std::max(out.owned_device_peak_bytes,out.diagrams.owned_device_peak_bytes);
 if(!out.diagrams.complete){out.reason="diagram_"+out.diagrams.reason;return out;}auto audit_options=o.tree_audit;audit_options.maximum_device_bytes=std::min(audit_options.maximum_device_bytes,o.maximum_device_bytes);out.audit=sd::audit_factors(input,out.diagrams,audit_options,stop,device);out.CUDA_executed|=out.audit.CUDA_executed;out.owned_device_peak_bytes=std::max(out.owned_device_peak_bytes,out.audit.owned_device_peak_bytes);
 if(!out.audit.complete){out.reason="factor_audit_"+out.audit.reason;return out;}if(halt()){out.reason="cancelled_after_tree_congruence";return out;}out.complete=true;out.reason="complete_GPU_partition_diagram_universal_words";return out;
}
Packed pack(const std::array<aa::Graph,7>&g,Options o,const Stop&stop,int device){
 Packed out;std::array<Descriptor,7>d{};U nn=0,na=0;for(int c=0;c<7;++c){aa::validate_transport(g[c]);need(g[c].source_binding==g[0].source_binding&&g[c].domain_binding==g[0].domain_binding&&g[c].domain.lo==g[0].domain.lo&&g[c].domain.hi==g[0].domain.hi&&g[c].domain.allowed==g[0].domain.allowed,"packing graph domain/source mismatch");d[c]={nn,g[c].nodes.size(),na,g[c].arcs.size(),g[c].root};nn=add(nn,g[c].nodes.size());na=add(na,g[c].arcs.size());}
 if(nn>o.maximum_packed_nodes||na>o.maximum_packed_arcs){out.reason="packed_graph_capacity";return out;}U plan=add(add(mul(nn,2*sizeof(Node)),mul(na,2*sizeof(Arc))),7*sizeof(Descriptor)+7*sizeof(U)+sizeof(Status));if(plan>o.maximum_device_bytes){out.reason="packed_device_capacity";return out;}auto halt=[&]{return stop&&stop();};if(halt()){out.reason="cancelled_before_pack";return out;}
 std::vector<Node>nodes;std::vector<Arc>arcs;nodes.reserve(nn);arcs.reserve(na);for(const auto&a:g){nodes.insert(nodes.end(),a.nodes.begin(),a.nodes.end());arcs.insert(arcs.end(),a.arcs.begin(),a.arcs.end());}
 cu(cudaSetDevice(device));Budget budget{o.maximum_device_bytes};Buf<Descriptor>desc(budget,7);Buf<Node>in(budget,nn),result(budget,nn);Buf<Arc>ia(budget,na),ra(budget,na);Buf<U>roots(budget,7);Buf<Status>status(budget,1);desc.put(d.data());in.put(nodes.data());ia.put(arcs.data());U launch=std::max(U(7),std::max(nn,na));pack_copy<<<blocks(launch),128>>>(desc.p,in.p,ia.p,nn,na,result.p,ra.p,roots.p,status.p);out.CUDA_executed=true;cu(cudaGetLastError());auto s=status.get()[0];need(!s.bad,"source score packing generation failed");out.owned_device_peak_bytes=budget.peak;
 if(halt()){out.reason="cancelled_after_pack_copy";return out;}pack_check<<<blocks(launch),128>>>(desc.p,in.p,ia.p,nn,na,result.p,ra.p,roots.p,status.p);cu(cudaGetLastError());s=status.get()[0];need(!s.bad&&s.nodes==nn&&s.arcs==na,"source score independent packing audit failed");out.audited_nodes=s.nodes;out.audited_arcs=s.arcs;if(halt()){out.reason="cancelled_after_pack_audit";return out;}
 auto&r=out.scores;r.complete=true;r.CUDA_executed=true;r.source_binding=g[0].source_binding;r.domain_binding=g[0].domain_binding;r.domain=g[0].domain;r.nodes=result.get();r.arcs=ra.get();auto rt=roots.get();std::copy(rt.begin(),rt.end(),r.roots.begin());r.counters.nodes=nn;r.counters.arcs=na;r.owned_device_peak_bytes=budget.peak;r.reason="GPU_exact_seven_pool_pack_independently_audited";sd::validate_transport(r);out.complete=true;out.reason=r.reason;return out;
}
}
QuotientAudit audit_quotient(const lp::Result&p,const sd::Result&r,U cap,U batch,U bytecap,const Stop&stop,int device){
 validate_source_transport(p);sd::validate_transport(r);need(r.source_binding==p.source.binding&&r.domain_binding==p.rank_sha256&&cap&&batch&&batch<=1048576&&bytecap,"source quotient binding/options");QuotientAudit out;auto halt=[&]{return stop&&stop();};if(bytecap<sizeof(Status)){out.reason="audit_device_cap";return out;}if(halt()){out.reason="cancelled_before_quotient";return out;}cu(cudaSetDevice(device));Budget budget{bytecap};Buf<Status>status(budget,1);quotient_count<<<1,1>>>(p.domain,r.domain,cap,status.p);out.CUDA_executed=true;cu(cudaGetLastError());auto s=status.get()[0];need(!s.bad,"source quotient domain invalid");out.owned_device_peak_bytes=budget.peak;if(s.cap){out.reason="source_quotient_cell_cap";return out;}
 const auto&src=p.source.value;U bytes=sizeof(Status);bytes=add(bytes,mul(src.leaves.size(),sizeof(aa::b::Leaf)));bytes=add(bytes,mul(add(src.offsets.size(),src.channels.size()),sizeof(int)));bytes=add(bytes,7*sizeof(float)+7*sizeof(U));bytes=add(bytes,mul(r.nodes.size(),sizeof(Node)));bytes=add(bytes,mul(r.arcs.size(),sizeof(Arc)));if(bytes>bytecap){out.reason="audit_device_cap";return out;}
 Buf<aa::b::Leaf>leaves(budget,src.leaves.size());Buf<int>offsets(budget,src.offsets.size()),channels(budget,src.channels.size());Buf<float>bias(budget,7);Buf<Node>nodes(budget,r.nodes.size());Buf<Arc>arcs(budget,r.arcs.size());Buf<U>roots(budget,7);leaves.put(src.leaves.data());offsets.put(src.offsets.data());channels.put(src.channels.data());bias.put(src.bias.data());nodes.put(r.nodes.data());arcs.put(r.arcs.data());roots.put(r.roots.data());out.owned_device_peak_bytes=budget.peak;
 for(U base=0;base<s.cells;){if(halt()){out.reason="cancelled_during_quotient";return out;}U n=std::min(batch,s.cells-base);quotient_words<<<blocks(mul(n,7)),128>>>(p.domain,leaves.p,offsets.p,channels.p,src.channels.size(),bias.p,nodes.p,r.nodes.size(),arcs.p,r.arcs.size(),roots.p,base,n,status.p);cu(cudaGetLastError());auto observed=status.get()[0];need(!observed.bad,"independent GPU direct ordered source quotient differs");s=observed;base+=n;}
 if(halt()){out.reason="cancelled_before_quotient_acceptance";return out;}need(s.words==mul(s.cells,7),"source quotient incomplete word count");out.complete=true;out.cells=s.cells;out.score_words=s.words;out.reason="GPU_complete_direct_source_ordered_RN32_quotient_equal";return out;
}
}