#include "class_rank_gpu_dag_collect.hpp"
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <openssl/sha.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>
namespace rank_gpu_dag_collect {namespace {
using W=unsigned long long;constexpr W none=~W(0);
void need(bool x,const char*s){if(!x)throw std::invalid_argument(s);}
void ck(cudaError_t e,const char*s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
W add(W a,W b){need(a<=none-b,"collector extent overflow");return a+b;}
W mul(W a,W b){need(!b||a<=none/b,"collector extent overflow");return a*b;}
unsigned blocks(W n){return unsigned(std::max<W>(1,std::min<W>(65535,(n+255)/256)));}
template<class T>struct Buffer {
 T*p=nullptr;W n=0;explicit Buffer(W count):n(count){need(n<=SIZE_MAX/sizeof(T),"collector allocation overflow");if(n)ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"collector allocation");}
 ~Buffer(){if(p)cudaFree(p);}Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 void put(const T*x){if(n)ck(cudaMemcpy(p,x,n*sizeof(T),cudaMemcpyHostToDevice),"collector upload");}
 void zero(){if(n)ck(cudaMemset(p,0,n*sizeof(T)),"collector zero");}
 void absent(){if(n)ck(cudaMemset(p,0xff,n*sizeof(T)),"collector absent");}
 std::vector<T>get(){std::vector<T>x(n);if(n)ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"collector download");return x;}
};
struct State {W count=0,nodes=0,terms=0,edges=0;int bad=0;};
void word(std::string&s,W x){for(unsigned i=0;i<8;++i)s.push_back(char(x>>(8*i)));}
std::string digest(const std::string&s){unsigned char bytes[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(s.data()),s.size(),bytes);std::string r;constexpr char h[]="0123456789abcdef";for(auto b:bytes){r.push_back(h[b>>4]);r.push_back(h[b&15]);}return r;}
std::string payload_digest(const std::vector<dl::Node>&nodes,const std::vector<dl::Term>&terms,const std::vector<U>&roots,const std::string&binding){
 std::string s="DAG_COLLECT_PAYLOAD_V1";word(s,binding.size());s+=binding;word(s,nodes.size());word(s,terms.size());word(s,roots.size());
 for(const auto&n:nodes)for(W v:std::array<W,10>{n.id,n.left,n.right,n.first_term,n.term_count,std::uint32_t(n.label),std::uint32_t(n.kind),std::uint32_t(n.feature),n.cut_bits,n.threshold_bits})word(s,v);
 for(const auto&t:terms)for(W v:std::array<W,3>{t.id,std::uint32_t(t.feature),t.weight_bits})word(s,v);for(auto r:roots)word(s,r);return digest(s);
}
__device__ bool finite32(unsigned v){return (v&0x7f800000u)!=0x7f800000u;}
__device__ bool finite64(W v){return (v&0x7ff0000000000000ULL)!=0x7ff0000000000000ULL;}
__global__ void validate(const dl::Node*n,W count,const dl::Term*t,W nt,const W*roots,W nr,State*s){
 for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<max(max(count,nt),nr);i+=W(gridDim.x)*blockDim.x){
  if(i<nr&&roots[i]>=count)atomicExch(&s->bad,1);
  if(i<nt&&(t[i].id!=i||t[i].feature<0||t[i].feature>=54||!finite32(t[i].weight_bits)))atomicExch(&s->bad,1);
  if(i>=count)continue;const auto x=n[i];bool bad=x.id!=i||x.kind<0||x.kind>2;
  if(x.kind==2)bad|=x.label<0||x.label>=7||x.left!=none||x.right!=none||x.first_term||x.term_count||x.feature!=-1||x.cut_bits||x.threshold_bits;
  else{bad|=x.label!=-1||x.left>=i||x.right>=i;if(x.kind==0)bad|=x.feature<0||x.feature>=54||!finite32(x.cut_bits)||x.first_term||x.term_count||x.threshold_bits;
   else bad|=x.feature!=-1||x.cut_bits||!finite64(x.threshold_bits)||x.term_count<1||x.first_term>nt||x.term_count>nt-x.first_term;}
  if(bad)atomicExch(&s->bad,1);
 }
}
__global__ void seed(const W*roots,W nr,W*marked,W*root_flags,W*queue,State*s){for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<nr;i+=W(gridDim.x)*blockDim.x){const auto r=roots[i];atomicCAS(root_flags+r,W(0),i+1);if(atomicCAS(marked+r,W(0),W(1))==0)queue[atomicAdd(&s->count,W(1))]=r;}}
__global__ void descend(const dl::Node*nodes,const W*frontier,W count,W*marked,W*witness,W*next,State*s){
 for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=W(gridDim.x)*blockDim.x){const auto id=frontier[i];const auto n=nodes[id];if(n.kind==2)continue;atomicAdd(&s->edges,W(2));
  const W children[2]={n.left,n.right};for(const W c:children)if(atomicCAS(marked+c,W(0),W(1))==0){witness[c]=id;next[atomicAdd(&s->count,W(1))]=c;}
 }
}
__global__ void mark_terms(const dl::Node*nodes,W count,const W*marked,W*terms,W*owner){
 for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=W(gridDim.x)*blockDim.x)if(marked[i]){const auto n=nodes[i];for(W k=0;k<n.term_count;++k){const W t=n.first_term+k;if(atomicCAS(terms+t,W(0),W(1))==0)owner[t]=i;}}
}
__global__ void map_and_count(const W*flags,W*map,W n,State*s,int type){
 for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=W(gridDim.x)*blockDim.x){const W prefix=map[i];if(i+1==n){if(type==0)s->nodes=prefix+flags[i];else s->terms=prefix+flags[i];}if(!flags[i])map[i]=none;}
}
__global__ void scatter(const dl::Node*nodes,W n,const dl::Term*terms,W nt,const W*roots,W nr,const W*nm,const W*tm,dl::Node*outn,dl::Term*outt,W*outr){
 for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<max(max(n,nt),nr);i+=W(gridDim.x)*blockDim.x){
  if(i<nr)outr[i]=nm[roots[i]];
  if(i<nt&&tm[i]!=none){auto t=terms[i];t.id=tm[i];outt[t.id]=t;}
  if(i>=n||nm[i]==none)continue;auto x=nodes[i];x.id=nm[i];if(x.kind!=2){x.left=nm[x.left];x.right=nm[x.right];}if(x.term_count)x.first_term=tm[x.first_term];outn[x.id]=x;
 }
}
__device__ bool same_node(const dl::Node&a,const dl::Node&b){return a.id==b.id&&a.left==b.left&&a.right==b.right&&a.first_term==b.first_term&&a.term_count==b.term_count&&a.label==b.label&&a.kind==b.kind&&a.feature==b.feature&&a.cut_bits==b.cut_bits&&a.threshold_bits==b.threshold_bits;}
__global__ void audit(const dl::Node*nodes,W n,const dl::Term*terms,W nt,const W*roots,W nr,const W*marked,const W*root_flags,const W*witness,const W*termflags,const W*owner,const W*nm,const W*tm,const dl::Node*outn,W nn,const dl::Term*outt,W ntt,const W*outr,W*node_seen,W*term_seen,State*s){
 for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<max(max(n,nt),nr);i+=W(gridDim.x)*blockDim.x){bool bad=false;
  if(i<nr)bad|=!marked[roots[i]]||outr[i]!=nm[roots[i]]||outr[i]>=nn;
  if(i<nt){bad|=termflags[i]>1;if(!termflags[i])bad|=tm[i]!=none;else{const W o=owner[i];bad|=tm[i]>=ntt||o>=n;if(o<n){const auto x=nodes[o];bad|=!marked[o]||x.kind!=1||i<x.first_term||i-x.first_term>=x.term_count;}if(tm[i]<ntt){bad|=atomicCAS(term_seen+tm[i],W(0),W(1))!=0;const auto t=outt[tm[i]];bad|=t.id!=tm[i]||t.feature!=terms[i].feature||t.weight_bits!=terms[i].weight_bits;}}}
  if(i<n){bad|=marked[i]>1;if(!marked[i])bad|=nm[i]!=none;else{bad|=nm[i]>=nn;if(root_flags[i]){const W pos=root_flags[i]-1;bad|=pos>=nr;if(pos<nr)bad|=roots[pos]!=i;}else{const W p=witness[i];bad|=p<=i||p>=n;if(p<n)bad|=!marked[p]||nodes[p].kind==2||(nodes[p].left!=i&&nodes[p].right!=i);}
    auto x=nodes[i];x.id=nm[i];if(x.kind!=2){bad|=!marked[x.left]||!marked[x.right];x.left=nm[x.left];x.right=nm[x.right];bad|=x.left>=x.id||x.right>=x.id;}
    if(x.term_count){const W old=x.first_term;x.first_term=tm[old];for(W k=0;k<x.term_count;++k)bad|=!termflags[old+k]||tm[old+k]!=x.first_term+k;}
    if(nm[i]<nn){bad|=atomicCAS(node_seen+nm[i],W(0),W(1))!=0;bad|=!same_node(x,outn[nm[i]]);}
  }}if(bad)atomicExch(&s->bad,1);
}
}
__global__ void audit_surjective(const W*nodes,W n,const W*terms,W nt,State*s){for(W i=W(blockIdx.x)*blockDim.x+threadIdx.x;i<max(n,nt);i+=W(gridDim.x)*blockDim.x)if((i<n&&nodes[i]!=1)||(i<nt&&terms[i]!=1))atomicExch(&s->bad,1);}
}
Result collect(Input input,Options options,const Stop&stop,int device){
 Result out;out.binding=input.binding;need(!input.binding.empty(),"collector binding absent");
 const W n=input.nodes.size(),nt=input.terms.size(),nr=input.roots.size();
 out.work.stored_nodes_before=n;out.work.stored_terms_before=nt;
 need(options.maximum_nodes<=INT_MAX&&options.maximum_terms<=INT_MAX&&options.maximum_roots<=INT_MAX,"collector bounded scan capacities invalid");
 if(n>options.maximum_nodes||nt>options.maximum_terms||nr>options.maximum_roots){out.reason="collector_resident_capacity";return out;}
 // Peak includes both complete input and maximum-size compact output. CUB's
 // separately queried temporary storage is checked before its allocation.
 W extent=add(add(mul(n,2*sizeof(dl::Node)+7*sizeof(W)),mul(nt,2*sizeof(dl::Term)+4*sizeof(W))),add(mul(nr,2*sizeof(W)),sizeof(State)));
 if(extent>options.maximum_device_bytes){out.reason="collector_device_budget";return out;}
 if(stop&&stop()){out.reason="cancelled_before_collection";return out;}
 out.input_sha256=payload_digest(input.nodes,input.terms,input.roots,input.binding);
 ck(cudaSetDevice(device),"collector device");
 Buffer<dl::Node>nodes(n);Buffer<dl::Term>terms(nt);Buffer<W>roots(nr),flags(n),root_flags(n),witness(n),qa(n),qb(n),nm(n),tf(nt),owner(nt),tm(nt);Buffer<State>state(1);
 nodes.put(input.nodes.data());terms.put(input.terms.data());roots.put(reinterpret_cast<const W*>(input.roots.data()));flags.zero();root_flags.zero();witness.absent();tf.zero();owner.absent();state.zero();
 validate<<<blocks(std::max({n,nt,nr})),256>>>(nodes.p,n,terms.p,nt,roots.p,nr,state.p);ck(cudaGetLastError(),"collector validation launch");need(!state.get()[0].bad,"CUDA collector invalid topology/payload/root");out.CUDA_executed=true;
 if(stop&&stop()){out.reason="cancelled_after_collection_validation";return out;}
 seed<<<blocks(nr),256>>>(roots.p,nr,flags.p,root_flags.p,qa.p,state.p);ck(cudaGetLastError(),"collector root marking");auto st=state.get()[0];W pending=st.count;W*current=qa.p,*next=qb.p;
 while(pending){if(stop&&stop()){out.reason="cancelled_during_reachability";return out;}state.zero();descend<<<blocks(pending),256>>>(nodes.p,current,pending,flags.p,witness.p,next,state.p);ck(cudaGetLastError(),"collector branch marking");st=state.get()[0];need(st.count<=n,"collector frontier overflow");pending=st.count;out.work.visited_edges=add(out.work.visited_edges,st.edges);++out.work.frontier_rounds;std::swap(current,next);}
 mark_terms<<<blocks(n),256>>>(nodes.p,n,flags.p,tf.p,owner.p);ck(cudaGetLastError(),"collector term marking");
 std::size_t scratch=0,a=0,b=0;if(n)ck(cub::DeviceScan::ExclusiveSum(nullptr,a,flags.p,nm.p,int(n)),"collector node scan sizing");if(nt)ck(cub::DeviceScan::ExclusiveSum(nullptr,b,tf.p,tm.p,int(nt)),"collector term scan sizing");scratch=std::max(a,b);
 if(add(extent,scratch)>options.maximum_device_bytes){out.reason="collector_scan_storage_budget";return out;}Buffer<unsigned char>temp(scratch);
 if(n){auto available=scratch;ck(cub::DeviceScan::ExclusiveSum(temp.p,available,flags.p,nm.p,int(n)),"collector node scan");}if(nt){auto available=scratch;ck(cub::DeviceScan::ExclusiveSum(temp.p,available,tf.p,tm.p,int(nt)),"collector term scan");}
 state.zero();map_and_count<<<blocks(n),256>>>(flags.p,nm.p,n,state.p,0);map_and_count<<<blocks(nt),256>>>(tf.p,tm.p,nt,state.p,1);ck(cudaGetLastError(),"collector map construction");st=state.get()[0];need(st.nodes<=n&&st.terms<=nt,"collector count overflow");
 if(stop&&stop()){out.reason="cancelled_before_compact_copy";return out;}
 Buffer<dl::Node>newnodes(st.nodes);Buffer<dl::Term>newterms(st.terms);Buffer<W>newroots(nr),node_seen(st.nodes),term_seen(st.terms);node_seen.zero();term_seen.zero();
 // scatter writes semantic Node members; define all copied object bytes,
 // including padding, before transporting the compact output.
 newnodes.zero();
 scatter<<<blocks(std::max({n,nt,nr})),256>>>(nodes.p,n,terms.p,nt,roots.p,nr,nm.p,tm.p,newnodes.p,newterms.p,newroots.p);ck(cudaGetLastError(),"collector compact copy");
 audit<<<blocks(std::max({n,nt,nr})),256>>>(nodes.p,n,terms.p,nt,roots.p,nr,flags.p,root_flags.p,witness.p,tf.p,owner.p,nm.p,tm.p,newnodes.p,st.nodes,newterms.p,st.terms,newroots.p,node_seen.p,term_seen.p,state.p);ck(cudaGetLastError(),"collector independent closure/minimality/remap audit");
 audit_surjective<<<blocks(std::max(st.nodes,st.terms)),256>>>(node_seen.p,st.nodes,term_seen.p,st.terms,state.p);ck(cudaGetLastError(),"collector compact namespace audit");need(!state.get()[0].bad,"CUDA collector structural audit failed");
 auto ns=newnodes.get();auto ts=newterms.get();auto rs=newroots.get();auto nmap=nm.get();auto tmap=tm.get();
 if(stop&&stop()){out.reason="cancelled_before_collection_publication";return out;}
 out.nodes=std::move(ns);out.terms=std::move(ts);out.roots.assign(rs.begin(),rs.end());out.node_map.assign(nmap.begin(),nmap.end());out.term_map.assign(tmap.begin(),tmap.end());
 out.output_sha256=payload_digest(out.nodes,out.terms,out.roots,input.binding);out.work.retained_nodes=st.nodes;out.work.retained_terms=st.terms;out.complete=true;out.structural_verified=true;out.reason="exact_reachable_subgraph_copied_into_fresh_namespace";return out;
}
}
