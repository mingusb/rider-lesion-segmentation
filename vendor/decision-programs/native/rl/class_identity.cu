#include "class_identity.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include "class_rank_gpu_score_pool_identity.hpp"
namespace rl_qualified_session::detail {
namespace {
namespace ca=rank_gpu_class_apply;
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U add(U a,U b){need(b<=UINT64_MAX-a,"identity size sum overflow");return a+b;}
U mul(U a,U b){need(!b||a<=UINT64_MAX/b,"identity size product overflow");return a*b;}
struct Budget{U cap,used=0,peak=0;void take(U n){need(n<=cap-used,"identity device cap");used+=n;peak=std::max(peak,used);}};
template<class T>struct Buf{
  Budget&b;U n,bytes;T*p=nullptr;
  Buf(Budget&owner,U count):b(owner),n(count),bytes(mul(count,sizeof(T))){b.take(bytes);if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){b.used-=bytes;cu(e);}e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);b.used-=bytes;cu(e);}}}
  ~Buf(){if(p)cudaFree(p);b.used-=bytes;}
  void put(const T*x){if(bytes)cu(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}
  std::vector<T>get()const{std::vector<T>x(n);if(bytes)cu(cudaMemcpy(x.data(),p,bytes,cudaMemcpyDeviceToHost));return x;}
};
unsigned blocks(U n){need(n&&n<=U(INT_MAX)*128,"identity launch extent");return unsigned((n+127)/128);}
struct Layout{U root,nn,na,ns,ne,nc,total;ca::sd::b::Box domain;U cuts[10]{};};
__global__ void pack(Layout q,const ca::Node*n,const ca::Arc*a,const ca::State*s,
                     const ca::Arc*e,const unsigned*c,U*out){
  U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=q.total)return;
  U at=i,value=0;
  if(at<37){
    if(at==0)value=q.root;
    else if(at<=10)value=unsigned(q.domain.lo[at-1]);
    else if(at<=20)value=unsigned(q.domain.hi[at-11]);
    else if(at==21)value=q.domain.allowed;
    else if(at==22)value=q.nn;else if(at==23)value=q.na;
    else if(at==24)value=q.ns;else if(at==25)value=q.ne;
    else if(at==26)value=q.nc;else value=q.cuts[at-27];
  }else{
    at-=37;
    if(at<7*q.nn){const auto&v=n[at/7];switch(at%7){case 0:value=v.kind;break;case 1:value=v.dimension;break;case 2:value=v.score_bits;break;case 3:value=v.reserved;break;case 4:value=v.first_arc;break;case 5:value=v.arc_count;break;default:value=v.hash_next;}}
    else{at-=7*q.nn;
      if(at<4*q.na){const auto&v=a[at/4];switch(at%4){case 0:value=unsigned(v.lo);break;case 1:value=unsigned(v.hi);break;case 2:value=v.allowed;break;default:value=v.child;}}
      else{at-=4*q.na;
        if(at<32*q.ns){const auto&v=s[at/32];U k=at%32;
          if(k<7)value=v.tuple[k];else if(k<17)value=unsigned(v.witness.rank[k-7]);
          else if(k==17)value=v.witness.categories;else if(k==18)value=v.first_edge;
          else if(k==19)value=v.edge_count;else if(k==20)value=v.result;
          else if(k==21)value=v.hash_next;else if(k<29)value=v.probability_bits[k-22];
          else if(k==29)value=v.level;else if(k==30)value=v.status;else value=unsigned(v.label);
        }else{at-=32*q.ns;
          if(at<4*q.ne){const auto&v=e[at/4];switch(at%4){case 0:value=unsigned(v.lo);break;case 1:value=unsigned(v.hi);break;case 2:value=v.allowed;break;default:value=v.child;}}
          else value=c[at-4*q.ne];
        }
      }
    }
  }
  out[i]=value;
}
__global__ void compare(const U*a,const U*b,U n,int*error){
  U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n&&a[i]!=b[i])atomicCAS(error,0,1);
}
__global__ void policy_order(const unsigned*expected,const rl_category_policy::Trace*trace,U version,int*error){
  unsigned k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=44)return;
  unsigned group=k<4?0:1,local=k<4?k:k-4;
  const auto&t=trace[group];
  if(t.version!=version||t.group!=group||t.steps!=(group?40u:4u)||t.fallback_mask||
     t.seed!=trace[0].seed||t.episode!=trace[0].episode||
     t.allowed_mask!=(group?rl_category_policy::soil_mask:rl_category_policy::wild_mask)||
     t.actions[local]!=expected[k])atomicCAS(error,0,2);
}
__global__ void choose_cost(U candidate,U incumbent,int*better){if(!threadIdx.x&&!blockIdx.x)*better=candidate<incumbent;}
__global__ void effective_order_validate(const unsigned*order,int*error){if(blockIdx.x||threadIdx.x)return;
  U seen=0;for(unsigned k=0;k<44;++k){unsigned bit=order[k];if(bit>=44||(k<4?bit>=4:bit<4)||(seen&(U(1)<<bit))){*error=3;return;}seen|=U(1)<<bit;}
  if(seen!=((U(1)<<44)-1))*error=3;
}
__global__ void effective_order_same(const unsigned*a,const unsigned*b,int*error){unsigned k=threadIdx.x;if(k<44&&a[k]!=b[k])atomicCAS(error,0,4);}
__global__ void effective_key(const U*source,U nn,const unsigned*order,bool injected,U*out,int*error){
  if(*error)return;
  U id=U(blockIdx.x)*blockDim.x+threadIdx.x;
  if(id==0){out[0]=source[0];out[1]=nn;}
  if(id>=nn)return;U*key=out+2+45*id;
  key[0]=id;key[1]=12;key[2]=UINT64_MAX;key[3]=0;key[4]=0;
  for(unsigned k=0;k<40;++k)key[5+k]=UINT_MAX;
  const U*node=source+37+7*id;U kind=node[0],dim=node[1],first=node[4],count=node[5];
  if(!kind||dim<10)return;
  U na=source[23];if(kind!=1||dim>=12||count<2||first>na||count>na-first){atomicCAS(error,0,5);return;}
  const U*arcs=source+37+7*nn;U fallback=0,seen=0;int largest=-1;
  U group=dim==10?15:((U(1)<<44)-1)^U(15),allowed=source[21]&group;
  for(U j=0;j<count;++j){const U*arc=arcs+4*(first+j);U mask=arc[2];
    if(!mask||(mask&~allowed)||(mask&seen)||arc[3]>=id){atomicCAS(error,0,6);return;}seen|=mask;
    int size=__popcll(mask);if(size>largest){largest=size;fallback=j;}
  }
  if(seen!=allowed){atomicCAS(error,0,7);return;}
  U selected=allowed&~arcs[4*(first+fallback)+2];unsigned at=0;
  key[1]=dim;key[2]=fallback;key[3]=selected;key[4]=__popcll(selected);
  if(injected){unsigned begin=dim==10?0:4,end=dim==10?4:44;
    for(unsigned k=begin;k<end;++k){if(order[k]>=44){atomicCAS(error,0,3);return;}if(selected&(U(1)<<order[k]))key[5+at++]=order[k];}
  }else{
    // Default lower_build emits arcs descending and bits ascending. Its root
    // chain therefore visits arcs ascending and bits descending within each.
    for(U j=0;j<count;++j)if(j!=fallback){U mask=arcs[4*(first+j)+2];
      while(mask){unsigned bit=63-__clzll(mask);mask^=U(1)<<bit;key[5+at++]=bit;}
    }
  }
  if(at!=key[4]||at>40)atomicCAS(error,0,8);
}
__global__ void effective_find(const U*current,const U*const*keys,U entries,U words,U*match){
  U entry=blockIdx.x;if(entry>=entries)return;__shared__ int different;
  if(!threadIdx.x)different=0;__syncthreads();
  const U*key=keys[entry];for(U k=threadIdx.x;k<words;k+=blockDim.x)if(current[k]!=key[k])atomicExch(&different,1);
  __syncthreads();if(!threadIdx.x&&!different)atomicMin(reinterpret_cast<unsigned long long*>(match),static_cast<unsigned long long>(entry));
}
__global__ void effective_trace(const U*key,const rl_category_lowering::Trace*trace,U nn,int*error){
  U id=U(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=nn)return;const U*k=key+2+45*id;const auto&t=trace[id];
  if(t.source_node!=k[0]||t.dimension!=k[1]||t.fallback_arc!=k[2]||t.selected_mask!=k[3]||t.count!=k[4]){atomicCAS(error,0,9);return;}
  for(unsigned j=0;j<40;++j)if(t.test_order[j]!=k[5+j])atomicCAS(error,0,10);
}
__global__ void cache_fixture(ca::Node*n,ca::Arc*a,unsigned*orders){if(blockIdx.x||threadIdx.x)return;
  for(unsigned k=0;k<6;++k){n[k]={};n[k].dimension=12;n[k].score_bits=k%3;}
  U soil=((U(1)<<44)-1)^U(15);U six=U(1)<<6,thirteen=U(1)<<13,fourteen=U(1)<<14;
  n[3].kind=1;n[3].dimension=11;n[3].score_bits=0;n[3].first_arc=0;n[3].arc_count=3;
  a[0]={0,0,soil&~(six|thirteen),2};a[1]={0,0,six,0};a[2]={0,0,thirteen,1};
  n[4].kind=1;n[4].dimension=11;n[4].score_bits=0;n[4].first_arc=3;n[4].arc_count=3;
  a[3]={0,0,soil&~(thirteen|fourteen),0};a[4]={0,0,thirteen,1};a[5]={0,0,fourteen,2};
  n[5].kind=1;n[5].dimension=0;n[5].score_bits=0;n[5].first_arc=6;n[5].arc_count=2;
  a[6]={0,0,0,3};a[7]={1,1,0,4};
  for(unsigned variant=0;variant<5;++variant)for(unsigned k=0;k<44;++k)orders[variant*44+k]=k;
  // B changes only categories inactive in both local selected masks.
  orders[44+0]=3;orders[44+3]=0;orders[44+4]=5;orders[44+5]=4;orders[44+7]=8;orders[44+8]=7;
  orders[88+6]=13;orders[88+13]=6; // Bit6 is fallback in node4 but active in node3.
  orders[132+13]=14;orders[132+14]=13;
  orders[176+6]=14;orders[176+13]=13;orders[176+14]=6;
}
// Literal contract-only fixture, never a source-qualified class graph.
__global__ void identity_fixture(Layout*q,ca::Node*n,ca::Arc*a,ca::State*s,ca::Arc*e,unsigned*c,U mutation){
  if(threadIdx.x||blockIdx.x)return;
  *q={};q->root=1;q->nn=2;q->na=2;q->ns=1;q->ne=1;q->nc=2;q->total=97;
  q->domain.allowed=(U(1)<<44)-1;
  for(unsigned k=0;k<10;++k){q->domain.lo[k]=0;q->domain.hi[k]=k?0:2;}
  q->cuts[0]=2;c[0]=0x00000000u;c[1]=0x3f800000u;
  n[0]={};n[0].kind=1;n[0].dimension=12;n[0].score_bits=6;
  n[1]={};n[1].kind=0;n[1].dimension=0;n[1].first_arc=0;n[1].arc_count=2;
  a[0]={0,0,0,0};a[1]={1,2,0,0};e[0]={0,2,0,0};s[0]={};
  for(unsigned k=0;k<7;++k){s[0].tuple[k]=k;s[0].probability_bits[k]=k?0x3d800000u:0x00000000u;}
  s[0].witness.categories=U(1)|(U(1)<<4);s[0].first_edge=0;s[0].edge_count=1;
  s[0].result=1;s[0].level=0;s[0].status=3;s[0].label=6;
  if(mutation==UINT64_MAX)return;
  U at=mutation;
  if(at<22){if(!at)q->root^=1;else if(at<=10)q->domain.lo[at-1]^=1;
    else if(at<=20)q->domain.hi[at-11]^=1;else q->domain.allowed^=1;return;}
  if(at<37)return;at-=37;
  if(at<14){auto&v=n[at/7];switch(at%7){case 0:v.kind^=1;break;case 1:v.dimension^=1;break;
    case 2:v.score_bits^=1;break;case 3:v.reserved^=1;break;case 4:v.first_arc^=1;break;
    case 5:v.arc_count^=1;break;default:v.hash_next^=1;}return;}
  at-=14;if(at<8){auto&v=a[at/4];switch(at%4){case 0:v.lo^=1;break;case 1:v.hi^=1;break;
    case 2:v.allowed^=1;break;default:v.child^=1;}return;}
  at-=8;if(at<32){auto&v=s[0];if(at<7)v.tuple[at]^=1;else if(at<17)v.witness.rank[at-7]^=1;
    else if(at==17)v.witness.categories^=1;else if(at==18)v.first_edge^=1;else if(at==19)v.edge_count^=1;
    else if(at==20)v.result^=1;else if(at==21)v.hash_next^=1;else if(at<29)v.probability_bits[at-22]^=0x80000000u;
    else if(at==29)v.level^=1;else if(at==30)v.status^=1;else v.label^=1;return;}
  at-=32;if(at<4){auto&v=e[0];switch(at){case 0:v.lo^=1;break;case 1:v.hi^=1;break;
    case 2:v.allowed^=1;break;default:v.child^=1;}return;}
  c[at-4]^=1;
}
Layout layout(const ca::Result&r,const std::array<std::vector<unsigned>,10>&cuts){
  Layout q{};q.root=r.root;q.nn=r.nodes.size();q.na=r.arcs.size();q.ns=r.states.size();q.ne=r.edges.size();q.domain=r.domain;
  for(unsigned k=0;k<10;++k){q.cuts[k]=cuts[k].size();q.nc=add(q.nc,q.cuts[k]);}
  q.total=identity_word_count(q.nn,q.na,q.ns,q.ne,q.nc);return q;
}
std::vector<unsigned>flatten(const std::array<std::vector<unsigned>,10>&cuts){
  std::vector<unsigned>out;for(const auto&v:cuts)out.insert(out.end(),v.begin(),v.end());return out;
}
void upload_pack(Budget&b,Layout q,const ca::Result&r,const std::array<std::vector<unsigned>,10>&cuts,U*out){
  Buf<ca::Node>nodes(b,q.nn);Buf<ca::Arc>arcs(b,q.na),edges(b,q.ne);Buf<ca::State>states(b,q.ns);Buf<unsigned>rank(b,q.nc);
  auto words=flatten(cuts);nodes.put(r.nodes.data());arcs.put(r.arcs.data());states.put(r.states.data());edges.put(r.edges.data());rank.put(words.data());
  pack<<<blocks(q.total),128>>>(q,nodes.p,arcs.p,states.p,edges.p,rank.p,out);cu(cudaGetLastError());cu(cudaDeviceSynchronize());
}
}
U identity_word_count(U nodes,U arcs,U states,U edges,U ranks){
  U n=37;n=add(n,mul(nodes,7));n=add(n,mul(arcs,4));n=add(n,mul(states,32));n=add(n,mul(edges,4));return add(n,ranks);
}
struct ClassIdentity::Impl{
  int device;Budget budget;Layout shape;Buf<U>reference;
  Impl(const ca::Result&r,const std::array<std::vector<unsigned>,10>&cuts,U cap,int gpu):device(gpu),budget{cap},shape(layout(r,cuts)),reference(budget,shape.total){
    cu(cudaSetDevice(device));upload_pack(budget,shape,r,cuts,reference.p);
  }
};
ClassIdentity::ClassIdentity(const ca::Result&r,const std::array<std::vector<unsigned>,10>&cuts,U cap,int device){cu(cudaSetDevice(device));p_=std::make_unique<Impl>(r,cuts,cap,device);}
ClassIdentity::~ClassIdentity()=default;
void ClassIdentity::audit(const ca::Result&r,const std::array<std::vector<unsigned>,10>&cuts){
  cu(cudaSetDevice(p_->device));auto q=layout(r,cuts);const auto&v=p_->shape;
  need(q.nn==v.nn&&q.na==v.na&&q.ns==v.ns&&q.ne==v.ne&&q.nc==v.nc&&q.total==v.total,"class identity extent changed");
  Buf<U>current(p_->budget,q.total);Buf<int>error(p_->budget,1);upload_pack(p_->budget,q,r,cuts,current.p);
  compare<<<blocks(q.total),128>>>(p_->reference.p,current.p,q.total,error.p);cu(cudaGetLastError());
  need(error.get()[0]==0,"CUDA immutable class graph word audit failed");
}
std::vector<U>ClassIdentity::words()const{cu(cudaSetDevice(p_->device));return p_->reference.get();}
U ClassIdentity::retained_device_bytes()const{return p_->reference.bytes;}
U ClassIdentity::owned_device_peak_bytes()const{return p_->budget.peak;}
struct EffectiveCache::Impl {
  int device;Budget budget;const U*source;U nn,words,limit;bool injected=false;U last_hit=UINT64_MAX;
  Buf<U>current,match;Buf<unsigned>order;Buf<int>error;Buf<const U*>table;
  std::vector<std::unique_ptr<Buf<U>>>keys;
  Impl(const U*ref,U nodes,U maximum,U cap,int gpu):device(gpu),budget{cap},source(ref),nn(nodes),
    words(add(2,mul(45,nodes))),limit(maximum),current(budget,words),match(budget,1),order(budget,44),error(budget,1),table(budget,maximum){
      need(nn&&limit&&limit<=INT_MAX,"effective cache metadata extent");
  }
};
EffectiveCache::EffectiveCache(const ClassIdentity&identity,U maximum,U cap,int device){
  cu(cudaSetDevice(device));need(identity.p_->device==device,"effective cache device identity");
  p_=std::make_unique<Impl>(identity.p_->reference.p,identity.p_->shape.nn,maximum,cap,device);
}
EffectiveCache::~EffectiveCache()=default;
EffectiveAction EffectiveCache::normalize(const unsigned*borrowed){cu(cudaSetDevice(p_->device));cu(cudaMemset(p_->error.p,0,4));p_->injected=borrowed!=nullptr;
  if(borrowed){cudaPointerAttributes a{};cu(cudaPointerGetAttributes(&a,borrowed));need(a.type==cudaMemoryTypeDevice&&a.device==p_->device,"effective order device identity");
    cu(cudaMemcpy(p_->order.p,borrowed,44*sizeof(unsigned),cudaMemcpyDeviceToDevice));effective_order_validate<<<1,1>>>(p_->order.p,p_->error.p);
  }
  effective_key<<<blocks(p_->nn),128>>>(p_->source,p_->nn,p_->order.p,p_->injected,p_->current.p,p_->error.p);cu(cudaGetLastError());
  need(p_->error.get()[0]==0,"CUDA per-node effective normalization failed");
  cu(cudaMemset(p_->match.p,255,sizeof(U)));
  if(!p_->keys.empty())effective_find<<<unsigned(p_->keys.size()),128>>>(p_->current.p,p_->table.p,p_->keys.size(),p_->words,p_->match.p);
  cu(cudaGetLastError());EffectiveAction out;out.hit=p_->match.get()[0];p_->last_hit=out.hit;
  rank_gpu_score_identity::Hash hash;hash.text("per-node-fixed-fallback-effective-order-1");for(auto w:p_->current.get())hash.word(w,8);out.key_sha256=hash.finish();
  if(borrowed){auto words=p_->order.get();std::copy(words.begin(),words.end(),out.order.begin());audit_order(borrowed);}return out;
}
void EffectiveCache::audit_order(const unsigned*borrowed){cu(cudaSetDevice(p_->device));
  need((borrowed!=nullptr)==p_->injected,"effective order mode changed");if(!borrowed)return;
  effective_order_same<<<1,64>>>(borrowed,p_->order.p,p_->error.p);cu(cudaGetLastError());need(p_->error.get()[0]==0,"effective order mutated");
}
void EffectiveCache::audit_traces(const std::vector<rl_category_lowering::Trace>&traces){
  cu(cudaSetDevice(p_->device));need(traces.size()==p_->nn,"effective trace extent");Buf<rl_category_lowering::Trace>trace(p_->budget,p_->nn);trace.put(traces.data());
  effective_trace<<<blocks(p_->nn),128>>>(p_->current.p,trace.p,p_->nn,p_->error.p);cu(cudaGetLastError());need(p_->error.get()[0]==0,"effective key differs from actual universally audited lowering trace");
}
U EffectiveCache::admit(){cu(cudaSetDevice(p_->device));need(p_->last_hit==UINT64_MAX,"effective cache duplicate admission");
  if(p_->keys.size()>=p_->limit||p_->current.bytes>p_->budget.cap-p_->budget.used)return UINT64_MAX;
  auto key=std::make_unique<Buf<U>>(p_->budget,p_->words);cu(cudaMemcpy(key->p,p_->current.p,key->bytes,cudaMemcpyDeviceToDevice));
  U id=p_->keys.size();const U*ptr=key->p;cu(cudaMemcpy(p_->table.p+id,&ptr,sizeof(ptr),cudaMemcpyHostToDevice));p_->keys.push_back(std::move(key));return id;
}
U EffectiveCache::retained_device_bytes()const{return p_->budget.used;}
U EffectiveCache::entries()const{return p_->keys.size();}
std::vector<std::vector<U>> EffectiveCache::snapshot_keys()const{
  cu(cudaSetDevice(p_->device));std::vector<std::vector<U>>out;
  for(const auto&key:p_->keys)out.push_back(key->get());return out;
}
void audit_policy_order(const std::array<unsigned,44>&expected,
                        const std::array<rl_category_policy::Trace,2>&trace,
                        U version,U cap,int device){
  cu(cudaSetDevice(device));Budget budget{cap};Buf<unsigned>order(budget,44);
  Buf<rl_category_policy::Trace>sample(budget,2);Buf<int>error(budget,1);
  order.put(expected.data());sample.put(trace.data());
  policy_order<<<1,64>>>(order.p,sample.p,version,error.p);cu(cudaGetLastError());
  need(error.get()[0]==0,"CUDA proposal/current policy sample identity differs");
}
bool smaller_encoded_cost(U candidate,U incumbent,int device){cu(cudaSetDevice(device));Budget b{4};Buf<int>out(b,1);
  choose_cost<<<1,1>>>(candidate,incumbent,out.p);cu(cudaGetLastError());return out.get()[0]!=0;}
namespace testing {
Report cache_gpu_checks(int device){cu(cudaSetDevice(device));Report out;Budget b{65536};Buf<ca::Node>n(b,6);Buf<ca::Arc>a(b,8);Buf<unsigned>orders(b,5*44),bad(b,44);
  cache_fixture<<<1,1>>>(n.p,a.p,orders.p);cu(cudaGetLastError());ca::Result graph;graph.root=5;graph.domain.allowed=(U(1)<<44)-1;graph.domain.hi[0]=1;
  graph.nodes=n.get();graph.arcs=a.get();graph.complete=graph.local_semantics_audited=true;graph.source_binding="literal-cache-contract-only";graph.domain_binding="literal-domain-only";
  std::array<std::vector<unsigned>,10>cuts;cuts[0]={0};ClassIdentity identity(graph,cuts,65536,device);EffectiveCache cache(identity,3,65536,device);
  auto check=[&](bool v,const char*s){need(v,s);++out.assertions;};auto reject=[&](auto fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}check(failed,"cache rejection absent");++out.rejections;};
  auto normalize=[&](const unsigned*order){auto action=cache.normalize(order);out.word_checks+=272;return action;};
  auto base=normalize(nullptr);check(base.hit==UINT64_MAX,"default cache unexpectedly hit");check(cache.admit()==0,"default cache admission");
  auto first=normalize(orders.p);check(first.hit==0,"default/injected effective chain differs");
  auto inactive=normalize(orders.p+44);check(inactive.hit==0&&inactive.key_sha256==base.key_sha256,"globally inactive permutation changed local key");
  auto c=normalize(orders.p+88);check(c.hit==UINT64_MAX&&c.key_sha256!=base.key_sha256,"node-specific fallback projection lost active bit");check(cache.admit()==1,"second key admission");
  check(normalize(orders.p+88).hit==1,"second key exact lookup");
  check(normalize(orders.p+132).hit==UINT64_MAX,"second node order mutation aliased");check(cache.admit()==2,"third key admission");
  check(normalize(orders.p).hit==0,"first exact key not retained");
  auto lower=rl_category_lowering::lower_binary(graph,{orders.p,1},1024,65536,{},device);check(lower.lowered.complete&&lower.trace_audited,"cache fixture frozen universal lowering failed");cache.audit_traces(lower.traces);++out.assertions;
  for(unsigned k=0;k<5;++k){normalize(orders.p);auto trace=lower.traces;
    if(k==0)trace[3].fallback_arc^=1;else if(k==1)trace[4].selected_mask^=U(1)<<6;else if(k==2)trace[3].test_order[0]=13;else if(k==3)trace[4].dimension=10;else trace[4].count^=1;
    reject([&]{cache.audit_traces(trace);});
  }
  cu(cudaMemcpy(bad.p,orders.p,44*sizeof(unsigned),cudaMemcpyDeviceToDevice));unsigned duplicate=0;cu(cudaMemcpy(bad.p+1,&duplicate,4,cudaMemcpyHostToDevice));reject([&]{cache.normalize(bad.p);});
  unsigned out_of_range=64;cu(cudaMemcpy(bad.p+1,&out_of_range,4,cudaMemcpyHostToDevice));reject([&]{cache.normalize(bad.p);});
  normalize(orders.p);cu(cudaMemcpy(bad.p,orders.p,44*sizeof(unsigned),cudaMemcpyDeviceToDevice));normalize(bad.p);cu(cudaMemcpy(bad.p+1,&duplicate,4,cudaMemcpyHostToDevice));reject([&]{cache.audit_order(bad.p);});
  check(normalize(orders.p+176).hit==UINT64_MAX,"fourth effective key aliased");check(cache.admit()==UINT64_MAX&&cache.entries()==3,"cache storage limit leaked partial key");
  identity.audit(graph,cuts);++out.assertions;out.passed=out.CUDA_executed=true;return out;
}
Report gpu_checks(int device){cu(cudaSetDevice(device));Report out;Budget b{65536};
  Buf<Layout>q(b,1);Buf<ca::Node>n(b,2);Buf<ca::Arc>a(b,2),e(b,1);Buf<ca::State>s(b,1);Buf<unsigned>c(b,2);
  auto fixture=[&](U mutation){identity_fixture<<<1,1>>>(q.p,n.p,a.p,s.p,e.p,c.p,mutation);cu(cudaGetLastError());
    auto shape=q.get()[0];ca::Result r;r.root=shape.root;r.domain=shape.domain;r.nodes=n.get();r.arcs=a.get();r.states=s.get();r.edges=e.get();
    std::array<std::vector<unsigned>,10>cuts;cuts[0]=c.get();return std::pair(std::move(r),std::move(cuts));};
  auto [baseline,cuts]=fixture(UINT64_MAX);ClassIdentity identity(baseline,cuts,65536,device);
  need(identity.words().size()==97&&identity.retained_device_bytes()==776,"identity fixture word shape");++out.assertions;
  identity.audit(baseline,cuts);++out.assertions;out.word_checks+=97;
  for(U k=0;k<97;++k){if(k>=22&&k<37)continue;auto [changed,words]=fixture(k);bool rejected=false;
    try{identity.audit(changed,words);}catch(const std::exception&){rejected=true;}
    need(rejected,"identity field mutation escaped audit");++out.assertions;++out.rejections;out.word_checks+=97;
  }
  // Extents are host metadata only; changed payloads still need CUDA equality.
  for(unsigned k=0;k<5;++k){auto r=baseline;auto words=cuts;
    if(k==0)r.nodes.push_back(r.nodes[0]);else if(k==1)r.arcs.push_back(r.arcs[0]);
    else if(k==2)r.states.push_back(r.states[0]);else if(k==3)r.edges.push_back(r.edges[0]);else words[0].push_back(words[0][0]);
    bool rejected=false;try{identity.audit(r,words);}catch(const std::exception&){rejected=true;}
    need(rejected,"identity extent mutation escaped");++out.assertions;++out.rejections;
  }
  for(unsigned k=1;k<10;++k){auto words=cuts;words[k]=std::move(words[0]);words[0].clear();bool rejected=false;
    try{identity.audit(baseline,words);}catch(const std::exception&){rejected=true;}
    need(rejected,"rank feature-boundary mutation escaped");++out.assertions;++out.rejections;out.word_checks+=97;
  }
  rl_category_policy::Policy policy(device);auto view=policy.sample44(123,7,1000);auto traces=policy.trace();
  std::array<unsigned,44>order{};cu(cudaMemcpy(order.data(),view.order44,sizeof(order),cudaMemcpyDeviceToHost));
  audit_policy_order(order,traces,view.version,65536,device);++out.assertions;
  for(unsigned k=0;k<6;++k){auto changed=traces;
    if(k==0)changed[0].actions[0]^=1;else if(k==1)changed[1].version^=1;
    else if(k==2)changed[1].seed^=1;else if(k==3)changed[1].episode^=1;
    else if(k==4)changed[1].group=0;else changed[1].fallback_mask=1;
    bool rejected=false;try{audit_policy_order(order,changed,view.version,65536,device);}catch(const std::exception&){rejected=true;}
    need(rejected,"policy sample identity mutation escaped");++out.assertions;++out.rejections;
  }
  need(smaller_encoded_cost(2,3,device)&&!smaller_encoded_cost(3,3,device)&&!smaller_encoded_cost(UINT64_MAX,3,device)&&smaller_encoded_cost(3,UINT64_MAX,device),"CUDA artifact cost comparison failed");++out.assertions;
  identity.audit(baseline,cuts);++out.assertions;out.word_checks+=97;
  out.passed=true;out.CUDA_executed=true;return out;
}
}
}
