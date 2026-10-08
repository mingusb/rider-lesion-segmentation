#include "class_rank_gpu_score_diagram.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace rank_gpu_score_diagram { namespace {
using Factor=rank_gpu_score_factors::Factor;
constexpr U categories=(U(1)<<44)-1;
void need(bool x,const char* why){if(!x)throw std::invalid_argument(why);}
void cu(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
struct Budget {U limit,used=0,peak=0;void add(U n){need(n<=limit-used,"score-diagram device budget");used+=n;peak=std::max(peak,used);}};
template<class T>struct Buf {
    Budget& b;U n,bytes;T* p=nullptr;
    Buf(Budget& a,U count):b(a),n(count),bytes(0){
        need(count<=none/sizeof(T),"score-diagram allocation overflow");bytes=count*sizeof(T);b.add(bytes);
        if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){b.used-=bytes;cu(e);}
            e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);p=nullptr;b.used-=bytes;cu(e);}}
    }
    ~Buf(){if(p)cudaFree(p);b.used-=bytes;}
    Buf(const Buf&)=delete;
    void put(const T* x){if(bytes)cu(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}
    std::vector<T> get(U count)const{need(count<=n,"score-diagram download range");std::vector<T>x(count);if(count)cu(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
struct Meta {Counters c;U roots[7],accepted[7],cursor=0;int bad=0,cap=0;};
static_assert(sizeof(Meta)+8*sizeof(U)<=4096);
struct Work {
    Options options;b::Box domain;Meta* m;State* states;Row* rows;Arc* edges;
    Node* nodes;Arc* arcs;U* state_heads;U* node_heads;Row* scratch;Arc* scratch_arcs;
};
__device__ U mix(U h,U x){h^=x;return h*1099511628211ull;}
__device__ bool box_equal(const b::Box&a,const b::Box&z){
    if(a.allowed!=z.allowed)return false;for(int d=0;d<10;++d)if(a.lo[d]!=z.lo[d]||a.hi[d]!=z.hi[d])return false;return true;
}
__device__ int row_compare(const Row&a,const Row&z){
    if(a.score_bits!=z.score_bits)return a.score_bits<z.score_bits?-1:1;
    for(int d=0;d<10;++d){if(a.box.lo[d]!=z.box.lo[d])return a.box.lo[d]<z.box.lo[d]?-1:1;if(a.box.hi[d]!=z.box.hi[d])return a.box.hi[d]<z.box.hi[d]?-1:1;}
    return a.box.allowed==z.box.allowed?0:a.box.allowed<z.box.allowed?-1:1;
}
__device__ U hash_row(U h,const Row&r){
    h=mix(h,r.score_bits);for(int d=0;d<10;++d){h=mix(h,unsigned(r.box.lo[d]));h=mix(h,unsigned(r.box.hi[d]));}return mix(h,r.box.allowed);
}
__device__ U canonicalize(Row* r,U n){
    for(U i=1;i<n;++i){Row x=r[i];U j=i;while(j&&row_compare(x,r[j-1])<0){r[j]=r[j-1];--j;}r[j]=x;}
    U count=0;for(U i=0;i<n;++i)if(!count||row_compare(r[i],r[count-1]))r[count++]=r[i];return count;
}
__device__ U state(Work w,unsigned level,U count){
    auto& m=*w.m;++m.c.state_attempts;count=canonicalize(w.scratch,count);
    if(!count){m.bad=20;return none;}U h=mix(1469598103934665603ull,level);
    for(U i=0;i<count;++i)h=hash_row(h,w.scratch[i]);U bucket=h%w.options.state_buckets;
    for(U id=w.state_heads[bucket];id!=none;id=w.states[id].hash_next){const auto&s=w.states[id];bool same=s.level==level&&s.row_count==count;
        for(U i=0;same&&i<count;++i){++m.c.row_comparisons;same=row_compare(w.rows[s.first_row+i],w.scratch[i])==0;}
        if(same){++m.c.state_hits;return id;}++m.c.state_collisions;
    }
    if(m.c.states==w.options.maximum_states){m.cap=2;return none;}
    if(count>w.options.maximum_rows-m.c.rows){m.cap=3;return none;}
    U id=m.c.states++;State s{};s.first_row=m.c.rows;s.row_count=count;s.level=level;s.hash_next=w.state_heads[bucket];
    for(U i=0;i<count;++i)w.rows[m.c.rows++]=w.scratch[i];w.states[id]=s;w.state_heads[bucket]=id;return id;
}
struct Wide {U x[4]{};};
__device__ bool times(Wide&a,U n){U carry=0;for(int k=0;k<4;++k){U lo=a.x[k]*n,hi=__umul64hi(a.x[k],n),v=lo+carry;hi+=v<lo;a.x[k]=v;carry=hi;}return !carry;}
__device__ bool plus(Wide&a,const Wide&b){U carry=0;for(int k=0;k<4;++k){U v=a.x[k]+b.x[k],c=v<a.x[k],z=v+carry;c+=z<v;a.x[k]=z;carry=c;}return !carry;}
__device__ bool volume(const b::Box&q,Wide&v){v.x[0]=1;for(int d=0;d<10;++d)if(!times(v,U(q.hi[d]-q.lo[d])+1))return false;return times(v,__popcll(q.allowed&b::wilderness))&&times(v,__popcll(q.allowed&b::soil));}
__device__ bool valid(const b::Box&q){
    if((q.allowed&~categories)||!(q.allowed&b::wilderness)||!(q.allowed&b::soil))return false;
    for(int d=0;d<10;++d)if(q.lo[d]<0||q.hi[d]>16777216||q.lo[d]>q.hi[d])return false;return true;
}
__global__ void validate(const Factor*f,U n,const U* offsets,Work w){
    if(blockIdx.x||threadIdx.x)return;auto&m=*w.m;
    if(!valid(w.domain)||offsets[0]||offsets[7]!=n){m.bad=1;return;}
    Wide target{};if(!volume(w.domain,target)){m.bad=2;return;}
    for(int c=0;c<7;++c){if(offsets[c]>=offsets[c+1]||offsets[c+1]>n){m.bad=3;return;}
        if(offsets[c+1]-offsets[c]>w.options.maximum_factors_per_class){m.cap=1;return;}
        Wide sum{};
        for(U i=offsets[c];i<offsets[c+1];++i){const auto&a=f[i];
            if(a.channel!=unsigned(c)||!valid(a.box)||!isfinite(__uint_as_float(a.score_bits))||(a.box.allowed&~w.domain.allowed)){m.bad=4;return;}
            for(int d=0;d<10;++d)if(a.box.lo[d]<w.domain.lo[d]||a.box.hi[d]>w.domain.hi[d]){m.bad=5;return;}
            Wide v{};if(!volume(a.box,v)||!plus(sum,v)){m.bad=6;return;}
            for(U j=i+1;j<offsets[c+1];++j){bool overlap=bool((a.box.allowed&f[j].box.allowed)&b::wilderness)&&bool((a.box.allowed&f[j].box.allowed)&b::soil);
                for(int d=0;d<10;++d)overlap=overlap&&max(a.box.lo[d],f[j].box.lo[d])<=min(a.box.hi[d],f[j].box.hi[d]);
                ++m.c.partition_pairs;if(overlap){m.bad=7;return;}
            }
        }
        for(int k=0;k<4;++k)if(sum.x[k]!=target.x[k]){m.bad=8;return;}++m.c.partition_volumes;
    }
}
__global__ void initialize(const Factor*f,const U* offsets,Work w){
    if(blockIdx.x||threadIdx.x)return;
    for(U i=0;i<w.options.state_buckets;++i)w.state_heads[i]=none;
    for(U i=0;i<w.options.node_buckets;++i)w.node_heads[i]=none;
    for(int c=0;c<7;++c)w.m->roots[c]=w.m->accepted[c]=none;
    for(int c=0;c<7;++c){U count=offsets[c+1]-offsets[c];
        for(U i=0;i<count;++i){Row r{};r.box=f[offsets[c]+i].box;r.score_bits=f[offsets[c]+i].score_bits;w.scratch[i]=r;}
        w.m->roots[c]=state(w,0,count);if(w.m->cap||w.m->bad)return;
    }
}
__device__ bool edge(Work w,State&s,Arc a){
    // Consolidate repeated child states before their descendants are processed.
    if(s.level<10&&s.edge_count){auto& last=w.edges[s.first_edge+s.edge_count-1];
        if(last.child==a.child&&last.hi+1==a.lo){last.hi=a.hi;++w.m->c.adjacent_arc_merges;return true;}}
    if(s.level>=10)for(U k=0;k<s.edge_count;++k){auto& old=w.edges[s.first_edge+k];if(old.child==a.child){old.allowed|=a.allowed;return true;}}
    if(w.m->c.edges==w.options.maximum_edges){w.m->cap=4;return false;}
    w.edges[w.m->c.edges++]=a;++s.edge_count;return true;
}
__device__ U restrict_rows(Work w,const State&s,int rank,U category){
    U count=0;for(U i=0;i<s.row_count;++i){Row r=w.rows[s.first_row+i];
        if(s.level<10){if(rank<r.box.lo[s.level]||rank>r.box.hi[s.level])continue;r.box.lo[s.level]=w.domain.lo[s.level];r.box.hi[s.level]=w.domain.hi[s.level];}
        else{if(!(r.box.allowed&category))continue;U group=s.level==10?b::wilderness:b::soil;r.box.allowed=(r.box.allowed&~group)|(w.domain.allowed&group);}
        w.scratch[count++]=r;
    }return count;
}
__global__ void expand(Work w,U limit){
    if(blockIdx.x||threadIdx.x)return;auto&m=*w.m;
    for(U done=0;done<limit&&m.cursor<m.c.states;++done){U id=m.cursor;auto&s=w.states[id];
        bool constant=true;unsigned bits=w.rows[s.first_row].score_bits;
        for(U i=1;i<s.row_count;++i)constant=constant&&w.rows[s.first_row+i].score_bits==bits;
        if(constant){s.status=2;++m.c.terminal_states;}
        else{
            if(s.level>=12){m.bad=21;return;}s.first_edge=m.c.edges;
            if(s.level<10){int at=w.domain.lo[s.level],last=w.domain.hi[s.level];
                while(at<=last){int next=last+1;
                    for(U i=0;i<s.row_count;++i){const auto&r=w.rows[s.first_row+i];int a=r.box.lo[s.level],z=r.box.hi[s.level]+1;if(a>at)next=min(next,a);if(z>at)next=min(next,z);}
                    U child=state(w,s.level+1,restrict_rows(w,s,at,0));if(m.cap||m.bad)return;
                    Arc a{};a.lo=at;a.hi=next-1;a.child=child;if(!edge(w,s,a))return;at=next;
                }
            }else{U allowed=w.domain.allowed&(s.level==10?b::wilderness:b::soil);
                while(allowed){U bit=allowed&(~allowed+1);allowed^=bit;
                    U child=state(w,s.level+1,restrict_rows(w,s,0,bit));if(m.cap||m.bad)return;
                    Arc a{};a.allowed=bit;a.child=child;if(!edge(w,s,a))return;
                }
            }s.status=1;
        }++m.cursor;++m.c.expanded;
    }
}
__device__ bool arc_equal(const Arc&a,const Arc&z){return a.lo==z.lo&&a.hi==z.hi&&a.allowed==z.allowed&&a.child==z.child;}
__device__ U node(Work w,unsigned kind,unsigned d,unsigned bits,U count){
    auto&m=*w.m;++m.c.node_attempts;U h=mix(mix(mix(1469598103934665603ull,kind),d),bits);
    for(U i=0;i<count;++i){const auto&a=w.scratch_arcs[i];h=mix(mix(mix(mix(h,unsigned(a.lo)),unsigned(a.hi)),a.allowed),a.child);}U bucket=h%w.options.node_buckets;
    for(U id=w.node_heads[bucket];id!=none;id=w.nodes[id].hash_next){const auto&n=w.nodes[id];bool same=n.kind==kind&&n.dimension==d&&n.score_bits==bits&&n.arc_count==count;
        for(U i=0;same&&i<count;++i)same=arc_equal(w.arcs[n.first_arc+i],w.scratch_arcs[i]);
        if(same){++m.c.node_hits;return id;}++m.c.node_collisions;
    }
    if(m.c.nodes==w.options.maximum_nodes){m.cap=5;return none;}
    if(count>w.options.maximum_arcs-m.c.arcs){m.cap=6;return none;}
    U id=m.c.nodes++;Node n{};n.kind=kind;n.dimension=d;n.score_bits=bits;n.first_arc=m.c.arcs;n.arc_count=count;n.hash_next=w.node_heads[bucket];
    for(U i=0;i<count;++i)w.arcs[m.c.arcs++]=w.scratch_arcs[i];w.nodes[id]=n;w.node_heads[bucket]=id;return id;
}
__global__ void reduce(Work w,U limit){
    if(blockIdx.x||threadIdx.x)return;auto&m=*w.m;
    for(U done=0;done<limit&&m.c.reduced<m.c.states;++done){U id=m.c.states-1-m.c.reduced;auto&s=w.states[id];
        if(s.status==2)s.result=node(w,0,12,w.rows[s.first_row].score_bits,0);
        else{if(s.status!=1||!s.edge_count){m.bad=22;return;}U count=0;
            for(U i=0;i<s.edge_count;++i){Arc a=w.edges[s.first_edge+i];
                // Breadth-first level construction guarantees a later state ID.
                if(a.child<=id||a.child>=m.c.states||w.states[a.child].status!=3){m.bad=23;return;}
                a.child=w.states[a.child].result;
                if(s.level<10&&count&&w.scratch_arcs[count-1].child==a.child){w.scratch_arcs[count-1].hi=a.hi;++m.c.adjacent_arc_merges;continue;}
                bool found=false;if(s.level>=10)for(U k=0;k<count;++k)if(w.scratch_arcs[k].child==a.child){w.scratch_arcs[k].allowed|=a.allowed;found=true;break;}
                if(!found)w.scratch_arcs[count++]=a;
            }
            if(count==1){s.result=w.scratch_arcs[0].child;++m.c.equal_child_reductions;}
            else s.result=node(w,1,s.level,0,count);
        }
        if(m.cap||m.bad)return;s.status=3;++m.c.reduced;
    }
}
__global__ void publish(Work w){if(blockIdx.x||threadIdx.x)return;auto&m=*w.m;
    if(m.cap||m.bad||m.c.reduced!=m.c.states){m.bad=24;return;}
    for(int c=0;c<7;++c){U id=m.roots[c];if(id>=m.c.states||w.states[id].status!=3||w.states[id].result>=m.c.nodes){m.bad=25;return;}m.accepted[c]=w.states[id].result;}
}
const char* cap_reason(int cap){switch(cap){case 1:return "factors_per_class_cap";case 2:return "state_cap";case 3:return "residual_row_cap";case 4:return "transition_cap";case 5:return "node_cap";case 6:return "arc_cap";default:return "unknown_cap";}}
struct AuditMeta {U cells=0,words=0;int bad=0,capped=0;};
__global__ void audit_size(b::Box q,b::Box other,U cap,AuditMeta*m){if(blockIdx.x||threadIdx.x)return;
    if(!valid(q)||!box_equal(q,other)){m->bad=1;return;}U n=1;
    for(int d=0;d<12;++d){U radix=d<10?U(q.hi[d]-q.lo[d])+1:U(__popcll(q.allowed&(d==10?b::wilderness:b::soil)));if(n>cap/radix){m->capped=1;return;}n*=radix;}m->cells=n;
}
__device__ U category_at(U mask,U index){while(index--)mask&=mask-1;return mask&(~mask+1);}
__global__ void audit_cells(b::Box domain,const Factor*f,const U*offsets,const Node*nodes,U nn,const Arc*arcs,U na,const U*roots,AuditMeta*m){
    U id=U(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=m->cells*7)return;int c=int(id%7);U cell=id/7;int ranks[10];U cats=0;
    for(int d=11;d>=0;--d){U radix=d<10?U(domain.hi[d]-domain.lo[d])+1:U(__popcll(domain.allowed&(d==10?b::wilderness:b::soil))),value=cell%radix;cell/=radix;
        if(d<10)ranks[d]=domain.lo[d]+int(value);else cats|=category_at(domain.allowed&(d==10?b::wilderness:b::soil),value);
    }
    unsigned expected=0;U matches=0;
    for(U i=offsets[c];i<offsets[c+1];++i){bool hit=(f[i].box.allowed&cats)==cats;for(int d=0;d<10;++d)hit=hit&&ranks[d]>=f[i].box.lo[d]&&ranks[d]<=f[i].box.hi[d];if(hit){++matches;expected=f[i].score_bits;}}
    if(matches!=1){atomicCAS(&m->bad,0,2);return;}U at=roots[c];int previous=-1;
    for(int step=0;step<14;++step){if(at>=nn){atomicCAS(&m->bad,0,3);return;}const auto&n=nodes[at];
        if(!n.kind){if(n.score_bits!=expected){atomicCAS(&m->bad,0,4);return;}atomicAdd(reinterpret_cast<unsigned long long*>(&m->words),1ull);return;}
        if(n.kind!=1||n.dimension>=12||int(n.dimension)<=previous||n.first_arc>na||n.arc_count>na-n.first_arc){atomicCAS(&m->bad,0,5);return;}
        U next=none,hits=0;for(U i=0;i<n.arc_count;++i){const auto&a=arcs[n.first_arc+i];bool hit=n.dimension<10?ranks[n.dimension]>=a.lo&&ranks[n.dimension]<=a.hi:bool(a.allowed&cats);if(hit){++hits;next=a.child;}}
        if(hits!=1||next>=at){atomicCAS(&m->bad,0,6);return;}previous=int(n.dimension);at=next;
    }atomicCAS(&m->bad,0,7);
}
}

Result construct(const Input& in,Options o,const Stop& stop,int device){
    Result out;out.source_binding=in.factors.source_binding;out.domain_binding=in.domain_binding;out.domain=in.domain;
    need(in.factors.complete&&!out.source_binding.empty()&&!out.domain_binding.empty(),"score-diagram qualified input/binding absent");
    U n=in.factors.factors.size();need(n==in.factors.compatible&&n,"score-diagram input counts differ");
    U planned=planned_device_bytes(n,o);if(n>o.maximum_factors){out.reason="input_factor_cap";return out;}
    if(planned>o.maximum_device_bytes){out.reason="device_budget";return out;}
    auto cancelled=[&]{return stop&&stop();};if(cancelled()){out.reason="cancelled_before_allocation";return out;}
    cu(cudaSetDevice(device));Budget budget{o.maximum_device_bytes};
    Buf<Factor> factors(budget,n);Buf<U> offsets(budget,8);Buf<Meta> metadata(budget,1);
    Buf<State> states(budget,o.maximum_states);Buf<Row> rows(budget,o.maximum_rows),scratch(budget,n);
    Buf<Arc> edges(budget,o.maximum_edges),arcs(budget,o.maximum_arcs),scratch_arcs(budget,n*2+44);
    Buf<Node> nodes(budget,o.maximum_nodes);Buf<U> sh(budget,o.state_buckets),nh(budget,o.node_buckets);
    factors.put(in.factors.factors.data());offsets.put(in.factors.factor_offsets.data());
    Work w{o,in.domain,metadata.p,states.p,rows.p,edges.p,nodes.p,arcs.p,sh.p,nh.p,scratch.p,scratch_arcs.p};
    Meta m{};auto checked=[&]{cu(cudaGetLastError());m=metadata.get(1)[0];if(m.bad)throw std::runtime_error("CUDA score-diagram validation failed: "+std::to_string(m.bad));};
    auto finish=[&](bool complete,const char* why){out.complete=complete;out.reason=why;out.counters=m.c;out.owned_device_peak_bytes=budget.peak;
        out.nodes=nodes.get(m.c.nodes);out.arcs=arcs.get(m.c.arcs);
        if(complete)for(int c=0;c<7;++c)out.roots[c]=m.accepted[c];
        else{out.frontier_states=states.get(m.c.states);out.frontier_rows=rows.get(m.c.rows);out.frontier_edges=edges.get(m.c.edges);}
        return out;};
    validate<<<1,1>>>(factors.p,n,offsets.p,w);out.CUDA_executed=true;checked();
    if(m.cap)return finish(false,cap_reason(m.cap));if(cancelled())return finish(false,"cancelled_after_partition_validation");
    initialize<<<1,1>>>(factors.p,offsets.p,w);checked();if(m.cap)return finish(false,cap_reason(m.cap));
    while(m.cursor<m.c.states){if(cancelled())return finish(false,"cancelled_before_expansion");expand<<<1,1>>>(w,o.states_per_launch);checked();if(m.cap)return finish(false,cap_reason(m.cap));}
    while(m.c.reduced<m.c.states){if(cancelled())return finish(false,"cancelled_before_reduction");reduce<<<1,1>>>(w,o.states_per_launch);checked();if(m.cap)return finish(false,cap_reason(m.cap));}
    if(cancelled())return finish(false,"cancelled_before_publication");publish<<<1,1>>>(w);checked();
    auto result=finish(true,"CUDA_complete_fixed_order_score_diagrams_no_class_authority");validate_transport(result);return result;
}

Audit audit_quotient(const Input& in,const Result&r,U maximum_cells,int device){
    validate_transport(r);need(in.factors.complete&&in.factors.compatible==in.factors.factors.size()&&in.factors.source_binding==r.source_binding&&in.domain_binding==r.domain_binding,"score-diagram audit input binding");
    need(maximum_cells&&maximum_cells<=U(INT_MAX)/7,"score-diagram audit cell cap invalid");
    need(in.factors.factor_offsets[0]==0&&in.factors.factor_offsets[7]==in.factors.factors.size(),"score-diagram audit offsets");
    for(int c=0;c<7;++c)need(in.factors.factor_offsets[c]<in.factors.factor_offsets[c+1],"score-diagram audit offset order");
    Audit out;cu(cudaSetDevice(device));Budget budget{none};Buf<AuditMeta> meta(budget,1);
    audit_size<<<1,1>>>(in.domain,r.domain,maximum_cells,meta.p);cu(cudaGetLastError());auto m=meta.get(1)[0];out.CUDA_executed=true;out.owned_device_peak_bytes=budget.peak;
    need(!m.bad,"score-diagram audit domain differs/invalid");if(m.capped){out.reason="audit_quotient_cell_cap";return out;}
    Buf<Factor> f(budget,in.factors.factors.size());Buf<U> offsets(budget,8),roots(budget,7);Buf<Node> nodes(budget,r.nodes.size());Buf<Arc> arcs(budget,r.arcs.size());
    f.put(in.factors.factors.data());offsets.put(in.factors.factor_offsets.data());roots.put(r.roots.data());nodes.put(r.nodes.data());arcs.put(r.arcs.data());
    audit_cells<<<unsigned((m.cells*7+127)/128),128>>>(in.domain,f.p,offsets.p,nodes.p,r.nodes.size(),arcs.p,r.arcs.size(),roots.p,meta.p);cu(cudaGetLastError());m=meta.get(1)[0];
    need(!m.bad&&m.words==m.cells*7,"CUDA score-diagram independent quotient audit differs");out.complete=true;out.quotient_cells=m.cells;out.score_words=m.words;out.owned_device_peak_bytes=budget.peak;out.reason="GPU_exhaustive_quotient_factor_words_equal";return out;
}
}
