#include "streaming_file_hash.hpp"
#include "class_rank_gpu_leaf_partitions.hpp"
#include "class_io.hpp"
#include "class_xgboost.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace rank_gpu_leaf_partitions { namespace {
constexpr U all_categories = (U(1)<<44)-1;
constexpr const char* pinned_library = "462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4";
void need(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U mul(U a,U c){need(!c||a<=UINT64_MAX/c,"leaf-partition byte overflow");return a*c;}
struct Budget {U maximum,used=0,peak=0;void add(U n){need(n<=maximum-used,"leaf-partition device budget exceeded");used+=n;peak=std::max(peak,used);}};
template<class T> struct Buf {
    Budget& owner; U count,bytes; T* p=nullptr;
    Buf(Budget& b,U n):owner(b),count(n),bytes(mul(n,sizeof(T))){
        b.add(bytes);if(bytes){auto e=cudaMalloc(reinterpret_cast<void**>(&p),bytes);if(e!=cudaSuccess){b.used-=bytes;cu(e);}e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){cudaFree(p);p=nullptr;b.used-=bytes;cu(e);}}
    }
    ~Buf(){if(p)cudaFree(p);owner.used-=bytes;}
    Buf(const Buf&)=delete;
    void put(const T* x){if(bytes)cu(cudaMemcpy(p,x,bytes,cudaMemcpyHostToDevice));}
    std::vector<T> get(U n)const{need(n<=count,"leaf-partition download range");std::vector<T> x(n);if(n)cu(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
struct Profile {int offset[10]{},count[10]{},minimum[10]{};U cuts=0;};
struct State {int bad=0;U kept=0,empty=0,pairs=0,volumes=0;};
struct View {
    int nodes,trees,leaves,depth;
    const int *feature,*left,*right,*roots,*channels,*tree_offsets,*leaf_nodes,*ancestors;
    const unsigned char* sides;const float *cut,*value,*bias;
};
unsigned blocks(U n){need(n&&n<=U(INT_MAX)*128,"leaf-partition launch extent");return unsigned((n+127)/128);}
__device__ void bad(State* s,int code){atomicCAS(&s->bad,0,code);}
__device__ unsigned ordered(unsigned b){return b&0x80000000u?~b:b^0x80000000u;}
__device__ unsigned inverse(unsigned b){return b&0x80000000u?b^0x80000000u:~b;}
__global__ void keys(View v,U* output,State* s){
    U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=U(v.nodes))return;
    output[i]=UINT64_MAX;
    if(v.left[i]<0){if(v.right[i]!=-1||!isfinite(v.value[i]))bad(s,1);return;}
    int f=v.feature[i];float x=v.cut[i];
    if(f<0||f>=54||!isfinite(x)||v.right[i]<0){bad(s,2);return;}
    if(f<10){if(x==0.f)x=0.f;output[i]=(U(unsigned(f))<<32)|ordered(__float_as_uint(x));}
}
__global__ void make_profile(const U* sorted,int n,unsigned* cuts,Profile* out,State* s){
    if(blockIdx.x||threadIdx.x)return;Profile p{};
    for(int i=0;i<n&&sorted[i]!=UINT64_MAX;++i){if(i&&sorted[i]==sorted[i-1])continue;
        int f=int(sorted[i]>>32);if(f<0||f>=10){bad(s,3);return;}
        if(!p.count[f])p.offset[f]=int(p.cuts);cuts[p.cuts++]=inverse(unsigned(sorted[i]));++p.count[f];
    }
    for(int f=0;f<10;++f)p.minimum[f]=p.count[f]&&cuts[p.offset[f]]==0xff7fffffu;
    *out=p;
}
__device__ int rank_of(float x,Profile p,const unsigned* cuts,int f){
    int a=0,z=p.count[f];while(a<z){int m=a+(z-a)/2;if(__uint_as_float(cuts[p.offset[f]+m])<=x)a=m+1;else z=m;}return a;
}
__device__ bool nonempty(const b::Box& x){
    if(!(x.allowed&b::wilderness)||!(x.allowed&b::soil))return false;
    for(int f=0;f<10;++f)if(x.lo[f]>x.hi[f])return false;return true;
}
__device__ bool same(const b::Box& a,const b::Box& c){
    if(a.allowed!=c.allowed)return false;for(int f=0;f<10;++f)if(a.lo[f]!=c.lo[f]||a.hi[f]!=c.hi[f])return false;return true;
}
__device__ bool contains(const b::Box& a,const int* ranks,U categories){
    if((a.allowed&categories)!=categories)return false;
    for(int f=0;f<10;++f)if(ranks[f]<a.lo[f]||ranks[f]>a.hi[f])return false;return true;
}
__device__ int tree_of(int leaf,View v){int a=0;while(a+1<v.trees&&leaf>=v.tree_offsets[a+1])++a;return a;}
// Direct rank intersections and a separate raw interval lowering are checked
// against one another. Raw lowering follows the qualified initialize_source
// ancestor algorithm; masks are converted by explicit exactly-one semantics.
__global__ void extract_leaves(View v,Profile p,const unsigned* cuts,b::Leaf* staged,unsigned char* keep,State* s){
    int i=int(U(blockIdx.x)*blockDim.x+threadIdx.x);if(i>=v.leaves)return;
    int tree=tree_of(i,v),at=v.roots[tree];b::Box box;box.allowed=all_categories;
    float lo[10],hi[10];U zero=all_categories,one=all_categories;
    for(int f=0;f<10;++f){box.lo[f]=p.minimum[f];box.hi[f]=p.count[f];lo[f]=-FLT_MAX;hi[f]=FLT_MAX;}
    bool ended=false;
    for(int k=0;k<v.depth;++k){int q=v.ancestors[U(i)*v.depth+k];unsigned side=v.sides[U(i)*v.depth+k];
        if(q==-1){ended=true;if(side)bad(s,4);continue;}
        if(ended||q!=at||q<0||q>=v.nodes||v.left[q]<0||side>1){bad(s,5);return;}
        int f=v.feature[q];float cut=v.cut[q];
        if(f<10){int r=rank_of(cut,p,cuts,f);if(r<1||r>p.count[f]||!(__uint_as_float(cuts[p.offset[f]+r-1])==cut)){bad(s,6);return;}
            if(side){box.lo[f]=max(box.lo[f],r);lo[f]=fmaxf(lo[f],cut);}
            else{box.hi[f]=min(box.hi[f],r-1);hi[f]=fminf(hi[f],nextafterf(cut,-CUDART_INF_F));}
        }else{
            int start=f<14?10:14,extent=f<14?4:40;U group=f<14?b::wilderness:b::soil,allowed=0;
            for(int k=0;k<extent;++k)if((float(start+k==f)<cut)==(!side))allowed|=U(1)<<(start+k-10);
            box.allowed&=(~group)|allowed;
            U bit=U(1)<<(f-10);if((0.f<cut)!=(!side))zero&=~bit;if((1.f<cut)!=(!side))one&=~bit;
        }
        at=side?v.right[q]:v.left[q];
    }
    if(at!=v.leaf_nodes[i]||at<0||at>=v.nodes||v.left[at]!=-1||v.right[at]!=-1){bad(s,7);return;}
    b::Box raw;raw.allowed=0;
    for(int f=0;f<10;++f){raw.lo[f]=rank_of(lo[f],p,cuts,f);raw.hi[f]=rank_of(hi[f],p,cuts,f);}
    for(int k=0;k<44;++k){U bit=U(1)<<k,group=k<4?b::wilderness:b::soil;
        if((one&bit)&&((zero&(group^bit))==(group^bit)))raw.allowed|=bit;
    }
    if(!same(box,raw)){bad(s,8);return;}
    staged[i].box=box;staged[i].ordinal=i;staged[i].value=v.value[at];keep[i]=nonempty(box);
}
__global__ void compact_layout(View v,const unsigned char* keep,int* destination,int* offsets,State* s){
    if(blockIdx.x||threadIdx.x)return;int count=0;
    for(int t=0;t<v.trees;++t){offsets[t]=count;
        for(int i=v.tree_offsets[t];i<v.tree_offsets[t+1];++i){destination[i]=count;count+=keep[i];}
        if(offsets[t]==count){bad(s,9);return;}
    }
    offsets[v.trees]=count;s->kept=count;s->empty=v.leaves-count;
}
__global__ void compact_leaves(int n,const b::Leaf* staged,const unsigned char* keep,const int* dst,b::Leaf* output){
    U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i<U(n)&&keep[i]){const auto& a=staged[i];auto& c=output[dst[i]];c.box=a.box;c.ordinal=a.ordinal;c.value=a.value;}
}
struct Wide {U word[4]{};};
__device__ bool times(Wide& a,U b){U carry=0;for(int k=0;k<4;++k){U low=a.word[k]*b,high=__umul64hi(a.word[k],b);U x=low+carry;high+=x<low;a.word[k]=x;carry=high;}return carry==0;}
__device__ bool plus(Wide& a,const Wide& b){U carry=0;for(int k=0;k<4;++k){U x=a.word[k]+b.word[k],c=x<a.word[k],y=x+carry;c+=y<x;a.word[k]=y;carry=c;}return carry==0;}
__device__ bool volume(const b::Box& box,Wide& w){w.word[0]=1;for(int f=0;f<10;++f)if(!times(w,U(box.hi[f]-box.lo[f])+1))return false;return times(w,__popcll(box.allowed&b::wilderness))&&times(w,__popcll(box.allowed&b::soil));}
__global__ void partition_audit(View v,Profile p,const b::Leaf* leaves,const int* offsets,State* s){
    int t=int(U(blockIdx.x)*blockDim.x+threadIdx.x);if(t>=v.trees)return;
    b::Box domain;for(int f=0;f<10;++f){domain.lo[f]=p.minimum[f];domain.hi[f]=p.count[f];}
    Wide target{},sum{};if(!volume(domain,target)){bad(s,10);return;}
    for(int i=offsets[t];i<offsets[t+1];++i){const auto& a=leaves[i];Wide size{};
        if(!nonempty(a.box)||!volume(a.box,size)||!plus(sum,size)){bad(s,11);return;}
        if(a.ordinal<v.tree_offsets[t]||a.ordinal>=v.tree_offsets[t+1]){bad(s,12);return;}
        for(int f=0;f<10;++f)if(a.box.lo[f]<domain.lo[f]||a.box.hi[f]>domain.hi[f])bad(s,13);
        for(int j=i+1;j<offsets[t+1];++j){const auto& c=leaves[j];bool overlap=bool((a.box.allowed&c.box.allowed)&b::wilderness)&&bool((a.box.allowed&c.box.allowed)&b::soil);
            for(int f=0;f<10;++f)overlap=overlap&&max(a.box.lo[f],c.box.lo[f])<=min(a.box.hi[f],c.box.hi[f]);
            if(overlap)bad(s,14);atomicAdd(reinterpret_cast<unsigned long long*>(&s->pairs),1ull);
        }
    }
    for(int k=0;k<4;++k)if(sum.word[k]!=target.word[k])bad(s,15);
    atomicAdd(reinterpret_cast<unsigned long long*>(&s->volumes),1ull);
}
__device__ void raw_from_leaf(const b::Leaf& leaf,Profile p,const unsigned* cuts,float* raw){
    for(int f=0;f<54;++f)raw[f]=0.f;
    for(int f=0;f<10;++f)raw[f]=leaf.box.lo[f]?__uint_as_float(cuts[p.offset[f]+leaf.box.lo[f]-1]):-FLT_MAX;
    raw[10+__ffsll(leaf.box.allowed&b::wilderness)-1]=1.f;
    raw[10+__ffsll(leaf.box.allowed&b::soil)-1]=1.f;
}
__global__ void witnesses(U begin,U count,U width,Profile p,const unsigned* cuts,const b::Leaf* leaves,float* raw){
    U row=U(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=count)return;U id=begin+row,variant=id%width;
    float* x=raw+row*54;raw_from_leaf(leaves[id/width],p,cuts,x);if(!variant)return;--variant;
    if(variant<60){int f=int(variant/6),k=int(variant%6);unsigned words[6]={0xff7fffffu,0x80000001u,0x80000000u,0u,1u,0x7f7fffffu};x[f]=__uint_as_float(words[k]);return;}variant-=60;
    if(variant<3*p.cuts){U q=variant/3;int side=int(variant%3),f=0;while(f<9&&q>=U(p.offset[f]+p.count[f]))++f;
        float cut=__uint_as_float(cuts[q]),v=side==0?nextafterf(cut,-CUDART_INF_F):side==2?nextafterf(cut,CUDART_INF_F):cut;
        x[f]=isfinite(v)?v:cut;return;}variant-=3*p.cuts;
    int first=variant<4?10:14,extent=variant<4?4:40,k=variant<4?int(variant):int(variant-4);
    for(int f=first;f<first+extent;++f)x[f]=float(f==first+k);
}
__global__ void audit_rows(U begin,U count,U width,View v,Profile p,const unsigned* cuts,const b::Leaf* leaves,const int* offsets,const float* raw,const float* native,State* s){
    U row=U(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=count)return;const float* x=raw+row*54;int ranks[10];U cats=0;
    for(int f=0;f<10;++f){if(!isfinite(x[f]))bad(s,16);ranks[f]=rank_of(x[f],p,cuts,f);}
    for(int f=10;f<54;++f)if(x[f]==1.f)cats|=U(1)<<(f-10);
    if(__popcll(cats&b::wilderness)!=1||__popcll(cats&b::soil)!=1){bad(s,17);return;}
    float acc[7],reference[7];for(int c=0;c<7;++c)acc[c]=reference[c]=v.bias[c];
    for(int t=0;t<v.trees;++t){int at=v.roots[t],steps=0;
        while(at>=0&&at<v.nodes&&v.left[at]>=0&&++steps<=v.nodes)at=x[v.feature[at]]<v.cut[at]?v.left[at]:v.right[at];
        if(at<0||at>=v.nodes||steps>v.nodes){bad(s,18);return;}
        int found=-1,matches=0;for(int k=offsets[t];k<offsets[t+1];++k)if(contains(leaves[k].box,ranks,cats)){found=k;++matches;}
        if(matches!=1||v.leaf_nodes[leaves[found].ordinal]!=at||__float_as_uint(leaves[found].value)!=__float_as_uint(v.value[at])){bad(s,19);return;}
        if((begin+row)%width==0){auto home=leaves[(begin+row)/width].ordinal;if(home>=v.tree_offsets[t]&&home<v.tree_offsets[t+1]&&leaves[found].ordinal!=home)bad(s,20);}
        int c=v.channels[t];acc[c]=__fadd_rn(acc[c],leaves[found].value);reference[c]=__fadd_rn(reference[c],v.value[at]);
    }
    for(int c=0;c<7;++c)if(!isfinite(acc[c])||__float_as_uint(acc[c])!=__float_as_uint(reference[c])||__float_as_uint(acc[c])!=__float_as_uint(native[row*7+c]))bad(s,21);
}
__global__ void audit_factor_rows(U count,Profile p,const unsigned* cuts,const rank_gpu_score_factors::Factor* factors,const U* offsets,const float* raw,const float* native,State* s){
    U row=U(blockIdx.x)*blockDim.x+threadIdx.x;if(row>=count)return;const float* x=raw+row*54;int ranks[10];U cats=0;
    for(int f=0;f<10;++f)ranks[f]=rank_of(x[f],p,cuts,f);for(int f=10;f<54;++f)if(x[f]==1.f)cats|=U(1)<<(f-10);
    for(int c=0;c<7;++c){U matches=0;unsigned score=0;
        for(U k=offsets[c];k<offsets[c+1];++k)if(contains(factors[k].box,ranks,cats)){++matches;score=factors[k].score_bits;if(factors[k].channel!=unsigned(c))bad(s,22);}
        if(matches!=1||score!=__float_as_uint(native[row*7+c]))bad(s,23);
    }
}
}
struct Bridge::Impl {
    Options options;Budget budget;int device;dpnative::SourceData source;std::string library_path,model_path,library_sha,rank_sha,binding;
    std::unique_ptr<Buf<int>> feature,left,right,roots,channels,tree_offsets,leaf_nodes,ancestors,destination,offsets;
    std::unique_ptr<Buf<unsigned char>> sides,keep;
    std::unique_ptr<Buf<float>> cut,value,bias,raw;
    std::unique_ptr<Buf<unsigned>> cuts;
    std::unique_ptr<Buf<Profile>> dp;
    std::unique_ptr<Buf<State>> state;
    std::unique_ptr<Buf<b::Leaf>> staged,leaves;
    std::unique_ptr<dpnative::XGBoostOracle> oracle;
    Profile profile{};State status{};U width=0,rows=0;bool extracted=false,attempted=false;
    template<class T>std::unique_ptr<Buf<T>> upload(const std::vector<T>& x){auto b=std::make_unique<Buf<T>>(budget,x.size());b->put(x.data());return b;}
    View view(){return {int(source.feature.size()),int(source.roots.size()),int(source.leaf_nodes.size()),source.depth,feature->p,left->p,right->p,roots->p,channels->p,tree_offsets->p,leaf_nodes->p,ancestors->p,sides->p,cut->p,value->p,bias->p};}
    void checked(){cu(cudaGetLastError());status=state->get(1)[0];if(status.bad)throw std::runtime_error("CUDA leaf-partition check failed: "+std::to_string(status.bad));}
    void identities(){need(dpnative::sha256(dpnative::read_text(model_path))==source.identity,"leaf-partition source changed");need(dp_streaming::sha256_file(library_path)==library_sha,"leaf-partition native library changed");}
    Impl(const std::string& lib,const std::string& model,const std::string& expected,Options o,int device):options(o),budget{o.maximum_device_bytes},device(device),source(dpnative::read_source(model)),library_path(lib),model_path(model){
        need(source.identity==expected,"leaf-partition expected source identity differs");library_sha=dp_streaming::sha256_file(lib);need(library_sha==pinned_library,"leaf-partition native library differs");
        need(o.maximum_source_nodes&&o.maximum_source_nodes<=4096&&o.maximum_source_trees&&o.maximum_source_trees<=64&&o.maximum_source_depth<=128&&o.maximum_native_rows&&o.native_batch_rows&&o.native_batch_rows<=1048576,"leaf-partition options invalid");
        need(source.features==54&&source.outputs==7&&source.feature.size()<=o.maximum_source_nodes&&source.roots.size()<=o.maximum_source_trees&&U(source.depth)<=o.maximum_source_depth,"leaf-partition source dimensions/capacity");
        need(source.bias.size()==7&&source.leaf_offsets.size()==source.roots.size()+1&&source.ancestors.size()==mul(source.leaf_nodes.size(),U(source.depth)),"leaf-partition source layout");
        cu(cudaSetDevice(device));feature=upload(source.feature);left=upload(source.left);right=upload(source.right);roots=upload(source.roots);channels=upload(source.channels);tree_offsets=upload(source.leaf_offsets);leaf_nodes=upload(source.leaf_nodes);ancestors=upload(source.ancestors);sides=upload(source.ancestor_right);cut=upload(source.cut);value=upload(source.value);bias=upload(source.bias);
        U n=source.feature.size(),l=source.leaf_nodes.size();cuts=std::make_unique<Buf<unsigned>>(budget,n);dp=std::make_unique<Buf<Profile>>(budget,1);state=std::make_unique<Buf<State>>(budget,1);
        staged=std::make_unique<Buf<b::Leaf>>(budget,l);leaves=std::make_unique<Buf<b::Leaf>>(budget,l);keep=std::make_unique<Buf<unsigned char>>(budget,l);destination=std::make_unique<Buf<int>>(budget,l);offsets=std::make_unique<Buf<int>>(budget,source.roots.size()+1);
    }
    bool halt(const Stop& stop){return stop&&stop();}
    Audit audit(const rank_gpu_score_factors::Result* factors,const Stop& stop){
        Audit out;out.CUDA_executed=true;cu(cudaSetDevice(device));
        auto finish=[&](){out.owned_device_peak_bytes=budget.peak;return out;};
        std::unique_ptr<Buf<rank_gpu_score_factors::Factor>> f;std::unique_ptr<Buf<U>> fo;
        if(factors){need(factors->complete&&factors->source_binding==binding&&factors->factors.size()==factors->compatible,"leaf-partition factor binding/completion differs");need(factors->factor_offsets[0]==0&&factors->factor_offsets[7]==factors->factors.size(),"leaf-partition factor offsets");for(int c=0;c<7;++c)need(factors->factor_offsets[c]<factors->factor_offsets[c+1],"leaf-partition factor class absent");f=upload(factors->factors);fo=std::make_unique<Buf<U>>(budget,8);fo->put(factors->factor_offsets.data());}
        for(U begin=0;begin<rows;){if(halt(stop)){out.reason="cancelled_native_audit";return finish();}U n=std::min(options.native_batch_rows,rows-begin);
            witnesses<<<blocks(n),128>>>(begin,n,width,profile,cuts->p,leaves->p,raw->p);checked();
            const float* native=oracle->predict_values(raw->p,n,true);
            audit_rows<<<blocks(n),128>>>(begin,n,width,view(),profile,cuts->p,leaves->p,offsets->p,raw->p,native,state->p);checked();
            if(factors){audit_factor_rows<<<blocks(n),128>>>(n,profile,cuts->p,f->p,fo->p,raw->p,native,state->p);checked();out.factor_margin_words+=n*7;}
            out.rows+=n;out.native_margin_words+=n*7;begin+=n;
        }
        if(halt(stop)){out.reason="cancelled_after_native_audit";return finish();}
        identities();out.complete=true;out.reason="GPU_source_partition_ordered_native_margin_witnesses_equal";return finish();
    }
};
Bridge::Bridge(const std::string& l,const std::string& m,const std::string& e,Options o,int d):p_(std::make_unique<Impl>(l,m,e,o,d)){}
Bridge::~Bridge()=default;
Result Bridge::extract(const Stop& stop){auto& x=*p_;need(!x.attempted,"leaf-partition extraction is single-use");x.attempted=true;cu(cudaSetDevice(x.device));Result r;r.source_sha256=x.source.identity;r.library_sha256=x.library_sha;
    auto finish=[&](){r.owned_device_peak_bytes=x.budget.peak;r.owned_device_resident_bytes=x.budget.used;return r;};
    auto halt=[&](const char* why){if(x.halt(stop)){r.reason=why;return true;}return false;};
    if(halt("cancelled_before_profile"))return finish();U n=x.source.feature.size();
    {Buf<U> a(x.budget,n),z(x.budget,n);keys<<<blocks(n),128>>>(x.view(),a.p,x.state->p);x.checked();r.CUDA_executed=true;
     size_t temporary=0;cu(cub::DeviceRadixSort::SortKeys(nullptr,temporary,a.p,z.p,int(n)));Buf<unsigned char> temp(x.budget,temporary);cu(cub::DeviceRadixSort::SortKeys(temp.p,temporary,a.p,z.p,int(n)));make_profile<<<1,1>>>(z.p,int(n),x.cuts->p,x.dp->p,x.state->p);x.checked();x.profile=x.dp->get(1)[0];}
    if(halt("cancelled_after_profile"))return finish();
    extract_leaves<<<blocks(x.source.leaf_nodes.size()),128>>>(x.view(),x.profile,x.cuts->p,x.staged->p,x.keep->p,x.state->p);x.checked();
    compact_layout<<<1,1>>>(x.view(),x.keep->p,x.destination->p,x.offsets->p,x.state->p);x.checked();
    compact_leaves<<<blocks(x.source.leaf_nodes.size()),128>>>(int(x.source.leaf_nodes.size()),x.staged->p,x.keep->p,x.destination->p,x.leaves->p);x.checked();
    partition_audit<<<blocks(x.source.roots.size()),128>>>(x.view(),x.profile,x.leaves->p,x.offsets->p,x.state->p);x.checked();
    if(halt("cancelled_after_partition_audit"))return finish();
    x.width=105+3*x.profile.cuts;x.rows=mul(x.width,x.status.kept);
    if(x.rows>x.options.maximum_native_rows){r.reason="native_witness_row_cap";return finish();}
    auto cuts=x.cuts->get(x.profile.cuts);std::string rank_bytes;
    for(int f=0;f<10;++f){r.domain.lo[f]=x.profile.minimum[f];r.domain.hi[f]=x.profile.count[f];r.rank_cut_bits[f].assign(cuts.begin()+x.profile.offset[f],cuts.begin()+x.profile.offset[f]+x.profile.count[f]);unsigned count=x.profile.count[f];rank_bytes.append(reinterpret_cast<const char*>(&count),sizeof(count));if(count)rank_bytes.append(reinterpret_cast<const char*>(r.rank_cut_bits[f].data()),count*sizeof(unsigned));}
    x.rank_sha=dpnative::sha256(rank_bytes);x.binding=dpnative::sha256(x.source.identity+x.library_sha+x.rank_sha+"finite-FP32-valid-Forest-leaf-partitions-v1");
    x.raw=std::make_unique<Buf<float>>(x.budget,mul(std::min(x.rows,x.options.native_batch_rows),54));
    x.oracle=std::make_unique<dpnative::XGBoostOracle>(x.library_path,x.model_path,54,7,x.device);
    auto audit=x.audit(nullptr,stop);if(!audit.complete){r.reason=audit.reason;return finish();}
    if(halt("cancelled_before_publication"))return finish();
    r.source.binding=x.binding;r.source.value.leaves=x.leaves->get(x.status.kept);r.source.value.offsets=x.offsets->get(x.source.roots.size()+1);r.source.value.channels=x.source.channels;std::copy(x.source.bias.begin(),x.source.bias.end(),r.source.value.bias.begin());
    r.original_leaves=x.source.leaf_nodes.size();r.feasible_leaves=x.status.kept;r.empty_leaves=x.status.empty;r.disjoint_pairs_checked=x.status.pairs;r.complete_tree_volumes_checked=x.status.volumes;r.native_witness_rows=audit.rows;r.native_margin_words=audit.native_margin_words;r.rank_sha256=x.rank_sha;r.owned_device_peak_bytes=x.budget.peak;r.owned_device_resident_bytes=x.budget.used;r.complete=true;r.reason="GPU_disjoint_complete_source_leaf_partitions_native_margin_witnesses_equal";x.extracted=true;return finish();
}
Audit Bridge::audit_factors(const rank_gpu_score_factors::Result& factors,const Stop& stop){need(p_->extracted,"leaf-partition factors require completed extraction");return p_->audit(&factors,stop);}
}
