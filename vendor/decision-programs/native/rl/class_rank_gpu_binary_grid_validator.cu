#include "streaming_file_hash.hpp"
#include "class_rank_gpu_binary_grid_validator.hpp"
#include "class_xgboost.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
namespace rank_gpu_binary_grid_validator { namespace {
void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
U mul(U a,U b){need(!b||a<=UINT64_MAX/b,"grid capacity arithmetic overflow");return a*b;}
struct Budget {U maximum,used=0,peak=0;void add(U n){need(n<=maximum-used,"grid owned-device-byte capacity exceeded");used+=n;peak=std::max(peak,used);}void sub(U n){used-=n;}};
template<class T>struct Buf {
 T*p=nullptr;U n=0;Budget*b=nullptr;
 Buf(Budget&budget,U count):n(count),b(&budget){U bytes=mul(n,sizeof(T));b->add(bytes);if(bytes){auto e=cudaMalloc(&p,bytes);if(e==cudaSuccess)e=cudaMemset(p,0,bytes);if(e!=cudaSuccess){if(p)cudaFree(p);b->sub(bytes);p=nullptr;b=nullptr;cu(e);}}}
 ~Buf(){if(p)cudaFree(p);if(b)b->sub(n*sizeof(T));}Buf(const Buf&)=delete;Buf&operator=(const Buf&)=delete;
 void set(const std::vector<T>&v){need(v.size()==n,"grid transport buffer size");if(n)cu(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
 std::vector<T>get(U count=UINT64_MAX)const{if(count==UINT64_MAX)count=n;need(count<=n,"grid transport count");std::vector<T>v(count);if(count)cu(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
struct Predicate {int feature=-1;std::uint32_t bits=0;};
struct DeviceProfile {
 U count=0,cells=0;int error=0;
 unsigned offsets[10]{},counts[10]{},minimum[10]{};
 unsigned cat_count[2]{},tested_count[2]{},representatives[2][40]{},features[2][40]{},cat_bits[2][40]{};
 U radix[12]{},stride[12]{};
};
__device__ unsigned ordered_word(unsigned b){return (b&0x80000000u)?~b:(b^0x80000000u);}
__device__ unsigned from_ordered(unsigned b){return (b&0x80000000u)?(b^0x80000000u):~b;}
__global__ void extract_keys(const int*feature,const int*left,const float*cut,U n,U*keys,int*error){
 U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;keys[i]=UINT64_MAX;if(left[i]<0)return;
 int f=feature[i];float c=cut[i];if(f<0||f>=54||!isfinite(c)){atomicExch(error,1);return;}if(c==0)c=0.f;
 keys[i]=(U(unsigned(f))<<32)|ordered_word(__float_as_uint(c));
}
__global__ void make_profile(const U*keys,U n,Predicate*questions,DeviceProfile*out,U maximum_cells){
 if(threadIdx.x||blockIdx.x)return;DeviceProfile p{};U previous=UINT64_MAX;
 unsigned active[2][40]{},chosen[2][40]{};
 for(U i=0;i<n;++i){U key=keys[i];if(key==UINT64_MAX)break;if(i&&key==previous)continue;previous=key;
  Predicate q{int(key>>32),from_ordered(unsigned(key))};questions[p.count]=q;
  if(q.feature<10){unsigned f=q.feature;if(p.counts[f]==0)p.offsets[f]=unsigned(p.count);++p.counts[f];}
  else {int g=q.feature<14?0:1,k=q.feature-(g?14:10);float c=__uint_as_float(q.bits);if(c>0.f&&c<=1.f&&!active[g][k]){active[g][k]=1;chosen[g][k]=q.bits;}}
  ++p.count;
 }
 for(int f=0;f<10;++f){p.minimum[f]=p.counts[f]&&questions[p.offsets[f]].bits==0xff7fffffu;p.radix[f]=U(p.counts[f])+1-p.minimum[f];}
 for(int g=0;g<2;++g){unsigned extent=g?40:4;int other=-1;for(unsigned k=0;k<extent;++k){if(active[g][k]){unsigned at=p.tested_count[g]++;p.representatives[g][at]=k;p.features[g][at]=(g?14:10)+k;p.cat_bits[g][at]=chosen[g][k];}else if(other<0)other=int(k);}p.cat_count[g]=p.tested_count[g];if(other>=0)p.representatives[g][p.cat_count[g]++]=unsigned(other);p.radix[10+g]=p.cat_count[g];}
 U cells=1;for(int d=11;d>=0;--d){p.stride[d]=cells;if(!p.radix[d]||p.radix[d]>maximum_cells/cells){p.error=2;*out=p;return;}cells*=p.radix[d];}p.cells=cells;*out=p;
}
__global__ void audit_profile(DeviceProfile p,const Predicate*q,const int*feature,const int*left,const float*cut,U n,int*error){
 U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n||left[i]<0)return;
 if(p.count>n||feature[i]<0||feature[i]>=54){atomicExch(error,2);return;}
 float c=cut[i];if(c==0)c=0.f;bool found=false;int f=feature[i];
 if(f<10){if(p.offsets[f]>p.count||p.counts[f]>p.count-p.offsets[f]){atomicExch(error,2);return;}for(unsigned j=0;j<p.counts[f];++j){auto a=q[p.offsets[f]+j];found=found||(a.feature==f&&a.bits==__float_as_uint(c));}}
 else for(U k=0;k<p.count;++k)found=found||(q[k].feature==f&&q[k].bits==__float_as_uint(c));
 if(!found)atomicExch(error,2);
}
// Independent profile checker covers every category in the exactly-one domain,
// including all members of OTHER; no representative-only category assumption.
__global__ void audit_partition(DeviceProfile p,const Predicate*q,U capacity,int*error){
 if(threadIdx.x||blockIdx.x)return;
 if(p.count>capacity){atomicExch(error,3);return;}
 for(int f=0;f<10;++f)if(p.offsets[f]>p.count||p.counts[f]>p.count-p.offsets[f]||p.minimum[f]>1||p.radix[f]!=U(p.counts[f])+1-p.minimum[f]){atomicExch(error,3);return;}
 for(int g=0;g<2;++g){unsigned extent=g?40:4;if(!p.cat_count[g]||p.cat_count[g]>extent||p.tested_count[g]>extent||p.cat_count[g]!=p.tested_count[g]+unsigned(p.tested_count[g]<extent)||p.radix[10+g]!=p.cat_count[g]){atomicExch(error,5);return;}}
 for(int f=0;f<10;++f){for(unsigned j=0;j<p.counts[f];++j){auto a=q[p.offsets[f]+j];if(a.feature!=f||!isfinite(__uint_as_float(a.bits))||(j&& !(__uint_as_float(q[p.offsets[f]+j-1].bits)<__uint_as_float(a.bits))))atomicExch(error,3);}if(p.minimum[f]!=(p.counts[f]&&q[p.offsets[f]].bits==0xff7fffffu))atomicExch(error,4);}
 for(int g=0;g<2;++g){int first=g?14:10,extent=g?40:4;
  for(int category=0;category<extent;++category){unsigned matches=0;
   for(unsigned r=0;r<p.cat_count[g];++r){unsigned rep=p.representatives[g][r];if(rep>=unsigned(extent))atomicExch(error,5);bool same=true;
    for(U k=0;k<p.count;++k)if(q[k].feature>=first&&q[k].feature<first+extent){float c=__uint_as_float(q[k].bits);same= same && ((float(q[k].feature==first+category)<c)==(float(q[k].feature==first+int(rep))<c));}matches+=same;
   }if(matches!=1)atomicExch(error,6);
  }
  for(unsigned j=0;j+1<p.cat_count[g];++j){if(j>=p.tested_count[g]||p.features[g][j]!=unsigned(first)+p.representatives[g][j]||!(__uint_as_float(p.cat_bits[g][j])>0.f&&__uint_as_float(p.cat_bits[g][j])<=1.f))atomicExch(error,7);bool found=false;for(U k=0;k<p.count;++k)found=found||(q[k].feature==int(p.features[g][j])&&q[k].bits==p.cat_bits[g][j]);if(!found)atomicExch(error,8);}
 }
 if(p.error==2)return;
 U cells=1;for(int d=11;d>=0;--d){if(p.stride[d]!=cells||!p.radix[d]||cells>UINT64_MAX/p.radix[d]){atomicExch(error,9);return;}cells*=p.radix[d];}if(cells!=p.cells)atomicExch(error,10);
}
__global__ void witnesses(DeviceProfile p,const Predicate*q,U base,U n,float*raw,rank_gpu_dag_inference::Row*ranks){
 U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;U s=base+i;auto&r=ranks[i];r={};for(int f=0;f<54;++f)raw[i*54+f]=0.f;
 for(int f=0;f<10;++f){unsigned rank=unsigned((s/p.stride[f])%p.radix[f])+p.minimum[f];r.rank[f]=rank;raw[i*54+f]=rank?__uint_as_float(q[p.offsets[f]+rank-1].bits):-FLT_MAX;}
 unsigned a=p.representatives[0][(s/p.stride[10])%p.radix[10]],b=p.representatives[1][s%p.radix[11]];raw[i*54+10+a]=raw[i*54+14+b]=1.f;r.categories=(1ull<<a)|(1ull<<(4+b));
}
__global__ void audit_witnesses(DeviceProfile p,const Predicate*q,U base,U n,const float*raw,const rank_gpu_dag_inference::Row*ranks,int*error){
 U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;U s=base+i;const auto&r=ranks[i];int a=0,b=0;
 for(int f=0;f<54;++f){float v=raw[i*54+f];if(!isfinite(v))atomicExch(error,11);if(f>=10){if(v!=0.f&&v!=1.f)atomicExch(error,12);if(f<14)a+=int(v);else b+=int(v);if(((r.categories>>(f-10))&1)!=U(v))atomicExch(error,13);}}
 if(a!=1||b!=1)atomicExch(error,14);
 unsigned category_a=p.representatives[0][(s/p.stride[10])%p.radix[10]],category_b=p.representatives[1][(s/p.stride[11])%p.radix[11]];
 U expected_mask=(1ull<<category_a)|(1ull<<(4+category_b));if(r.categories!=expected_mask||raw[i*54+10+category_a]!=1.f||raw[i*54+14+category_b]!=1.f)atomicExch(error,14);
 for(int f=0;f<10;++f){unsigned rank=0;for(unsigned j=0;j<p.counts[f];++j)rank+=raw[i*54+f]>=__uint_as_float(q[p.offsets[f]+j].bits);if(rank!=r.rank[f]||rank!=unsigned((s/p.stride[f])%p.radix[f])+p.minimum[f])atomicExch(error,15);}
}
__global__ void labels_kernel(U base,U n,const float*prob,int*labels,int*error){U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;int best=0;for(int k=0;k<7;++k){float v=prob[i*7+k];if(!isfinite(v))atomicExch(error,16);if(v>prob[i*7+best])best=k;}labels[base+i]=best;}
__global__ void margins_kernel(U n,const float*raw,const float*native,const int*roots,const int*channels,U trees,const int*feature,const int*left,const int*right,const float*cut,const float*value,const float*bias,U nodes,int*error){
 U s=U(blockIdx.x)*blockDim.x+threadIdx.x;if(s>=n)return;float acc[7];for(int k=0;k<7;++k)acc[k]=bias[k];
 for(U t=0;t<trees;++t){int at=roots[t];U steps=0;while(at>=0&&U(at)<nodes&&left[at]>=0&&++steps<=nodes){int f=feature[at];if(f<0||f>=54){atomicExch(error,17);return;}at=raw[s*54+f]<cut[at]?left[at]:right[at];}if(at<0||U(at)>=nodes||steps>nodes||channels[t]<0||channels[t]>=7){atomicExch(error,18);return;}acc[channels[t]]=__fadd_rn(acc[channels[t]],value[at]);}
 for(int k=0;k<7;++k)if(__float_as_uint(acc[k])!=__float_as_uint(native[s*7+k]))atomicExch(error,19);
}
__global__ void audit_graph(U n,const rank_gpu_dag_inference::Row*ranks,const int*labels,const dl::Node*nodes,U count,U root,int*error,U*checked){
 U s=U(blockIdx.x)*blockDim.x+threadIdx.x;if(s>=n)return;U at=root;
 for(U step=0;step<=count;++step){if(at>=count){atomicExch(error,24);return;}auto v=nodes[at];if(v.kind==2){if(v.label!=labels[s])atomicExch(error,25);atomicAdd(reinterpret_cast<unsigned long long*>(checked),1ull);return;}if(v.kind!=0||v.feature<0||v.feature>=54||v.left>=at||v.right>=at){atomicExch(error,26);return;}float x=v.feature<10?float(ranks[s].rank[v.feature]):float((ranks[s].categories>>(v.feature-10))&1);at=x<__uint_as_float(v.cut_bits)?v.left:v.right;}
 atomicExch(error,27);
}
void sync(){cu(cudaGetLastError());cu(cudaDeviceSynchronize());}
int blocks(U n){need(n>0&&n<=U(INT_MAX)*128,"grid CUDA launch capacity exceeded");return int((n+127)/128);}
struct Product {U words[4]{};int fits=0,within=0,error=0;};
__global__ void count_product(DeviceProfile p,U budget,Product*out){
 if(threadIdx.x||blockIdx.x)return;Product x{};x.words[0]=1;
 for(int d=0;d<12;++d){U factor=p.radix[d],carry=0;if(!factor||factor>16777217ull){x.error=1;break;}
  for(int k=0;k<4;++k){U lo=x.words[k]*factor,hi=__umul64hi(x.words[k],factor),sum=lo+carry;x.words[k]=sum;carry=hi+(sum<lo);}if(carry){x.error=2;break;}
 }
 x.fits=!x.error&&!x.words[1]&&!x.words[2]&&!x.words[3];x.within=x.fits&&x.words[0]&&x.words[0]<=budget;
 if(x.fits&&(p.error||p.cells!=x.words[0]))x.error=3;if(!x.fits&&p.error!=2)x.error=4;*out=x;
}

// New independent persisted-binary certificate begins here.
__global__ void exact_native_words(const float*observed,const float*expected,U n,U*checked,int*error){U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;if(__float_as_uint(observed[i])!=__float_as_uint(expected[i])||!isfinite(observed[i]))atomicExch(error,51);atomicAdd(reinterpret_cast<unsigned long long*>(checked),1ull);}
struct Certificate {U nodes=0,numeric=0,entries=0,values=0,comparisons=0;};
__device__ int dimension(const dl::Node&v){return v.kind==2?12:v.feature<10?v.feature:v.feature<14?10:11;}
__device__ void increment(U*p,U n=1){atomicAdd(reinterpret_cast<unsigned long long*>(p),n);}
__global__ void binary_structure(DeviceProfile p,const dl::Node*nodes,U n,U root,codec::tasks::Box scope,unsigned*entry,Certificate*cert,int*error){
 U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 if(!i){if(root>=n||scope.allowed!=(1ull<<44)-1)atomicExch(error,31);for(int f=0;f<10;++f)if(scope.lo[f]!=int(p.minimum[f])||scope.hi[f]!=int(p.counts[f]))atomicExch(error,32);if(root<n)atomicExch(entry+root,1u);}
 auto v=nodes[i];if(v.id!=i||v.term_count||v.first_term||v.threshold_bits){atomicExch(error,33);return;}
 increment(&cert->nodes);if(v.kind==2){if(v.label<0||v.label>=7)atomicExch(error,34);return;}
 if(v.kind!=0||v.feature<0||v.feature>=54||v.left>=i||v.right>=i||!isfinite(__uint_as_float(v.cut_bits))){atomicExch(error,35);return;}
 int d=dimension(v);for(U child:{v.left,v.right}){auto c=nodes[child];if(c.kind!=0&&c.kind!=2){atomicExch(error,36);return;}int cd=dimension(c);if(cd<d){atomicExch(error,37);return;}if(cd>d&&cd<12)atomicExch(entry+child,1u);}
 if(d<10){bool boundary=false;for(unsigned k=1;k<=p.counts[d];++k)boundary|=v.cut_bits==__float_as_uint(float(k));if(!boundary)atomicExch(error,38);increment(&cert->numeric);}
}
__device__ U category_exit(const dl::Node*nodes,U n,U at,int group,int category,int*error){
 int d=10+group,first=group?14:10;for(U steps=0;steps<=n;++steps){if(at>=n){atomicExch(error,39);return UINT64_MAX;}auto v=nodes[at];if(v.kind==2||dimension(v)>d)return at;if(v.kind!=0||dimension(v)!=d||v.left>=at||v.right>=at){atomicExch(error,40);return UINT64_MAX;}at=float(v.feature==first+category)<__uint_as_float(v.cut_bits)?v.left:v.right;}atomicExch(error,41);return UINT64_MAX;
}
__global__ void categorical_constancy(DeviceProfile p,const Predicate*q,const dl::Node*nodes,U n,const unsigned*entry,Certificate*cert,int*error){
 U i=U(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n||!entry[i]||nodes[i].kind==2)return;int d=dimension(nodes[i]);if(d<10)return;int g=d-10,extent=g?40:4,first=g?14:10;U exits[40];
 for(int c=0;c<extent;++c)exits[c]=category_exit(nodes,n,i,g,c,error);
 for(int c=0;c<extent;++c){unsigned matches=0;for(unsigned r=0;r<p.cat_count[g];++r){auto rep=p.representatives[g][r];bool same=true;for(U k=0;k<p.count;++k)if(q[k].feature>=first&&q[k].feature<first+extent){float cut=__uint_as_float(q[k].bits);same= same && ((float(q[k].feature==first+c)<cut)==(float(q[k].feature==first+int(rep))<cut));}if(same){++matches;increment(&cert->comparisons);if(rep>=unsigned(extent)||exits[c]!=exits[rep])atomicExch(error,42);}}if(matches!=1)atomicExch(error,43);}
 increment(&cert->entries);increment(&cert->values,extent);
}
void stopped(const Stop&stop){if(stop&&stop())throw std::runtime_error("validator cancelled before complete coverage");}
}
struct Validator::Impl {
 std::string native_contract;Options options;Budget budget;dpnative::SourceData source;std::unique_ptr<dpnative::XGBoostOracle>oracle;
 std::unique_ptr<Buf<int>>feature,left,right,roots,channels,error,labels;
 std::unique_ptr<Buf<float>>cut,value,bias,raw;
 std::unique_ptr<Buf<Predicate>>questions;std::unique_ptr<Buf<DeviceProfile>>device_profile;
 std::unique_ptr<Buf<rank_gpu_dag_inference::Row>>ranks;
 DeviceProfile p{};ProfileReport report;bool profiled=false,failed=false;U batch=0;
 Impl(const std::string&library,const std::string&model,Options o):options(std::move(o)),budget{options.maximum_device_bytes},source(dpnative::read_source(model)){
  validate_options(options);const auto&v=options;need(v.maximum_cells>0&&v.maximum_device_bytes>0&&v.native_batch_rows>0&&v.native_batch_rows<=1048576,"stream grid invalid capacity options");
  const auto&s=source;need(!s.feature.empty()&&s.features==54&&s.outputs==7&&s.feature.size()<=v.maximum_source_nodes&&s.roots.size()<=v.maximum_source_trees,"stream grid source dimensions/capacity");need(s.feature.size()<=16777216&&s.roots.size()<=U(INT_MAX),"stream grid source index/rank capacity");
  need(dp_streaming::sha256_file(library)=="462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4","validator native library hash differs");oracle=std::make_unique<dpnative::XGBoostOracle>(library,model,54,7);need(s.bytes==dpnative::read_text(model),"stream grid source changed during native load");
  cudaDeviceProp prop{};int driver=0,runtime=0;cu(cudaGetDeviceProperties(&prop,0));cu(cudaDriverGetVersion(&driver));cu(cudaRuntimeGetVersion(&runtime));need(prop.major==8&&prop.minor==6&&std::string(prop.name)=="NVIDIA RTX A5000 Laptop GPU"&&driver==13040&&runtime==13040,"validator native device/runtime unsupported");using J=dpnative::json;auto info=J::parse(oracle->build_info());need(dpnative::sha256(info.dump())=="66af94fc63f2d242b37516a7b27ee0c8270cdd8de874dfc9c0315e9d3775379d","validator native build unsupported");native_contract=J{{"source_revision","v3.4.1"},{"library_sha256","462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4"},{"gpu",prop.name},{"driver",driver},{"runtime",runtime},{"native_build_info",info},{"transform_source","https://github.com/dmlc/xgboost/blob/v3.4.1/src/objective/multiclass_obj.cu#L138-L149"},{"softmax_source","https://github.com/dmlc/xgboost/blob/v3.4.1/src/common/math.h#L63-L80"},{"trusted_semantic_premise","pinned native predictor respects finite source leaf paths and ordered scores; transform is deterministic and row-local"},{"CUDA_implementation_formally_refined",false},{"loaded_image_recaptured",false}}.dump();
  auto ints=[&](const std::vector<int>&a){auto b=std::make_unique<Buf<int>>(budget,a.size());b->set(a);return b;};auto floats=[&](const std::vector<float>&a){auto b=std::make_unique<Buf<float>>(budget,a.size());b->set(a);return b;};
  feature=ints(s.feature);left=ints(s.left);right=ints(s.right);roots=ints(s.roots);channels=ints(s.channels);cut=floats(s.cut);value=floats(s.value);bias=floats(s.bias);
  error=std::make_unique<Buf<int>>(budget,1);cu(cudaMemset(error->p,0,sizeof(int)));questions=std::make_unique<Buf<Predicate>>(budget,s.feature.size());device_profile=std::make_unique<Buf<DeviceProfile>>(budget,1);
 }
 void checked(){int e=error->get()[0];need(e==0,("stream grid GPU verification code "+std::to_string(e)).c_str());}
 Profile transport_profile(){Profile out;out.distinct_count=p.count;out.cells=p.cells;out.owned_device_peak_bytes=budget.peak;auto q=questions->get(p.count);for(int f=0;f<10;++f){out.minimum_rank[f]=p.minimum[f];for(unsigned j=0;j<p.counts[f];++j)out.cuts[f].push_back(q[p.offsets[f]+j].bits);}for(int g=0;g<2;++g){auto&c=out.categories[g];c.representatives.assign(p.representatives[g],p.representatives[g]+p.cat_count[g]);c.tested_features.assign(p.features[g],p.features[g]+p.tested_count[g]);c.cut_bits.assign(p.cat_bits[g],p.cat_bits[g]+p.tested_count[g]);}for(int d=0;d<12;++d){out.radices[d]=p.radix[d];out.strides[d]=p.stride[d];}return out;}
 void profile(){if(profiled)return;need(!failed,"stream grid compiler failed");U n=source.feature.size();
  {Buf<U>keys(budget,n),sorted(budget,n);extract_keys<<<blocks(n),128>>>(feature->p,left->p,cut->p,n,keys.p,error->p);sync();checked();std::size_t bytes=0;cu(cub::DeviceRadixSort::SortKeys(nullptr,bytes,keys.p,sorted.p,int(n)));Buf<unsigned char>temporary(budget,bytes);cu(cub::DeviceRadixSort::SortKeys(temporary.p,bytes,keys.p,sorted.p,int(n)));sync();make_profile<<<1,1>>>(sorted.p,n,questions->p,device_profile->p,UINT64_MAX);sync();p=device_profile->get()[0];need(p.error==0||p.error==2,"stream grid invalid source profile");}
  audit_profile<<<blocks(n),128>>>(p,questions->p,feature->p,left->p,cut->p,n,error->p);cu(cudaGetLastError());audit_partition<<<1,1>>>(p,questions->p,n,error->p);sync();checked();Buf<Product>count(budget,1);count_product<<<1,1>>>(p,options.maximum_cells,count.p);sync();auto x=count.get()[0];need(!x.error,"stream grid exact product audit failed");report.profile=transport_profile();for(int k=0;k<4;++k)report.cell_count_words[k]=x.words[k];report.fits_uint64=x.fits;report.within_cell_budget=x.within;profiled=true;
 }
 void buffers(){if(raw)return;need(report.fits_uint64&&report.within_cell_budget,"stream grid feasible-cell capacity exceeded; inspect profile_only");batch=std::min(p.cells,options.native_batch_rows);raw=std::make_unique<Buf<float>>(budget,mul(batch,54));labels=std::make_unique<Buf<int>>(budget,batch);ranks=std::make_unique<Buf<rank_gpu_dag_inference::Row>>(budget,batch);}
 void evaluate(U base,U n){
  witnesses<<<blocks(n),128>>>(p,questions->p,base,n,raw->p,ranks->p);sync();audit_witnesses<<<blocks(n),128>>>(p,questions->p,base,n,raw->p,ranks->p,error->p);sync();checked();
  const float*prob=oracle->predict_values(raw->p,n,false);labels_kernel<<<blocks(n),128>>>(0,n,prob,labels->p,error->p);sync();checked();
  const float*margin=oracle->predict_values(raw->p,n,true);margins_kernel<<<blocks(n),128>>>(n,raw->p,margin,roots->p,channels->p,source.roots.size(),feature->p,left->p,right->p,cut->p,value->p,bias->p,source.feature.size(),error->p);sync();checked();
 }
};
U planned_batch_device_bytes(U rows){need(rows>0&&rows<=1048576,"validator batch rows must be1..1048576");return mul(rows,54*sizeof(float)+sizeof(int)+sizeof(rank_gpu_dag_inference::Row));}
void validate_options(const Options&o){planned_batch_device_bytes(o.native_batch_rows);need(o.maximum_cells&&o.maximum_cells<=UINT64_MAX/14&&o.maximum_device_bytes&&o.maximum_source_nodes&&o.maximum_source_trees&&o.maximum_candidate_nodes,"validator options must be positive");}
Validator::Validator(const std::string&library,const std::string&source,Options o):p_(std::make_unique<Impl>(library,source,std::move(o))){}Validator::~Validator()=default;
ProfileReport Validator::profile_only(){p_->profile();return p_->report;}
Result Validator::audit(const codec::Model&m,const Stop&stop){auto&x=*p_;Result result;try{
 stopped(stop);need(!m.nodes.empty()&&m.nodes.size()<=x.options.maximum_candidate_nodes&&m.root<m.nodes.size()&&m.terms.empty(),"validator candidate shape/capacity");need(m.source_sha256==x.source.identity&&m.rank_sha256==codec::rank_digest(m.rank_cut_bits),"validator candidate source/rank identity mismatch");x.profile();result.CUDA_executed=true;result.profile=x.report.profile;result.cells=x.p.cells;result.native_configuration=x.oracle->configuration_json();result.native_contract=x.native_contract;need(m.rank_cut_bits==x.report.profile.cuts,"persisted runtime rank table differs from complete source profile");stopped(stop);need(x.report.fits_uint64&&x.report.within_cell_budget,"validator complete-cell work budget exceeded");x.buffers();
 cu(cudaMemset(x.error->p,0,sizeof(int)));Buf<dl::Node>graph(x.budget,m.nodes.size());graph.set(m.nodes);Buf<unsigned>entry(x.budget,m.nodes.size());Buf<Certificate>cert(x.budget,1);Buf<U>checked(x.budget,1);
 binary_structure<<<blocks(m.nodes.size()),128>>>(x.p,graph.p,m.nodes.size(),m.root,m.scope,entry.p,cert.p,x.error->p);sync();x.checked();categorical_constancy<<<blocks(m.nodes.size()),128>>>(x.p,x.questions->p,graph.p,m.nodes.size(),entry.p,cert.p,x.error->p);sync();x.checked();auto c=cert.get()[0];result.checked_nodes=c.nodes;result.numeric_questions=c.numeric;result.category_entries=c.entries;result.category_values=c.values;result.category_equivalence_comparisons=c.comparisons;result.candidate_source_cell_constancy=true;stopped(stop);
 U remaining=x.p.cells;while(remaining){stopped(stop);U n=std::min(x.batch,remaining),base=remaining-n;x.evaluate(base,n);audit_graph<<<blocks(n),128>>>(n,x.ranks->p,x.labels->p,graph.p,m.nodes.size(),m.root,x.error->p,checked.p);sync();x.checked();remaining=base;result.native_rows+=n;result.native_margin_words+=7*n;++result.batches;if(x.options.progress)x.options.progress(result.native_rows,x.p.cells);}
 need(checked.get()[0]==x.p.cells&&result.native_rows==x.p.cells,"validator full native coverage differs");stopped(stop);result.complete=true;result.reason="every source quotient cell passed native public class and ordered margin words; persisted binary source-cell constancy proved by full coordinate checks";
 }catch(const std::exception&e){result.reason=e.what();}result.owned_device_peak_bytes=x.budget.peak;return result;}
BatchAudit Validator::audit_batch_words(U first,U second,const Stop&stop){auto&x=*p_;BatchAudit out;try{planned_batch_device_bytes(first);planned_batch_device_bytes(second);U small=std::min(first,second),large=std::max(first,second);need(large%small==0,"batch word audit requires nested aligned batch sizes");stopped(stop);x.profile();out.CUDA_executed=true;need(x.report.fits_uint64&&x.report.within_cell_budget,"batch word audit complete-cell budget exceeded");U capacity=std::min(large,x.p.cells);cu(cudaMemset(x.error->p,0,sizeof(int)));Buf<float>raw(x.budget,mul(capacity,54)),margin(x.budget,mul(capacity,7)),probability(x.budget,mul(capacity,7));Buf<rank_gpu_dag_inference::Row>ranks(x.budget,capacity);Buf<U>checked(x.budget,1);
 for(U base=0;base<x.p.cells;){stopped(stop);U count=std::min(capacity,x.p.cells-base);witnesses<<<blocks(count),128>>>(x.p,x.questions->p,base,count,raw.p,ranks.p);audit_witnesses<<<blocks(count),128>>>(x.p,x.questions->p,base,count,raw.p,ranks.p,x.error->p);sync();x.checked();
 const float*m=x.oracle->predict_values(raw.p,count,true);++out.native_calls;out.native_rows+=count;margins_kernel<<<blocks(count),128>>>(count,raw.p,m,x.roots->p,x.channels->p,x.source.roots.size(),x.feature->p,x.left->p,x.right->p,x.cut->p,x.value->p,x.bias->p,x.source.feature.size(),x.error->p);sync();x.checked();cu(cudaMemcpy(margin.p,m,mul(count,7*sizeof(float)),cudaMemcpyDeviceToDevice));
 const float*p=x.oracle->predict_values(raw.p,count,false);++out.native_calls;out.native_rows+=count;cu(cudaMemcpy(probability.p,p,mul(count,7*sizeof(float)),cudaMemcpyDeviceToDevice));stopped(stop);
 for(U offset=0;offset<count;){stopped(stop);U n=std::min(small,count-offset);const float*observed_margin=x.oracle->predict_values(raw.p+offset*54,n,true);++out.native_calls;out.native_rows+=n;exact_native_words<<<blocks(mul(n,7)),128>>>(observed_margin,margin.p+offset*7,n*7,checked.p,x.error->p);sync();x.checked();out.margin_words+=n*7;const float*observed_probability=x.oracle->predict_values(raw.p+offset*54,n,false);++out.native_calls;out.native_rows+=n;exact_native_words<<<blocks(mul(n,7)),128>>>(observed_probability,probability.p+offset*7,n*7,checked.p,x.error->p);sync();x.checked();out.probability_words+=n*7;offset+=n;}
 base+=count;out.cells=base;}
 need(checked.get()[0]==mul(x.p.cells,14)&&out.cells==x.p.cells,"native batch word audit coverage mismatch");stopped(stop);out.complete=true;out.reason="all source quotient cells exact native margin and public probability words equal under both batch partitions";
 }catch(const std::exception&e){out.reason=e.what();}out.owned_device_peak_bytes=x.budget.peak;return out;}

}
