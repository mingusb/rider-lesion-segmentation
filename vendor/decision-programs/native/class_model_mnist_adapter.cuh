#pragma once
// Dataset adapter: exact historical IDX transport, fixed split and scoring.
// Shared Runtime owns all persisted-model inference; this file only packs
// pixels, verifies labels and scores already computed class IDs on CUDA.
#include "class_model_io.cuh"
#include <iomanip>
#include <map>
void need(bool b,const std::string&m){if(!b)throw std::runtime_error(m);}
void cu(cudaError_t e){ck(e,"MNIST CUDA");}
void gpu_sync(){done();}
std::string mn_read(const fs::path&p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"input read failed");auto n=f.tellg();need(n>=0&&U(n)<=1024ULL*1024*1024,"input extent");std::string s(size_t(n),'\0');f.seekg(0);if(n)f.read(s.data(),n);need(bool(f)&&f.peek()==EOF&&!f.bad(),"fresh input read incomplete");return s;}
void Cpath(const fs::path&p){need(p.is_absolute()&&p.string().find('\0')==std::string::npos,"absolute dataset or output path required");(void)fs::weakly_canonical(p);}
unsigned be32(const char*p){return (unsigned(uint8_t(p[0]))<<24)|(unsigned(uint8_t(p[1]))<<16)|(unsigned(uint8_t(p[2]))<<8)|unsigned(uint8_t(p[3]));}
#include "class_model_native_adapter.cuh"
struct PackStats{unsigned bad,labels[2][10];};
__global__ void synthetic(uint8_t*x,uint8_t*y,unsigned n){unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r<n){y[r]=r%10;for(unsigned c=0;c<784;++c)x[U(r)*784+c]=(r*13+c*7)%256;}}
__global__ void pack(const uint8_t*raw,const uint8_t*labels,unsigned n,unsigned fit,float*x,float*y,unsigned*ids,unsigned*seen,PackStats*s){unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;unsigned original=(U(r)*17+1337)%n;ids[r]=original;atomicAdd(seen+original,1);unsigned label=labels[original];y[r]=float(label);if(label>=10)atomicAdd(&s->bad,1);else atomicAdd(&s->labels[r>=fit][label],1);for(unsigned c=0;c<784;++c)x[U(r)*784+c]=float(raw[U(original)*784+c]);}
__global__ void audit_pack(const uint8_t*raw,const uint8_t*labels,unsigned n,const float*x,const float*y,const unsigned*ids,const unsigned*seen,PackStats*s){unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;unsigned original=(U(r)*17+1337)%n;bool bad=ids[r]!=original||seen[r]!=1||y[r]!=float(labels[original]);for(unsigned c=0;c<784;++c)bad|=x[U(r)*784+c]!=float(raw[U(original)*784+c]);if(bad)atomicAdd(&s->bad,1);}
struct Score{unsigned bad,errors,classes[10],confusion[100];};
__global__ void score(const float*p,const float*y,unsigned n,Score*s){unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;unsigned label=unsigned(y[r]),best=0;float v=p[U(r)*10],sum=0;bool bad=label>=10||y[r]!=float(label);for(unsigned k=0;k<10;++k){float q=p[U(r)*10+k];bad|=!isfinite(q)||q<0||q>1;sum+=q;if(q>v){v=q;best=k;}}bad|=fabsf(sum-1)>0.0001f;if(bad){atomicAdd(&s->bad,1);return;}atomicAdd(&s->classes[label],1);atomicAdd(&s->confusion[label*10+best],1);atomicAdd(&s->errors,unsigned(label!=best));}
__global__ void synthetic_probabilities(const float*y,unsigned n,float*p,unsigned mode){unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;unsigned label=unsigned(y[r]);for(unsigned k=0;k<10;++k)p[U(r)*10+k]=mode==0?(k==label?1.f:0.f):0.1f;}
J mnist_fixture(){J cases=J::array();for(unsigned n:{128u,257u}){need(std::gcd(n,17u)==1,"fixture split bijection");unsigned f=n*5/6;Dev<uint8_t>raw(U(n)*784),labels(n);Dev<float>x(U(n)*784),y(n);Dev<unsigned>ids(n),seen(n);Dev<PackStats>s(1);seen.zero();s.zero();synthetic<<<(n+255)/256,256>>>(raw.p,labels.p,n);pack<<<(n+255)/256,256>>>(raw.p,labels.p,n,f,x.p,y.p,ids.p,seen.p,s.p);audit_pack<<<(n+255)/256,256>>>(raw.p,labels.p,n,x.p,y.p,ids.p,seen.p,s.p);gpu_sync();auto a=s.get()[0];need(!a.bad,"fixture pack mismatch");unsigned sizes[2]={};for(unsigned part=0;part<2;++part)for(unsigned k=0;k<10;++k)sizes[part]+=a.labels[part][k];need(sizes[0]==f&&sizes[1]==n-f,"fixture population");Dev<float>probs(U(n)*10);for(unsigned mode=0;mode<2;++mode){Dev<Score>result(1);result.zero();synthetic_probabilities<<<(n+255)/256,256>>>(y.p,n,probs.p,mode);score<<<(n+255)/256,256>>>(probs.p,y.p,n,result.p);gpu_sync();auto q=result.get()[0];need(!q.bad,"fixture score probability failure");unsigned expected=mode? n-a.labels[0][0]-a.labels[1][0]:0;need(q.errors==expected,"fixture score error count");for(unsigned k=0;k<10;++k){need(q.classes[k]==a.labels[0][k]+a.labels[1][k],"fixture score class counts");for(unsigned j=0;j<10;++j){unsigned want=mode?(j==0?q.classes[k]:0):(j==k?q.classes[k]:0);need(q.confusion[k*10+j]==want,"fixture score confusion");}}}cases.push_back({{"rows",n},{"FIT",f},{"VALID",n-f},{"bijection_and_all784_pixel_words_exact",true}});}return {{"complete",true},{"CUDA_executed",true},{"cases",cases},{"actual_data_read",false},{"TEST_read",false}};}

__global__ void word_probe_rows(float*x,const u32*feature,const u32*word,u32 rows){for(u64 t=u64(blockIdx.x)*blockDim.x+threadIdx.x;t<u64(rows)*784;t+=u64(blockDim.x)*gridDim.x){u32 r=t/784,c=t%784;x[t]=c==feature[r]?__uint_as_float(word[r]):0.f;}}
__global__ void probability_norm(const float*p,u32 rows,u32*bad){for(u32 r=blockIdx.x*blockDim.x+threadIdx.x;r<rows;r+=blockDim.x*gridDim.x){float sum=0;bool invalid=false;for(u32 k=0;k<10;++k){float q=p[u64(r)*10+k];invalid|=!isfinite(q)||q<0||q>1;sum+=q;}invalid|=fabsf(sum-1)>0.0001f;if(invalid)atomicAdd(bad,1);}}
__global__ void fixture_truth(const float*x,float*y,u32*out,u32 rows){for(u32 r=blockIdx.x*blockDim.x+threadIdx.x;r<rows;r+=u32(blockDim.x)*gridDim.x){u32 k=__float_as_uint(x[u64(r)*784])==0x80000001u?0:1;y[r]=float(k);out[r]=k;}}
J counts(const Score&s,u32 rows){need(!s.bad,"invalid label or native probability");J cl=J::array(),cf=J::array();for(auto k:s.classes)cl.push_back(k);for(auto k:s.confusion)cf.push_back(k);return {{"CUDA_computed",true},{"rows",rows},{"errors",s.errors},{"accuracy",double(rows-s.errors)/rows},{"class_counts",cl},{"confusion_rows_true_columns_predicted",cf}};}

__global__ void score_direct_classes(const float*p,const float*y,unsigned n,unsigned*out,Score*s){
 unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 float q=p[r],labelword=y[r];out[r]=10;
 if(!isfinite(q)||q<0||q>=10||q!=floorf(q)||!isfinite(labelword)||labelword<0||labelword>=10||labelword!=floorf(labelword)){
  atomicAdd(&s->bad,1);return;
 }
 unsigned best=unsigned(q),label=unsigned(labelword);out[r]=best;
 atomicAdd(&s->classes[label],1);atomicAdd(&s->confusion[label*10+best],1);atomicAdd(&s->errors,unsigned(label!=best));
}
__global__ void direct_fixture_words(float*y,float*p,unsigned n,unsigned mode){
 unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 y[r]=float(r%10);p[r]=mode==0?y[r]:mode==1?0.f:mode==2?10.f:mode==3?-1.f:mode==4?0.5f:__int_as_float(0x7fc00000);
}
__global__ void direct_fixture_ids(const unsigned*actual,unsigned n,unsigned mode,unsigned*bad){
 unsigned r=blockIdx.x*blockDim.x+threadIdx.x;if(r<n){unsigned expected=mode==0?r%10:mode==1?0:10;if(actual[r]!=expected)atomicAdd(bad,1);}
}
J direct_fixture(){
 const unsigned n=257;Dev<float>p(n),y(n);Dev<unsigned>ids(n),bad(1);J cases=J::array();
 for(unsigned mode=0;mode<6;++mode){Dev<Score>s(1);s.zero();bad.zero();direct_fixture_words<<<2,256>>>(y.p,p.p,n,mode);score_direct_classes<<<2,256>>>(p.p,y.p,n,ids.p,s.p);direct_fixture_ids<<<2,256>>>(ids.p,n,mode,bad.p);done();auto a=s.get()[0];need(!bad.at(0),"native direct ID transport mismatch");
  if(mode<2){need(!a.bad,"valid direct-class fixture rejected");need(a.errors==(mode?231u:0u),"direct-class errors differ");for(unsigned k=0;k<10;++k){unsigned expected=n/10+(k<n%10);need(a.classes[k]==expected,"direct class population");for(unsigned j=0;j<10;++j)need(a.confusion[k*10+j]==((mode?j==0:j==k)?expected:0),"direct-class confusion");}}
  else need(a.bad==n&&!a.errors,"invalid direct-class word accepted");
  cases.push_back({{"mode",mode},{"rows",n},{"invalid_words",a.bad},{"errors",a.errors},{"ID_copy_mismatches",0}});
 }
 return {{"complete",true},{"CUDA_executed",true},{"CPU_predictions",false},{"cases",cases},{"TEST_read",false}};
}

u32 inference_block_size(const J&p){
 auto b=p.value("inference_block_size",J(256));
 need(b.is_number_integer(),"inference block size must be an integer");
 auto n=b.get<int64_t>();need(n==64||n==128||n==256,"inference block size must be64,128or256");return u32(n);
}
u32 inference_blocks(u32 rows,u32 threads){need(threads==64||threads==128||threads==256,"inference block size invalid");return u32((u64(rows)+threads-1)/threads);}


__global__ void score_class_ids(const u32*predicted,const float*y,u32 rows,Score*s){
 for(u32 r=blockIdx.x*blockDim.x+threadIdx.x;r<rows;r+=blockDim.x*gridDim.x){u32 best=predicted[r];float word=y[r];
  if(best>=10||!isfinite(word)||word<0||word>=10||word!=floorf(word)){atomicAdd(&s->bad,1u);continue;}
  u32 label=u32(word);atomicAdd(&s->classes[label],1u);atomicAdd(&s->confusion[label*10+best],1u);atomicAdd(&s->errors,u32(label!=best));
 }
}
