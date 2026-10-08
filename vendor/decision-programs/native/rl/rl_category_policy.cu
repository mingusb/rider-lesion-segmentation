#include "rl_category_policy.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace rl_category_policy {namespace {
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void sync(){cu(cudaGetLastError());cu(cudaDeviceSynchronize());}
struct State {double logits[categories]{};U version=1,updates=0;};
struct Meta {U version=0,seed=0,episode=0;unsigned error=0,sampled=0;double reward=0;U updates=0,baseline_bytes=0;};
static_assert(sizeof(State)==368&&sizeof(Meta)==56);
template<class T>struct Buffer {T*p=nullptr;U count;explicit Buffer(U n):count(n){cu(cudaMalloc(&p,n*sizeof(T)));}~Buffer(){if(p)cudaFree(p);}Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;T read()const{T v{};cu(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost));return v;}};
__device__ U mix(U x){x+=0x9e3779b97f4a7c15ull;x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;return x^(x>>31);}
__device__ U rng(U seed,U episode,U version,unsigned group,unsigned step){return mix(seed^mix(episode)^mix(version)^mix(U(group)+0x632be59bd9b4e019ull)^mix(U(step)+0x8cb92baa7f3d8dd7ull));}
__device__ bool distribution(const State&s,U mask,double*weights,double&sum,double&maximum){maximum=-CUDART_INF;sum=0;for(unsigned j=0;j<categories;++j){weights[j]=0;if(mask&(U(1)<<j)){if(!isfinite(s.logits[j]))return false;maximum=fmax(maximum,s.logits[j]);}}if(!isfinite(maximum))return false;for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j)){weights[j]=exp(s.logits[j]-maximum);sum+=weights[j];}return isfinite(sum)&&sum>0;}
__device__ bool sample(const State&s,Request request,Trace&out){
 out=Trace{};out.version=s.version;out.seed=request.seed;out.episode=request.episode;out.allowed_mask=request.allowed_mask;out.fallback_mask=request.fallback_mask;out.group=request.group;
 if(request.group>1||request.reserved||!s.version)return false;U group=request.group?soil_mask:wild_mask;
 if(!request.allowed_mask||(request.allowed_mask&~group)||(request.fallback_mask&~request.allowed_mask))return false;
 U mask=request.allowed_mask&~request.fallback_mask;
 for(unsigned step=0;mask;++step){if(step>=maximum_actions)return false;double weights[categories],sum,maximum;if(!distribution(s,mask,weights,sum,maximum))return false;
  double u=double(rng(request.seed,request.episode,s.version,request.group,step)>>11)*0x1p-53;double target=u*sum,cumulative=0;unsigned selected=categories,last=categories;
  for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j)){if(weights[j]>0)last=j;cumulative+=weights[j];if(selected==categories&&target<cumulative)selected=j;}
  if(selected==categories)selected=last;if(selected==categories||!(weights[selected]>0))return false;
  double lp=s.logits[selected]-maximum-log(sum);if(!isfinite(lp))return false;out.actions[step]=selected;out.remaining[step]=mask;out.log_probability[step]=lp;out.total_log_probability+=lp;
  for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j))out.gradient[j]-=weights[j]/sum;out.gradient[selected]+=1;mask&=~(U(1)<<selected);++out.steps;
 }
 return isfinite(out.total_log_probability);
}
__global__ void initialize(State*s,Meta*m){if(blockIdx.x||threadIdx.x)return;*s=State{};*m=Meta{};}
__global__ void sample_order(const State*s,Trace*traces,unsigned*order,Meta*m,U seed,U episode,U baseline_bytes){
 if(blockIdx.x||threadIdx.x)return;Meta next{};next.version=s->version;next.seed=seed;next.episode=episode;next.baseline_bytes=baseline_bytes;
 if(baseline_bytes>(U(1)<<53)){next.error=11;*m=next;return;}
 for(unsigned g=0;g<2;++g){Request q{seed,episode,g?soil_mask:wild_mask,0,g,0};if(!sample(*s,q,traces[g])){next.error=1;*m=next;return;}unsigned offset=g?4:0;for(unsigned j=0;j<traces[g].steps;++j)order[offset+j]=traces[g].actions[j];}
 next.sampled=1;*m=next;
}
__device__ void apply_update(State*s,const Trace*traces,Meta*m,Credit credit){
 m->error=0;m->reward=0;
 if(!m->sampled||credit.sampled_version!=s->version||m->version!=s->version||credit.sampled_seed!=m->seed||credit.sampled_episode!=m->episode){m->error=2;return;}
 if(credit.complete!=1||credit.independently_validated!=1||credit.scope_qualified!=1||credit.reserved){m->error=3;return;}
 if(credit.encoded_bytes>(U(1)<<53)||credit.baseline_bytes>(U(1)<<53)||credit.baseline_bytes!=m->baseline_bytes){m->error=4;return;}
 double lr=__longlong_as_double(credit.learning_rate_bits);if(!isfinite(lr)||lr<=0||lr>1){m->error=5;return;}
 if(s->version==UINT64_MAX||s->updates==UINT64_MAX){m->error=6;return;}
 double reward=double(credit.baseline_bytes)-double(credit.encoded_bytes),proposed[categories];
 for(unsigned g=0;g<2;++g)if(traces[g].version!=s->version||traces[g].seed!=m->seed||traces[g].episode!=m->episode||traces[g].group!=g){m->error=7;return;}
 for(unsigned j=0;j<categories;++j){double gradient=traces[0].gradient[j]+traces[1].gradient[j];proposed[j]=s->logits[j]+lr*reward*gradient;if(!isfinite(gradient)||!isfinite(proposed[j])){m->error=8;return;}}
 for(unsigned j=0;j<categories;++j)s->logits[j]=proposed[j];++s->version;++s->updates;m->version=s->version;m->updates=s->updates;m->reward=reward;m->sampled=0;
}
__global__ void update(State*s,const Trace*traces,Meta*m,Credit credit){if(!blockIdx.x&&!threadIdx.x)apply_update(s,traces,m,credit);}
__global__ void upload_words(State*s,Meta*m,const U*words,U version){if(blockIdx.x||threadIdx.x)return;m->error=0;if(!version){m->error=9;return;}for(unsigned j=0;j<categories;++j)if(!isfinite(__longlong_as_double(words[j]))){m->error=10;return;}for(unsigned j=0;j<categories;++j)s->logits[j]=__longlong_as_double(words[j]);s->version=version;s->updates=0;*m=Meta{};m->version=version;}
__global__ void download_words(const State*s,U*words){unsigned j=threadIdx.x;if(j<categories)words[j]=__double_as_longlong(s->logits[j]);}
// Fixtures invoke the same sampler and update device functions as production.
// Their synthetic credit declarations confer no native classifier authority.
struct DeviceReport {U assertions=0,rejections=0,gradient_cases=0,update_cases=0,first_failure=0;};
__device__ void expect(DeviceReport&r,bool ok,U id){++r.assertions;if(!ok&&!r.first_failure)r.first_failure=id;}
__device__ bool same_state(const State&a,const State&b){if(a.version!=b.version||a.updates!=b.updates)return false;for(unsigned j=0;j<categories;++j)if(__double_as_longlong(a.logits[j])!=__double_as_longlong(b.logits[j]))return false;return true;}
__device__ bool same_trace(const Trace&a,const Trace&b){if(a.version!=b.version||a.seed!=b.seed||a.episode!=b.episode||a.allowed_mask!=b.allowed_mask||a.fallback_mask!=b.fallback_mask||a.group!=b.group||a.steps!=b.steps)return false;for(unsigned i=0;i<maximum_actions;++i)if(a.actions[i]!=b.actions[i]||a.remaining[i]!=b.remaining[i]||__double_as_longlong(a.log_probability[i])!=__double_as_longlong(b.log_probability[i]))return false;for(unsigned j=0;j<categories;++j)if(__double_as_longlong(a.gradient[j])!=__double_as_longlong(b.gradient[j]))return false;return __double_as_longlong(a.total_log_probability)==__double_as_longlong(b.total_log_probability);}
// Independent fixed-action likelihood, recomputing eligible masks from actions
// rather than using the production gradient or retained remaining[] arrays.
__device__ double fixed_likelihood(const State&s,const Trace&t){U mask=t.allowed_mask&~t.fallback_mask;double ll=0;for(unsigned i=0;i<t.steps;++i){double maximum=-CUDART_INF;for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j))maximum=fmax(maximum,s.logits[j]);double sum=0;for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j))sum+=exp(s.logits[j]-maximum);ll+=s.logits[t.actions[i]]-maximum-log(sum);mask&=~(U(1)<<t.actions[i]);}return ll;}
__device__ double expected_synthetic_reward(const State&s){double value=0;for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)if(a!=b){Trace t{};t.allowed_mask=7;t.steps=3;t.actions[0]=a;t.actions[1]=b;t.actions[2]=3-a-b;double reward=120.0-double(100+5*a+3*b);value+=exp(fixed_likelihood(s,t))*reward;}return value;}
__device__ void check_trace(DeviceReport&r,const State&s,const Request&q,const Trace&t,U base){
 U mask=q.allowed_mask&~q.fallback_mask,seen=0;expect(r,t.steps==unsigned(__popcll(mask)),base);expect(r,t.version==s.version&&t.seed==q.seed&&t.episode==q.episode&&t.group==q.group,base+1);
 for(unsigned i=0;i<t.steps;++i){expect(r,t.remaining[i]==mask,base+2);unsigned bit=t.actions[i];bool valid=bit<categories&&(mask&(U(1)<<bit));expect(r,valid,base+3);if(!valid)return;seen|=U(1)<<bit;mask&=~(U(1)<<bit);expect(r,isfinite(t.log_probability[i])&&t.log_probability[i]<=0,base+4);}
 expect(r,!mask&&seen==(q.allowed_mask&~q.fallback_mask),base+5);double sum=0;for(unsigned j=0;j<categories;++j){expect(r,isfinite(t.gradient[j]),base+6);if(!(q.allowed_mask&(U(1)<<j))||(q.fallback_mask&(U(1)<<j)))expect(r,t.gradient[j]==0,base+7);sum+=t.gradient[j];}expect(r,fabs(sum)<1e-12,base+8);expect(r,fabs(t.total_log_probability-fixed_likelihood(s,t))<1e-12,base+9);
}
__device__ Meta sampled_meta(const State&s,U baseline){Meta m{};m.version=s.version;m.seed=123;m.episode=7;m.baseline_bytes=baseline;m.sampled=1;m.updates=s.updates;return m;}
__device__ bool episode_traces(const State&s,const Meta&m,Trace*traces){for(unsigned g=0;g<2;++g)if(!sample(s,Request{m.seed,m.episode,g?soil_mask:wild_mask,0,g,0},traces[g]))return false;return true;}
__device__ Credit credit_for(const Meta&m,U bytes,double lr){Credit c{};c.sampled_version=m.version;c.sampled_seed=m.seed;c.sampled_episode=m.episode;c.encoded_bytes=bytes;c.baseline_bytes=m.baseline_bytes;c.learning_rate_bits=__double_as_longlong(lr);c.complete=c.independently_validated=c.scope_qualified=1;return c;}
__global__ void checks_kernel(DeviceReport*out){
 if(blockIdx.x||threadIdx.x)return;DeviceReport r{};State uniform{};Trace t{},again{};
 // Complete, sparse and fixed-fallback masks in both categorical groups.
 for(unsigned g=0;g<2;++g)for(unsigned kind=0;kind<4;++kind){U allowed=g?soil_mask:wild_mask;U fallback=0;if(kind==1)allowed=g?((U(1)<<4)|(U(1)<<17)|(U(1)<<43)):5;if(kind==2)fallback=g?(U(1)<<19):2;if(kind==3)fallback=allowed;Request q{91,12,allowed,fallback,g,0};expect(r,sample(uniform,q,t),100+kind);check_trace(r,uniform,q,t,110+kind*10);expect(r,sample(uniform,q,again)&&same_trace(t,again),160+kind);}
 // Uniform two-item likelihood has exactly +/- 1/2 score gradient.
 Request pair{23,11,3,0,0,0};expect(r,sample(uniform,pair,t),200);expect(r,t.steps==2&&t.log_probability[0]==-log(2.0)&&t.log_probability[1]==0,201);expect(r,t.gradient[t.actions[0]]==0.5&&t.gradient[t.actions[1]]==-0.5,202);r.gradient_cases+=2;
 // Validate input eligibility independently of learning or native qualification.
 for(unsigned kind=0;kind<6;++kind){Request q{1,2,wild_mask,0,0,0};if(kind==0)q.group=2;if(kind==1)q.reserved=1;if(kind==2)q.allowed_mask=0;if(kind==3)q.allowed_mask=U(1)<<4;if(kind==4)q.fallback_mask=U(1)<<4;if(kind==5)q.allowed_mask|=U(1)<<63;expect(r,!sample(uniform,q,t),220+kind);++r.rejections;}
 State nonfinite=uniform;nonfinite.logits[0]=CUDART_NAN;expect(r,!sample(nonfinite,pair,t),230);++r.rejections;nonfinite.logits[0]=CUDART_INF;expect(r,!sample(nonfinite,pair,t),231);++r.rejections;
 // Nonuniform gradients checked by CUDA central finite differences for all44
 // coordinates and both groups, including zero gradients for absent features.
 State varying{};for(unsigned j=0;j<categories;++j)varying.logits[j]=(int(j%7)-3)*0.2;
 for(unsigned g=0;g<2;++g){Request q{567,89,g?soil_mask:wild_mask,0,g,0};expect(r,sample(varying,q,t),300+g);check_trace(r,varying,q,t,310+10*g);for(unsigned j=0;j<categories;++j){State plus=varying,minus=varying;plus.logits[j]+=1e-5;minus.logits[j]-=1e-5;double derivative=(fixed_likelihood(plus,t)-fixed_likelihood(minus,t))/(2e-5);expect(r,isfinite(derivative)&&fabs(derivative-t.gradient[j])<2e-8,350+j);++r.gradient_cases;}}
 // Reproducibility binds version, seed and episode; changing seed exercises
 // multiple actions without claiming a statistical RNG-quality certificate.
 expect(r,sample(uniform,pair,t),400);unsigned differences=0;for(U seed=0;seed<16;++seed){Request q=pair;q.seed=seed;expect(r,sample(uniform,q,again),401);differences+=again.actions[0]!=t.actions[0];}expect(r,differences>0,402);
 // Three-category PL probability normalization and expected score gradient.
 double probability_sum=0,expected_gradient[3]{};for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)if(a!=b){unsigned c=3-a-b;Trace path{};path.allowed_mask=7;path.steps=3;path.actions[0]=a;path.actions[1]=b;path.actions[2]=c;double probability=exp(fixed_likelihood(varying,path));probability_sum+=probability;U mask=7;for(unsigned k=0;k<3;++k){double weights[categories],sum,maximum;expect(r,distribution(varying,mask,weights,sum,maximum),420);for(unsigned j=0;j<3;++j)if(mask&(U(1)<<j))expected_gradient[j]-=probability*weights[j]/sum;expected_gradient[path.actions[k]]+=probability;mask&=~(U(1)<<path.actions[k]);}}
 expect(r,fabs(probability_sum-1)<1e-12,421);for(unsigned j=0;j<3;++j)expect(r,fabs(expected_gradient[j])<1e-12,422+j);r.gradient_cases+=3;
 // Six complete synthetic outcomes: score-function derivative of expected
 // encoded-byte reward agrees with differentiating its enumerated expectation.
 // Two different fixed baselines produce the same expected derivative.
 double reward_gradient[3]{},other_baseline_gradient[3]{};for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)if(a!=b){Trace path{};path.allowed_mask=7;path.steps=3;path.actions[0]=a;path.actions[1]=b;path.actions[2]=3-a-b;double p=exp(fixed_likelihood(varying,path));double cost=double(100+5*a+3*b),gradient[3]{};U mask=7;for(unsigned k=0;k<3;++k){double weights[categories],sum,maximum;expect(r,distribution(varying,mask,weights,sum,maximum),430);for(unsigned j=0;j<3;++j)if(mask&(U(1)<<j))gradient[j]-=weights[j]/sum;gradient[path.actions[k]]+=1;mask&=~(U(1)<<path.actions[k]);}for(unsigned j=0;j<3;++j){reward_gradient[j]+=p*(120-cost)*gradient[j];other_baseline_gradient[j]+=p*(-cost)*gradient[j];}}
 for(unsigned j=0;j<3;++j){State plus=varying,minus=varying;plus.logits[j]+=1e-5;minus.logits[j]-=1e-5;double derivative=(expected_synthetic_reward(plus)-expected_synthetic_reward(minus))/(2e-5);expect(r,fabs(derivative-reward_gradient[j])<2e-8,431+j);expect(r,fabs(reward_gradient[j]-other_baseline_gradient[j])<1e-12,434+j);++r.gradient_cases;}
 // Actual production REINFORCE updates use one joint trajectory score. A
 // positive complete-byte improvement raises its fixed-action likelihood;
 // a worse encoding lowers it; a zero reward preserves all logit words.
 Trace traces[2]{};for(unsigned mode=0;mode<3;++mode){State s=uniform,before=s;Meta m=sampled_meta(s,100);expect(r,episode_traces(s,m,traces),500);Credit c=credit_for(m,mode==0?90:mode==1?110:100,1e-4);double old_ll=fixed_likelihood(s,traces[0])+fixed_likelihood(s,traces[1]);apply_update(&s,traces,&m,c);expect(r,!m.error&&!m.sampled&&s.version==2&&s.updates==1,501);double new_ll=fixed_likelihood(s,traces[0])+fixed_likelihood(s,traces[1]);expect(r,mode==0?new_ll>old_ll:mode==1?new_ll<old_ll:new_ll==old_ll,502);unsigned changed=0;for(unsigned j=0;j<categories;++j)changed+=__double_as_longlong(s.logits[j])!=__double_as_longlong(before.logits[j]);expect(r,mode==2?changed==0:changed>0,503);expect(r,m.reward==(mode==0?10.0:mode==1?-10.0:0.0),504);State committed=s;apply_update(&s,traces,&m,c);expect(r,m.error==2&&same_state(s,committed),505);++r.rejections;++r.update_cases;}
 // Every rejection must leave all policy words, version and count unchanged.
 for(unsigned kind=0;kind<15;++kind){State s=uniform;Meta m=sampled_meta(s,100);expect(r,episode_traces(s,m,traces),600);Credit c=credit_for(m,90,1e-4);if(kind==0)c.complete=0;if(kind==1)c.independently_validated=0;if(kind==2)c.scope_qualified=0;if(kind==3)c.reserved=1;if(kind==4)++c.sampled_version;if(kind==5)++c.sampled_seed;if(kind==6)++c.sampled_episode;if(kind==7)c.encoded_bytes=(U(1)<<53)+1;if(kind==8)c.baseline_bytes=101;if(kind==9)c.learning_rate_bits=__double_as_longlong(0.0);if(kind==10)c.learning_rate_bits=__double_as_longlong(CUDART_NAN);if(kind==11)c.learning_rate_bits=__double_as_longlong(1.01);if(kind==12)traces[1].version=7;if(kind==13){s.version=m.version=c.sampled_version=UINT64_MAX;traces[0].version=traces[1].version=UINT64_MAX;}if(kind==14)m.sampled=0;State before=s;apply_update(&s,traces,&m,c);expect(r,m.error&&same_state(s,before),610+kind);++r.rejections;++r.update_cases;}
 // Stable softmax under a large finite common shift; rejected nonfinite is
 // separate from softmax underflow. Finite-difference cases stay moderate.
 State shifted=uniform;for(unsigned j=0;j<categories;++j)shifted.logits[j]=1e300;expect(r,sample(shifted,pair,t),700);expect(r,t.gradient[t.actions[0]]==0.5&&isfinite(t.total_log_probability),701);
 *out=r;
}
} // anonymous
struct Policy::Impl {
 int device;U current=1;bool has_trace=false;Buffer<State>state{1};Buffer<Meta>meta{1};Buffer<Trace>traces{2};Buffer<unsigned>order{44};Buffer<U>words{44};
 explicit Impl(int d):device(d){initialize<<<1,1>>>(state.p,meta.p);sync();}
};
Policy::Policy(int device){cu(cudaSetDevice(device));p_=std::make_unique<Impl>(device);}Policy::~Policy(){if(p_){cudaSetDevice(p_->device);p_.reset();}}Policy::Policy(Policy&&)noexcept=default;Policy&Policy::operator=(Policy&&)noexcept=default;
OrderView Policy::sample44(U seed,U episode,U baseline_bytes){cu(cudaSetDevice(p_->device));p_->has_trace=false;sample_order<<<1,1>>>(p_->state.p,p_->traces.p,p_->order.p,p_->meta.p,seed,episode,baseline_bytes);sync();auto m=p_->meta.read();if(m.error)throw std::runtime_error("GPU categorical policy sampling rejected");p_->has_trace=true;return {p_->order.p,p_->traces.p,m.version,seed,episode,p_->device,baseline_bytes};}
std::array<Trace,2>Policy::trace()const{if(!p_->has_trace)throw std::runtime_error("GPU policy trace not sampled");cu(cudaSetDevice(p_->device));std::array<Trace,2>x{};cu(cudaMemcpy(x.data(),p_->traces.p,sizeof(x),cudaMemcpyDeviceToHost));return x;}
Update Policy::reinforce(const Credit&credit){cu(cudaSetDevice(p_->device));U old=p_->current;update<<<1,1>>>(p_->state.p,p_->traces.p,p_->meta.p,credit);sync();auto m=p_->meta.read();Update r{!m.error,old,p_->current,m.updates,int(m.error),m.error?"GPU_update_rejected":"GPU_REINFORCE_complete_episode_update"};if(!m.error){p_->current=m.version;r.new_version=m.version;}return r;}
std::array<U,categories>Policy::logit_words()const{cu(cudaSetDevice(p_->device));download_words<<<1,64>>>(p_->state.p,p_->words.p);sync();std::array<U,categories>x{};cu(cudaMemcpy(x.data(),p_->words.p,sizeof(x),cudaMemcpyDeviceToHost));return x;}
void Policy::upload(const std::array<U,categories>&words,U version){cu(cudaSetDevice(p_->device));cu(cudaMemcpy(p_->words.p,words.data(),sizeof(words),cudaMemcpyHostToDevice));upload_words<<<1,1>>>(p_->state.p,p_->meta.p,p_->words.p,version);sync();if(p_->meta.read().error)throw std::runtime_error("GPU categorical policy word upload rejected");p_->current=version;p_->has_trace=false;}
U Policy::version()const{return p_->current;}U Policy::owned_device_bytes()const{return sizeof(State)+sizeof(Meta)+2*sizeof(Trace)+44*sizeof(unsigned)+44*sizeof(U);}
CheckResult gpu_checks(int device){
 cu(cudaSetDevice(device));Buffer<DeviceReport>report{1};checks_kernel<<<1,1>>>(report.p);sync();auto d=report.read();CheckResult r{!d.first_failure,true,false,d.assertions,d.rejections,d.gradient_cases,d.update_cases,d.first_failure};
 auto host_expect=[&](bool ok,U id){++r.assertions;if(!ok&&!r.first_failure)r.first_failure=id;};
 // Public adapter smoke: host compares only integer metadata and FP words;
 // all probability, score and reward arithmetic runs inside CUDA.
 Policy policy(device);bool rejected=false;try{(void)policy.trace();}catch(const std::exception&){rejected=true;}host_expect(rejected,800);++r.rejections;
 auto initial=policy.logit_words();for(U word:initial)host_expect(word==0,801);host_expect(policy.version()==1&&policy.owned_device_bytes()==3368,802);
 auto view=policy.sample44(23,7,100);host_expect(view.version==1&&view.seed==23&&view.episode==7&&view.baseline_bytes==100&&view.order44&&view.traces,803);
 std::array<unsigned,44>order{};cu(cudaMemcpy(order.data(),view.order44,sizeof(order),cudaMemcpyDeviceToHost));U seen=0;for(unsigned i=0;i<44;++i){unsigned bit=order[i];bool good=bit<44&&((i<4&&bit<4)||(i>=4&&bit>=4))&&!(seen&(U(1)<<bit));host_expect(good,804);if(bit<44)seen|=U(1)<<bit;}host_expect(seen==((U(1)<<44)-1),805);
 Credit credit{};credit.sampled_version=1;credit.sampled_seed=23;credit.sampled_episode=7;credit.encoded_bytes=90;credit.baseline_bytes=100;credit.learning_rate_bits=0x3f1a36e2eb1c432dull;credit.complete=credit.independently_validated=credit.scope_qualified=1;
 auto updated=policy.reinforce(credit);host_expect(updated.applied&&updated.old_version==1&&updated.new_version==2&&updated.updates==1,806);auto learned=policy.logit_words();host_expect(learned!=initial,807);auto duplicate=policy.reinforce(credit);host_expect(!duplicate.applied&&duplicate.error==2&&policy.logit_words()==learned,808);++r.rejections;
 policy.upload(initial,9);host_expect(policy.version()==9&&policy.logit_words()==initial,809);rejected=false;try{(void)policy.trace();}catch(const std::exception&){rejected=true;}host_expect(rejected,810);++r.rejections;
 auto invalid=initial;invalid[0]=0x7ff0000000000000ull;rejected=false;try{policy.upload(invalid,10);}catch(const std::exception&){rejected=true;}host_expect(rejected&&policy.version()==9&&policy.logit_words()==initial,811);++r.rejections;
 rejected=false;try{policy.upload(initial,0);}catch(const std::exception&){rejected=true;}host_expect(rejected&&policy.version()==9&&policy.logit_words()==initial,812);++r.rejections;
 rejected=false;try{(void)policy.sample44(0,0,(U(1)<<53)+1);}catch(const std::exception&){rejected=true;}host_expect(rejected&&policy.version()==9&&policy.logit_words()==initial,813);++r.rejections;
 r.passed=!r.first_failure;return r;
}
} // namespace
