#pragma once
#include "rl_category_policy.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
namespace rl_qualified_session::resident_policy1 {
using rl_category_policy::U;using rl_category_policy::Request;
using rl_category_policy::Trace;using rl_category_policy::Credit;
using rl_category_policy::categories;using rl_category_policy::maximum_actions;
using rl_category_policy::wild_mask;using rl_category_policy::soil_mask;
struct State {double logits[categories]{};U version=1,updates=0;};
struct Meta {U version=0,seed=0,episode=0;unsigned error=0,sampled=0;double reward=0;U updates=0,baseline_bytes=0;};
static_assert(sizeof(State)==368&&sizeof(Meta)==56);
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
__device__ void sample_order(const State*s,Trace*traces,unsigned*order,Meta*m,U seed,U episode,U baseline_bytes){
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
} // namespace rl_qualified_session::resident_policy1
