#pragma once
#include "resident_policy1_device.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
// Isolated qualified block64 body extracted as a cooperative device helper.
// Caller MUST invoke uniformly from all threads of exactly one block, with
// blockDim.x >=44. State/Meta/Trace lifetime spans the synchronized call.
// Call outside any thread0-only branch. Resident block128/inlining requires
// fresh ROOT all-word/control qualification before integration acceptance.
namespace resident_cooperative_policy1 {
using resident_policy1::State;using resident_policy1::Meta;
using resident_policy1::rng;
using rl_category_policy::U;using rl_category_policy::Trace;
using rl_category_policy::categories;using rl_category_policy::maximum_actions;
using rl_category_policy::wild_mask;using rl_category_policy::soil_mask;
__device__ void cooperative_order(const State*s,Trace*traces,unsigned*order,Meta*m,U seed,U episode,U baseline_bytes){
 __shared__ U mask;
 __shared__ unsigned selected;
 __shared__ double maximum,sum,weights[44];
 if(blockIdx.x)return;
 if(!threadIdx.x){*m=Meta{};m->version=s->version;m->seed=seed;m->episode=episode;m->baseline_bytes=baseline_bytes;
  if(baseline_bytes>(U(1)<<53))m->error=11;else if(!s->version)m->error=1;}
 __syncthreads();if(m->error)return;
 for(unsigned g=0;g<2;++g){
  if(!threadIdx.x){traces[g]=Trace{};auto&t=traces[g];t.version=s->version;t.seed=seed;t.episode=episode;
   t.allowed_mask=g?soil_mask:wild_mask;t.group=g;mask=t.allowed_mask;}
  __syncthreads();
  for(unsigned step=0;mask;++step){
   if(!threadIdx.x){maximum=-CUDART_INF;sum=0;
    if(step>=maximum_actions)m->error=1;
    for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j)){
     if(!isfinite(s->logits[j]))m->error=1;maximum=fmax(maximum,s->logits[j]);}
    if(!isfinite(maximum))m->error=1;}
   __syncthreads();if(m->error)return;
   unsigned j=threadIdx.x;
   if(j<categories){weights[j]=0;if(mask&(U(1)<<j))weights[j]=exp(s->logits[j]-maximum);}
   __syncthreads();
   if(!threadIdx.x){for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j))sum+=weights[j];
    if(!isfinite(sum)||sum<=0)m->error=1;
    else {
     double u=double(rng(seed,episode,s->version,g,step)>>11)*0x1p-53;double target=u*sum,cumulative=0;selected=categories;unsigned last=categories;
     for(unsigned j=0;j<categories;++j)if(mask&(U(1)<<j)){if(weights[j]>0)last=j;cumulative+=weights[j];if(selected==categories&&target<cumulative)selected=j;}
     if(selected==categories)selected=last;
     if(selected==categories||!(weights[selected]>0))m->error=1;
     else {double lp=s->logits[selected]-maximum-log(sum);if(!isfinite(lp))m->error=1;
      else {auto&t=traces[g];t.actions[step]=selected;t.remaining[step]=mask;t.log_probability[step]=lp;t.total_log_probability+=lp;}}
    }}
   __syncthreads();if(m->error)return;
   if(j<categories&&mask&(U(1)<<j)){traces[g].gradient[j]-=weights[j]/sum;if(j==selected)traces[g].gradient[j]+=1;}
   __syncthreads();
   if(!threadIdx.x){mask&=~(U(1)<<selected);++traces[g].steps;}
   __syncthreads();
  }
  if(!threadIdx.x){if(!isfinite(traces[g].total_log_probability))m->error=1;
   else {unsigned offset=g?4:0;for(unsigned j=0;j<traces[g].steps;++j)order[offset+j]=traces[g].actions[j];}}
  __syncthreads();if(m->error)return;
 }
 if(!threadIdx.x)m->sampled=1;
}
}
