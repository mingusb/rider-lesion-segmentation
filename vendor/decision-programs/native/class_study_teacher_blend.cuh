#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace class_study::teacher_blend {
inline constexpr const char* arithmetic_policy="oof_brier_fixed256_pairwise_fp64_alpha32_rn_v1";
inline constexpr unsigned reduction_width=256;
struct Pair {double numerator,denominator;};
// Consecutive fixed 256-row leaves, padded with exact zero. Every higher level
// uses the same pairwise tree on consecutive prior-level partials. No atomics
// participate in the numerical reductions; status atomics only report refusal.
__global__ static void partials(const float* raw,unsigned raw_width,unsigned teacher_column,
    const float* responses,unsigned response_width,unsigned probability_column,
    const unsigned* labels,const float* weights,unsigned rows,Pair* output,unsigned* bad){
  __shared__ Pair values[reduction_width];unsigned lane=threadIdx.x;
  if(blockDim.x!=reduction_width||!raw||!responses||!labels||!output||!bad||!rows||teacher_column>=raw_width||probability_column>=response_width){if(bad&&lane==0)atomicOr(bad,1u);return;}
  std::uint64_t row=std::uint64_t(blockIdx.x)*reduction_width+lane;Pair value{0,0};
  if(row<rows){float c=raw[row*raw_width+teacher_column],p=responses[row*response_width+probability_column],w=weights?weights[row]:1.f;unsigned y=labels[row];
    if(!isfinite(c)||c<0||c>1||!isfinite(p)||p<0||p>1||!isfinite(w)||w<0||y>1)atomicOr(bad,2u);
    else{double d=__dsub_rn(double(p),double(c)),e=__dsub_rn(double(y),double(c));value={__dmul_rn(double(w),__dmul_rn(d,e)),__dmul_rn(double(w),__dmul_rn(d,d))};}
  }
  values[lane]=value;__syncthreads();
  for(unsigned stride=reduction_width/2;stride;stride/=2){if(lane<stride){values[lane].numerator=__dadd_rn(values[lane].numerator,values[lane+stride].numerator);values[lane].denominator=__dadd_rn(values[lane].denominator,values[lane+stride].denominator);}__syncthreads();}
  if(lane==0)output[blockIdx.x]=values[0];
}
__global__ static void reduce_pairs(const Pair* input,unsigned count,Pair* output,unsigned* bad){
  __shared__ Pair values[reduction_width];unsigned lane=threadIdx.x;
  if(blockDim.x!=reduction_width||!input||!output||!bad||!count){if(bad&&lane==0)atomicOr(bad,1u);return;}
  std::uint64_t i=std::uint64_t(blockIdx.x)*reduction_width+lane;values[lane]=i<count?input[i]:Pair{0,0};__syncthreads();
  for(unsigned stride=reduction_width/2;stride;stride/=2){if(lane<stride){values[lane].numerator=__dadd_rn(values[lane].numerator,values[lane+stride].numerator);values[lane].denominator=__dadd_rn(values[lane].denominator,values[lane+stride].denominator);}__syncthreads();}
  if(lane==0)output[blockIdx.x]=values[0];
}
__global__ static void coefficient(const Pair* sums,double* alpha64,float* alpha32,unsigned* bad){
  if(blockIdx.x||threadIdx.x)return;
  if(!sums||!alpha64||!alpha32||!bad){if(bad)atomicOr(bad,1u);return;}if(*bad)return;
  double n=sums->numerator,d=sums->denominator;if(!isfinite(n)||!isfinite(d)||d<0||(d==0&&n!=0)){atomicOr(bad,4u);return;}
  double a=d==0||n<=0?0.:n>=d?1.:__ddiv_rn(n,d);float f=__double2float_rn(a);
  if(!isfinite(a)||!isfinite(f)||a<0||a>1||f<0||f>1){atomicOr(bad,4u);return;}*alpha64=a;*alpha32=f;
}
__global__ static void apply(const float* raw,unsigned raw_width,unsigned teacher_column,
    const float* responses,unsigned response_width,unsigned probability_column,
    unsigned rows,float alpha,float* output,unsigned* bad){
  if(!raw||!responses||!output||!bad||!rows||teacher_column>=raw_width||probability_column>=response_width||!isfinite(alpha)||alpha<0||alpha>1){if(bad&&threadIdx.x==0)atomicOr(bad,1u);return;}
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=std::uint64_t(blockDim.x)*gridDim.x){float c=raw[i*raw_width+teacher_column],p=responses[i*response_width+probability_column];
    if(!isfinite(c)||c<0||c>1||!isfinite(p)||p<0||p>1){atomicOr(bad,2u);continue;}
    float q=alpha==0?c:alpha==1?p:__double2float_rn(__dadd_rn(double(c),__dmul_rn(double(alpha),__dsub_rn(double(p),double(c)))));
    if(!isfinite(q)||q<0||q>1){atomicOr(bad,4u);continue;}output[i]=q;
  }
}
__global__ static void classify(const float* probabilities,unsigned rows,bool strict_half,float threshold,float* output,unsigned* bad){
  if(!probabilities||!output||!bad||!rows||!isfinite(threshold)||threshold<0||threshold>1){if(bad&&threadIdx.x==0)atomicOr(bad,1u);return;}
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=std::uint64_t(blockDim.x)*gridDim.x){float p=probabilities[i];if(!isfinite(p)||p<0||p>1){atomicOr(bad,2u);continue;}output[i]=strict_half?p>.5f:p>=threshold;}
}
} // namespace class_study::teacher_blend
