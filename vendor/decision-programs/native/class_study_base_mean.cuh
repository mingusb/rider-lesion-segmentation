#pragma once
#include "class_study_oof_aggregation.cuh"

namespace class_study::base_mean {
inline constexpr const char* arithmetic_policy=oof_aggregation::arithmetic_policy;
// The response bank contains copied immutable FP32 class pairs in declared
// model order. Accumulation is identical to accumulate(weight=1), finish(B),
// with two register sums per row and one pair output, not a response cube.
static __global__ void probabilities(const float* responses,std::uint64_t rows,
    unsigned width,unsigned members,float* output,unsigned* status){
  if(!responses||!output||!status||!rows||!members||std::uint64_t(members)*2>width){
    if(status&&!blockIdx.x&&!threadIdx.x)atomicOr(status,1u);return;
  }
  for(std::uint64_t row=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=std::uint64_t(blockDim.x)*gridDim.x){
    double sums[2]={0.,0.};bool valid=true;
    for(unsigned member=0;member<members;++member)for(unsigned cls=0;cls<2;++cls){
      float p=responses[row*width+std::uint64_t(member)*2+cls];
      if(!isfinite(p)||p<0.f||p>1.f){atomicOr(status,2u);valid=false;}
      else sums[cls]=__dadd_rn(sums[cls],__dmul_rn(1.,double(p)));
    }
    if(!valid)continue;
    for(unsigned cls=0;cls<2;++cls){if(!isfinite(sums[cls])||sums[cls]<0.||sums[cls]>double(members)){atomicOr(status,4u);continue;}
      output[2*row+cls]=__double2float_rn(__ddiv_rn(sums[cls],double(members)));
    }
  }
}
} // namespace class_study::base_mean
