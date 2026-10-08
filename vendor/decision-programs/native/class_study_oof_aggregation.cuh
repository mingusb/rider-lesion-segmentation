#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace class_study::oof_aggregation {
inline constexpr const char* arithmetic_policy = "ordered_fp64_mul_add_div_rn_then_fp32_rn_v1";
// The caller checks positive integer bank weights and total <= 2^53, clears
// sums/status, and consumes status before any output is used for scoring.
static __global__ void accumulate(const float* probabilities, std::uint64_t elements,
                          std::uint64_t weight, double* sums, unsigned* status) {
  if (!weight || weight > (std::uint64_t(1)<<53)) { if (!blockIdx.x && !threadIdx.x) atomicOr(status, 1u); return; }
  for (std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<elements; i+=std::uint64_t(blockDim.x)*gridDim.x) {
    float p=probabilities[i]; double previous=sums[i];
    if (!isfinite(p) || p<0.f || p>1.f || !isfinite(previous) || previous<0.) { atomicOr(status,2u); continue; }
    double next=__dadd_rn(previous,__dmul_rn(double(weight),double(p)));
    if (!isfinite(next)) { atomicOr(status,4u); continue; }
    sums[i]=next;
  }
}
static __global__ void finish(const double* sums, std::uint64_t elements,
                       std::uint64_t total_weight, float* probabilities, unsigned* status) {
  if (!total_weight || total_weight > (std::uint64_t(1)<<53)) { if (!blockIdx.x && !threadIdx.x) atomicOr(status,1u); return; }
  for (std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<elements; i+=std::uint64_t(blockDim.x)*gridDim.x) {
    double s=sums[i];
    if (!isfinite(s) || s<0. || s>double(total_weight)) { atomicOr(status,4u); continue; }
    probabilities[i]=__double2float_rn(__ddiv_rn(s,double(total_weight)));
  }
}
static __global__ void binary_argmax(const float* probabilities, std::uint64_t rows,
                             float* classes, unsigned* status) {
  for (std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<rows; i+=std::uint64_t(blockDim.x)*gridDim.x) {
    float a=probabilities[2*i],b=probabilities[2*i+1];
    if (!isfinite(a)||!isfinite(b)||a<0.f||a>1.f||b<0.f||b>1.f) {if(status)atomicOr(status,2u); continue;}
    classes[i]=b>a?1.f:0.f;
  }
}
} // namespace class_study::oof_aggregation
