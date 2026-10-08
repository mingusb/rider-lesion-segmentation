#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace class_study::meta_bootstrap {
inline constexpr const char* draw_policy = "sorted_fit_patients_splitmix64_rejection_v1";
inline constexpr const char* weight_policy = "fp32_rn_integer_multiplicity_times_existing_weight_v1";
// Unsigned operations wrap modulo 2^64. This is a portable seeded sampling
// stream, not a cryptographic random generator. Patient IDs and order bind it.
inline std::uint64_t mix(std::uint64_t z) {
  z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
  return z^(z>>31);
}
inline std::vector<std::uint32_t> multiplicities(
    const std::vector<std::uint32_t>& patients,std::uint64_t seed,std::uint32_t replicate) {
  if(patients.empty()||patients.size()>(UINT64_C(1)<<24)||
     !std::is_sorted(patients.begin(),patients.end())||
     std::adjacent_find(patients.begin(),patients.end())!=patients.end())
    throw std::invalid_argument("bootstrap requires 1..2^24 distinct sorted FIT patients");
  auto state=mix(seed^UINT64_C(0x6d657461626f6f74)); // versioned 'metaboot' namespace
  state=mix(state^std::uint64_t(patients.size()));
  for(auto patient:patients)state=mix(state^std::uint64_t(patient));
  state=mix(state^std::uint64_t(replicate));
  const auto n=std::uint64_t(patients.size()),reject_below=(std::uint64_t(0)-n)%n;
  std::vector<std::uint32_t> counts(patients.size(),0);
  for(std::uint64_t draw=0;draw<n;++draw){
    std::uint64_t word;
    do {state+=UINT64_C(0x9e3779b97f4a7c15);word=mix(state);} while(word<reject_below);
    ++counts[word%n];
  }
  return counts;
}
static __global__ void apply_weights(const float* original,const std::uint32_t* row_ids,
    const std::uint32_t* all_patients,std::uint32_t all_rows,const std::uint32_t* labels,
    std::uint64_t rows,const std::uint32_t* groups,const std::uint32_t* multiplicity,
    std::uint32_t group_count,float* output,double* class_mass,std::uint32_t* status) {
  if(!group_count||group_count>(1u<<24)){if(!blockIdx.x&&!threadIdx.x)atomicOr(status,2u);return;}
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=std::uint64_t(blockDim.x)*gridDim.x){
    if(row_ids[i]>=all_rows||labels[i]>1){atomicOr(status,1u);continue;}
    const auto patient=all_patients[row_ids[i]];std::uint32_t lo=0,hi=group_count;
    while(lo<hi){auto middle=lo+(hi-lo)/2;if(groups[middle]<patient)lo=middle+1;else hi=middle;}
    if(lo==group_count||groups[lo]!=patient){atomicOr(status,1u);continue;}
    const auto m=multiplicity[lo];const float w=original[i];
    if(m>group_count||group_count>(1u<<24)||!isfinite(w)||w<0){atomicOr(status,2u);continue;}
    const float weighted=__fmul_rn(float(m),w);
    if(!isfinite(weighted)){atomicOr(status,4u);continue;}
    output[i]=weighted;atomicAdd(class_mass+std::uint64_t(lo)*2+labels[i],double(weighted));
  }
}
static __global__ void argmax_disagreements(const float* native_classes,const float* probabilities,
    std::uint64_t rows,unsigned long long* mismatches,std::uint32_t* status) {
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=std::uint64_t(blockDim.x)*gridDim.x){
    const float a=probabilities[i*2],b=probabilities[i*2+1],c=native_classes[i];
    if(!isfinite(a)||!isfinite(b)||a<0||a>1||b<0||b>1||(c!=0&&c!=1)){atomicOr(status,1u);continue;}
    if(c!=float(b>a))atomicAdd(mismatches,1ULL);
  }
}
} // namespace class_study::meta_bootstrap
