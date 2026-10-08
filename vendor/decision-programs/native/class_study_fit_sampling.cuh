#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace class_study::fit_sampling {
using U=std::uint32_t;
using L=unsigned long long;
inline constexpr const char* arithmetic_policy=
    "legacy_mass_three_strata_fp64_then_fp32_v1";
struct Counts {
  L full[3]{}, retained[3]{};
  L baseline=0, eligible=0, eligible_positive=0;
};
static_assert(sizeof(Counts)==9*sizeof(L));
inline constexpr U invalid_input=1, missing_support=2, inconsistent_counts=4,
                   invalid_weight=8, count_overflow=16;

// Sorted unique ID lists and bounded source storage are host preconditions.
static __device__ int locate(U value,const U* keys,U count) {
  U first=0,last=count;
  while(first<last) {
    U middle=first+(last-first)/2;
    if(keys[middle]<value)first=middle+1;else last=middle;
  }
  return first<count&&keys[first]==value?int(first):-1;
}
static __device__ void shape_error(U* bad) {
  if(bad&&!blockIdx.x&&!threadIdx.x)atomicOr(bad,invalid_input);
}
static __device__ U stratum(U label,float mask) {
  return label?0u:mask==1.f?1u:2u;
}

// Counts are zeroed by the owner. Validation covers every source row, while
// retaining only scalar scan summaries. No labels become prediction features.
static __global__ void count_scans(const float* x,const U* y,const U* eligible,
    const U* sid,L rows,U features,U mask,const U* sorted_scans,U S,
    Counts* out,U* bad) {
  if(!x||!y||!eligible||!sid||!sorted_scans||!out||!bad||!rows||
      !features||mask>=features||!S||rows>(~L(0))/features) {
    shape_error(bad);return;
  }
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;
      row+=L(blockDim.x)*gridDim.x) {
    const float m=x[row*features+mask];
    const U label=y[row],e=eligible[row];
    const int scan=locate(sid[row],sorted_scans,S);
    if((m!=0.f&&m!=1.f)||label>1||e>1||scan<0) {
      atomicOr(bad,invalid_input);continue;
    }
    Counts* c=out+scan;
    const U s=stratum(label,m);
    atomicAdd(c->full+s,1ULL);
    if(e||label||m==1.f)atomicAdd(c->retained+s,1ULL);
    if(e||label)atomicAdd(&c->baseline,1ULL);
    if(e)atomicAdd(&c->eligible,1ULL);
    if(e&&label)atomicAdd(&c->eligible_positive,1ULL);
  }
}

// Use immutable original row IDs for both raw and OOF-response FIT matrices.
// The mask is never read from a response column in a meta-learning matrix.
static __global__ void count_retained(const float* source_x,const U* source_y,
    const U* source_pid,U source_rows,U features,U mask,const U* ids,
    const U* labels,U n,const U* sorted_patients,U G,L* counts3,U* bad) {
  if(!source_x||!source_y||!source_pid||!ids||!labels||!sorted_patients||
      !counts3||!bad||!source_rows||!features||mask>=features||!n||!G) {
    shape_error(bad);return;
  }
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
      row+=L(blockDim.x)*gridDim.x) {
    const U id=ids[row],label=labels[row];
    if(id>=source_rows||label>1) {atomicOr(bad,invalid_input);continue;}
    const float m=source_x[L(id)*features+mask];
    const int group=locate(source_pid[id],sorted_patients,G);
    if(source_y[id]!=label||(m!=0.f&&m!=1.f)||group<0) {
      atomicOr(bad,invalid_input);continue;
    }
    atomicAdd(counts3+L(group)*3+stratum(label,m),1ULL);
  }
}

// n0 is the OLD retained count for this enclosing FIT, not the expanded row
// count. This preserves the prior total loss mass and equal-patient target.
// The exact declared operation order has no fused multiply-add or floor.
static __global__ void coefficients(const L* full3,const L* retained3,L n0,
    U G,float* coeff3,U* bad) {
  if(!full3||!retained3||!coeff3||!bad||!n0||!G) {
    shape_error(bad);return;
  }
  for(L group=L(blockIdx.x)*blockDim.x+threadIdx.x;group<G;
      group+=L(blockDim.x)*gridDim.x) {
    const L at=group*3;
    L full=0,retained=0;U error=0;
    for(U s=0;s<3;++s) {
      coeff3[at+s]=0.f;
      const L N=full3[at+s],n=retained3[at+s];
      if(N>(~L(0))-full||n>(~L(0))-retained)error|=count_overflow;
      else {full+=N;retained+=n;}
      if(n>N||(s<2&&n!=N))error|=inconsistent_counts;
      if(N&&!n)error|=missing_support;
    }
    if(!full||!retained)error|=missing_support;
    if(error) {atomicOr(bad,error);continue;}
    const double mass=__ddiv_rn(double(n0),double(G));
    if(!isfinite(mass)||mass<=0.) {atomicOr(bad,invalid_weight);continue;}
    for(U s=0;s<3;++s) {
      const L N=full3[at+s],n=retained3[at+s];
      if(!N)continue; // empty population and retained stratum: exact +0.
      const double fraction=__ddiv_rn(double(N),double(full));
      const double w=__ddiv_rn(__dmul_rn(mass,fraction),double(n));
      const float stored=__double2float_rn(w);
      if(!isfinite(w)||w<=0.||!isfinite(stored)||stored<=0.f) {
        atomicOr(bad,invalid_weight);continue;
      }
      coeff3[at+s]=stored;
    }
  }
}

// Called only after count_retained and coefficients succeed. Sums are actual
// FP32 weights accumulated in FP64 for the receipt, not fitting reductions.
static __global__ void apply(const float* source_x,const U* source_pid,
    U features,U mask,const U* ids,const U* labels,U n,
    const U* sorted_patients,U G,const float* coeff3,float* weights,
    double* sums3) {
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
      row+=L(blockDim.x)*gridDim.x) {
    const U id=ids[row];
    const U group=U(locate(source_pid[id],sorted_patients,G));
    const U s=stratum(labels[row],source_x[L(id)*features+mask]);
    const L at=L(group)*3+s;
    const float w=coeff3[at];
    weights[row]=w;
    atomicAdd(sums3+at,double(w));
  }
}

// Data has already audited the mask, labels, eligibility and ID inventory.
// Evaluation ignores FIT eligibility and retains every row of chosen patients.
static __global__ void select_flags(const float* x,const U* pid,
    const U* eligible,const U* labels,U n,U features,U mask,
    const U* selected,U count,bool fit,U* flags) {
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
      row+=L(blockDim.x)*gridDim.x) {
    flags[row]=locate(pid[row],selected,count)>=0&&
        (!fit||eligible[row]||labels[row]==1||x[row*features+mask]==1.f);
  }
}
} // namespace class_study::fit_sampling
