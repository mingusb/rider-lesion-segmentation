#pragma once
// Included only by the shared trainer translation unit.
#include "class_study_group_loss_cuda.hpp"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace class_study::group_loss {
namespace {
__device__ void audit_max(double*where,double value){
  atomicMax(reinterpret_cast<unsigned long long*>(where),static_cast<unsigned long long>(__double_as_longlong(value)));
}
__global__ void audit_native_step(const float*before,const float*after,std::uint64_t rows,
    float eta,float cap,double real_radius,StepAudit*receipt,std::uint32_t*bad){
  // Pinned native binary scalar-tree inference accumulates each class in tree
  // order. One new tree/class adds one stored FP32 leaf; its magnitude is at
  // most RN(eta*cap). Monotonic correctly rounded addition gives exact endpoint
  // envelopes without an arbitrary tolerance or a descent-certificate claim.
  const float leaf=__fmul_rn(eta,cap);
  for(std::uint64_t row=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=std::uint64_t(blockDim.x)*gridDim.x){
    const float b0=before[2*row],b1=before[2*row+1],a0=after[2*row],a1=after[2*row+1];
    const float lo0=__fadd_rn(b0,-leaf),hi0=__fadd_rn(b0,leaf),lo1=__fadd_rn(b1,-leaf),hi1=__fadd_rn(b1,leaf);
    if(!isfinite(b0)||!isfinite(b1)||!isfinite(a0)||!isfinite(a1)||!isfinite(lo0)||!isfinite(hi0)||!isfinite(lo1)||!isfinite(hi1)){
      atomicOr(bad,1024u);continue;
    }
    if(a0<lo0||a0>hi0||a1<lo1||a1>hi1)atomicOr(bad,2048u);
    const double delta=(double(a1)-double(b1))-(double(a0)-double(b0));
    const double positive=(double(hi1)-double(b1))-(double(lo0)-double(b0));
    const double negative=(double(hi0)-double(b0))-(double(lo1)-double(b1));
    audit_max(&receipt->maximum_abs_logit_step,fabs(delta));
    audit_max(&receipt->maximum_rounding_envelope,fmax(positive,negative));
    atomicAdd(&receipt->rows_checked,1ULL);
    if(fabs(delta)>real_radius)atomicAdd(&receipt->steps_exceeding_real_radius,1ULL);
  }
}
__device__ void fail(u32* status,u32 code) { if(code) atomicOr(status,code); }
__device__ bool failed(u32* status) { return atomicAdd(status,0u)!=0; }
__global__ void initialize(GroupStats* stats,u32 groups,u32* status) {
  if(blockIdx.x==0&&threadIdx.x==0) *status=ok;
  for(u64 g=u64(blockIdx.x)*blockDim.x+threadIdx.x;g<groups;g+=u64(blockDim.x)*gridDim.x)
    stats[g]=GroupStats{};
}
__global__ void count_rows(DeviceInputs in,DeviceOutputs out) {
  for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<in.rows;i+=u64(blockDim.x)*gridDim.x) {
    const u32 g=in.fit_group_ids[i];
    const float label=in.labels[i];
    if(label!=0.0f&&label!=1.0f) { fail(out.status,bad_label);continue; }
    const u32 y=u32(label);
    if(g>=in.groups) { fail(out.status,bad_group);continue; }
    if(y>1) { fail(out.status,bad_label);continue; }
    if(!finite(double(in.margins[2*i]))||!finite(double(in.margins[2*i+1])))
      fail(out.status,nonfinite_margin);
    atomicAdd(&out.stats[g].retained[y],1ULL);
  }
}
__global__ void validate_groups(DeviceInputs in,DeviceOutputs out,Config config) {
  for(u64 g=u64(blockIdx.x)*blockDim.x+threadIdx.x;g<in.groups;g+=u64(blockDim.x)*gridDim.x)
    fail(out.status,validate_group(in.group_specs[g],out.stats[g],config));
}
__global__ void reduce_rows(DeviceInputs in,DeviceOutputs out) {
  if(failed(out.status)) return;
  for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<in.rows;i+=u64(blockDim.x)*gridDim.x) {
    const u32 g=in.fit_group_ids[i],y=u32(in.labels[i]);RowState r;
    const u32 error=row_state(in.margins[2*i],in.margins[2*i+1],y,
                              in.group_specs[g],out.stats[g],r);
    if(error) { fail(out.status,error);continue; }
    // Discover only lanes that reached this point: tail lanes and rejected
    // rows are absent. Partition those active lanes by their arbitrary group
    // IDs, including noncontiguous equal IDs. Every lane in each partition
    // participates in all three collectives before its elected leader writes.
    // This changes FP64 addition order, not the moment definition; it does not
    // promise bit identity with the prior per-row atomic reduction.
    namespace cg=cooperative_groups;
    const auto peers=cg::labeled_partition(cg::coalesced_threads(),g);
    const double intersection=cg::reduce(peers,r.weight*double(y)*r.positive,cg::plus<double>());
    const double prediction_mass=cg::reduce(peers,r.weight*r.positive,cg::plus<double>());
    const double ce_sum=cg::reduce(peers,r.weight*r.ce,cg::plus<double>());
    if(peers.thread_rank()==0) {
      atomicAdd(&out.stats[g].intersection,intersection);
      atomicAdd(&out.stats[g].prediction_mass,prediction_mass);
      atomicAdd(&out.stats[g].ce_sum,ce_sum);
    }
  }
}
__global__ void finalize_groups(DeviceInputs in,DeviceOutputs out,Config config) {
  if(failed(out.status)) return;
  for(u64 g=u64(blockIdx.x)*blockDim.x+threadIdx.x;g<in.groups;g+=u64(blockDim.x)*gridDim.x)
    fail(out.status,finalize_group(in.group_specs[g],out.stats[g],config));
}
__global__ void output_rows(DeviceInputs in,DeviceOutputs out,Config config) {
  if(failed(out.status)) return;
  for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<in.rows;i+=u64(blockDim.x)*gridDim.x) {
    const u32 g=in.fit_group_ids[i],y=u32(in.labels[i]);RowState r;OutputPair p;
    u32 error=row_state(in.margins[2*i],in.margins[2*i+1],y,in.group_specs[g],out.stats[g],r);
    if(!error) error=row_output(r,y,in.group_specs[g],out.stats[g],config,in.groups,p);
    if(error) { fail(out.status,error);continue; }
    out.gradients[2*i]=p.g0;out.gradients[2*i+1]=p.g1;
    out.hessians[2*i]=p.h0;out.hessians[2*i+1]=p.h1;
  }
}
unsigned blocks(u64 rows) { return unsigned(rows>65535ULL*256?65535:(rows+255)/256); }
} // namespace

cudaError_t enqueue_step_audit(const float*before,const float*after,u64 rows,
    float eta,float cap,double real_radius,StepAudit*receipt,u32*status,cudaStream_t stream){
  if(!before||!after||!rows||rows>u64(SIZE_MAX/2/sizeof(float))||!receipt||!status||
     !finite(eta)||eta<=0||!finite(cap)||cap<=0||!finite(real_radius)||real_radius<=0)
    return cudaErrorInvalidValue;
  audit_native_step<<<blocks(rows),256,0,stream>>>(before,after,rows,eta,cap,real_radius,receipt,status);
  return cudaPeekAtLastError();
}

cudaError_t enqueue_objective(DeviceInputs in,DeviceOutputs out,Config config,
                             cudaStream_t stream,const PhaseEvents* timing) {
  if(validate_config(config,in.rows,in.groups)||!in.margins||!in.labels||
     !in.fit_group_ids||!in.group_specs||!out.gradients||!out.hessians||
     !out.stats||!out.status) return cudaErrorInvalidValue;
  if(timing&&(!timing->start||!timing->after_counts||!timing->after_reduction||!timing->after_output))
    return cudaErrorInvalidValue;
  if(timing) if(auto e=cudaEventRecord(timing->start,stream);e!=cudaSuccess) return e;
  // Stream ordering separates status reset, count validation and reductions.
  initialize<<<blocks(in.groups),256,0,stream>>>(out.stats,in.groups,out.status);
  if(auto e=cudaPeekAtLastError();e!=cudaSuccess) return e;
  count_rows<<<blocks(in.rows),256,0,stream>>>(in,out);
  if(auto e=cudaPeekAtLastError();e!=cudaSuccess) return e;
  validate_groups<<<blocks(in.groups),256,0,stream>>>(in,out,config);
  if(auto e=cudaPeekAtLastError();e!=cudaSuccess) return e;
  if(timing) if(auto e=cudaEventRecord(timing->after_counts,stream);e!=cudaSuccess) return e;
  reduce_rows<<<blocks(in.rows),256,0,stream>>>(in,out);
  if(auto e=cudaPeekAtLastError();e!=cudaSuccess) return e;
  finalize_groups<<<blocks(in.groups),256,0,stream>>>(in,out,config);
  if(auto e=cudaPeekAtLastError();e!=cudaSuccess) return e;
  if(timing) if(auto e=cudaEventRecord(timing->after_reduction,stream);e!=cudaSuccess) return e;
  output_rows<<<blocks(in.rows),256,0,stream>>>(in,out,config);
  if(auto e=cudaPeekAtLastError();e!=cudaSuccess) return e;
  return timing?cudaEventRecord(timing->after_output,stream):cudaSuccess;
}
} // namespace class_study::group_loss
