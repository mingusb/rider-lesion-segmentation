#pragma once
#include "class_study_group_loss_math.hpp"
#include <cuda_runtime_api.h>

namespace class_study::group_loss {
inline constexpr const char* moment_reduction_policy="keyed_warp_fp64_atomic_v1";
struct DeviceInputs {
  const float* margins=nullptr; // borrowed contiguous [rows,2] FP32 raw margins
  const float* labels=nullptr; // trusted staged native labels, exactly 0 or 1
  const u32* fit_group_ids=nullptr; // dense FIT-only [0,groups) IDs
  const GroupSpec* group_specs=nullptr; // device-resident FIT-only counts
  u64 rows=0;
  u32 groups=0;
};
struct DeviceOutputs {
  float* gradients=nullptr; // contiguous [rows,2], row-major
  float* hessians=nullptr; // contiguous [rows,2], row-major
  GroupStats* stats=nullptr; // [groups] workspace and diagnostics
  u32* status=nullptr; // exactly one device u32; any nonzero invalidates ALL outputs
};
struct StepAudit {
  double maximum_abs_logit_step=0,maximum_rounding_envelope=0;
  unsigned long long rows_checked=0,steps_exceeding_real_radius=0;
};
static_assert(sizeof(StepAudit)==32);
// Accumulates audit receipt; caller zeros it once and checks status after every
// update. This enqueues no training and makes no descent-certificate claim.
cudaError_t enqueue_step_audit(const float* before,const float* after,u64 rows,
    float eta,float cap,double real_radius,StepAudit* receipt,u32* status,cudaStream_t stream);
struct PhaseEvents {
  // Optional caller-owned events; all four must be provided together.
  cudaEvent_t start=nullptr,after_counts=nullptr,after_reduction=nullptr,after_output=nullptr;
};
// Caller owns allocation/provenance/lifetime/nonaliasing checks. Enqueues
// only; it does not synchronize, read status, train, allocate, or call XGBoost.
// The returned code covers launches. Caller must wait and inspect status before
// using outputs. Separate GPU integration qualification is required.
cudaError_t enqueue_objective(DeviceInputs in,DeviceOutputs out,Config config,
                             cudaStream_t stream,const PhaseEvents* timing=nullptr);
} // namespace class_study::group_loss
