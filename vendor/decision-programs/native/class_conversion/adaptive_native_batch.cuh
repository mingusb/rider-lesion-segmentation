#pragma once
#include "adaptive_engine.cuh"

namespace class_conversion_adaptive {
// Only terminal states with all source contributions consumed enter this batch.
// Their full witnesses, not projected cache keys, reproduce native margins.
__global__ void materialize_native_batch(EngineView e, const u32* ids,
                                         u32 count, float* x) {
  for (u64 row = u64(blockIdx.x) * blockDim.x + threadIdx.x; row < count;
       row += u64(blockDim.x) * gridDim.x) {
    const u32 id = ids[row];
    if (id >= e.status->states || e.arena.states[id].phase != 1 ||
        e.arena.states[id].predicate != none) {
      atomicCAS(&e.status->error, 0u, 41u); continue;
    }
    if (!domain::region_witness(e.domain, witness_region(e, id),
                                x + row * e.source.features, 0))
      atomicCAS(&e.status->error, 0u, 30u);
  }
}
__global__ void check_native_margin_batch(EngineView e, const u32* ids,
                                          u32 count, const float* margins) {
  const u32 columns = e.source.native_margin_classes ? e.source.native_margin_classes : e.source.classes;
  if (!columns || columns > e.source.classes) { atomicCAS(&e.status->error, 0u, 41u); return; }
  const u64 extent = u64(count) * columns;
  for (u64 index = u64(blockIdx.x) * blockDim.x + threadIdx.x; index < extent;
       index += u64(blockDim.x) * gridDim.x) {
    const u32 id = ids[index / columns];
    if (id >= e.status->states || e.arena.states[id].phase != 1 ||
        e.arena.states[id].predicate != none) {
      atomicCAS(&e.status->error, 0u, 41u); continue;
    }
    if (__float_as_uint(margins[index]) !=
        e.arena.words[u64(id) * e.source.classes + index % columns])
      atomicCAS(&e.status->error, 0u, 31u);
  }
}
// Decode native public results only after the margin audit has synchronized.
// Publishing labels into graph nodes remains a serial, retryable commit.
__global__ void decode_native_class_batch(EngineView e, const float* output,
                                          u32 count, bool direct, u32* labels) {
  const u32 classes = output_class_count(e);
  for (u64 row = u64(blockIdx.x) * blockDim.x + threadIdx.x; row < count;
       row += u64(blockDim.x) * gridDim.x) {
    u32 label = 0;
    if (direct) {
      float value = output[row];
      if (!isfinite(value) || value < 0 || double(value) >= classes ||
          value != floorf(value)) {
        atomicCAS(&e.status->error, 0u, 32u); labels[row] = none; continue;
      }
      label = u32(value);
    } else {
      const float* p = output + row * classes;
      float best = p[0]; bool valid = true;
      for (u32 c = 0; c < classes; ++c) {
        float value = p[c];
        if (!isfinite(value) || value < 0 || value > 1) valid = false;
        if (value > best) { best = value; label = c; }
      }
      if (!valid) { atomicCAS(&e.status->error, 0u, 33u); labels[row] = none; continue; }
    }
    labels[row] = label;
  }
}
} // namespace class_conversion_adaptive
