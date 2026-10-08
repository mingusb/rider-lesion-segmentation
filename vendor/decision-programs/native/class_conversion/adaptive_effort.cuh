#pragma once
#include "adaptive_engine.cuh"

namespace class_conversion_adaptive::effort {
// Caller owns the private stack, range buffers and runtime work policy. This
// result is observation data; it grants no native RuntimeGate authority.
struct Result {
  int label=-1,static_label=-1;
  bool attempted=false,additional_prune=false;
  u32 visited=0,refined_roots=0,tightened_roots=0,rejected_refinements=0;
  u64 fallback_frontiers=0;
};

// The same admitted residual state, original tree order and RN32 additions are
// retained. R never changes during a walk. Unvisited frontiers use their static
// enclosure, so exhausted work remains conservative. Both decisions call the
// engine's unchanged qualified_range_label rule under its authentic gate flag.
__device__ Result interval_label(EngineView e,u32 id,domain::RegionView R,
    u32* private_stack,u32 stack_capacity,u32 global_visit_budget) {
  Result out;
  if(!e.qualified_gap)return out;
  out.static_label=out.label=qualified_interval_label(e,id);
  if(out.label>=0||!global_visit_budget||!e.source.trees)return out;
  out.attempted=true;
  const u32* words=e.arena.words+u64(id)*e.source.classes;
  const auto* roots=e.arena.residual+u64(id)*e.source.trees;
  for(u32 c=0;c<e.source.classes;++c)
    e.range_lower[c]=e.range_upper[c]=__uint_as_float(words[c]);
  u32 remaining=global_visit_budget;
  const u32 capacity=private_stack?stack_capacity:0;
  for(u32 t=0;t<e.source.trees;++t)if(roots[t]>=0){
    const u32 root=u32(roots[t]),channel=u32(e.source.channels[t]);
    u32 lo=e.minimum[root],hi=e.maximum[root];
    if(remaining){
      const auto refined=conditioned_subtree_extrema(e,R,root,private_stack,capacity,remaining);
      remaining-=refined.visited;out.visited+=refined.visited;
      out.fallback_frontiers+=refined.fallbacks;++out.refined_roots;
      const float a=__uint_as_float(refined.minimum),b=__uint_as_float(refined.maximum);
      const float original_a=__uint_as_float(lo),original_b=__uint_as_float(hi);
      if(isfinite(a)&&isfinite(b)&&a<=b&&a>=original_a&&b<=original_b){
        if(a>original_a||b<original_b)++out.tightened_roots;
        lo=refined.minimum;hi=refined.maximum;
      }else{++out.rejected_refinements;++out.fallback_frontiers;}
    }else{++out.fallback_frontiers;}
    e.range_lower[channel]=__fadd_rn(e.range_lower[channel],__uint_as_float(lo));
    e.range_upper[channel]=__fadd_rn(e.range_upper[channel],__uint_as_float(hi));
  }
  out.label=qualified_range_label(e);
  out.additional_prune=out.label>=0;
  return out;
}
} // namespace class_conversion_adaptive::effort