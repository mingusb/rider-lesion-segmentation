#pragma once
#include "adaptive_effort.cuh"

namespace class_conversion_adaptive::cover {
enum class Rejection : u32 {
  none=0, gate_unavailable, budget_disabled, invalid_state, invalid_predicate,
  invalid_region, invalid_scratch, uncertified_case, differing_labels,
  no_feasible_case, invalid_source, visit_budget_violation
};
struct Result {
  std::int32_t label=-1, branch_labels[2]={-1,-1};
  Rejection rejection=Rejection::none;
  u32 attempted=0, success=0, feasible_cases=0, certified_cases=0;
  u32 visited=0, visit_budget=0;
  u64 refined_roots=0, tightened_roots=0, rejected_refinements=0, fallback_frontiers=0;
};
static_assert(sizeof(Result)==72);

namespace detail {
struct Span { const void* pointer; u64 bytes; };
__device__ inline bool overlap(Span a,Span b) {
  if(!a.pointer||!b.pointer||!a.bytes||!b.bytes)return false;
  const auto x=reinterpret_cast<std::uintptr_t>(a.pointer);
  const auto y=reinterpret_cast<std::uintptr_t>(b.pointer);
  return x<=y ? u64(y-x)<a.bytes : u64(x-y)<b.bytes;
}
__device__ inline bool private_scratch(EngineView e,domain::RegionView original,
    domain::RegionView scratch,u32* stack,u32 capacity) {
  const u64 numeric=u64(e.domain.numeric_features)*sizeof(u32);
  const u64 masks=u64(e.domain.mask_words)*sizeof(u64);
  if((numeric&&(!scratch.lower||!scratch.upper||!scratch.missing))||
     (masks&&!scratch.allowed)||!e.range_lower||!e.range_upper||
     (capacity&&!stack))return false;
  const Span output[]={{scratch.lower,numeric},{scratch.upper,numeric},
    {scratch.missing,numeric},{scratch.allowed,masks},
    {stack,u64(capacity)*sizeof(u32)},
    {e.range_lower,u64(e.source.classes)*sizeof(float)},
    {e.range_upper,u64(e.source.classes)*sizeof(float)}};
  const Span parent[]={{original.lower,numeric},{original.upper,numeric},
    {original.missing,numeric},{original.allowed,masks},
    {e.arena.words,u64(e.arena.state_capacity)*e.source.classes*sizeof(u32)},
    {e.arena.positions,u64(e.arena.state_capacity)*e.source.classes*sizeof(u32)},
    {e.arena.residual,u64(e.arena.state_capacity)*e.source.trees*sizeof(std::int32_t)}};
  for(u32 i=0;i<7;++i){
    for(u32 j=0;j<i;++j)if(overlap(output[i],output[j]))return false;
    for(const auto& p:parent)if(overlap(output[i],p))return false;
  }
  return true;
}
} // namespace detail

// Caller supplies an immutable validated source/state and exclusive scratch.
// Scratch, ranges and stack must be disjoint from ALL source/arena storage;
// direct parent/payload overlap and scratch overlap are additionally refused.
// Each successful region_restrict partitions the ORIGINAL region using the
// exact source predicate, including missing routing and one-hot constraints.
// Prefix words, positions and residual roots are never normalized or changed.
// The unchanged effort rule folds every residual in original RN32 source order.
// Exhausted visits/stack use existing conservative static enclosures. Every
// feasible side must certify the same label; no incomplete side is omitted.
// This result does not create native RuntimeGate authority.
__device__ inline Result interval_label(EngineView e,u32 id,
    domain::RegionView original,u32 predicate,domain::RegionView private_region,
    u32* private_stack,u32 stack_capacity,u32 global_visit_budget) {
  Result out;out.visit_budget=global_visit_budget;
  if(!e.qualified_gap){out.rejection=Rejection::gate_unavailable;return out;}
  if(!global_visit_budget){out.rejection=Rejection::budget_disabled;return out;}
  if(id>=e.arena.state_capacity||!e.arena.states||!e.arena.words||
     e.arena.states[id].phase==free_phase){out.rejection=Rejection::invalid_state;return out;}
  if(!e.source.classes||!e.source.features||e.source.features!=e.domain.features||
     !e.source.feature||!e.source.left||!e.source.right||!e.source.cut||
     !e.source.missing||(e.source.trees&&(!e.arena.residual||!e.source.channels||
       !e.minimum||!e.maximum))){out.rejection=Rejection::invalid_source;return out;}
  if(predicate>=e.source.nodes||e.source.left[predicate]<0||
     e.source.right[predicate]<0||u32(e.source.left[predicate])>=e.source.nodes||
     u32(e.source.right[predicate])>=e.source.nodes||e.source.feature[predicate]<0||
     u32(e.source.feature[predicate])>=e.source.features||e.source.missing[predicate]>1||
     domain::nan_word(__float_as_uint(e.source.cut[predicate]))){
    out.rejection=Rejection::invalid_predicate;return out;
  }
  if((e.domain.numeric_features&&(!original.lower||!original.upper||!original.missing))||
     (e.domain.mask_words&&!original.allowed)||!domain::region_valid(e.domain,original)){
    out.rejection=Rejection::invalid_region;return out;
  }
  if(!detail::private_scratch(e,original,private_region,private_stack,stack_capacity)){
    out.rejection=Rejection::invalid_scratch;return out;
  }
  for(u32 t=0;t<e.source.trees;++t){
    const auto root=e.arena.residual[u64(id)*e.source.trees+t];
    if((root>=0&&u32(root)>=e.source.nodes)||e.source.channels[t]<0||
       u32(e.source.channels[t])>=e.source.classes){
      out.rejection=Rejection::invalid_source;return out;
    }
  }
  const u32 feature=u32(e.source.feature[predicate]);
  const u32 cut=__float_as_uint(e.source.cut[predicate]);
  const bool default_left=bool(e.source.missing[predicate]);
  out.attempted=1;u32 remaining=global_visit_budget;
  for(u32 side=0;side<2;++side){
    if(!domain::region_split_feasible(e.domain,original,feature,cut,default_left,bool(side)))continue;
    ++out.feasible_cases;copy_region(e,private_region,original);
    if(!domain::region_restrict(e.domain,private_region,feature,cut,default_left,bool(side))){
      out.rejection=Rejection::invalid_region;return out;
    }
    const auto result=effort::interval_label(e,id,private_region,private_stack,stack_capacity,remaining);
    if(result.visited>remaining){out.rejection=Rejection::visit_budget_violation;return out;}
    remaining-=result.visited;out.visited+=result.visited;
    out.refined_roots+=result.refined_roots;out.tightened_roots+=result.tightened_roots;
    out.rejected_refinements+=result.rejected_refinements;
    out.fallback_frontiers+=result.fallback_frontiers;
    out.branch_labels[side]=result.label;
    if(result.label>=0){++out.certified_cases;}
  }
  if(!out.feasible_cases){out.rejection=Rejection::no_feasible_case;return out;}
  if(out.certified_cases!=out.feasible_cases){out.rejection=Rejection::uncertified_case;return out;}
  int common=-1;
  for(u32 side=0;side<2;++side)if(out.branch_labels[side]>=0){
    if(common>=0&&common!=out.branch_labels[side]){
      out.rejection=Rejection::differing_labels;return out;
    }
    common=out.branch_labels[side];
  }
  out.label=common;out.success=1;return out;
}
} // namespace class_conversion_adaptive::cover
