#pragma once
#include "adaptive_engine.cuh"
#include "adaptive_proof_modules.hpp"

namespace class_conversion_adaptive::proof_modules {
inline InputV1 make_input(EngineView e,u64 token,const u32*ids,const u32*parents){
  InputV1 v;v.struct_bytes=sizeof(InputV1);v.session_token=token;
  v.features=e.source.features;v.classes=e.source.classes;v.nodes=e.source.nodes;v.trees=e.source.trees;v.state_capacity=e.arena.state_capacity;
  v.feature=e.source.feature;v.left=e.source.left;v.right=e.source.right;v.channels=e.source.channels;v.residual=e.arena.residual;
  v.minimum=e.minimum;v.maximum=e.maximum;v.state_ids=ids;v.parent_predicates=parents;return v;
}
enum class ProposalRejection:u32 {none=0,flags,invalid_state,invalid_predicate,not_residual,forced};
// Advisory only: malformed proposals cannot become class certificates. The
// core construction predicate remains unchanged; this result is for cover work.
__device__ inline u32 checked_predicate(EngineView e,u32 id,ProposalV1 proposal,
    u32 fallback,ProposalRejection*rejection=nullptr){
  auto refuse=[&](ProposalRejection why){if(rejection)*rejection=why;return fallback;};
  if(rejection)*rejection=ProposalRejection::none;
  if(proposal.flags)return refuse(ProposalRejection::flags);
  if(id>=e.arena.state_capacity||!e.arena.states||e.arena.states[id].phase==free_phase)return refuse(ProposalRejection::invalid_state);
  const u32 p=proposal.predicate;
  if(p>=e.source.nodes||!e.source.feature||!e.source.left||!e.source.right||!e.source.cut||!e.source.missing||
     e.source.left[p]<0||e.source.right[p]<0||u32(e.source.left[p])>=e.source.nodes||u32(e.source.right[p])>=e.source.nodes||
     e.source.feature[p]<0||u32(e.source.feature[p])>=e.source.features||e.source.missing[p]>1||
     domain::nan_word(__float_as_uint(e.source.cut[p])))return refuse(ProposalRejection::invalid_predicate);
  if(e.source.trees&&!e.arena.residual)return refuse(ProposalRejection::not_residual);
  bool found=false;for(u32 t=0;t<e.source.trees;++t)found|=e.arena.residual[u64(id)*e.source.trees+t]==std::int32_t(p);
  if(!found)return refuse(ProposalRejection::not_residual);
  const auto R=region(e,id);if(!domain::region_valid(e.domain,R))return refuse(ProposalRejection::invalid_state);
  bool right=false;if(domain::forced_side(e.domain,R,u32(e.source.feature[p]),__float_as_uint(e.source.cut[p]),bool(e.source.missing[p]),right))return refuse(ProposalRejection::forced);
  return p;
}
} // namespace class_conversion_adaptive::proof_modules
