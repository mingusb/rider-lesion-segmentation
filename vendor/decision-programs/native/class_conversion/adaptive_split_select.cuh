#pragma once
#include "adaptive_engine.cuh"
#include <bit>
#include <map>
#include <tuple>
#include <vector>

// Advisory source-predicate priorities for the sole adaptive constructor.
// These never certify a class, alter a prefix, or drop a residual source tree.
namespace class_conversion_adaptive::split {
enum class Policy : u32 {
  source_order = 0,
  widest_residual = 1,
  aggregate_residual = 2,
  contracting_residual = 3
};
struct HostMetadata {
  std::vector<u32> group_ids;
  u32 groups = 0;
};
struct View {
  const u32* group_ids = nullptr;
  u32 groups = 0;
  Policy policy = Policy::source_order;
};
inline bool valid_policy(Policy policy) {
  return policy == Policy::source_order || policy == Policy::widest_residual ||
         policy == Policy::aggregate_residual || policy == Policy::contracting_residual;
}
// Exact structural grouping only. +/-0 cuts compare identically. A missing
// direction is irrelevant only when this feature cannot admit NaN in the
// declared domain: all numeric features of a finite-only domain, and exactly-
// one finite 0/1 group members. Every original predicate word stays in Source.
inline HostMetadata make_metadata(const std::vector<std::int32_t>& feature,
    const std::vector<float>& cut, const std::vector<std::uint8_t>& missing,
    const domain::HostMetadata& domain_metadata) {
  require(feature.size() == cut.size() && feature.size() == missing.size() &&
          feature.size() <= UINT32_MAX, "split metadata source extents");
  require(domain_metadata.features && domain_metadata.allow_nan <= 1 &&
          domain_metadata.feature_group.size() == domain_metadata.features,
          "split metadata domain shape");
  for(auto group : domain_metadata.feature_group)
    require(group == -1 || (group >= 0 && u32(group) < domain_metadata.groups),
            "split metadata domain group index");
  HostMetadata out;out.group_ids.assign(feature.size(),none);
  std::map<std::tuple<u32,u32,u32>,u32> groups;
  for(u32 node = 0; node < feature.size(); ++node) {
    const auto f = feature[node];
    require(f >= -1 && (f < 0 || u32(f) < domain_metadata.features),
            "split metadata source feature index");
    if(f < 0)continue;
    u32 word = std::bit_cast<u32>(cut[node]);
    require((word & 0x7fffffffu) <= 0x7f800000u && missing[node] <= 1,
            "split metadata source predicate words");
    if((word & 0x7fffffffu) == 0)word = 0;
    const bool admits_nan = domain_metadata.allow_nan &&
                            domain_metadata.feature_group[u32(f)] < 0;
    const auto key = std::tuple{u32(f),word,admits_nan ? u32(missing[node]) : 0u};
    auto found = groups.find(key);
    if(found == groups.end()) {
      require(out.groups != none, "split metadata group extent");
      found = groups.emplace(key,out.groups++).first;
    }
    out.group_ids[node] = found->second;
  }
  return out;
}
struct Storage {
  Buffer<u32> group_ids;
  u32 groups = 0;
  Storage(Budget& budget,const HostMetadata& metadata)
      :group_ids(budget,metadata.group_ids.size()),groups(metadata.groups) {
    for(u32 group : metadata.group_ids)
      require(group == none || group < groups,"split storage group index");
    group_ids.upload(metadata.group_ids);
  }
  View view(Policy policy = Policy::source_order) const {
    require(valid_policy(policy),"invalid adaptive split policy");
    return {group_ids.data,groups,policy};
  }
};
namespace detail {
__device__ inline bool node_width(EngineView e,std::int32_t node,double& width) {
  width = 0;
  if(node < 0 || u32(node) >= e.source.nodes)return false;
  const float lo = __uint_as_float(e.minimum[node]),hi = __uint_as_float(e.maximum[node]);
  if(!isfinite(lo) || !isfinite(hi) || lo > hi)return false;
  width = __dsub_rn(double(hi),double(lo));
  return true;
}
// A strict current residual-root split ensures the selected source tree
// descends on both children. A candidate with a constant numerical enclosure
// remains the source-order fallback; signed leaf words/native ties may differ.
__device__ inline bool candidate(EngineView e,domain::RegionView R,
                                std::int32_t root,double& width) {
  width = 0;
  if(root < 0 || u32(root) >= e.source.nodes || e.source.left[root] < 0)return false;
  const auto feature = e.source.feature[root];
  if(feature < 0 || u32(feature) >= e.source.features)return false;
  bool right = false;
  if(domain::forced_side(e.domain,R,u32(feature),__float_as_uint(e.source.cut[root]),
                         bool(e.source.missing[root]),right))return false;
  return node_width(e,root,width) && width > 0;
}
// Static child enclosures are an advisory screen, not a conditioned enclosure
// or a class certificate. Width arithmetic stays FP64 on CUDA, using the
// existing GPU-computed subtree extrema without changing their RN32 leaves.
__device__ inline bool contraction(EngineView e,domain::RegionView R,
    std::int32_t root,double& gain,double& width) {
  gain = 0;
  if(!candidate(e,R,root,width))return false;
  double left = 0,right = 0;
  if(!node_width(e,e.source.left[root],left) ||
     !node_width(e,e.source.right[root],right))return false;
  const double difference = __dsub_rn(width,left > right ? left : right);
  gain = difference > 0 ? difference : 0;
  return true;
}
}
// group_scores has view.groups FP64 words private to this parent/job when
// aggregate_residual is selected. All admitted source/arena/context words are
// read-only. The caller commits the chosen predicate before publishing children;
// the exact state key remains region + RN32 prefix/positions + residual roots.
// Scores and ties are computed on CUDA, in original source-tree order. No
// competitive-channel filter or native-class assumption enters these policies.
__device__ inline u32 choose(EngineView e,u32 parent_id,View view,double* group_scores) {
  if(parent_id >= e.arena.state_capacity)return none;
  const u32 fallback = e.arena.states[parent_id].predicate;
  if(fallback == none || view.policy == Policy::source_order || !e.source.trees)
    return fallback;
  if(view.policy != Policy::widest_residual && view.policy != Policy::aggregate_residual &&
     view.policy != Policy::contracting_residual)
    return fallback;
  const auto R = region(e,parent_id);
  const auto* roots = e.arena.residual + u64(parent_id)*e.source.trees;
  u32 selected = fallback;double best = 0;
  if(view.policy == Policy::contracting_residual) {
    double best_gain = 0;
    detail::contraction(e,R,std::int32_t(fallback),best_gain,best);
    for(u32 tree = 0; tree < e.source.trees; ++tree) {
      double gain = 0,width = 0;
      if(detail::contraction(e,R,roots[tree],gain,width) &&
         (gain > best_gain || (gain == best_gain && width > best))) {
        best_gain = gain;best = width;selected = u32(roots[tree]);
      }
    }
    return selected;
  }
  if(view.policy == Policy::widest_residual) {
    detail::candidate(e,R,std::int32_t(fallback),best);
    for(u32 tree = 0; tree < e.source.trees; ++tree) {
      double width = 0;
      if(detail::candidate(e,R,roots[tree],width) && width > best) {
        best = width;selected = u32(roots[tree]);
      }
    }
    return selected;
  }
  if(!view.group_ids || !view.groups || !group_scores || fallback >= e.source.nodes)
    return fallback;
  const u32 fallback_group = view.group_ids[fallback];
  if(fallback_group >= view.groups)return fallback;
  for(u32 group = 0; group < view.groups; ++group)group_scores[group] = 0;
  for(u32 tree = 0; tree < e.source.trees; ++tree) {
    double width = 0;
    if(!detail::candidate(e,R,roots[tree],width))continue;
    const u32 group = view.group_ids[u32(roots[tree])];
    if(group >= view.groups)return fallback;
    group_scores[group] = __dadd_rn(group_scores[group],width);
  }
  best = group_scores[fallback_group];
  for(u32 tree = 0; tree < e.source.trees; ++tree) {
    double width = 0;
    if(!detail::candidate(e,R,roots[tree],width))continue;
    const u32 group = view.group_ids[u32(roots[tree])];
    if(group_scores[group] > best) {
      best = group_scores[group];selected = u32(roots[tree]);
    }
  }
  return selected;
}
} // namespace class_conversion_adaptive::split
