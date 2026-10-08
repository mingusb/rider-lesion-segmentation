#pragma once
#include "adaptive_engine.cuh"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

namespace class_conversion_adaptive::grid {
// Setup metadata only. Numerical regional counts and branch weights run on CUDA.
// The catalog must accompany the exact source/domain from which it was made.
// Missing is a separate atom, even if its predicate signature matches a finite
// bin; this defines an explicit refinement grid when NaN inputs are admitted.
struct HostCatalog {
  u32 features=0,numeric_features=0,groups=0,mask_words=0,allow_nan=0;
  std::vector<u32> numeric_offsets,numeric_cut_keys;
  std::vector<u64> singleton_masks,remainder_masks,base_axis_bins;
};
struct CatalogView {
  u32 features=0,numeric_features=0,groups=0,mask_words=0,allow_nan=0,cut_count=0;
  const u32* numeric_offsets=nullptr;
  const u32* numeric_cut_keys=nullptr;
  const u64* singleton_masks=nullptr;
  const u64* remainder_masks=nullptr;
  const u64* base_axis_bins=nullptr;
};
namespace detail {
inline u32 host_key(u32 word) {
  if((word&0x7fffffffu)==0)word=0;
  return word&0x80000000u?~word:word^0x80000000u;
}
inline bool host_nan(u32 word){return (word&0x7fffffffu)>0x7f800000u;}
inline void check_metadata(const domain::HostMetadata& d) {
  require(d.features&&d.allow_nan<=1&&d.numeric_features<=d.features,
          "grid domain shape");
  require(d.feature_group.size()==d.features&&d.feature_numeric.size()==d.features&&
          d.feature_bit.size()==d.features&&d.group_widths.size()==d.groups&&
          d.group_word_offsets.size()==u64(d.groups)+1&&
          d.group_feature_offsets.size()==u64(d.groups)+1&&
          d.initial_masks.size()==d.mask_words,"grid domain metadata extents");
  require(d.group_word_offsets.front()==0&&d.group_word_offsets.back()==d.mask_words&&
          d.group_feature_offsets.front()==0&&
          d.group_feature_offsets.back()==d.group_features.size(),"grid domain flattened extents");
  std::vector<u32> seen_numeric(d.numeric_features),seen_group(d.features);
  for(u32 g=0;g<d.groups;++g) {
    const u32 width=d.group_widths[g],begin=d.group_word_offsets[g],end=d.group_word_offsets[g+1];
    require(width&&begin<=end&&end<=d.mask_words&&u64(end)-begin==(u64(width)+63)/64&&
            d.group_feature_offsets[g]<=d.group_feature_offsets[g+1]&&
            d.group_feature_offsets[g+1]<=d.group_features.size()&&
            u64(d.group_feature_offsets[g+1])-d.group_feature_offsets[g]==width,
            "grid group extent");
    bool any=false;
    for(u32 w=begin;w<end;++w) {
      const u32 tail=width%64;
      const u64 valid=w+1==end&&tail?(u64(1)<<tail)-1:UINT64_MAX;
      require(!(d.initial_masks[w]&~valid),"grid initial group mask extent");
      any|=d.initial_masks[w]!=0;
    }
    require(any,"grid empty initial group");
    for(u32 bit=0;bit<width;++bit) {
      const u32 f=d.group_features[d.group_feature_offsets[g]+bit];
      require(f<d.features&&!seen_group[f]++&&d.feature_group[f]==std::int32_t(g)&&
              d.feature_numeric[f]==-1&&d.feature_bit[f]==bit,"grid group mapping");
    }
  }
  for(u32 f=0;f<d.features;++f) {
    if(d.feature_group[f]<0) {
      const auto n=d.feature_numeric[f];
      require(d.feature_group[f]==-1&&n>=0&&u32(n)<d.numeric_features&&!seen_numeric[u32(n)]++,
              "grid numeric mapping");
    }else require(u32(d.feature_group[f])<d.groups&&seen_group[f]==1,"grid group feature mapping");
  }
  for(auto seen:seen_numeric)require(seen==1,"grid numeric mapping incomplete");
}
}
inline HostCatalog make_catalog(const domain::HostMetadata& d,
    const std::vector<std::int32_t>& source_feature,const std::vector<float>& source_cut) {
  detail::check_metadata(d);
  require(source_feature.size()==source_cut.size(),"grid source metadata extents");
  HostCatalog h;h.features=d.features;h.numeric_features=d.numeric_features;
  h.groups=d.groups;h.mask_words=d.mask_words;h.allow_nan=d.allow_nan;
  h.singleton_masks.assign(d.mask_words,0);
  std::vector<std::vector<u32>> keys(d.numeric_features);
  const u32 zero=detail::host_key(0),one=detail::host_key(0x3f800000u);
  for(size_t at=0;at<source_feature.size();++at) {
    const auto f=source_feature[at];
    require(f>=-1&&(f<0||u32(f)<d.features),"grid source feature index");
    if(f<0)continue; // SourceData's terminal marker; leaf response is not a cut.
    const u32 word=std::bit_cast<u32>(source_cut[at]);
    require(!detail::host_nan(word),"grid source NaN predicate");
    const u32 key=detail::host_key(word);
    if(d.feature_group[u32(f)]<0) {
      if(domain::finite_min_key<key&&key<=domain::finite_max_key)
        keys[u32(d.feature_numeric[u32(f)])].push_back(key);
    }else if(zero<key&&key<=one) {
      const u32 g=u32(d.feature_group[u32(f)]),bit=d.feature_bit[u32(f)];
      const u32 w=d.group_word_offsets[g]+bit/64;const u64 mask=u64(1)<<(bit%64);
      h.singleton_masks[w]|=d.initial_masks[w]&mask;
    }
  }
  h.numeric_offsets.push_back(0);
  for(auto& axis:keys) {
    std::sort(axis.begin(),axis.end());axis.erase(std::unique(axis.begin(),axis.end()),axis.end());
    require(axis.size()<=UINT32_MAX-h.numeric_cut_keys.size(),"grid cut catalog index extent");
    h.numeric_cut_keys.insert(h.numeric_cut_keys.end(),axis.begin(),axis.end());
    h.numeric_offsets.push_back(u32(h.numeric_cut_keys.size()));
    h.base_axis_bins.push_back(u64(axis.size())+1+h.allow_nan);
  }
  h.remainder_masks.resize(d.mask_words);
  for(u32 w=0;w<d.mask_words;++w)h.remainder_masks[w]=d.initial_masks[w]&~h.singleton_masks[w];
  for(u32 g=0;g<d.groups;++g) {
    u64 bins=0;bool remainder=false;
    for(u32 w=d.group_word_offsets[g];w<d.group_word_offsets[g+1];++w) {
      bins+=std::popcount(h.singleton_masks[w]);remainder|=h.remainder_masks[w]!=0;
    }
    h.base_axis_bins.push_back(bins+u64(remainder));
  }
  return h;
}
struct Storage {
  HostCatalog shape;
  Buffer<u32> numeric_offsets,numeric_cut_keys;
  Buffer<u64> singleton_masks,remainder_masks,base_axis_bins;
  Storage(Budget& budget,const HostCatalog& h):shape(h),
    numeric_offsets(budget,h.numeric_offsets.size()),numeric_cut_keys(budget,h.numeric_cut_keys.size()),
    singleton_masks(budget,h.singleton_masks.size()),remainder_masks(budget,h.remainder_masks.size()),
    base_axis_bins(budget,h.base_axis_bins.size()) {
    require(h.numeric_offsets.size()==u64(h.numeric_features)+1&&
            h.singleton_masks.size()==h.mask_words&&h.remainder_masks.size()==h.mask_words&&
            h.base_axis_bins.size()==u64(h.numeric_features)+h.groups,"grid storage extents");
    numeric_offsets.upload(h.numeric_offsets);numeric_cut_keys.upload(h.numeric_cut_keys);
    singleton_masks.upload(h.singleton_masks);remainder_masks.upload(h.remainder_masks);
    base_axis_bins.upload(h.base_axis_bins);
  }
  CatalogView view()const{return {shape.features,shape.numeric_features,shape.groups,shape.mask_words,
    shape.allow_nan,u32(numeric_cut_keys.size),numeric_offsets.data,numeric_cut_keys.data,
    singleton_masks.data,remainder_masks.data,base_axis_bins.data};}
};
struct BranchCounts {u64 left=0,right=0,total=0;bool valid=false;};
namespace detail {
__device__ inline bool same_shape(const EngineView& e,CatalogView c) {
  return c.features==e.source.features&&c.features==e.domain.features&&
    c.numeric_features==e.domain.numeric_features&&c.groups==e.domain.groups&&
    c.mask_words==e.domain.mask_words&&c.allow_nan==e.domain.allow_nan;
}
__device__ inline u32 upper_bound_key(CatalogView c,u32 begin,u32 end,u32 key) {
  while(begin<end){u32 middle=begin+(end-begin)/2;
    if(c.numeric_cut_keys[middle]<=key)begin=middle+1;else end=middle;}
  return begin;
}
__device__ inline u64 finite_bins(CatalogView c,u32 n,u32 lower,u32 upper) {
  if(lower>upper)return 0;
  const u32 begin=c.numeric_offsets[n],end=c.numeric_offsets[n+1];
  return u64(upper_bound_key(c,begin,end,upper))-upper_bound_key(c,begin,end,lower)+1;
}
__device__ inline bool numeric_valid(const EngineView& e,domain::RegionView r,u32 n) {
  if(!r.lower||!r.upper||!r.missing||r.missing[n]>1||(!e.domain.allow_nan&&r.missing[n]))return false;
  const u32 lo=r.lower[n],hi=r.upper[n];
  if(lo<=hi)return lo>=domain::finite_min_key&&hi<=domain::finite_max_key&&
    lo!=domain::negative_zero_hole&&hi!=domain::negative_zero_hole;
  return r.missing[n]!=0;
}
__device__ inline u64 group_bins(CatalogView c,const domain::DomainView& d,
    domain::RegionView r,u32 g,u32 split_word=none,u64 split_mask=0,bool keep_one=true,bool keep_zero=true) {
  u64 bins=0;bool remainder=false;
  for(u32 w=d.group_word_offsets[g];w<d.group_word_offsets[g+1];++w) {
    const u64 own=w==split_word?split_mask:0;
    const u64 keep=(keep_one?own:0)|(keep_zero?~own:0);
    const u64 allowed=r.allowed[w]&keep;
    bins+=__popcll(allowed&c.singleton_masks[w]);
    remainder|=(allowed&c.remainder_masks[w])!=0;
  }
  return bins+u64(remainder);
}
}
// Axis order is compact numeric coordinates followed by whole exactly-one groups.
// Caller supplies a validated admitted region from the catalog's exact domain.
__device__ inline u64 axis_bins(EngineView e,CatalogView c,domain::RegionView r,u32 axis) {
  if(!detail::same_shape(e,c)||axis>=u64(c.numeric_features)+c.groups)return 0;
  if(axis<c.numeric_features) {
    if(!detail::numeric_valid(e,r,axis))return 0;
    return detail::finite_bins(c,axis,r.lower[axis],r.upper[axis])+r.missing[axis];
  }
  if(!r.allowed)return 0;
  return detail::group_bins(c,e.domain,r,axis-c.numeric_features);
}
// Counts refer to the parent's split axis, before any child support projection.
// Non-catalog cuts which bisect a finite bin are refused by the additive check.
__device__ inline BranchCounts branch_counts(EngineView e,CatalogView c,
    domain::RegionView r,u32 feature,u32 cut_word,bool default_left) {
  BranchCounts out;
  if(!detail::same_shape(e,c)||feature>=e.domain.features||domain::nan_word(cut_word)||
     !e.domain.feature_group||!e.domain.feature_numeric)return out;
  const auto group=e.domain.feature_group[feature];
  if(group>=0) {
    const u32 g=u32(group);if(g>=c.groups||!r.allowed||!e.domain.initial_masks)return out;
    bool any=false;
    for(u32 w=e.domain.group_word_offsets[g];w<e.domain.group_word_offsets[g+1];++w) {
      if(r.allowed[w]&~e.domain.initial_masks[w])return out;any|=r.allowed[w]!=0;
    }
    if(!any)return out;
    const u32 bit=e.domain.feature_bit[feature],word=e.domain.group_word_offsets[g]+bit/64;
    const u64 mask=u64(1)<<(bit%64);const u32 key=domain::sortable_word(cut_word);
    const bool one=domain::sortable_word(0x3f800000u)<key,zero=domain::zero_key<key;
    out.total=detail::group_bins(c,e.domain,r,g);
    out.left=detail::group_bins(c,e.domain,r,g,word,mask,one,zero);
    out.right=detail::group_bins(c,e.domain,r,g,word,mask,!one,!zero);
  }else {
    const auto coordinate=e.domain.feature_numeric[feature];
    if(coordinate<0||u32(coordinate)>=c.numeric_features)return out;const u32 n=u32(coordinate);
    if(!detail::numeric_valid(e,r,n))return out;
    out.total=detail::finite_bins(c,n,r.lower[n],r.upper[n])+r.missing[n];
    u32 lo,hi;domain::restricted_finite(r,n,cut_word,false,lo,hi);
    out.left=detail::finite_bins(c,n,lo,hi)+(r.missing[n]&&default_left);
    domain::restricted_finite(r,n,cut_word,true,lo,hi);
    out.right=detail::finite_bins(c,n,lo,hi)+(r.missing[n]&&!default_left);
  }
  out.valid=out.total>0&&out.left<=out.total&&out.right<=out.total&&out.left+out.right==out.total;
  return out;
}
__device__ inline BranchCounts branch_counts(EngineView e,CatalogView c,u32 state_id) {
  if(!e.arena.states||state_id>=e.arena.state_capacity)return {};
  const u32 predicate=e.arena.states[state_id].predicate;
  if(predicate>=e.source.nodes||!e.source.feature||!e.source.cut||!e.source.missing)return {};
  const auto f=e.source.feature[predicate];if(f<0)return {};
  return branch_counts(e,c,region(e,state_id),u32(f),__float_as_uint(e.source.cut[predicate]),
                       e.source.missing[predicate]!=0);
}
} // namespace class_conversion_adaptive::grid