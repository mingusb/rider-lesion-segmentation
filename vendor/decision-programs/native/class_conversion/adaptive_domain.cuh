#pragma once
// Domain restrictions for the sole adaptive class constructor. Host code
// validates caller metadata; every FP32 restriction and witness runs on CUDA.
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace class_conversion_adaptive_domain {
using u32=std::uint32_t;
using u64=std::uint64_t;
constexpr u32 finite_min_key=0x00800000u,finite_max_key=0xff7fffffu;
constexpr u32 zero_key=0x80000000u,negative_zero_hole=0x7fffffffu;
struct DomainView {
 u32 features=0,groups=0,mask_words=0,allow_nan=1;
 const std::int32_t*feature_group=nullptr;
 const u32*feature_bit=nullptr;
 const u32*group_word_offsets=nullptr;
 const u32*group_feature_offsets=nullptr;
 const u32*group_widths=nullptr;
 const u32*group_features=nullptr;
 const u64*initial_masks=nullptr;
 // Physical features map to compact numeric coordinates; grouped features
 // need only their exactly-one mask, and map to -1.
 u32 numeric_features=0;
 const std::int32_t*feature_numeric=nullptr;
};
// Bounds/missing have numeric_features entries, allowed has mask_words.
// An all-grouped domain has zero numeric entries and permits null arrays.
struct RegionView {u32*lower=nullptr,*upper=nullptr,*missing=nullptr;u64*allowed=nullptr;};
struct HostMetadata {
 u32 features=0,groups=0,mask_words=0,allow_nan=1;
 std::vector<std::int32_t>feature_group;
 std::vector<u32>feature_bit,group_word_offsets,group_feature_offsets,group_widths,group_features;
 std::vector<u64>initial_masks;
 u32 numeric_features=0;
 std::vector<std::int32_t>feature_numeric;
};
// Groups are disjoint, nonempty, exactly-one constraints. Bits identify the
// caller's feature order inside each group; wider groups use additional words.
inline HostMetadata prepare_domain_metadata(u32 features,const std::vector<std::vector<u32>>&groups,bool allow_nan){
 if(!features||features>u32(INT32_MAX)||groups.size()>u32(INT32_MAX))throw std::runtime_error("adaptive domain metadata extent");
 HostMetadata h;h.features=features;h.groups=u32(groups.size());h.allow_nan=allow_nan?1:0;
 h.feature_group.assign(features,-1);h.feature_bit.assign(features,0);
 h.group_word_offsets.push_back(0);h.group_feature_offsets.push_back(0);
 for(u32 g=0;g<h.groups;++g){const auto&members=groups[g];
  if(members.empty()||members.size()>UINT32_MAX)throw std::runtime_error("adaptive one-hot group extent");
  const u32 width=u32(members.size());const u64 words=(u64(width)+63)/64;
  if(words>UINT32_MAX-h.initial_masks.size()||members.size()>UINT32_MAX-h.group_features.size())throw std::runtime_error("adaptive domain flattened extent");
  h.group_widths.push_back(width);
  for(u32 bit=0;bit<width;++bit){u32 f=members[bit];if(f>=features||h.feature_group[f]>=0)throw std::runtime_error("adaptive one-hot groups overlap or feature is absent");
   h.feature_group[f]=std::int32_t(g);h.feature_bit[f]=bit;h.group_features.push_back(f);}
  for(u64 word=0;word<words;++word){u32 tail=width%64;h.initial_masks.push_back(word+1==words&&tail?(u64(1)<<tail)-1:UINT64_MAX);}
  h.group_word_offsets.push_back(u32(h.initial_masks.size()));h.group_feature_offsets.push_back(u32(h.group_features.size()));
 }
 h.feature_numeric.assign(features,-1);
 for(u32 f=0;f<features;++f)if(h.feature_group[f]<0)h.feature_numeric[f]=std::int32_t(h.numeric_features++);
 h.mask_words=u32(h.initial_masks.size());return h;
}
__device__ inline bool nan_word(u32 word){return (word&0x7fffffffu)>0x7f800000u;}
__device__ inline u32 sortable_word(u32 word){if((word&0x7fffffffu)==0)word=0;return word&0x80000000u?~word:word^0x80000000u;}
__device__ inline u32 word_from_key(u32 key){return key&0x80000000u?key^0x80000000u:~key;}
__device__ inline u32 predecessor_key(u32 key){return key==zero_key?negative_zero_hole-1:key-1;}
__device__ inline bool finite_nonempty(u32 lower,u32 upper){return lower<=upper;}
__device__ inline bool initial_domain(DomainView d,RegionView r){
 if(!d.features||!d.feature_group||!d.feature_numeric||
    (d.numeric_features&&(!r.lower||!r.upper||!r.missing))||
    (d.mask_words&&(!r.allowed||!d.initial_masks)))return false;
 for(u32 n=0;n<d.numeric_features;++n){r.lower[n]=finite_min_key;r.upper[n]=finite_max_key;r.missing[n]=d.allow_nan;}
 for(u32 w=0;w<d.mask_words;++w)r.allowed[w]=d.initial_masks[w];return true;
}
__device__ inline bool region_valid(DomainView d,RegionView r){
 if(!d.features||!d.feature_group||!d.feature_numeric||
    (d.numeric_features&&(!r.lower||!r.upper||!r.missing))||
    (d.mask_words&&(!r.allowed||!d.initial_masks)))return false;
 for(u32 n=0;n<d.numeric_features;++n){u32 lo=r.lower[n],hi=r.upper[n];if(r.missing[n]>1||(!d.allow_nan&&r.missing[n]))return false;
  if(finite_nonempty(lo,hi)){if(lo<finite_min_key||hi>finite_max_key||lo==negative_zero_hole||hi==negative_zero_hole)return false;}
  else if(!r.missing[n])return false;
 }
 for(u32 g=0;g<d.groups;++g){bool any=false;for(u32 w=d.group_word_offsets[g];w<d.group_word_offsets[g+1];++w){if(r.allowed[w]&~d.initial_masks[w])return false;any|=r.allowed[w]!=0;}if(!any)return false;}
 return true;
}
__device__ inline void restricted_finite(RegionView r,u32 f,u32 cut_word,bool right,u32&lo,u32&hi){
 lo=r.lower[f];hi=r.upper[f];if(!finite_nonempty(lo,hi))return;u32 key=sortable_word(cut_word);
 if(right){if(lo<key)lo=key;if(lo==negative_zero_hole)lo=zero_key;}
 else{u32 end=predecessor_key(key);if(hi>end)hi=end;if(hi==negative_zero_hole)hi=negative_zero_hole-1;}
 if(!finite_nonempty(lo,hi)){lo=finite_min_key;hi=finite_min_key-1;}
}
__device__ inline bool group_branch_nonempty(DomainView d,RegionView r,u32 f,u32 cut_word,bool right){
 u32 g=u32(d.feature_group[f]),bit=d.feature_bit[f],word=d.group_word_offsets[g]+bit/64;u64 mask=u64(1)<<(bit%64);
 bool one=(r.allowed[word]&mask)!=0,zero=false;
 for(u32 w=d.group_word_offsets[g];w<d.group_word_offsets[g+1];++w)zero|=(r.allowed[w]&(w==word?~mask:UINT64_MAX))!=0;
 float cut=__uint_as_float(cut_word);return (one&&((1.f<cut)==!right))||(zero&&((0.f<cut)==!right));
}
__device__ inline bool region_split_feasible(DomainView d,RegionView r,u32 f,u32 cut_word,bool default_left,bool right){
 if(f>=d.features||nan_word(cut_word))return false;
 if(d.feature_group[f]>=0)return group_branch_nonempty(d,r,f,cut_word,right);
 const std::int32_t slot=d.feature_numeric[f];if(slot<0||u32(slot)>=d.numeric_features)return false;
 const u32 n=u32(slot);u32 lo,hi;restricted_finite(r,n,cut_word,right,lo,hi);
 return finite_nonempty(lo,hi)||(r.missing[n]&&(default_left!=right));
}
// Check before mutation: refusal leaves the caller's region unchanged. The
// two successful children partition the parent by the exact source predicate.
__device__ inline bool region_restrict(DomainView d,RegionView r,u32 f,u32 cut_word,bool default_left,bool right){
 if(!region_split_feasible(d,r,f,cut_word,default_left,right))return false;
 if(d.feature_group[f]>=0){u32 g=u32(d.feature_group[f]),bit=d.feature_bit[f],word=d.group_word_offsets[g]+bit/64;u64 mask=u64(1)<<(bit%64);float cut=__uint_as_float(cut_word);
  bool keep_one=(1.f<cut)==!right,keep_zero=(0.f<cut)==!right;
  for(u32 w=d.group_word_offsets[g];w<d.group_word_offsets[g+1];++w){u64 own=w==word?mask:0;u64 keep=(keep_one?own:0)|(keep_zero?~own:0);r.allowed[w]&=keep;}
 }else{const u32 n=u32(d.feature_numeric[f]);u32 lo,hi;restricted_finite(r,n,cut_word,right,lo,hi);r.lower[n]=lo;r.upper[n]=hi;r.missing[n]=r.missing[n]&&(default_left!=right);}
 return true;
}
__device__ inline bool forced_side(DomainView d,RegionView r,u32 f,u32 cut_word,bool default_left,bool&right){
 bool l=region_split_feasible(d,r,f,cut_word,default_left,false),q=region_split_feasible(d,r,f,cut_word,default_left,true);if(l==q)return false;right=q;return true;
}
// Modes0/1 choose lower/upper finite endpoints and first/last valid categories.
// Mode2 chooses permitted NaN, otherwise -0; mode3 prefers -0 even when NaN
// is also admitted, and mode4 prefers +0. NaN-only regions stay NaN.
// These are valid witnesses, not a universal native-class proof by observation.
__device__ inline bool region_witness(DomainView d,RegionView r,float*out,int variant=0){
 if(!out||!region_valid(d,r))return false;
 for(u32 f=0;f<d.features;++f){if(d.feature_group[f]>=0){out[f]=0.f;continue;}
  const std::int32_t slot=d.feature_numeric[f];if(slot<0||u32(slot)>=d.numeric_features)return false;
  const u32 n=u32(slot);u32 lo=r.lower[n],hi=r.upper[n];bool finite=finite_nonempty(lo,hi);u32 word;
  if(!finite||(variant==2&&r.missing[n]))word=variant&1?0xffc12345u:0x7fc00001u;
  else if((variant==2||variant==3||variant==4)&&lo<=zero_key&&zero_key<=hi)word=variant==4?0:0x80000000u;
  else word=word_from_key(variant==1||variant==3?hi:lo);out[f]=__uint_as_float(word);
 }
 for(u32 g=0;g<d.groups;++g){u32 chosen=UINT32_MAX;
  if(variant==1||variant==3){for(u32 w=d.group_word_offsets[g+1];w>d.group_word_offsets[g];){--w;u64 bits=r.allowed[w];if(bits){chosen=(w-d.group_word_offsets[g])*64+63-u32(__clzll(bits));break;}}}
  else{for(u32 w=d.group_word_offsets[g];w<d.group_word_offsets[g+1];++w){u64 bits=r.allowed[w];if(bits){chosen=(w-d.group_word_offsets[g])*64+u32(__ffsll(static_cast<long long>(bits))-1);break;}}}
  if(chosen>=d.group_widths[g])return false;out[d.group_features[d.group_feature_offsets[g]+chosen]]=1.f;
 }
 return true;
}
} // namespace class_conversion_adaptive_domain