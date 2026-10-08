#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Host shape/resource arithmetic only. The converter's model arithmetic and
// exact key/node admission remain on CUDA and do not depend on this policy.
namespace class_conversion_growth {
using u32=std::uint32_t;
using u64=std::uint64_t;
inline u64 add(u64 a,u64 b) {
  if(a>UINT64_MAX-b)throw std::runtime_error("adaptive growth byte sum overflow");
  return a+b;
}
inline u64 multiply(u64 a,u64 b) {
  if(b&&a>UINT64_MAX/b)throw std::runtime_error("adaptive growth byte extent overflow");
  return a*b;
}
inline u32 bucket_count(u32 capacity) {
  if(!capacity||capacity>0x3fffffffu)throw std::runtime_error("adaptive growth index capacity");
  u32 count=1;while(u64(count)<u64(capacity)*2)count*=2;return count;
}
struct Shape {
  u32 numeric_features=0,classes=0,trees=0,mask_words=0;
  u64 state_record_bytes=0,node_record_bytes=0;
};
inline u64 state_bytes(const Shape& shape,u32 capacity) {
  u64 one=add(shape.state_record_bytes,4); // traversal stack
  one=add(one,multiply(shape.classes,8));
  one=add(one,multiply(shape.numeric_features,24));
  one=add(one,multiply(shape.trees,4));
  one=add(one,multiply(shape.mask_words,16));
  return add(multiply(capacity,one),multiply(bucket_count(capacity),8));
}
inline u64 queue_bytes(u32 capacity) {return multiply(capacity,32);}
inline u64 node_bytes(const Shape& shape,u32 capacity) {
  return add(multiply(capacity,shape.node_record_bytes),multiply(bucket_count(capacity),4));
}
struct Plan {
  u32 current=0,required=0,preferred=0,selected=0;
  u64 additional_bytes=0,available_bytes=0,minimum_bytes=0;
  bool affordable=false;
};
struct Outcome {
  Plan plan;
  u64 allocation_refusals=0,transaction_rollbacks=0;
  bool complete=false;
};
// Cost includes all replacement owners that coexist with the old owners. It
// is monotone, including table bucket cliffs; capacities need not be powers2.
template<class Cost>inline Plan choose(u32 current,u64 minimum,u64 ceiling,
                                      u64 available,Cost&& cost,u32 retry_ceiling=0) {
  if(!current||current>=ceiling||ceiling>0x3fffffffu)
    throw std::runtime_error("adaptive growth index limit reached");
  minimum=std::max<u64>(minimum,u64(current)+1);
  if(minimum>ceiling)throw std::runtime_error("adaptive required growth exceeds index limit");
  Plan out;out.current=current;out.required=u32(minimum);
  out.preferred=u32(std::min<u64>(ceiling,std::max<u64>(minimum,u64(current)*2)));
  out.available_bytes=available;out.minimum_bytes=cost(out.required);
  u32 upper=retry_ceiling?std::min(out.preferred,retry_ceiling):out.preferred;
  if(upper<out.required||out.minimum_bytes>available)return out;
  u32 low=out.required,high=upper;
  while(low<high) {
    u32 middle=low+(high-low+1)/2;
    if(cost(middle)<=available)low=middle;else high=middle-1;
  }
  out.selected=low;out.additional_bytes=cost(low);out.affordable=true;return out;
}
// A real allocation refusal can reveal fragmentation or allocation drift not
// visible in a free-byte snapshot. Geometric retry reaches the required minimum
// in finitely many attempts, without interpreting non-allocation CUDA errors.
inline u32 retry_ceiling(const Plan& plan) {
  return plan.selected>plan.required
    ?plan.required+(plan.selected-plan.required)/2:0;
}
} // namespace class_conversion_growth
