#pragma once
#include "adaptive_work_estimate.cuh"
#include <type_traits>

// CUDA reporting reduction only. It estimates sampled, unshared construction
// work under the sampler's frozen controls; it supplies neither an ETA nor a
// confidence interval. Censored prefixes remain separate from completed paths.
namespace class_conversion_adaptive::work {
struct Metric {
  double mean=0.,standard_error=0.,minimum=0.,maximum=0.,max_fraction_of_total=0.;
  u64 count=0,standard_error_available=0,max_fraction_available=0;
  u64 invalid_values=0,numeric_overflow=0;
};
struct Metrics {
  Metric decisions,expanding_vertices,class_prunes,native_terminals,refinement_visits;
};
struct Depth {
  u64 count=0,minimum=0,maximum=0;
  double mean=0.;
};
struct Summary {
  u64 paths=0,done=0,censored=0,errors=0,overflows=0,invalid_status=0;
  u64 valid_total=0,completed_selection_conditioned=0;
  // Direct fields include finite, error-free complete and censored paths.
  // If valid_total is false, these are only observed partial-work statistics.
  Metric decisions,expanding_vertices,class_prunes,native_terminals,refinement_visits;
  Metrics completed,partial_censored;
  Depth depth;
};
static_assert(sizeof(Metric)==80&&sizeof(Metrics)==400&&sizeof(Depth)==32&&sizeof(Summary)==1296);
static_assert(std::is_trivially_copyable_v<Summary>);

namespace summary_detail {
enum class Group:u32 { all=0,completed=1,censored=2 };
enum class Field:u32 { decisions=0,expanding_vertices=1,class_prunes=2,native_terminals=3,refinement_visits=4 };
__device__ inline bool status_valid(const Result& r) {
  return r.done<=1&&r.censored<=1&&r.weight_overflow<=1&&r.done+r.censored==1;
}
__device__ inline bool selected(const Result& r,Group group) {
  if(!status_valid(r)||r.error||r.weight_overflow)return false;
  return group==Group::all||(group==Group::completed?r.done!=0:r.censored!=0);
}
__device__ inline double value(const Result& r,Field field) {
  switch(field) {
    case Field::decisions:return r.weighted_decisions;
    case Field::expanding_vertices:return r.weighted_expanding_vertices;
    case Field::class_prunes:return r.weighted_class_prunes;
    case Field::native_terminals:return r.weighted_native_terminals;
    case Field::refinement_visits:return r.weighted_refinement_visits;
  }
  return 0.;
}
__device__ inline bool value_valid(double x) { return isfinite(x)&&x>=0.; }
__device__ Metric reduce(const Result* samples,u32 paths,Group group,Field field) {
  Metric result;
  for(u32 i=0;i<paths;++i)if(selected(samples[i],group)) {
    const double x=value(samples[i],field);
    if(!value_valid(x)){++result.invalid_values;continue;}
    if(!result.count||x<result.minimum)result.minimum=x;
    if(x>result.maximum)result.maximum=x;
    ++result.count;
  }
  if(!result.count)return result;
  // Every normalized observation lies in [0,1]. This avoids overflowing the
  // raw sum or square even when an individual finite path is near DBL_MAX.
  double sum=0.;
  if(result.maximum>0.)for(u32 i=0;i<paths;++i)if(selected(samples[i],group)) {
    const double x=value(samples[i],field);
    if(value_valid(x))sum=__dadd_rn(sum,__ddiv_rn(x,result.maximum));
  }
  const double normalized_mean=__ddiv_rn(sum,double(result.count));
  result.mean=__dmul_rn(result.maximum,normalized_mean);
  if(result.maximum>0.&&sum>0.) {
    result.max_fraction_of_total=__ddiv_rn(1.,sum);
    result.max_fraction_available=1;
  }
  if(result.count>1) {
    double squared_deviations=0.;
    for(u32 i=0;i<paths;++i)if(selected(samples[i],group)) {
      const double x=value(samples[i],field);
      if(!value_valid(x))continue;
      const double normalized=result.maximum>0.?__ddiv_rn(x,result.maximum):0.;
      const double delta=__dsub_rn(normalized,normalized_mean);
      squared_deviations=__dadd_rn(squared_deviations,__dmul_rn(delta,delta));
    }
    const double denominator=__dmul_rn(double(result.count),double(result.count-1));
    result.standard_error=__dmul_rn(result.maximum,
      __dsqrt_rn(__ddiv_rn(squared_deviations,denominator)));
    result.standard_error_available=1;
  }
  if(!isfinite(result.mean)||!isfinite(result.standard_error)||
      !isfinite(result.max_fraction_of_total)) {
    result.numeric_overflow=1;result.standard_error_available=0;
  }
  return result;
}
__device__ inline Metrics bundle(const Result* samples,u32 paths,Group group) {
  return {reduce(samples,paths,group,Field::decisions),
    reduce(samples,paths,group,Field::expanding_vertices),
    reduce(samples,paths,group,Field::class_prunes),
    reduce(samples,paths,group,Field::native_terminals),
    reduce(samples,paths,group,Field::refinement_visits)};
}
__device__ inline bool valid_metric(const Metric& metric,u32 paths) {
  return metric.count==paths&&!metric.invalid_values&&!metric.numeric_overflow;
}
__global__ void aggregate(const Result* samples,u32 paths,Summary* output) {
  // The public helper launches exactly one thread, keeping input order fixed.
  if(blockIdx.x||blockIdx.y||blockIdx.z||threadIdx.x||threadIdx.y||threadIdx.z)return;
  Summary result;result.paths=paths;
  u64 depth_sum=0;
  for(u32 i=0;i<paths;++i) {
    const auto& r=samples[i];
    result.done+=r.done!=0;result.censored+=r.censored!=0;
    result.errors+=r.error!=0;result.overflows+=r.weight_overflow!=0;
    result.invalid_status+=!status_valid(r);
    if(!i||r.depth<result.depth.minimum)result.depth.minimum=r.depth;
    if(r.depth>result.depth.maximum)result.depth.maximum=r.depth;
    depth_sum+=r.depth;
  }
  result.depth.count=paths;
  if(paths)result.depth.mean=__ddiv_rn(double(depth_sum),double(paths));
  const auto all=bundle(samples,paths,Group::all);
  result.decisions=all.decisions;result.expanding_vertices=all.expanding_vertices;
  result.class_prunes=all.class_prunes;result.native_terminals=all.native_terminals;
  result.refinement_visits=all.refinement_visits;
  result.completed=bundle(samples,paths,Group::completed);
  result.partial_censored=bundle(samples,paths,Group::censored);
  result.valid_total=paths&&!result.censored&&!result.errors&&!result.overflows&&!result.invalid_status&&
    valid_metric(result.decisions,paths)&&valid_metric(result.expanding_vertices,paths)&&
    valid_metric(result.class_prunes,paths)&&valid_metric(result.native_terminals,paths)&&
    valid_metric(result.refinement_visits,paths);
  result.completed_selection_conditioned=!result.valid_total;
  *output=result;
}
} // namespace summary_detail

// Output is caller-budgeted; no source/dataset/library I/O occurs. Numerical
// aggregation remains on CUDA, and only one compact metadata object is copied.
inline Summary summarize(const Result* samples,u32 paths,Buffer<Summary>& output) {
  require(output.data&&output.size>=1,"work summary output shape");
  require(!paths||samples,"work summary input shape");
  output.zero();
  summary_detail::aggregate<<<1,1>>>(samples,paths,output.data);
  synchronize();
  return output.download(1).front();
}
} // namespace class_conversion_adaptive::work
