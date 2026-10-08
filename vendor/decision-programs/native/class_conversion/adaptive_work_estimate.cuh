#pragma once
#include "adaptive_effort.cuh"
#include "adaptive_split_select.cuh"
#include <cfloat>

// Reporting only: no state/node publication, cache lookup, native prediction,
// or class certificate is created. The caller retains validated immutable
// source/domain/support/extrema and authentic gate ownership during every call.
namespace class_conversion_adaptive::work {
enum class Stop : u32 { partial=0, native_obligation=1, class_proof=2,
                        decision_budget=3, source_error=4, weight_overflow=5 };
struct Options {
  u32 paths=0;
  u64 seed=0;
  u32 maximum_decisions=0;
  u32 refinement_visit_budget=0; // fixed, clamped to source-derived maximum
};
struct Result {
  u64 decisions=0,expanding_vertices=0,class_prunes=0,native_terminals=0;
  u64 refinement_attempts=0,refinement_visits=0,refinement_additional_prunes=0;
  u64 normalization_steps=0,prefix_additions=0,rng_state=0;
  double inverse_probability=1.,weighted_decisions=0.,weighted_expanding_vertices=0.;
  double weighted_class_prunes=0.,weighted_native_terminals=0.,weighted_refinement_visits=0.;
  double weighted_refinement_additional_prunes=0.,weighted_normalization_steps=0.,weighted_prefix_additions=0.;
  u32 depth=0,done=0,censored=1,error=0,weight_overflow=0;
  u32 stop_kind=u32(Stop::partial),pruned_label=none,reserved=0;
};
static_assert(sizeof(Result)==184); // all fields initialized; no transport padding
struct View {
  EngineView base;
  Options options;
  State* states;
  Status* status;
  Result* results;
  u32 *words,*positions,*blocked,*lower,*upper,*missing;
  u32 *witness_lower,*witness_upper,*witness_missing,*refinement_stack;
  std::int32_t* residual;
  u64 *allowed,*witness_allowed,*active_support;
  float *range_lower,*range_upper;
  u32 refinement_stack_capacity;
};
namespace detail {
template<class T> __device__ inline T* at(T* pointer,u64 stride,u32 path) {
  return stride ? pointer+u64(path)*stride : nullptr;
}
__device__ inline EngineView private_view(const View& v,u32 path) {
  auto e=v.base;
  e.arena={};
  e.arena.states=v.states+path;
  e.arena.words=at(v.words,e.source.classes,path);
  e.arena.positions=at(v.positions,e.source.classes,path);
  e.arena.residual=at(v.residual,e.source.trees,path);
  e.arena.lower=at(v.lower,e.domain.numeric_features,path);
  e.arena.upper=at(v.upper,e.domain.numeric_features,path);
  e.arena.missing=at(v.missing,e.domain.numeric_features,path);
  e.arena.allowed=at(v.allowed,e.domain.mask_words,path);
  e.arena.witness_lower=at(v.witness_lower,e.domain.numeric_features,path);
  e.arena.witness_upper=at(v.witness_upper,e.domain.numeric_features,path);
  e.arena.witness_missing=at(v.witness_missing,e.domain.numeric_features,path);
  e.arena.witness_allowed=at(v.witness_allowed,e.domain.mask_words,path);
  e.arena.state_capacity=1;
  e.draft=e.arena.states;
  e.draft_words=e.arena.words;e.draft_positions=e.arena.positions;
  e.draft_residual=e.arena.residual;
  e.draft_region=region(e,0);e.draft_witness=witness_region(e,0);
  e.blocked=at(v.blocked,e.source.classes,path);
  e.active_support=at(v.active_support,e.support_words,path);
  e.range_lower=at(v.range_lower,e.source.classes,path);
  e.range_upper=at(v.range_upper,e.source.classes,path);
  e.status=v.status+path;
  e.refinement_stack_capacity=v.refinement_stack_capacity;
  return e;
}
__device__ inline u64 random_word(u64& state) {
  // SplitMix64; path index and seed determine the stream, independently of chunks.
  u64 x=(state+=0x9e3779b97f4a7c15ull);
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;
  x=(x^(x>>27))*0x94d049bb133111ebull;
  return x^(x>>31);
}
__device__ inline void overflow(Result& r) {
  r.weight_overflow=1;r.censored=1;r.stop_kind=u32(Stop::weight_overflow);
}
__device__ inline bool add(Result& r,double& target,double amount) {
  const double sum=__dadd_rn(target,amount);
  if(!isfinite(amount)||!isfinite(sum)){overflow(r);return false;}
  target=sum;return true;
}
__device__ inline bool add_cost(Result& r,double& target,u64 count) {
  return add(r,target,__dmul_rn(r.inverse_probability,double(count)));
}
__device__ inline void source_error(Result& r,u32 error) {
  r.error=error;r.censored=1;r.stop_kind=u32(Stop::source_error);
}
__device__ inline bool account_normalization(Result& r,const Status& s,u64 old_steps,u64 old_adds) {
  const u64 steps=s.normalization_steps-old_steps,adds=s.prefix_additions-old_adds;
  r.normalization_steps+=steps;r.prefix_additions+=adds;
  return add_cost(r,r.weighted_normalization_steps,steps)&&
         add_cost(r,r.weighted_prefix_additions,adds);
}
__global__ void initialize(View v) {
  const u32 path=blockIdx.x*blockDim.x+threadIdx.x;if(path>=v.options.paths)return;
  auto e=private_view(v,path);auto& r=v.results[path];
  *e.status=Status{};*e.draft=State{};r=Result{};
  r.rng_state=v.options.seed+0x9e3779b97f4a7c15ull*(u64(path)+1);
  if(!domain::initial_domain(e.domain,e.draft_witness)){source_error(r,2);return;}
  copy_region(e,e.draft_region,e.draft_witness);
  for(u32 c=0;c<e.source.classes;++c){
    e.draft_words[c]=__float_as_uint(e.source.bias[c]);e.draft_positions[c]=0;
  }
  for(u32 t=0;t<e.source.trees;++t)e.draft_residual[t]=e.source.roots[t];
  e.draft->predicate=normalize(e,e.draft_region,e.draft_words,e.draft_positions,e.draft_residual);
  if(!account_normalization(r,*e.status,0,0))return;
  if(e.status->error){source_error(r,e.status->error);return;}
  project_context(e,e.draft_region,e.draft_residual);
  if(!v.options.maximum_decisions)r.stop_kind=u32(Stop::decision_budget);
}
__global__ void advance(View v,u32 decisions_per_path) {
  const u32 path=blockIdx.x*blockDim.x+threadIdx.x;if(path>=v.options.paths)return;
  auto e=private_view(v,path);auto& r=v.results[path];
  if(r.done||r.error||r.weight_overflow)return;
  for(u32 iteration=0;iteration<decisions_per_path;++iteration){
    if(r.decisions>=v.options.maximum_decisions){
      r.censored=1;r.stop_kind=u32(Stop::decision_budget);return;
    }
    ++r.decisions;
    if(!add(r,r.weighted_decisions,r.inverse_probability))return;
    auto proof=effort::interval_label(e,0,e.draft_region,
      at(v.refinement_stack,v.refinement_stack_capacity,path),
      v.refinement_stack_capacity,v.options.refinement_visit_budget);
    if(proof.attempted){
      ++r.refinement_attempts;r.refinement_visits+=proof.visited;
      if(!add_cost(r,r.weighted_refinement_visits,proof.visited))return;
    }
    if(proof.label>=0){
      ++r.class_prunes;
      if(!add(r,r.weighted_class_prunes,r.inverse_probability))return;
      if(proof.additional_prune&&e.draft->predicate!=none){
        ++r.refinement_additional_prunes;
        if(!add(r,r.weighted_refinement_additional_prunes,r.inverse_probability))return;
      }
      r.pruned_label=u32(proof.label);r.done=1;r.censored=0;
      r.stop_kind=u32(Stop::class_proof);return;
    }
    if(e.draft->predicate==none){
      ++r.native_terminals;
      if(!add(r,r.weighted_native_terminals,r.inverse_probability))return;
      r.done=1;r.censored=0;r.stop_kind=u32(Stop::native_obligation);return;
    }
    const u32 predicate=split::choose(e,0,split::View{},nullptr);
    if(predicate>=e.source.nodes||e.source.left[predicate]<0){source_error(r,26);return;}
    const u32 feature=u32(e.source.feature[predicate]),cut=__float_as_uint(e.source.cut[predicate]);
    const bool missing=bool(e.source.missing[predicate]);
    // Unforced source predicates must have two valid branches. Do not invent a
    // probability for an empty branch or silently resample an invalid context.
    if(!domain::region_split_feasible(e.domain,e.draft_witness,feature,cut,missing,false)||
       !domain::region_split_feasible(e.domain,e.draft_witness,feature,cut,missing,true)){
      source_error(r,26);return;
    }
    ++r.expanding_vertices;
    if(!add(r,r.weighted_expanding_vertices,r.inverse_probability))return;
    if(r.inverse_probability>DBL_MAX/2){overflow(r);return;}
    const bool right=(random_word(r.rng_state)>>63)!=0;
    r.inverse_probability=__dmul_rn(r.inverse_probability,2.);++r.depth;
    if(!domain::region_restrict(e.domain,e.draft_witness,feature,cut,missing,right)){
      source_error(r,26);return;
    }
    copy_region(e,e.draft_region,e.draft_witness);
    const u64 old_steps=e.status->normalization_steps,old_adds=e.status->prefix_additions;
    e.draft->predicate=normalize(e,e.draft_region,e.draft_words,e.draft_positions,e.draft_residual);
    // Edge cost is weighted at CHILD inclusion probability: this estimates
    // both-child normalization work without materializing the unchosen branch.
    if(!account_normalization(r,*e.status,old_steps,old_adds))return;
    if(e.status->error){source_error(r,e.status->error);return;}
    project_context(e,e.draft_region,e.draft_residual);
  }
  r.censored=1;r.stop_kind=u32(r.decisions>=v.options.maximum_decisions?
                                  Stop::decision_budget:Stop::partial);
}
inline EngineView checked_base(EngineView e,const Options& options) {
  require(options.paths,"work estimator needs paths");
  require(e.source.features&&e.source.classes&&e.source.features==e.domain.features,
          "work estimator source/domain shape");
  require(e.domain.feature_group&&e.domain.feature_numeric&&e.source.bias,
          "work estimator immutable domain/bias binding");
  require(e.support_words==(u64(e.source.features)+63)/64,
          "work estimator source support shape");
  if(e.source.trees){
    require(e.source.nodes&&e.source.feature&&e.source.left&&e.source.right&&e.source.roots&&
            e.source.channels&&e.source.cut&&e.source.value&&e.source.missing&&e.support,
            "work estimator validated source binding");
    if(e.qualified_gap)require(e.minimum&&e.maximum,"work estimator qualified extrema binding");
  }
  return e;
}
} // namespace detail

// Every mutable buffer is private and charged to the caller's Budget. Borrowed
// source metadata must outlive this owner. No authentic authority is inferred
// from observed classes; qualified_gap is solely the caller's existing binding.
struct Storage {
  EngineView base;
  Options options;
  u32 stack_capacity;
  Buffer<State> states;
  Buffer<Status> status;
  Buffer<Result> results;
  Buffer<u32> words,positions,blocked,lower,upper,missing,wl,wh,wm,refinement_stack;
  Buffer<std::int32_t> residual;
  Buffer<u64> allowed,wa,active;
  Buffer<float> range_lower,range_upper;
  bool started=false;
  Storage(Budget& budget,EngineView source,Options requested)
    :base(detail::checked_base(source,requested)),options(requested),
     stack_capacity(base.qualified_gap&&requested.refinement_visit_budget&&base.source.trees?
                    base.refinement_stack_capacity:0),
     states(budget,options.paths),status(budget,options.paths),results(budget,options.paths),
     words(budget,multiply(options.paths,base.source.classes)),
     positions(budget,words.size),blocked(budget,words.size),
     lower(budget,multiply(options.paths,base.domain.numeric_features)),
     upper(budget,lower.size),missing(budget,lower.size),wl(budget,lower.size),
     wh(budget,lower.size),wm(budget,lower.size),
     refinement_stack(budget,multiply(options.paths,stack_capacity)),
     residual(budget,multiply(options.paths,base.source.trees)),
     allowed(budget,multiply(options.paths,base.domain.mask_words)),wa(budget,allowed.size),
     active(budget,multiply(options.paths,base.support_words)),
     range_lower(budget,words.size),range_upper(budget,words.size) {
    options.refinement_visit_budget=base.qualified_gap?
      std::min(requested.refinement_visit_budget,base.refinement_maximum_visits):0;
    // Also initializes the complete host-transport object representation.
    states.zero();status.zero();results.zero();
  }
  View view(){
    return {base,options,states.data,status.data,results.data,words.data,positions.data,blocked.data,
      lower.data,upper.data,missing.data,wl.data,wh.data,wm.data,refinement_stack.data,residual.data,
      allowed.data,wa.data,active.data,range_lower.data,range_upper.data,stack_capacity};
  }
  void initialize(){
    require(!started,"work estimator initialized twice");
    detail::initialize<<<u32((u64(options.paths)+127)/128),128>>>(view());synchronize();started=true;
  }
  void advance(u32 decisions_per_path){
    require(started&&decisions_per_path,"work estimator chunk shape");
    detail::advance<<<u32((u64(options.paths)+127)/128),128>>>(view(),decisions_per_path);synchronize();
  }
  std::vector<Result> download()const{
    require(started,"work estimator not initialized");return results.download(options.paths);
  }
  // Root may aggregate these records on CUDA; no host numerical model work.
  const Result* device_results()const{return results.data;}
};
} // namespace class_conversion_adaptive::work
