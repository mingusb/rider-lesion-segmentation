#pragma once
#include "adaptive_engine.cuh"
#include "adaptive_native_batch.cuh"
#include "adaptive_effort.cuh"
#include "adaptive_cover_proof.cuh"
#include "adaptive_checkpoint.cuh"
#include "adaptive_proof_modules.cuh"
#include "adaptive_grid_coverage.cuh"
#include "adaptive_split_select.cuh"
#include <functional>
#include <chrono>
#include <cmath>
#include "../class_study_convert.hpp"

namespace class_conversion_adaptive::frontier {
struct Limits {
  u32 batch_capacity=256;
  u64 max_states=0x3fffffffu,max_nodes=0x3fffffffu,max_expansions=0;
  bool dynamic_batching=false;
  u32 maximum_batch_capacity=0; // zero derives the ceiling from resources
  u32 admission_threads=0; // zero tunes a device-legal, payload-bounded width
  u32 draft_threads=0; // zero tunes independent draft workers per legal block
  bool dynamic_cache=false,dynamic_refinement=false,dynamic_cover=false;
  u32 completed_cache_limit=UINT32_MAX,refinement_visit_budget=0,cover_visit_budget=0;
  split::View split_selection{};
  u32 oldest_ready_jobs=0; // zero preserves LIFO; UINT32_MAX selects all oldest
  std::string checkpoint_path,resume_from,proof_module_directory,proof_module_request;
  double checkpoint_interval_seconds=0;
  bool checkpoint_on_completion=true;
  u64 checkpoint_host_byte_budget=0;
  nlohmann::json checkpoint_identity;
  std::function<bool()> stop_requested;
  std::function<bool()> checkpoint_requested;
  std::function<class_study::CheckpointPublication(u64 gpu_captured_bytes)> checkpoint_publication;
};
struct Snapshot {
  Status status;
  coverage::Result grid_coverage;
  u64 grid_coverage_state_creations=0,grid_coverage_samples=0;
  u64 ready_states=0,completion_events=0;
  u64 oldest_selected_jobs=0; // selection events, including returned/retried jobs
  u32 native_pending=0,batch_jobs=0,maximum_batch_jobs=0;
  u64 state_growths=0,node_growths=0,native_margin_rows=0,native_public_rows=0;
  u64 state_growth_required_capacity=0,state_growth_preferred_capacity=0,state_growth_selected_capacity=0,state_growth_additional_bytes=0;
  u64 node_growth_required_capacity=0,node_growth_preferred_capacity=0,node_growth_selected_capacity=0,node_growth_additional_bytes=0;
  u64 growth_allocation_refusals=0,growth_transaction_rollbacks=0;
  u64 native_batches=0,draft_batches=0,resolved_edges=0,pending_cache_merges=0;
  u64 completed_cache_edges=0,completion_growth_retries=0;
  u64 completed_cached_states=0;
  u64 priority_split_changes=0;
  u32 active_batch_capacity=0,allocated_batch_capacity=0,autotune_best_capacity=0;
  u64 batch_adjustments=0,batch_growth_refusals=0,batch_memory_backoffs=0,autotune_samples=0,autotune_probes=0;
  u64 autotune_timed_jobs=0,autotune_excluded_growth_batches=0,batch_returned_jobs=0;
  u64 autotune_upward_probes=0,autotune_downward_probes=0;
  u32 admission_threads=0,maximum_admission_threads=0,best_admission_threads=0;
  u64 admission_thread_adjustments=0,admission_thread_probes=0;
  u32 draft_threads=0,maximum_draft_threads=0,best_draft_threads=0;
  u64 draft_thread_adjustments=0,draft_thread_probes=0;
  u32 completed_cache_limit=UINT32_MAX,refinement_visit_budget=0;
  u64 cache_limit_adjustments=0,cache_limit_probes=0,cache_policy_evictions=0;
  u64 refinement_adjustments=0,refinement_probes=0,refinement_attempts=0,refinement_visits=0;
  u64 refinement_tightened_roots=0,refinement_rejected_roots=0,refinement_fallback_frontiers=0,refinement_additional_prunes=0;
  u32 cover_visit_budget=0;
  u64 cover_adjustments=0,cover_probes=0,cover_attempts=0,cover_visits=0;
  u64 cover_feasible_cases=0,cover_certified_cases=0,cover_additional_prunes=0,cover_failures=0;
  u64 retune_backoff_batches=0,retune_cooldown_batches=0,retune_drift_resets=0;
  double autotune_last_work_equivalents_per_second=0,autotune_best_work_equivalents_per_second=0;
  double autotune_last_jobs_per_second=0,autotune_best_jobs_per_second=0;
  double autotune_last_batch_seconds=0;
  bool resumed=false,stopped=false;
  nlohmann::json checkpoint_status,proof_module_status;
};
using NativePredict=std::function<const float*(const float*,u64,bool)>;
using Progress=std::function<void(const Snapshot&)>;
// Initialization/source extrema stay in the sole converter. This replaces its
// serial advance loop. Hooks run only at synchronous host boundaries.
inline Snapshot run(EngineView&,std::unique_ptr<StateStorage>&,
                    std::unique_ptr<NodeStorage>&,Budget&,Limits,
                    const NativePredict&,bool direct_public_classes,
                    const Progress&,const std::function<void()>& check_native_gate={},
                    const grid::HostCatalog* coverage_catalog=nullptr,
                    Buffer<Status>* status_owner=nullptr);
} // namespace class_conversion_adaptive::frontier

namespace class_conversion_adaptive::frontier {
namespace detail {
constexpr u32 ready_once=1u,finish_once=2u,cached_once=16u;
constexpr u32 tuning_window_batches=4;
enum Stage:u32 {idle=0,draft_commit=1,native_query=2,native_commit=3};
struct Control {
  u64 ready_count=0,ready_head=0,finish_head=0,finish_tail=0;
  u64 cache_head=0,cache_tail=0;
  u64 resolved_edges=0,pending_merges=0,completed_edges=0;
  u64 priority_split_changes=0;
  u64 oldest_selected_jobs=0;
  u64 cache_policy_evictions=0,refinement_attempts=0,refinement_visits=0;
  u64 refinement_tightened_roots=0,refinement_rejected_roots=0,refinement_fallback_frontiers=0,refinement_additional_prunes=0;
  u64 cover_attempts=0,cover_visits=0,cover_feasible_cases=0,cover_certified_cases=0;
  u64 cover_additional_prunes=0,cover_failures=0;
  u32 stage=idle,jobs=0,cursor=0,expanding=0;
  u32 native_count=0,native_cursor=0,root_state=0;
};
struct QueueView {
  u32* ready;u32* finish;u32* flags;u32* pending;
  u32* wait_head;u32* edge_next;u32* completed;
  u32 capacity;
};
struct DraftView {
  State* states;Status* status;
  u32 *words,*positions,*blocked,*lower,*upper,*missing;
  u32 *witness_lower,*witness_upper,*witness_missing;
  std::int32_t* residual;
  u64 *allowed,*witness_allowed,*active_support;
  float *range_lower,*range_upper;
  u32* refinement_stack;
};
struct View {
  QueueView queue;DraftView draft;Control* control;
  u32 *jobs,*kinds,*prune_labels,*native_ids,*native_labels;
  u32 batch;
  u32 refinement_visits=0;
  split::View split_selection{};
  u32* selected_predicates=nullptr;
  double* split_scores=nullptr;
  u32 oldest_ready_jobs=0;
  u32 cover_visits=0;
  const proof_modules::ProposalV1* proof_proposals=nullptr;
};
struct QueueStorage {
  Buffer<u32> ready,finish,flags,pending,wait_head,edge_next,completed;
  u32 capacity;
  QueueStorage(Budget& budget,u32 n):ready(budget,n),finish(budget,n),flags(budget,n),
    pending(budget,n),wait_head(budget,n),edge_next(budget,multiply(n,2)),completed(budget,n),capacity(n) {
    flags.zero();pending.zero();wait_head.empty_table();edge_next.empty_table();
  }
  QueueView view(){return {ready.data,finish.data,flags.data,pending.data,wait_head.data,edge_next.data,completed.data,capacity};}
};
struct DraftStorage {
  Buffer<State> states;Buffer<Status> status;
  Buffer<u32> words,positions,blocked,lower,upper,missing,witness_lower,witness_upper,witness_missing;
  Buffer<std::int32_t> residual;
  Buffer<u64> allowed,witness_allowed,active_support;
  Buffer<float> range_lower,range_upper;
  Buffer<u32> refinement_stack;
  DraftStorage(Budget& b,u32 slots,u32 N,u32 K,u32 T,u32 W,u32 S,u32 R):states(b,slots),status(b,slots),
    words(b,multiply(slots,K)),positions(b,multiply(slots,K)),blocked(b,multiply(slots,K)),
    lower(b,multiply(slots,N)),upper(b,multiply(slots,N)),missing(b,multiply(slots,N)),
    witness_lower(b,multiply(slots,N)),witness_upper(b,multiply(slots,N)),witness_missing(b,multiply(slots,N)),
    residual(b,multiply(slots,T)),allowed(b,multiply(slots,W)),witness_allowed(b,multiply(slots,W)),
    active_support(b,multiply(slots,S)),range_lower(b,multiply(slots,K)),range_upper(b,multiply(slots,K)),
    refinement_stack(b,multiply(slots,R)) {}
  DraftView view(){return {states.data,status.data,words.data,positions.data,blocked.data,lower.data,upper.data,missing.data,
    witness_lower.data,witness_upper.data,witness_missing.data,residual.data,allowed.data,witness_allowed.data,
    active_support.data,range_lower.data,range_upper.data,refinement_stack.data};}
};
// One owner for scratch whose contents are disposable ONLY at an idle boundary.
// Queue/control/state ownership is separate and never changes during resizing.
struct BatchStorage {
  DraftStorage draft;
  Buffer<u32> jobs,kinds,prune_labels,native_ids,native_labels;
  Buffer<u32> selected_predicates;
  Buffer<proof_modules::ProposalV1> proof_proposals;
  Buffer<double> split_scores;
  Buffer<float> native_rows;
  u32 capacity;
  BatchStorage(Budget& b,u32 n,const EngineView& e,split::View selection={})
      :draft(b,u32(multiply(n,2)),e.domain.numeric_features,e.source.classes,
             e.source.trees,e.domain.mask_words,e.support_words,e.refinement_stack_capacity),
       jobs(b,n),kinds(b,n),prune_labels(b,n),native_ids(b,n),native_labels(b,n),selected_predicates(b,n),proof_proposals(b,n),
       split_scores(b,selection.policy==split::Policy::aggregate_residual?multiply(n,selection.groups):0),
       native_rows(b,multiply(n,e.source.features)),capacity(n) {}
};
inline u64 plus(u64 a,u64 b) {
  require(a<=UINT64_MAX-b,"frontier byte estimate overflow");return a+b;
}
inline u64 scratch_bytes(const EngineView& e,u32 n,split::View selection={}) {
  u64 one=plus(sizeof(State),sizeof(Status));
  one=plus(one,multiply(e.source.classes,20));
  one=plus(one,multiply(e.domain.numeric_features,24));
  one=plus(one,multiply(e.source.trees,4));
  one=plus(one,multiply(e.domain.mask_words,16));
  one=plus(one,multiply(e.support_words,8));
  one=plus(one,multiply(e.refinement_stack_capacity,4));
  u64 scoring=selection.policy==split::Policy::aggregate_residual?multiply(selection.groups,8):0;
  return multiply(n,plus(multiply(one,2),plus(plus(24+sizeof(proof_modules::ProposalV1),multiply(e.source.features,4)),scoring)));
}
inline u64 state_owner_bytes(const EngineView& e,u32 n) {
  return growth::state_bytes(growth_shape(e),n);
}
// Current eligible completed entries can all be reclaimed before admission.
// Reserve the next state/queue owners when the worst two-child batch needs them;
// this accounts for coexistence with current owners, not just final arena bytes.
inline bool batch_resources(const EngineView& e,const Status& h,const Control& c,
    const Budget& b,u32 capacity,u64 max_states,u64 new_scratch_bytes) {
  require(h.free_count<=h.states,"frontier free-list resource extent");
  u64 cached=c.cache_tail-c.cache_head,live=h.states-h.free_count;
  require(cached<=live,"frontier completed-cache resource extent");
  u64 needed=plus(live-cached,multiply(capacity,2));
  if(needed>max_states)return false;
  if(b.used>b.limit||new_scratch_bytes>b.limit-b.used)return false;
  std::size_t free_bytes=0,total_bytes=0;
  auto error=cudaMemGetInfo(&free_bytes,&total_bytes);
  check(error,"frontier memory resource query");
  // Native outputs are borrowed/private to the callback, outside Budget. Reserve
  // their two dense K-word vectors as a declared minimum estimate, not a bound
  // on all native-library workspace or concurrent external GPU allocations.
  u64 native_output= multiply(multiply(capacity,e.source.classes),8);
  if(native_output>free_bytes||new_scratch_bytes>u64(free_bytes)-native_output)return false;
  u64 available=std::min(b.limit-b.used-new_scratch_bytes,u64(free_bytes)-native_output-new_scratch_bytes);
  if(needed<=e.arena.state_capacity)return true;
  auto plan=growth::choose(e.arena.state_capacity,needed,max_states,available,[&](u32 n){
    return plus(plus(state_owner_bytes(e,n),growth::queue_bytes(n)),sizeof(Control));
  });
  return plan.affordable;
}
// Measured scheduling heuristic. Launch axes compare jobs/time; cache and
// refinement axes also count directly avoided work (two child jobs per extra
// nonterminal prune and one per completed-cache edge). This is a local proxy,
// not a claim of globally optimal scheduling. No timing changes class semantics.
struct Tuner {
  u32 active=0,best=0,maximum=0,window_batches=0;
  u64 window_jobs=0,window_avoided=0,cooldown=0,backoff=0;
  double window_seconds=0,best_rate=0,best_work_rate=0;
  double window_min_cost=INFINITY,window_max_cost=0,previous_min_cost=0,previous_max_cost=INFINITY;
  bool probing=false;
  int probe_direction=1;
  u32 admission=1,best_admission=1,maximum_admission=1;
  int admission_direction=1;
  u32 draft=1,best_draft=1,maximum_draft=1;
  int draft_direction=1;
  u32 cache=UINT32_MAX,best_cache=UINT32_MAX,refinement=0,best_refinement=0,cover=0,best_cover=0;
  int cache_direction=-1,refinement_direction=1,cover_direction=1;
  u32 probe_axis=0,next_axis=0; // batch, copy, draft, cache retention, refinement, cover proof
  void reset_window(){window_batches=0;window_jobs=0;window_avoided=0;window_seconds=0;window_min_cost=INFINITY;window_max_cost=0;}
  void reject_probe(){backoff=backoff?std::min<u64>(UINT64_MAX/2,backoff)*2:tuning_window_batches;cooldown=backoff;}
  bool observe_drift(double lo,double hi){
    if(probing||!(lo>previous_max_cost||hi<previous_min_cost))return false;
    backoff=cooldown=0;return true;
  }
};
inline u32 neighboring_work(u32 active,u32 maximum,int& direction) {
  if(!maximum)return 0;
  if(active>=maximum)direction=-1;else if(!active)direction=1;
  return direction>0?u32(std::min<u64>(maximum,std::max<u64>(1,u64(active)*2))):active/2;
}
inline u32 neighboring_extent(u32 active,u32 maximum,int& direction) {
  if(maximum==1)return 1;
  if(active==maximum)direction=-1;
  else if(active==1)direction=1;
  return direction>0?u32(std::min<u64>(maximum,u64(active)*2)):std::max(1u,active/2);
}
inline u32 neighboring_capacity(Tuner& tuner) {
  return neighboring_extent(tuner.active,tuner.maximum,tuner.probe_direction);
}
inline u32 power_two_extent(u32 extent) {
  u32 width=1;while(width<extent&&width<=UINT32_MAX/2)width*=2;return width;
}

template<class T>__device__ T* offset(T* pointer,u64 count){return pointer?pointer+count:nullptr;}
__device__ EngineView private_view(EngineView e,View f,u32 slot) {
  auto d=f.draft;const auto N=e.domain.numeric_features,K=e.source.classes,T=e.source.trees,W=e.domain.mask_words;
  e.draft=d.states+slot;e.status=d.status+slot;
  e.draft_words=d.words+u64(slot)*K;e.draft_positions=d.positions+u64(slot)*K;
  e.blocked=d.blocked+u64(slot)*K;e.draft_residual=offset(d.residual,u64(slot)*T);
  e.draft_region={offset(d.lower,u64(slot)*N),offset(d.upper,u64(slot)*N),offset(d.missing,u64(slot)*N),offset(d.allowed,u64(slot)*W)};
  e.draft_witness={offset(d.witness_lower,u64(slot)*N),offset(d.witness_upper,u64(slot)*N),offset(d.witness_missing,u64(slot)*N),offset(d.witness_allowed,u64(slot)*W)};
  e.active_support=d.active_support+u64(slot)*e.support_words;
  e.range_lower=d.range_lower+u64(slot)*K;e.range_upper=d.range_upper+u64(slot)*K;return e;
}
__device__ void fail(EngineView e,u32 code){if(!e.status->error)e.status->error=code;}
__device__ void count_add(u64* target,u64 value){atomicAdd(reinterpret_cast<unsigned long long*>(target),static_cast<unsigned long long>(value));}
__device__ bool enqueue_ready(EngineView e,View f,u32 id) {
  if(id>=e.status->states||id>=f.queue.capacity){fail(e,51);return false;}
  if(f.queue.flags[id]&ready_once)return true;
  if(f.control->ready_count>=f.queue.capacity){fail(e,52);return false;}
  f.queue.ready[(f.control->ready_head+f.control->ready_count++)%f.queue.capacity]=id;
  f.queue.flags[id]|=ready_once;return true;
}
__device__ bool enqueue_finish(EngineView e,View f,u32 id) {
  if(id>=e.status->states||id>=f.queue.capacity){fail(e,51);return false;}
  if(f.queue.flags[id]&finish_once)return true;
  if(f.control->finish_tail-f.control->finish_head>=f.queue.capacity){fail(e,53);return false;}
  f.queue.finish[f.control->finish_tail++%f.queue.capacity]=id;f.queue.flags[id]|=finish_once;return true;
}
__device__ void resolve_edge(EngineView e,View f,u32 parent,u32 side,u32 node) {
  const u32 bit=4u<<side;
  if(parent>=e.status->states||node>=e.status->nodes||e.arena.states[parent].phase!=2||
      (f.queue.flags[parent]&bit)||!f.queue.pending[parent]){fail(e,54);return;}
  if(side)e.arena.states[parent].right=node;else e.arena.states[parent].left=node;
  f.queue.flags[parent]|=bit;--f.queue.pending[parent];++f.control->resolved_edges;
  if(!f.queue.pending[parent])enqueue_finish(e,f,parent);
}
__device__ void attach_edge(EngineView e,View f,u32 parent,u32 side,u32 child,bool fresh) {
  if(child>=e.status->states||child==parent){fail(e,55);return;}
  if(side)e.arena.states[parent].right=child;else e.arena.states[parent].left=child;
  if(e.arena.states[child].phase==3){++f.control->completed_edges;resolve_edge(e,f,parent,side,e.arena.states[child].node);}
  else {u32 edge=2*parent+side;f.queue.edge_next[edge]=f.queue.wait_head[child];f.queue.wait_head[child]=edge;
    if(!fresh)++f.control->pending_merges;}
}
__global__ void bootstrap(EngineView e,View f) {
  if(blockIdx.x||threadIdx.x)return;
  if(e.status->states!=1||e.arena.states[0].phase!=0){fail(e,56);return;}
  *f.control=Control{};f.control->root_state=0;e.status->depth=0;enqueue_ready(e,f,0);
}
__global__ void select_ready(EngineView e,View f) {
  if(blockIdx.x||threadIdx.x)return;
  e.status->request=0;
  if(f.control->stage!=idle){fail(e,57);return;}
  auto count=u32(f.control->ready_count<f.batch?f.control->ready_count:f.batch);
  f.control->jobs=count;f.control->cursor=0;f.control->expanding=0;
  f.control->native_count=0;f.control->native_cursor=0;
  const auto oldest=f.oldest_ready_jobs<count?f.oldest_ready_jobs:count;
  for(u32 job=0;job<count;++job){
    const bool front=job<oldest;
    u32 id;
    if(front){
      id=f.queue.ready[f.control->ready_head];
      f.control->ready_head=(f.control->ready_head+1)%f.queue.capacity;
      --f.control->ready_count;
    }else id=f.queue.ready[(f.control->ready_head+--f.control->ready_count)%f.queue.capacity];
    if(id>=e.status->states||e.arena.states[id].phase!=0){fail(e,58);return;}
    f.jobs[job]=id;if(front)++f.control->oldest_selected_jobs;
  }
  f.control->stage=draft_commit;
}
// Every worker reads immutable admitted states and writes only its private
// two-child scratch/status. Global error and expanding count use atomics.
__global__ void initialize_proposals(EngineView e,View f,proof_modules::ProposalV1* proposals) {
  const u32 job=blockIdx.x*blockDim.x+threadIdx.x;if(job>=f.control->jobs)return;
  f.selected_predicates[job]=e.arena.states[f.jobs[job]].predicate;
  proposals[job]={none,UINT32_MAX}; // a partial module write is never a proposal
}
__global__ void prepare_drafts(EngineView e,View f) {
  u32 job=blockIdx.x*blockDim.x+threadIdx.x;if(job>=f.control->jobs)return;
  u32 id=f.jobs[job];const auto parent=e.arena.states[id];
  for(u32 side=0;side<2;++side)*private_view(e,f,2*job+side).status=Status{};
  // The same authorized interval theorem also covers terminal point scores.
  // Ties, close gaps, out-of-range scores and disabled gates still use the
  // native public-output callback below.
  auto prune=private_view(e,f,2*job);
  auto proof=effort::interval_label(prune,id,region(e,id),
    offset(f.draft.refinement_stack,u64(2)*job*e.refinement_stack_capacity),
    e.refinement_stack_capacity,f.refinement_visits);
  int label=proof.label;
  if(proof.attempted){
    count_add(&f.control->refinement_attempts,1);
    count_add(&f.control->refinement_visits,proof.visited);
    count_add(&f.control->refinement_tightened_roots,proof.tightened_roots);
    count_add(&f.control->refinement_rejected_roots,proof.rejected_refinements);
    count_add(&f.control->refinement_fallback_frontiers,proof.fallback_frontiers);
    if(proof.additional_prune&&parent.predicate!=none)count_add(&f.control->refinement_additional_prunes,1);
  }
  if(label>=0){f.kinds[job]=1;f.prune_labels[job]=u32(label);return;}
  if(parent.predicate==none){f.kinds[job]=0;return;}
  u32 selected=split::choose(e,id,f.split_selection,
      offset(f.split_scores,u64(job)*f.split_selection.groups));
  if(f.selected_predicates)f.selected_predicates[job]=selected;
  else if(selected!=parent.predicate){atomicCAS(&e.status->error,0u,70u);return;}
  // A proof-only split covers the parent without publishing child states.
  // Reuse disposable private draft scratch; the original prefix, residual
  // roots, full witness and arena remain untouched. Both cases must certify
  // the same native-qualified class, otherwise normal construction follows.
  if(f.cover_visits){
    const auto proof_predicate=f.proof_proposals?
      proof_modules::checked_predicate(e,id,f.proof_proposals[job],selected):selected;
    const auto covered=cover::interval_label(prune,id,region(e,id),proof_predicate,
      prune.draft_region,offset(f.draft.refinement_stack,u64(2)*job*e.refinement_stack_capacity),
      e.refinement_stack_capacity,f.cover_visits);
    if(covered.attempted){
      count_add(&f.control->cover_attempts,1);
      count_add(&f.control->cover_visits,covered.visited);
      count_add(&f.control->cover_feasible_cases,covered.feasible_cases);
      count_add(&f.control->cover_certified_cases,covered.certified_cases);
      if(covered.label>=0){
        count_add(&f.control->cover_additional_prunes,1);
        f.kinds[job]=1;f.prune_labels[job]=u32(covered.label);return;
      }
      count_add(&f.control->cover_failures,1);
    }
  }
  f.kinds[job]=2;atomicAdd(&f.control->expanding,1u);
  for(u32 side=0;side<2;++side){auto child=private_view(e,f,2*job+side);u32 predicate=selected;
    copy_region(child,child.draft_witness,witness_region(e,id));
    if(!domain::region_restrict(e.domain,child.draft_witness,u32(e.source.feature[predicate]),
        __float_as_uint(e.source.cut[predicate]),bool(e.source.missing[predicate]),bool(side))){atomicCAS(&e.status->error,0u,26u);return;}
    copy_region(child,child.draft_region,child.draft_witness);
    for(u32 c=0;c<e.source.classes;++c){child.draft_words[c]=e.arena.words[u64(id)*e.source.classes+c];child.draft_positions[c]=e.arena.positions[u64(id)*e.source.classes+c];}
    for(u32 t=0;t<e.source.trees;++t)child.draft_residual[t]=e.arena.residual[u64(id)*e.source.trees+t];
    State state{};state.predicate=normalize(child,child.draft_region,child.draft_words,child.draft_positions,child.draft_residual);
    if(child.status->error){atomicCAS(&e.status->error,0u,child.status->error);return;}
    project_context(child,child.draft_region,child.draft_residual);
    state.hash=state_hash(child,child.draft_region,child.draft_words,child.draft_positions,child.draft_residual);
    *child.draft=state;
  }
}
__device__ void add_draft_counts(EngineView e,const Status& local) {
  e.status->normalization_steps+=local.normalization_steps;e.status->prefix_additions+=local.prefix_additions;
}
// One block admits jobs in the original order. Thread zero owns lookup, counters,
// queues and graph publication; the selected block width copies fresh payload bits.
// The host reserves 2*expanding jobs, so growth cannot half-admit a parent.
__global__ void commit_drafts(EngineView e,View f,u64 maximum_expansions) {
  if(gridDim.x!=1||gridDim.y!=1||gridDim.z!=1||!blockDim.x||(blockDim.x&(blockDim.x-1))||blockDim.y!=1||blockDim.z!=1){
    if(!blockIdx.x&&!blockIdx.y&&!blockIdx.z&&!threadIdx.x&&!threadIdx.y&&!threadIdx.z)fail(e,77);return;
  }
  const u32 lane=threadIdx.x;
  if(!lane)e.status->request=0;
  __syncthreads();
  for(;;){
    const u32 job=f.control->cursor;
    if(job>=f.control->jobs)break;
    const u32 id=f.jobs[job];auto& parent=e.arena.states[id];
    if(!lane&&parent.phase!=0)fail(e,59);
    __syncthreads();
    if(e.status->error)return;
    const u32 kind=f.kinds[job];
    if(kind==0){
      if(!lane){const auto*words=e.arena.words+u64(id)*e.source.classes;u32 cached=terminal_lookup(e,words,false,none);
        if(!e.status->error){
          if(cached!=none){
            if(e.arena.states[cached].phase!=3)fail(e,60);
            else {parent.node=e.arena.states[cached].node;parent.phase=3;++e.status->terminal_hits;enqueue_finish(e,f,id);}
          }else if(f.control->native_count>=f.batch)fail(e,61);
          else {f.native_ids[f.control->native_count++]=id;parent.phase=1;}
        }
      }
    }else if(kind==1){
      if(!lane){u32 node=intern_node(e,{-1,f.prune_labels[job],0,0});
        if(node!=none){parent.node=node;parent.phase=3;++e.status->class_pruned_states;
          if(parent.predicate==none)++e.status->terminal_gap_pruned_states;
          enqueue_finish(e,f,id);}
      }
    }else {
      if(!lane){
        if(maximum_expansions&&(e.status->expansions>=maximum_expansions||maximum_expansions-e.status->expansions<2))fail(e,25);
        else if(u64(e.arena.state_capacity)-e.status->states+e.status->free_count<2)e.status->request=2;
      }
      __syncthreads();
      if(e.status->error||e.status->request)return;
      u32 children[2];bool fresh[2]{};
      for(u32 side=0;side<2;++side){
        auto child=private_view(e,f,2*job+side);const auto* prepared_status=child.status;child.status=e.status;
        u64 before=0;if(!lane)before=e.status->state_creations;
        children[side]=intern_state_block(child,*child.draft,child.draft_region,child.draft_words,child.draft_positions,child.draft_residual,true);
        if(children[side]==none)return;
        if(!lane){fresh[side]=e.status->state_creations!=before;add_draft_counts(e,*prepared_status);}
        __syncthreads();
        if(e.status->error)return;
      }
      if(!lane){
        u32 selected=f.selected_predicates?f.selected_predicates[job]:parent.predicate;
        if(selected!=parent.predicate)++f.control->priority_split_changes;
        parent.predicate=selected;parent.phase=2;f.queue.pending[id]=2;
        attach_edge(e,f,id,0,children[0],fresh[0]);attach_edge(e,f,id,1,children[1],fresh[1]);
        // Admit left before right; push right first to retain left-first LIFO.
        if(fresh[1])enqueue_ready(e,f,children[1]);if(fresh[0])enqueue_ready(e,f,children[0]);
        e.status->expansions+=2;--f.control->expanding;
      }
    }
    __syncthreads();
    if(e.status->error||e.status->request)return;
    if(!lane)++f.control->cursor;
    __syncthreads();
  }
  if(!lane){f.control->stage=f.control->native_count?native_query:idle;f.control->jobs=0;f.control->cursor=0;f.control->expanding=0;}
}
// The head is not consumed until intern_node succeeds. Resolved labels/child
// edges remain owned across a node-growth retry, without another native call.
__global__ void drain_completions(EngineView e,View f) {
  if(blockIdx.x||threadIdx.x)return;e.status->request=0;
  while(f.control->finish_head<f.control->finish_tail&&!e.status->error){u32 id=f.queue.finish[f.control->finish_head%f.queue.capacity];auto& state=e.arena.states[id];
    if(state.phase!=3){if(state.phase!=2||f.queue.pending[id]||(f.queue.flags[id]&12u)!=12u||state.left==none||state.right==none||
        state.left>=e.status->nodes||state.right>=e.status->nodes){fail(e,62);return;}
      u32 predicate=state.predicate;std::int64_t feature=e.source.feature[predicate];if(e.source.missing[predicate])feature=-feature-2;
      u32 node=intern_node(e,{std::int32_t(feature),__float_as_uint(e.source.cut[predicate]),state.left,state.right});
      if(node==none)return;state.node=node;state.phase=3;
    }
    u32 edge=f.queue.wait_head[id];while(edge!=none){u32 next=f.queue.edge_next[edge];resolve_edge(e,f,edge/2,edge%2,state.node);if(e.status->error)return;edge=next;}
    f.queue.wait_head[id]=none;++f.control->finish_head;
    if(id==f.control->root_state){e.status->root=state.node;e.status->complete=1;}
    else {
      // No ready/native/unconsumed-completion reference remains. Incoming
      // parents now own stable node IDs, not this reusable state slot.
      if(f.queue.flags[id]&cached_once||f.control->cache_tail-f.control->cache_head>=f.queue.capacity){fail(e,67);return;}
      f.queue.completed[f.control->cache_tail++%f.queue.capacity]=id;
      f.queue.flags[id]|=cached_once;
    }
  }
}
// Completed cache entries remain exact, but are expendable on admission
// pressure. A slot is recycled only after its sole FIFO entry is consumed.
__global__ void reclaim_completed(EngineView e,View f,u64 needed_slots,u64 maximum_cached=UINT64_MAX) {
  if(blockIdx.x||threadIdx.x)return;
  while((u64(e.arena.state_capacity)-e.status->states+e.status->free_count<needed_slots||
         f.control->cache_tail-f.control->cache_head>maximum_cached)&&
        f.control->cache_head<f.control->cache_tail&&!e.status->error) {
    u32 id=f.queue.completed[f.control->cache_head%f.queue.capacity];
    if(id>=e.status->states||id==f.control->root_state||e.arena.states[id].phase!=3||
       !(f.queue.flags[id]&cached_once)||f.queue.wait_head[id]!=none||f.queue.pending[id]){fail(e,68);return;}
    if(!evict_completed_state(e,id))return;
    f.queue.flags[id]=0;f.queue.pending[id]=0;f.queue.wait_head[id]=none;
    f.queue.edge_next[u64(id)*2]=none;f.queue.edge_next[u64(id)*2+1]=none;
    if(f.control->cache_tail-f.control->cache_head>maximum_cached)++f.control->cache_policy_evictions;
    ++f.control->cache_head;
  }
}
// Return only the uncommitted selected tail. The reverse push restores its
// original LIFO pop order; committed native IDs and every completion cursor stay
// unchanged. Disposable private drafts for returned states are not published.
// An oldest-selected job returned this way becomes newest-end work; its former
// FIFO age is not restored. Selection accounting deliberately includes retries.
__global__ void trim_selected_tail(EngineView e,View f,u32 keep) {
  if(blockIdx.x||threadIdx.x)return;
  auto& c=*f.control;
  if(c.stage!=draft_commit||keep<c.cursor||keep>=c.jobs){fail(e,81);return;}
  for(u32 i=c.jobs;i>keep;--i) {
    u32 id=f.jobs[i-1];
    if(id>=e.status->states||e.arena.states[id].phase!=0||c.ready_count>=f.queue.capacity){fail(e,82);return;}
    f.queue.ready[(c.ready_head+c.ready_count++)%f.queue.capacity]=id;
  }
  c.jobs=keep;c.expanding=0;
  for(u32 i=c.cursor;i<keep;++i)c.expanding+=f.kinds[i]==2;
  e.status->request=0;
}

__global__ void labels_ready(EngineView e,View f) {
  if(blockIdx.x||threadIdx.x)return;
  if(e.status->error||f.control->stage!=native_query){fail(e,63);return;}f.control->stage=native_commit;f.control->native_cursor=0;
}
__global__ void commit_native(EngineView e,View f) {
  if(blockIdx.x||threadIdx.x)return;e.status->request=0;
  for(;f.control->native_cursor<f.control->native_count;++f.control->native_cursor){u32 row=f.control->native_cursor,id=f.native_ids[row],label=f.native_labels[row];
    if(label>=output_class_count(e)||e.arena.states[id].phase!=1){fail(e,64);return;}
    u32 node=intern_node(e,{-1,label,0,0});if(node==none)return;
    auto&state=e.arena.states[id];state.node=node;state.phase=3;
    u32 previous=terminal_lookup(e,e.arena.words+u64(id)*e.source.classes,true,id);
    if(e.status->error)return;if(previous!=none&&e.arena.states[previous].node!=node){fail(e,65);return;}
    ++e.status->native_terminals;enqueue_finish(e,f,id);if(e.status->error)return;
  }
  f.control->native_count=0;f.control->native_cursor=0;f.control->stage=idle;
}
__global__ void copy_queue_on_growth(QueueView old_queue,QueueView new_queue,Control* control) {
  for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<control->ready_count;i+=u64(blockDim.x)*gridDim.x)
    new_queue.ready[i]=old_queue.ready[(control->ready_head+i)%old_queue.capacity];
  u64 pending=control->finish_tail-control->finish_head;
  for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<pending;i+=u64(blockDim.x)*gridDim.x)new_queue.finish[i]=old_queue.finish[(control->finish_head+i)%old_queue.capacity];
  u64 cached=control->cache_tail-control->cache_head;
  for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<cached;i+=u64(blockDim.x)*gridDim.x)new_queue.completed[i]=old_queue.completed[(control->cache_head+i)%old_queue.capacity];
}
__global__ void reset_completion_indices(Control* control) {
  if(blockIdx.x||threadIdx.x)return;control->ready_head=0;
  control->finish_tail-=control->finish_head;control->finish_head=0;
  control->cache_tail-=control->cache_head;control->cache_head=0;
}
template<class T>inline T read_one(const T* device) {
  T value{};check(cudaMemcpy(&value,device,sizeof(T),cudaMemcpyDeviceToHost),"frontier control download");return value;
}
inline growth::Outcome grow_state_queue_transaction(EngineView& e,
    std::unique_ptr<StateStorage>& states,NodeStorage& nodes,
    std::unique_ptr<QueueStorage>& queue,Buffer<Control>& control,Budget& budget,
    const Status& h,u64 minimum,u64 ceiling,const GrowthObserver& observer={},
    const GrowthAllocationHook& hook={},u64 device_reserve=0) {
  require(states&&queue&&states->capacity==queue->capacity,"adaptive coupled growth owner extents");
  growth::Outcome out;u32 retry=0;
  for(;;) {
    plan_growth(out,budget,states->capacity,minimum,ceiling,[&](u32 n){
      return plus(plus(state_owner_bytes(e,n),growth::queue_bytes(n)),sizeof(Control));
    },retry,device_reserve,observer);
    std::unique_ptr<StateStorage> fresh_states;
    std::unique_ptr<QueueStorage> fresh_queue;
    Buffer<Control> fresh_control;
    try {
      if(hook)hook("states");
      fresh_states=std::make_unique<StateStorage>(budget,out.plan.selected,e.domain.numeric_features,
        e.source.classes,e.source.trees,e.domain.mask_words);
      if(hook)hook("queue");
      fresh_queue=std::make_unique<QueueStorage>(budget,out.plan.selected);
      if(hook)hook("control");
      fresh_control=Buffer<Control>(budget,1);
    }catch(const AllocationRefusal&) {
      // Partially constructed Buffer members already unwind. Release complete
      // staged owners before publishing/refitting; live bindings never changed.
      fresh_states.reset();fresh_queue.reset();fresh_control.reset();
      ++out.allocation_refusals;++out.transaction_rollbacks;
      if(observer)observer(out);
      retry=growth::retry_ceiling(out.plan);
      if(!retry)throw;
      continue;
    }
    copy(fresh_control,control,1);
    copy(fresh_queue->flags,queue->flags,h.states);
    copy(fresh_queue->pending,queue->pending,h.states);
    copy(fresh_queue->wait_head,queue->wait_head,h.states);
    copy(fresh_queue->edge_next,queue->edge_next,multiply(h.states,2));
    copy_queue_on_growth<<<128,256>>>(queue->view(),fresh_queue->view(),control.data);synchronize();
    reset_completion_indices<<<1,1>>>(fresh_control.data);synchronize();
    EngineView staged=stage_state_growth(e,*states,*fresh_states,nodes,h);
    // All device preparation succeeded. No launch observes the short host-only
    // swap sequence; the next view has matching state/queue/control owners.
    states.swap(fresh_states);queue.swap(fresh_queue);control.swap(fresh_control);e=staged;
    fresh_states.reset();fresh_queue.reset();fresh_control.reset();out.complete=true;
    if(observer)observer(out);return out;
  }
}
} // namespace detail
} // namespace class_conversion_adaptive::frontier

namespace class_conversion_adaptive::frontier {
namespace detail {
__global__ void request_boundary(EngineView e,u32 request,u32 error=0) {
  if(blockIdx.x||threadIdx.x)return;e.status->request=request;if(error&&!e.status->error)e.status->error=error;
}
struct AdmissionShape {u32 initial=1,maximum=1;};
inline AdmissionShape admission_shape(const EngineView& e,u32 fixed) {
  require(!fixed||!(fixed&(fixed-1)),"admission threads must be a power of two");
  cudaFuncAttributes attributes{};
  check(cudaFuncGetAttributes(&attributes,commit_drafts),"admission kernel attributes");
  int device=0,warp=0;check(cudaGetDevice(&device),"admission device");
  check(cudaDeviceGetAttribute(&warp,cudaDevAttrWarpSize,device),"admission warp size");
  require(attributes.maxThreadsPerBlock>0&&warp>0,"admission device launch limits");
  const u32 hardware_limit=u32(attributes.maxThreadsPerBlock);
  require(!fixed||fixed<=hardware_limit,"fixed admission threads exceed kernel limit");
  const u32 extent=std::max({1u,e.domain.numeric_features,e.domain.mask_words,e.source.classes,e.source.trees});
  u32 upper=1;
  while(upper<extent&&upper<=hardware_limit/2)upper*=2;
  if(fixed)upper=fixed;
  u32 legal=0;
  for(u32 width=1;width<=upper;){
    int blocks=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,commit_drafts,int(width),0),
                       "admission kernel resource capacity");
    if(blocks>0)legal=width;
    if(width>upper/2)break;
    width*=2;
  }
  require(legal&&(!fixed||legal==fixed),"admission width exceeds available kernel resources");
  // No lane above the largest slab extent can copy a payload element. The
  // remaining legal powers of two are measured; occupancy is a launch guard,
  // not a claim that its largest block maximizes throughput.
  return {fixed?fixed:std::min(u32(warp),legal),fixed?fixed:legal};
}
struct DraftShape {u32 initial=1,maximum=1,multiprocessors=1;};
// Cache launch resources once. The initial auto grouping exposes independent
// jobs across available SMs, within a warp and the positive-occupancy ceiling.
// Later timing chooses grouping; neither occupancy nor this seed is an optimum.
inline DraftShape draft_shape(u32 fixed,u32 available_jobs) {
  require(!fixed||!(fixed&(fixed-1)),"draft threads must be a power of two");
  cudaFuncAttributes attributes{};
  check(cudaFuncGetAttributes(&attributes,prepare_drafts),"draft kernel attributes");
  int device=0,warp=0,sms=0;check(cudaGetDevice(&device),"draft device");
  check(cudaDeviceGetAttribute(&warp,cudaDevAttrWarpSize,device),"draft warp size");
  check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"draft SM count");
  require(attributes.maxThreadsPerBlock>0&&warp>0&&sms>0,"draft device launch limits");
  u32 upper=fixed?fixed:u32(attributes.maxThreadsPerBlock);
  require(!fixed||fixed<=u32(attributes.maxThreadsPerBlock),"fixed draft threads exceed kernel limit");
  u32 legal=0;
  for(u32 width=1;width<=upper;){
    int blocks=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,prepare_drafts,int(width),0),
                       "draft kernel resource capacity");
    if(blocks>0)legal=width;
    if(width>upper/2)break;width*=2;
  }
  require(legal&&(!fixed||legal==fixed),"draft width exceeds available kernel resources");
  u32 per_sm=u32((u64(std::max(available_jobs,1u))+u32(sms)-1)/u32(sms));
  u32 initial=std::min({power_two_extent(per_sm),u32(warp),legal});
  return {fixed?fixed:initial,fixed?fixed:legal,u32(sms)};
}
}
} // namespace class_conversion_adaptive::frontier
#include "adaptive_checkpoint_frontier.cuh"
namespace class_conversion_adaptive::frontier {
inline Snapshot run(EngineView& e,std::unique_ptr<StateStorage>& states,
    std::unique_ptr<NodeStorage>& nodes,Budget& budget,Limits limits,
    const NativePredict& native,bool direct,const Progress& progress,
    const std::function<void()>& check_gate,const grid::HostCatalog* coverage_catalog,
    Buffer<Status>* status_owner) {
  require(limits.batch_capacity>0&&limits.batch_capacity<=65536&&native,
          "frontier batch declaration");
  require(!limits.maximum_batch_capacity||
          (limits.maximum_batch_capacity>=limits.batch_capacity&&limits.maximum_batch_capacity<=65536),
          "frontier maximum batch declaration");
  require(states&&nodes&&states->capacity<=limits.max_states&&nodes->capacity<=limits.max_nodes,
          "frontier initial capacities");
  require(std::isfinite(limits.checkpoint_interval_seconds)&&limits.checkpoint_interval_seconds>=0,
          "checkpoint interval must be finite and nonnegative");
  std::unique_ptr<proof_modules::Registry> modules;
  if(!limits.proof_module_directory.empty()||!limits.proof_module_request.empty()) {
    int device=0;check(cudaGetDevice(&device),"proof module device");
    modules=std::make_unique<proof_modules::Registry>(proof_modules::Session{
      0,u32(device),e.source.features,e.source.classes,e.source.nodes,e.source.trees});
    if(!limits.proof_module_directory.empty())modules->start_watching(limits.proof_module_directory);
    if(!limits.proof_module_request.empty())modules->request(limits.proof_module_request);
  }
  auto queue=std::make_unique<detail::QueueStorage>(budget,states->capacity);
  Buffer<detail::Control> control(budget,1);control.zero();
  bool resumed=false;
  if(!limits.resume_from.empty()) {
    require(status_owner&&status_owner->data==e.status,"resume requires current status owner");
    if(check_gate)check_gate();
    checkpoint_io::restore(limits.resume_from,limits.checkpoint_identity,e,states,nodes,queue,control,*status_owner,
                           budget,limits.max_states,limits.max_nodes);
    resumed=true;
  }
  checkpoint::AsyncWriter checkpoint_writer;
  u32 initial=limits.batch_capacity;u64 initial_backoffs=0;
  if(limits.dynamic_batching) {
    auto initial_status=detail::read_one(e.status);detail::Control empty{};
    while(initial>1&&!detail::batch_resources(e,initial_status,empty,budget,initial,limits.max_states,
                                            detail::scratch_bytes(e,initial,limits.split_selection))) {
      initial=(initial+1)/2;++initial_backoffs;
    }
  }
  std::unique_ptr<detail::BatchStorage> scratch;
  for(;;) {
    try {scratch=std::make_unique<detail::BatchStorage>(budget,initial,e,limits.split_selection);break;}
    catch(const AllocationRefusal&) {
      if(!limits.dynamic_batching||initial==1)throw;
      initial=(initial+1)/2;++initial_backoffs;
    }
  }
  detail::Tuner tuner{};tuner.active=tuner.best=initial;
  tuner.maximum=limits.maximum_batch_capacity?limits.maximum_batch_capacity:65536;
  const auto copy_shape=detail::admission_shape(e,limits.admission_threads);
  tuner.admission=tuner.best_admission=copy_shape.initial;tuner.maximum_admission=copy_shape.maximum;
  const bool tune_batch=limits.dynamic_batching&&tuner.maximum>1;
  const bool tune_admission=!limits.admission_threads&&tuner.maximum_admission>1;
  const auto draft_shape=detail::draft_shape(limits.draft_threads,initial);
  tuner.draft=tuner.best_draft=draft_shape.initial;tuner.maximum_draft=draft_shape.maximum;
  const bool tune_draft=!limits.draft_threads&&tuner.maximum_draft>1;
  const bool tune_cache=limits.dynamic_cache;
  const bool tune_refinement=limits.dynamic_refinement&&e.qualified_gap&&e.refinement_maximum_visits>0;
  const u32 cover_maximum=u32(std::min<u64>(UINT32_MAX,u64(e.refinement_maximum_visits)*2));
  const bool tune_cover=limits.dynamic_cover&&e.qualified_gap&&cover_maximum>0;
  tuner.cache=tuner.best_cache=tune_cache?states->capacity:limits.completed_cache_limit;
  tuner.refinement=tuner.best_refinement=e.qualified_gap?
    std::min(e.refinement_maximum_visits,tune_refinement?e.source.trees:limits.refinement_visit_budget):0;
  tuner.cover=tuner.best_cover=e.qualified_gap?
    u32(std::min<u64>(cover_maximum,tune_cover?u64(e.source.trees)*2:limits.cover_visit_budget)):0;
  bool have_proposals=false;
  auto view=[&] {return detail::View{queue->view(),scratch->draft.view(),control.data,
    scratch->jobs.data,scratch->kinds.data,scratch->prune_labels.data,
    scratch->native_ids.data,scratch->native_labels.data,scratch->capacity,tuner.refinement,
    limits.split_selection,scratch->selected_predicates.data,scratch->split_scores.data,limits.oldest_ready_jobs,tuner.cover,
    have_proposals?scratch->proof_proposals.data:nullptr};};
  Snapshot result{};result.batch_memory_backoffs=initial_backoffs;
  result.resumed=resumed;
  result.batch_adjustments=initial!=limits.batch_capacity;detail::Control current{};
  using Clock=std::chrono::steady_clock;
  bool timing=false,batch_grew=false;u32 timed_jobs=0,timed_capacity=0;
  u64 timed_prunes=0,timed_cache_edges=0;
  Clock::time_point batch_start{};double excluded_seconds=0;u32 exclusion_depth=0;
  auto last_checkpoint_attempt=Clock::now();
  bool explicit_checkpoint_pending=false;
  u64 checkpoint_captures=0,checkpoint_capture_failures=0;
  std::string checkpoint_capture_error;
  auto checkpoint_report=[&] {
    if(limits.checkpoint_path.empty()){result.checkpoint_status={{"enabled",false}};return;}
    const auto state=checkpoint_writer.poll();
    result.checkpoint_status={{"enabled",true},{"path",limits.checkpoint_path},
      {"periodic_interval_seconds",limits.checkpoint_interval_seconds},
      {"periodic_enabled",limits.checkpoint_interval_seconds>0},{"captures",checkpoint_captures},
      {"capture_failures",checkpoint_capture_failures},{"capture_error",checkpoint_capture_error},
      {"request_pending",explicit_checkpoint_pending},{"write_inflight",state.inflight},
      {"last_write_committed",state.committed},{"last_write_failed",state.failed},{"write_error",state.error},
      {"capture_bytes",state.captured_bytes},{"payload_bytes",state.payload_bytes},
      {"capture_seconds",state.capture_seconds},{"written_bytes",state.written_bytes},
      {"total_file_bytes",state.total_bytes},{"write_seconds",state.write_seconds}};
  };
  auto excluded=[&](auto&& fn) {
    const bool outer=exclusion_depth++==0;auto start=Clock::now();
    auto close=[&]{--exclusion_depth;if(timing&&outer)excluded_seconds+=std::chrono::duration<double>(Clock::now()-start).count();};
    try{fn();}catch(...){close();throw;}close();
  };
  auto refresh=[&] {
    result.status=detail::read_one(e.status);current=detail::read_one(control.data);
    result.ready_states=current.ready_count;result.completion_events=current.finish_tail-current.finish_head;
    result.oldest_selected_jobs=current.oldest_selected_jobs;
    result.native_pending=current.native_count;result.batch_jobs=current.jobs;
    result.maximum_batch_jobs=std::max(result.maximum_batch_jobs,current.jobs);
    result.resolved_edges=current.resolved_edges;result.pending_cache_merges=current.pending_merges;
    result.priority_split_changes=current.priority_split_changes;
    result.completed_cache_edges=current.completed_edges;
    result.completed_cached_states=current.cache_tail-current.cache_head;
    result.active_batch_capacity=tuner.active;result.allocated_batch_capacity=scratch?scratch->capacity:0;
    result.autotune_best_capacity=tuner.best;result.autotune_best_jobs_per_second=tuner.best_rate;
    result.admission_threads=tuner.admission;result.maximum_admission_threads=tuner.maximum_admission;
    result.best_admission_threads=tuner.best_admission;
    result.draft_threads=tuner.draft;result.maximum_draft_threads=tuner.maximum_draft;
    result.best_draft_threads=tuner.best_draft;
    result.completed_cache_limit=tuner.cache;result.refinement_visit_budget=tuner.refinement;
    result.cache_policy_evictions=current.cache_policy_evictions;
    result.refinement_attempts=current.refinement_attempts;result.refinement_visits=current.refinement_visits;
    result.refinement_tightened_roots=current.refinement_tightened_roots;
    result.refinement_rejected_roots=current.refinement_rejected_roots;
    result.refinement_fallback_frontiers=current.refinement_fallback_frontiers;
    result.refinement_additional_prunes=current.refinement_additional_prunes;
    result.cover_visit_budget=tuner.cover;
    result.cover_attempts=current.cover_attempts;result.cover_visits=current.cover_visits;
    result.cover_feasible_cases=current.cover_feasible_cases;result.cover_certified_cases=current.cover_certified_cases;
    result.cover_additional_prunes=current.cover_additional_prunes;result.cover_failures=current.cover_failures;
    result.retune_backoff_batches=tuner.backoff;result.retune_cooldown_batches=tuner.cooldown;
    result.autotune_best_work_equivalents_per_second=tuner.best_work_rate;
    if(modules)result.proof_module_status=modules->diagnostics();
    checkpoint_report();
  };
  auto last_coverage=Clock::now();double coverage_interval=1.;bool terminal_coverage_sampled=false;
  auto publish=[&] {
    refresh();
    if(progress)excluded([&]{
      const bool diagnostic_stop=result.status.error==25;
      if((!result.status.error||diagnostic_stop)&&(coverage_catalog||result.status.complete)) {
        if(result.status.complete) {
          result.grid_coverage={};result.grid_coverage.available=true;
          result.grid_coverage.lower=result.grid_coverage.upper=1.;
          result.grid_coverage_state_creations=result.status.state_creations;
        }else if((diagnostic_stop&&!terminal_coverage_sampled)||
                 std::chrono::duration<double>(Clock::now()-last_coverage).count()>=coverage_interval) {
          // Telemetry is optional and charged to the same owned byte budget.
          // Target <=1% reporting time using measured cost; a large DAG report
          // can time out without disrupting construction. It is not an ETA.
          double window=std::chrono::duration<double>(Clock::now()-last_coverage).count();
          auto measurement_start=Clock::now();const auto before=budget.used;
          result.grid_coverage={};
          try {
            // Catalog and scratch exist only during this report. Optional
            // telemetry cannot consume memory needed by later arena growth.
            grid::Storage catalog(budget,*coverage_catalog);
            auto catalog_bytes=budget.used-before;
            result.grid_coverage=coverage::measure(e,queue->view(),current.root_state,
                u32(result.status.states),budget,catalog.view(),std::max(diagnostic_stop?1.:.02,window*.01));
            result.grid_coverage.catalog_bytes=catalog_bytes;
          }catch(const AllocationRefusal&){result.grid_coverage.allocation_refused=true;}
          result.grid_coverage.seconds=std::chrono::duration<double>(Clock::now()-measurement_start).count();
          result.grid_coverage_state_creations=result.status.state_creations;
          ++result.grid_coverage_samples;
          if(diagnostic_stop)terminal_coverage_sampled=true;
          coverage_interval=std::max(1.,result.grid_coverage.seconds*100.);
          if(result.grid_coverage.timed_out||result.grid_coverage.allocation_refused)
            coverage_interval=std::max(coverage_interval,window*2.);
          last_coverage=Clock::now();
        }
      }
      progress(result);
    });
  };
  auto gate=[&] {if(check_gate)excluded([&]{check_gate();});};
  // Called only with all kernels complete at an idle boundary. Snapshot capture
  // owns host bytes before search resumes; the writer never reads live GPU data.
  auto checkpoint_boundary=[&](bool final) {
    if(limits.checkpoint_path.empty())return;
    if(limits.checkpoint_requested&&limits.checkpoint_requested())explicit_checkpoint_pending=true;
    const bool due=limits.checkpoint_interval_seconds>0&&
      std::chrono::duration<double>(Clock::now()-last_checkpoint_attempt).count()>=limits.checkpoint_interval_seconds;
    if(!final&&!explicit_checkpoint_pending&&!due)return;
    if(final)checkpoint_writer.finish();
    else if(checkpoint_writer.poll().inflight)return;
    last_checkpoint_attempt=Clock::now();explicit_checkpoint_pending=false;
    try {
      gate();
      auto snapshot=checkpoint_io::capture(e,*states,*nodes,*queue,control,
          limits.checkpoint_identity,limits.checkpoint_host_byte_budget);
      auto publication=limits.checkpoint_publication?limits.checkpoint_publication(snapshot.captured_bytes()):
          class_study::CheckpointPublication{limits.checkpoint_path,{},{},{}};
      require(!publication.snapshot_path.empty(),"checkpoint transaction has no snapshot destination");
      require(checkpoint_writer.try_submit(publication.snapshot_path,std::move(snapshot),
          std::move(publication.prepare_storage),std::move(publication.publish),std::move(publication.abort)),
              "checkpoint writer unexpectedly busy");
      ++checkpoint_captures;checkpoint_capture_error.clear();
    }catch(const std::exception& error) {
      ++checkpoint_capture_failures;checkpoint_capture_error=error.what();checkpoint_report();
      if(final){publish();throw;}return;
    }
    if(final) {
      const auto saved=checkpoint_writer.finish();checkpoint_report();
      if(saved.failed){publish();throw std::runtime_error("final checkpoint write failed: "+saved.error);}
    }
    checkpoint_report();
  };
  auto growth=[&](auto&& fn) {batch_grew=timing;excluded(fn);};
  auto grow_nodes_now=[&] {
    // A failed allocation must retain this exact GPU boundary in the receipt.
    publish();const u64 refusals=result.growth_allocation_refusals,rollbacks=result.growth_transaction_rollbacks;
    auto observe=[&](const class_conversion_growth::Outcome& out) {
      result.node_growth_required_capacity=out.plan.required;
      result.node_growth_preferred_capacity=out.plan.preferred;
      result.node_growth_selected_capacity=out.plan.selected;
      result.node_growth_additional_bytes=out.plan.affordable?out.plan.additional_bytes:out.plan.minimum_bytes;
      result.growth_allocation_refusals=refusals+out.allocation_refusals;
      result.growth_transaction_rollbacks=rollbacks+out.transaction_rollbacks;publish();
    };
    growth([&]{grow_nodes(e,*states,nodes,budget,result.status,limits.max_nodes,
      result.status.nodes+1,observe,{},multiply(multiply(scratch->capacity,e.source.classes),8));});++result.node_growths;
    detail::request_boundary<<<1,1>>>(e,0);synchronize();
  };
  auto set_active=[&](u32 n) {
    if(n!=tuner.active){tuner.active=n;++result.batch_adjustments;}
  };
  auto set_admission=[&](u32 n) {
    if(n!=tuner.admission){tuner.admission=n;++result.admission_thread_adjustments;}
  };
  auto set_draft=[&](u32 n) {
    if(n!=tuner.draft){tuner.draft=n;++result.draft_thread_adjustments;}
  };
  auto set_cache=[&](u32 n) {
    if(n!=tuner.cache){tuner.cache=n;++result.cache_limit_adjustments;}
  };
  auto set_refinement=[&](u32 n) {
    if(n!=tuner.refinement){tuner.refinement=n;++result.refinement_adjustments;}
  };
  auto set_cover=[&](u32 n) {
    if(n!=tuner.cover){tuner.cover=n;++result.cover_adjustments;}
  };
  auto invalidate_pair=[&] {
    tuner.best=tuner.active;tuner.best_admission=tuner.admission;tuner.best_draft=tuner.draft;
    tuner.best_cache=tuner.cache;tuner.best_refinement=tuner.refinement;tuner.best_cover=tuner.cover;
    tuner.best_rate=tuner.best_work_rate=0;tuner.probing=false;tuner.cooldown=tuner.backoff=0;
    tuner.previous_min_cost=0;tuner.previous_max_cost=INFINITY;tuner.reset_window();
  };
  auto resize_scratch=[&](u32 n) {
    require(current.stage==detail::idle&&!current.jobs&&!current.native_count&&!current.native_cursor,
            "frontier scratch resize ownership boundary");
    if(n<=scratch->capacity)return true;
    if(!detail::batch_resources(e,result.status,current,budget,n,limits.max_states,
                               detail::scratch_bytes(e,n,limits.split_selection))) {
      ++result.batch_growth_refusals;return false;
    }
    // Optional allocation failure preserves all old owners and does not abort
    // conversion. No queued job refers to the disposable old scratch at idle.
    try {auto fresh=std::make_unique<detail::BatchStorage>(budget,n,e,limits.split_selection);scratch.swap(fresh);}
    catch(const AllocationRefusal&) {++result.batch_growth_refusals;return false;}
    return true;
  };
  auto shrink_scratch=[&](u32 n) {
    require(current.stage==detail::idle&&!current.jobs&&!current.native_count&&!current.native_cursor,
            "frontier scratch shrink ownership boundary");
    if(n>=scratch->capacity)return;
    // Try a transactional shrink first. Under pressure its coexistence may be
    // impossible; idle scratch is disposable, so release it before allocating
    // the smaller owner. State/queue/node ownership is untouched either way.
    u64 bytes=detail::scratch_bytes(e,n,limits.split_selection);std::size_t free_bytes=0,total_bytes=0;
    auto error=cudaMemGetInfo(&free_bytes,&total_bytes);
    check(error,"frontier shrink memory resource query");
    if(budget.used<=budget.limit&&bytes<=budget.limit-budget.used&&bytes<=free_bytes) {
      try {auto fresh=std::make_unique<detail::BatchStorage>(budget,n,e,limits.split_selection);scratch.swap(fresh);return;}
      catch(const AllocationRefusal&) {}
    }
    publish();scratch.reset();
    for(;;) {
      try {scratch=std::make_unique<detail::BatchStorage>(budget,n,e,limits.split_selection);break;}
      catch(const AllocationRefusal&) {
        if(n==1){publish();throw;}
        n=(n+1)/2;++result.batch_memory_backoffs;
      }
    }
    set_active(std::min(tuner.active,n));
  };
  auto finish_timing=[&] {
    if(!timing)return;
    double seconds=std::max(0.0,std::chrono::duration<double>(Clock::now()-batch_start).count()-excluded_seconds);
    timing=false;result.autotune_last_batch_seconds=seconds;result.autotune_timed_jobs+=timed_jobs;
    result.autotune_last_jobs_per_second=seconds>0?double(timed_jobs)/seconds:0;
    const u64 avoided=2*(current.refinement_additional_prunes+current.cover_additional_prunes-timed_prunes)+current.completed_edges-timed_cache_edges;
    result.autotune_last_work_equivalents_per_second=seconds>0?(double(timed_jobs)+double(avoided))/seconds:0;
    if(!tune_batch&&!tune_admission&&!tune_draft&&!tune_cache&&!tune_refinement&&!tune_cover)return;
    if(batch_grew){++result.autotune_excluded_growth_batches;invalidate_pair();return;}
    if(!seconds||timed_jobs<(timed_capacity+1)/2)return;
    if(tuner.cooldown)--tuner.cooldown;
    ++result.autotune_samples;++tuner.window_batches;tuner.window_jobs+=timed_jobs;
    tuner.window_avoided+=avoided;tuner.window_seconds+=seconds;
    const double cost=seconds/double(timed_jobs);
    tuner.window_min_cost=std::min(tuner.window_min_cost,cost);tuner.window_max_cost=std::max(tuner.window_max_cost,cost);
    if(tuner.window_batches<detail::tuning_window_batches)return;
    const double rate=double(tuner.window_jobs)/tuner.window_seconds;
    const double work_rate=(double(tuner.window_jobs)+double(tuner.window_avoided))/tuner.window_seconds;
    const double min_cost=tuner.window_min_cost,max_cost=tuner.window_max_cost;
    // Disjoint observed cost ranges indicate a new workload regime. Reopen
    // exploration immediately; this is a scheduling heuristic, not a test of
    // statistical significance or a change to the proof/accuracy requirement.
    if(tuner.observe_drift(min_cost,max_cost))++result.retune_drift_resets;
    tuner.reset_window();
    auto accept=[&]{
      tuner.best=tuner.active;tuner.best_admission=tuner.admission;tuner.best_draft=tuner.draft;
      tuner.best_cache=tuner.cache;tuner.best_refinement=tuner.refinement;tuner.best_cover=tuner.cover;
      tuner.best_rate=rate;tuner.best_work_rate=work_rate;
      tuner.previous_min_cost=min_cost;tuner.previous_max_cost=max_cost;
    };
    // Refresh the current-capacity baseline as the source work changes. Only
    // the immediately paired neighboring probe compares against that saved window;
    // an old easy region's peak cannot veto every later capacity candidate.
    if(!tuner.probing) {
      accept();
    }else if(tuner.probe_axis>=3?work_rate>tuner.best_work_rate:rate>tuner.best_rate) {
      accept();tuner.probing=false;tuner.cooldown=tuner.backoff=0;
    }else {
      set_active(tuner.best);set_admission(tuner.best_admission);set_draft(tuner.best_draft);
      set_cache(tuner.best_cache);set_refinement(tuner.best_refinement);set_cover(tuner.best_cover);
      tuner.probing=false;
      tuner.reject_probe();
      if(tuner.probe_axis==0)tuner.probe_direction=-tuner.probe_direction;
      else if(tuner.probe_axis==1)tuner.admission_direction=-tuner.admission_direction;
      else if(tuner.probe_axis==2)tuner.draft_direction=-tuner.draft_direction;
      else if(tuner.probe_axis==3)tuner.cache_direction=-tuner.cache_direction;
      else if(tuner.probe_axis==4)tuner.refinement_direction=-tuner.refinement_direction;
      else tuner.cover_direction=-tuner.cover_direction;
    }
    // Explore only when current completed windows actually exercise >=half of
    // this extent. At a boundary explore inward; a rejected probe reverses the
    // next direction, so the maximum never becomes an absorbing batch size.
    if(!result.status.complete&&!tuner.probing&&!tuner.cooldown&&tuner.active==tuner.best) {
      u32 offered=u32(std::min<u64>(tuner.active,current.ready_count));
      u32 draft_ceiling=std::min(tuner.maximum_draft,detail::power_two_extent(std::max(offered,1u)));
      auto enabled=[&](u32 axis){return axis==0?tune_batch:axis==1?tune_admission:axis==3?tune_cache:axis==4?tune_refinement:axis==5?tune_cover:
        tune_draft&&draft_ceiling>1&&tuner.draft<=draft_ceiling;};
      u32 axis=tuner.next_axis,visited=0;
      while(visited<6&&!enabled(axis)){axis=(axis+1)%6;++visited;}
      if(visited==6)return;
      tuner.probe_axis=axis;tuner.next_axis=(axis+1)%6;
      if(axis==0){
        u32 candidate=detail::neighboring_capacity(tuner);++result.autotune_probes;
        if(candidate>tuner.active)++result.autotune_upward_probes;else ++result.autotune_downward_probes;
        if(resize_scratch(candidate)){set_active(candidate);tuner.probing=true;}
        else {tuner.cooldown=detail::tuning_window_batches;tuner.probe_direction=-tuner.probe_direction;}
      }else if(axis==1){
        u32 candidate=detail::neighboring_extent(tuner.admission,tuner.maximum_admission,tuner.admission_direction);
        ++result.admission_thread_probes;set_admission(candidate);tuner.probing=true;
      }else if(axis==2){
        u32 candidate=detail::neighboring_extent(tuner.draft,draft_ceiling,tuner.draft_direction);
        ++result.draft_thread_probes;set_draft(candidate);tuner.probing=true;
      }else if(axis==3){
        u32 candidate=detail::neighboring_work(tuner.cache,states->capacity,tuner.cache_direction);
        ++result.cache_limit_probes;set_cache(candidate);tuner.probing=true;
      }else if(axis==4){
        u32 candidate=detail::neighboring_work(tuner.refinement,e.refinement_maximum_visits,tuner.refinement_direction);
        ++result.refinement_probes;set_refinement(candidate);tuner.probing=true;
      }else{
        u32 candidate=detail::neighboring_work(tuner.cover,cover_maximum,tuner.cover_direction);
        ++result.cover_probes;set_cover(candidate);tuner.probing=true;
      }
    }
  };
  refresh();if(result.status.error){publish();return result;}
  if(!resumed){detail::bootstrap<<<1,1>>>(e,view());synchronize();}
  for(;;) {
    publish();if(result.status.error)return result;
    if(result.status.complete){checkpoint_boundary(limits.checkpoint_on_completion||
        (limits.stop_requested&&limits.stop_requested()));publish();return result;}
    gate();
    if(result.status.request==3){grow_nodes_now();continue;}
    // Finish parents before selecting another frontier. intern_node refusal
    // leaves the FIFO head unconsumed; all edge/label results remain owned.
    detail::drain_completions<<<1,1>>>(e,view());synchronize();refresh();
    if(result.status.error||result.status.complete){
      if(result.status.complete&&current.stage==detail::idle){finish_timing();checkpoint_boundary(
        limits.checkpoint_on_completion||(limits.stop_requested&&limits.stop_requested()));}
      publish();return result;
    }
    if(result.status.request==3){++result.completion_growth_retries;grow_nodes_now();continue;}
    if(current.stage==detail::idle) {
      finish_timing();
      const bool stop=limits.stop_requested&&limits.stop_requested();
      checkpoint_boundary(stop);
      if(stop){result.stopped=true;publish();return result;}
      if(modules)excluded([&]{if(modules->activate_pending_at_idle())invalidate_pair();});
      if(!current.ready_count) {
        detail::request_boundary<<<1,1>>>(e,0,66);synchronize();publish();return result;
      }
      if(limits.dynamic_batching) {
        u32 proposed=u32(std::min<u64>(tuner.active,current.ready_count)),affordable=proposed;
        while(affordable>1&&!detail::batch_resources(e,result.status,current,budget,affordable,limits.max_states,0))
          affordable=(affordable+1)/2;
        bool pressure=affordable<proposed||
          !detail::batch_resources(e,result.status,current,budget,affordable,limits.max_states,0);
        if(pressure) {
          set_active(affordable);++result.batch_memory_backoffs;
          shrink_scratch(affordable);refresh();
          // Releasing the historical peak owner changes the actual headroom.
          // Recheck this smaller selection before publishing any draft work.
          affordable=u32(std::min<u64>(tuner.active,current.ready_count));
          while(affordable>1&&!detail::batch_resources(e,result.status,current,budget,affordable,limits.max_states,0))
            affordable=(affordable+1)/2;
          set_active(affordable);shrink_scratch(affordable);
          invalidate_pair();
        }
      }
      // Preserve the resource seed/learned grouping through small frontiers.
      // prepare_drafts bounds-checks jobs in a partial block; job availability
      // only gates exploration, never destructively resets the current width.
      timing=true;batch_grew=false;excluded_seconds=0;timed_capacity=tuner.active;batch_start=Clock::now();
      timed_prunes=current.refinement_additional_prunes+current.cover_additional_prunes;timed_cache_edges=current.completed_edges;
      // Trim only fully resolved states. Charge eviction work to the next
      // measured batch. Freed slots are reusable; allocated arenas do not shrink.
      if(current.cache_tail-current.cache_head>tuner.cache){
        detail::reclaim_completed<<<1,1>>>(e,view(),0,tuner.cache);synchronize();refresh();
        if(result.status.error){publish();return result;}
      }
      auto selected=view();selected.batch=tuner.active;
      detail::select_ready<<<1,1>>>(e,selected);synchronize();refresh();timed_jobs=current.jobs;
      if(result.status.error){publish();return result;}
      have_proposals=false;
      if(modules&&tuner.cover) {
        detail::initialize_proposals<<<(current.jobs+127)/128,128>>>(e,view(),scratch->proof_proposals.data);
        check(cudaGetLastError(),"initialize proof proposals");
        have_proposals=modules->propose(proof_modules::make_input(e,modules->session().token,
          scratch->jobs.data,scratch->selected_predicates.data),current.jobs,scratch->proof_proposals.data);
        synchronize();
      }
      detail::prepare_drafts<<<(current.jobs+tuner.draft-1)/tuner.draft,tuner.draft>>>(e,view());synchronize();++result.draft_batches;
      refresh();if(result.status.error){publish();return result;}
    }
    if(current.stage==detail::draft_commit) {
      // Reserving the worst case makes each two-child admission atomic with
      // respect to arena growth. Scratch/cursors survive growth unchanged.
      u64 requested=multiply(current.expanding,2);
      if(requested>u64(states->capacity)-result.status.states+result.status.free_count) {
        detail::reclaim_completed<<<1,1>>>(e,view(),requested);synchronize();refresh();
        if(result.status.error){publish();return result;}
      }
      require(result.status.free_count<=result.status.states,"frontier free-list extent");
      u64 needed=result.status.states-result.status.free_count+multiply(current.expanding,2);
      // External allocation drift or a growing frontier may invalidate a prior
      // reserve estimate. Reduce ONLY the uncommitted tail; committed native
      // labels/IDs/cursors and all parent dependencies retain their owners.
      while(limits.dynamic_batching&&
            (needed>limits.max_states||(needed>states->capacity&&
             !detail::batch_resources(e,result.status,current,budget,current.expanding,limits.max_states,0)))&&
            current.jobs>current.cursor&&(current.jobs>1||scratch->capacity>1)) {
        u32 remaining=current.jobs-current.cursor;
        u32 keep=current.cursor+(remaining>1?(remaining+1)/2:0);
        u32 previous_jobs=current.jobs;
        detail::trim_selected_tail<<<1,1>>>(e,view(),keep);synchronize();refresh();
        if(result.status.error){publish();return result;}
        result.batch_returned_jobs+=previous_jobs-current.jobs;
        timed_jobs=current.jobs;
        ++result.batch_memory_backoffs;batch_grew=true;
        set_active(std::max(1u,keep-current.cursor));
        invalidate_pair();
        needed=result.status.states-result.status.free_count+multiply(current.expanding,2);
      }
      if(needed>limits.max_states) {
        detail::request_boundary<<<1,1>>>(e,2);synchronize();publish();
        throw std::runtime_error("frontier two-child admission exceeds state limit");
      }
      bool retry_selection=false;
      while(states->capacity<needed) {
        detail::request_boundary<<<1,1>>>(e,2);synchronize();publish();
        // The external progress hook may allocate memory. Recheck after that
        // publication, before committing a new owner; private drafts persist.
        if(limits.dynamic_batching&&current.jobs>current.cursor&&
           (current.jobs>1||scratch->capacity>1)&&
           !detail::batch_resources(e,result.status,current,budget,current.expanding,limits.max_states,0)) {
          retry_selection=true;break;
        }
        const u64 refusals=result.growth_allocation_refusals,rollbacks=result.growth_transaction_rollbacks;
        auto observe=[&](const class_conversion_growth::Outcome& out) {
          result.state_growth_required_capacity=out.plan.required;
          result.state_growth_preferred_capacity=out.plan.preferred;
          result.state_growth_selected_capacity=out.plan.selected;
          result.state_growth_additional_bytes=out.plan.affordable?out.plan.additional_bytes:out.plan.minimum_bytes;
          result.growth_allocation_refusals=refusals+out.allocation_refusals;
          result.growth_transaction_rollbacks=rollbacks+out.transaction_rollbacks;publish();
        };
        growth([&]{detail::grow_state_queue_transaction(e,states,*nodes,queue,control,budget,
          result.status,needed,limits.max_states,observe,{},multiply(multiply(current.expanding,e.source.classes),8));});++result.state_growths;
        refresh();
      }
      if(retry_selection)continue;
      detail::commit_drafts<<<1,tuner.admission>>>(e,view(),limits.max_expansions);synchronize();refresh();
      if(result.status.error){publish();return result;}
      if(result.status.request==2){publish();throw std::runtime_error("frontier admission reserve invariant");}
      if(result.status.request==3){grow_nodes_now();continue;}
    }
    if(current.stage==detail::native_query) {
      const u32 count=current.native_count;require(count>0&&count<=scratch->capacity,"frontier native batch extent");
      publish();gate();
      materialize_native_batch<<<(count+127)/128,128>>>(e,scratch->native_ids.data,count,scratch->native_rows.data);synchronize();refresh();
      if(result.status.error){publish();return result;}
      const float* margins=native(scratch->native_rows.data,count,true);
      require(margins!=nullptr,"frontier native margin callback returned null");
      const u64 words=multiply(count,e.source.classes);
      check_native_margin_batch<<<u32(std::min<u64>(65535,(words+255)/256)),256>>>(e,scratch->native_ids.data,count,margins);
      synchronize();result.native_margin_rows+=count;refresh();
      // The callback owns this pointer and may overwrite it at public predict.
      // Complete every bit check before requesting the next borrowed result.
      if(result.status.error){publish();return result;}
      gate();
      const float* public_output=native(scratch->native_rows.data,count,false);
      require(public_output!=nullptr,"frontier native public callback returned null");
      decode_native_class_batch<<<(count+127)/128,128>>>(e,public_output,count,direct,scratch->native_labels.data);
      synchronize();result.native_public_rows+=count;++result.native_batches;refresh();
      if(result.status.error){publish();return result;}
      detail::labels_ready<<<1,1>>>(e,view());synchronize();refresh();
    }
    if(current.stage==detail::native_commit) {
      detail::commit_native<<<1,1>>>(e,view());synchronize();refresh();
      if(result.status.error){publish();return result;}
      if(result.status.request==3){grow_nodes_now();continue;}
    }
  }
}
} // namespace class_conversion_adaptive::frontier
