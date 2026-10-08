#pragma once
#include "adaptive_grid_catalog.cuh"
#include <chrono>

// Read-only progress measurement for a synchronized adaptive dependency DAG.
// This never authorizes a class, changes the search, or traverses graph-node IDs
// stored on resolved edges as though they were live state slots.
namespace class_conversion_adaptive::coverage {
struct Interval { double lower, upper; };
struct Result {
  bool available=false,timed_out=false,allocation_refused=false;
  double lower=0,upper=1,seconds=0;
  u64 scratch_bytes=0,catalog_bytes=0;
  u32 waves=0,error=0;
};
struct Control { u32 count,error,root_ready,reserved; Interval root; };
struct ScratchView {
  Interval* values;
  u32 *pending,*notified,*next;
  Control* control;
  u32 count,root;
};
__device__ inline void error(ScratchView s,u32 code) {
  atomicCAS(&s.control->error,0u,code);
}
__device__ inline void enqueue(ScratchView s,u32 id) {
  u32 index=atomicAdd(&s.control->count,1u);
  if(index>=s.count){error(s,1);return;}
  s.next[index]=id;
}
__device__ inline Interval combine(Interval left,Interval right,grid::BranchCounts counts) {
  // Equal intervals need no arithmetic: branch weights sum to exactly one.
  if(left.lower==right.lower&&left.upper==right.upper)return left;
  double total=double(counts.total);
  double ll=__ddiv_rd(double(counts.left),total),lu=__ddiv_ru(double(counts.left),total);
  double rl=__ddiv_rd(double(counts.right),total),ru=__ddiv_ru(double(counts.right),total);
  return {fmax(0.,__dadd_rd(__dmul_rd(ll,left.lower),__dmul_rd(rl,right.lower))),
          fmin(1.,__dadd_ru(__dmul_ru(lu,left.upper),__dmul_ru(ru,right.upper)))};
}
template<class Queue> __global__ void initialize(EngineView e,Queue q,ScratchView s) {
  for(u32 id=blockIdx.x*blockDim.x+threadIdx.x;id<s.count;id+=blockDim.x*gridDim.x) {
    const State state=e.arena.states[id];
    s.pending[id]=0;
    s.notified[id]=0;s.values[id]={0.,1.};
    if(state.phase==free_phase){if(id==s.root)error(s,2);continue;}
    if(state.phase>3){error(s,3);continue;}
    u32 pending=0;
    if(state.phase==2) {
      for(u32 side=0;side<2;++side)if(!(q.flags[id]&(4u<<side))) {
        u32 child=side?state.right:state.left;
        if(child>=s.count||child==id||e.arena.states[child].phase==free_phase){error(s,4);continue;}
        ++pending;
      }
      if(pending!=q.pending[id])error(s,5);
    }
    s.pending[id]=pending;
    if(!pending) {
      double value=(state.phase==2||state.phase==3)?1.:0.;
      s.values[id]={value,value};
      if(id==s.root){s.control->root={value,value};s.control->root_ready=1;}
      if(q.wait_head[id]!=none)enqueue(s,id);
    }
  }
}
template<class Queue> __global__ void propagate(EngineView e,Queue q,grid::CatalogView catalog,
                                               ScratchView s,const u32* input,u32 input_count) {
  for(u32 index=blockIdx.x*blockDim.x+threadIdx.x;index<input_count;index+=blockDim.x*gridDim.x) {
    if(atomicAdd(&s.control->error,0u))return;
    u32 child=input[index],edge=q.wait_head[child];u64 visited=0;
    while(edge!=none) {
      if(++visited>u64(s.count)*2||u64(edge)>=u64(s.count)*2){error(s,6);break;}
      u32 parent=edge/2,side=edge%2;const State state=e.arena.states[parent];
      if(state.phase!=2||(q.flags[parent]&(4u<<side))||
          (side?state.right:state.left)!=child){error(s,7);break;}
      if(atomicOr(s.notified+parent,1u<<side)&(1u<<side)){error(s,12);break;}
      u32 previous=atomicSub(s.pending+parent,1u);
      if(!previous){error(s,8);break;}
      if(previous==1) {
        Interval left=(q.flags[parent]&4u)?Interval{1.,1.}:s.values[state.left];
        Interval right=(q.flags[parent]&8u)?Interval{1.,1.}:s.values[state.right];
        auto counts=grid::branch_counts(e,catalog,parent);
        if(!counts.valid||!counts.total||counts.left>counts.total||counts.right!=counts.total-counts.left||
            counts.total>(u64(1)<<53)){error(s,9);break;}
        Interval value=combine(left,right,counts);
        s.values[parent]=value;
        if(parent==s.root){s.control->root=value;s.control->root_ready=1;}
        if(q.wait_head[parent]!=none)enqueue(s,parent);
      }
      edge=q.edge_next[edge];
    }
  }
  // Every value consumed in this wave was finalized in an earlier kernel.
  // Atomic pending decrements elect one parent producer. The next kernel launch
  // publishes all newly produced values and frontier entries together.
}
inline u64 scratch_extent(u32 count) {
  return multiply(count,sizeof(Interval)+4*sizeof(u32))+sizeof(Control);
}
template<class Queue> inline Result measure(EngineView e,Queue q,u32 root,u32 count,
    Budget& budget,grid::CatalogView catalog,double maximum_seconds=0) {
  using Clock=std::chrono::steady_clock;
  const auto started=Clock::now();Result result;result.scratch_bytes=scratch_extent(count);
  auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-started).count();};
  if(!count||count>q.capacity||count>e.arena.state_capacity||root>=count){result.error=10;return result;}
  try {
    Buffer<Interval> values(budget,count);
    Buffer<u32> pending(budget,count),notified(budget,count),a(budget,count),b(budget,count);
    Buffer<Control> control(budget,1);control.zero();
    ScratchView s{values.data,pending.data,notified.data,a.data,control.data,count,root};
    const u32 blocks=std::min(4096u,(count+255u)/256u);
    initialize<<<blocks,256>>>(e,q,s);synchronize();
    auto current=control.download(1).front();
    for(;;) {
      if(current.error){result.error=current.error;break;}
      if(current.root_ready){result.available=true;result.lower=current.root.lower;result.upper=current.root.upper;break;}
      if(!current.count||result.waves>=count){result.error=11;break;}
      if(maximum_seconds>0&&elapsed()>=maximum_seconds){result.timed_out=true;break;}
      u32 count_in=current.count;std::swap(a,b);s.next=a.data;
      control.zero();
      propagate<<<std::min(4096u,(count_in+255u)/256u),256>>>(e,q,catalog,s,b.data,count_in);
      synchronize();++result.waves;current=control.download(1).front();
    }
  }catch(const AllocationRefusal&){result.allocation_refused=true;}
  result.seconds=elapsed();return result;
}
} // namespace class_conversion_adaptive::coverage
