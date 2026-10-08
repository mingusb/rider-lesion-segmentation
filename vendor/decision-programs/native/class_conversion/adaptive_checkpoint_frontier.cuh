#pragma once
// Include at GLOBAL scope after frontier::detail definitions, before run().
// This adapter has no cadence, process signals, source reload or gate transfer.
#include "adaptive_checkpoint.cuh"
namespace class_conversion_adaptive::frontier::checkpoint_io {
namespace cp=class_conversion_adaptive::checkpoint;
using Control=detail::Control;using QueueStorage=detail::QueueStorage;
struct RestoreReceipt {cp::Json metadata;cp::Receipt codec;};
inline void boundary(const Status& h,const Control& c,u32 state_capacity,u32 node_capacity){
 require(!h.error&&!h.request&&!h.depth,"checkpoint arena has an error/request/legacy stack");
 require(c.stage==detail::idle&&!c.jobs&&!c.cursor&&!c.expanding&&!c.native_count&&!c.native_cursor,
   "checkpoint frontier is not idle");
 require(h.states&&h.states<=state_capacity&&h.nodes<=node_capacity&&h.free_count<=h.states,
   "checkpoint arena extent");
 require(c.root_state==0&&c.root_state<h.states,"checkpoint root state extent");
 require(c.ready_head<state_capacity&&c.ready_count<=state_capacity&&
   c.finish_head<=c.finish_tail&&c.finish_tail-c.finish_head<=state_capacity&&
   c.cache_head<=c.cache_tail&&c.cache_tail-c.cache_head<=state_capacity,
   "checkpoint ring extent");
 require(h.complete? h.root<h.nodes:h.root==none,"checkpoint root completion extent");
 require(h.free_count?h.free_head<h.states:h.free_head==none,"checkpoint free-list head extent");
}
inline std::vector<cp::Range> ring_ranges(u64 head,u64 count,u32 capacity){
 require(capacity&&count<=capacity,"checkpoint sparse ring extent");
 if(!count)return{};head%=capacity;const auto tail=std::min<u64>(count,capacity-head);
 std::vector<cp::Range> out;if(count>tail)out.push_back({0,multiply(count-tail,sizeof(u32))});
 out.push_back({multiply(head,sizeof(u32)),multiply(tail,sizeof(u32))});return out;
}
template<class T> inline cp::DeviceSection section(const char*name,const Buffer<T>& b,u64 count){
 require(count<=b.size,"checkpoint section live extent");return{name,b.data,multiply(count,sizeof(T))};
}
inline std::vector<cp::DeviceSection> sections(const EngineView&e,const StateStorage&s,
 const NodeStorage&n,const QueueStorage&q,const Buffer<Control>&control,const Status&h,const Control&c){
 const auto H=h.states;const auto K=e.source.classes,T=e.source.trees,N=e.domain.numeric_features,W=e.domain.mask_words;
 std::vector<cp::DeviceSection> out{
  section("arena.states",s.states,H),section("arena.words",s.words,multiply(H,K)),
  section("arena.positions",s.positions,multiply(H,K)),section("arena.residual",s.residual,multiply(H,T)),
  section("arena.lower",s.lower,multiply(H,N)),section("arena.upper",s.upper,multiply(H,N)),section("arena.missing",s.missing,multiply(H,N)),
  section("arena.witness_lower",s.witness_lower,multiply(H,N)),section("arena.witness_upper",s.witness_upper,multiply(H,N)),section("arena.witness_missing",s.witness_missing,multiply(H,N)),
  section("arena.allowed",s.allowed,multiply(H,W)),section("arena.witness_allowed",s.witness_allowed,multiply(H,W)),
  section("arena.state_table",s.table,s.table.size),section("arena.terminal_table",s.terminal_table,s.terminal_table.size),
  section("arena.nodes",n.nodes,h.nodes),section("arena.node_table",n.table,n.table.size),
  {"arena.status",e.status,sizeof(Status)},section("frontier.control",control,1),
  {"frontier.ready",q.ready.data,multiply(q.capacity,sizeof(u32)),true,ring_ranges(c.ready_head,c.ready_count,q.capacity)},
  {"frontier.finish",q.finish.data,multiply(q.capacity,sizeof(u32)),true,ring_ranges(c.finish_head,c.finish_tail-c.finish_head,q.capacity)},
  {"frontier.completed",q.completed.data,multiply(q.capacity,sizeof(u32)),true,ring_ranges(c.cache_head,c.cache_tail-c.cache_head,q.capacity)},
  section("frontier.flags",q.flags,H),section("frontier.pending",q.pending,H),section("frontier.wait_head",q.wait_head,H),section("frontier.edge_next",q.edge_next,multiply(H,2))
 };return out;
}
inline cp::Json shape(const EngineView&e,const StateStorage&s,const NodeStorage&n,const Status&h,const Control&c){
 return{{"format","adaptive-frontier-checkpoint-1"},{"features",e.source.features},{"classes",e.source.classes},
  {"source_nodes",e.source.nodes},{"trees",e.source.trees},{"numeric_features",e.domain.numeric_features},
  {"mask_words",e.domain.mask_words},{"support_words",e.support_words},{"state_capacity",s.capacity},
  {"node_capacity",n.capacity},{"state_highwater",h.states},{"nodes",h.nodes},
  {"state_buckets",s.table.size},{"node_buckets",n.table.size},
  {"state_bytes",sizeof(State)},{"node_bytes",sizeof(Node)},{"status_bytes",sizeof(Status)},{"control_bytes",sizeof(Control)},
  {"ready_head",c.ready_head},{"ready_count",c.ready_count},{"finish_head",c.finish_head},{"finish_tail",c.finish_tail},
  {"cache_head",c.cache_head},{"cache_tail",c.cache_tail},{"native_gate_authority_serialized",false},
  {"legacy_stack_depth",0},{"host_tuner_state_serialized",false}};
}
inline cp::OwnedSnapshot capture(EngineView e,const StateStorage&s,const NodeStorage&n,
 const QueueStorage&q,const Buffer<Control>&control,const cp::Json&identity,u64 host_budget=0){
 synchronize();const auto h=detail::read_one(e.status);const auto c=detail::read_one(control.data);
 require(q.capacity==s.capacity&&control.size==1,"checkpoint coupled owner extent");
 boundary(h,c,s.capacity,n.capacity);cp::CaptureOptions options;options.host_byte_budget=host_budget;
 return cp::capture(identity,shape(e,s,n,h,c),sections(e,s,n,q,control,h,c),{},options);
}
struct Validation {
 u32 *state_seen,*node_seen,*ready_seen,*finish_seen,*cache_seen,*edge_seen,*free_seen,*error;
};
__device__ inline void reject(Validation v,u32 code){atomicCAS(v.error,0u,code);}
__global__ void check_tables(EngineView e,Validation v){
 const auto h=*e.status;
 for(u64 slot=u64(blockIdx.x)*blockDim.x+threadIdx.x;slot<e.arena.state_buckets;slot+=u64(blockDim.x)*gridDim.x){
  u32 id=e.arena.table[slot];if(id!=none){
   if(id>=h.states||e.arena.states[id].phase==free_phase){reject(v,1);continue;}
   if(atomicAdd(v.state_seen+id,1u)){reject(v,2);continue;}
   auto probe=hash_bucket(e.arena.states[id].hash,e.arena.state_buckets);bool found=false;
   for(u32 step=0;step<e.arena.state_buckets;++step){const auto at=e.arena.table[probe];if(at==none)break;
    if(probe==slot){found=true;break;}probe=(probe+1)&(e.arena.state_buckets-1);}
   if(!found)reject(v,3);
  }
  id=e.arena.terminal_table[slot];if(id!=none){
   if(id>=h.states||e.arena.states[id].phase!=3||e.arena.states[id].node>=h.nodes){reject(v,4);continue;}
   const auto* words=e.arena.words+u64(id)*e.source.classes;
   auto probe=hash_bucket(score_hash(e,words),e.arena.state_buckets);bool found=false;
   for(u32 step=0;step<e.arena.state_buckets;++step){if(e.arena.terminal_table[probe]==none)break;
    if(probe==slot){found=true;break;}probe=(probe+1)&(e.arena.state_buckets-1);}if(!found)reject(v,5);
  }
 }
 for(u64 slot=u64(blockIdx.x)*blockDim.x+threadIdx.x;slot<e.arena.node_buckets;slot+=u64(blockDim.x)*gridDim.x){
  const auto id=e.arena.node_table[slot];if(id==none)continue;
  if(id>=h.nodes){reject(v,6);continue;}if(atomicAdd(v.node_seen+id,1u)){reject(v,7);continue;}
  auto probe=hash_bucket(node_hash(e.arena.nodes[id]),e.arena.node_buckets);bool found=false;
  for(u32 step=0;step<e.arena.node_buckets;++step){if(e.arena.node_table[probe]==none)break;
   if(probe==slot){found=true;break;}probe=(probe+1)&(e.arena.node_buckets-1);}if(!found)reject(v,8);
 }
}
__global__ void check_queue(EngineView e,detail::QueueView q,const Control*control,Validation v){
 const auto h=*e.status;const auto c=*control;
 for(u64 at=u64(blockIdx.x)*blockDim.x+threadIdx.x;at<c.ready_count;at+=u64(blockDim.x)*gridDim.x){
  auto id=q.ready[(c.ready_head+at)%q.capacity];if(id>=h.states||e.arena.states[id].phase!=0){reject(v,9);continue;}
  if(atomicAdd(v.ready_seen+id,1u))reject(v,10);
 }
 for(u64 at=u64(blockIdx.x)*blockDim.x+threadIdx.x;at<c.finish_tail-c.finish_head;at+=u64(blockDim.x)*gridDim.x){
  auto id=q.finish[(c.finish_head+at)%q.capacity];if(id>=h.states||(e.arena.states[id].phase!=2&&e.arena.states[id].phase!=3)){reject(v,11);continue;}
  if(atomicAdd(v.finish_seen+id,1u))reject(v,12);
 }
 for(u64 at=u64(blockIdx.x)*blockDim.x+threadIdx.x;at<c.cache_tail-c.cache_head;at+=u64(blockDim.x)*gridDim.x){
  auto id=q.completed[(c.cache_head+at)%q.capacity];if(id>=h.states||id==c.root_state||e.arena.states[id].phase!=3){reject(v,13);continue;}
  if(atomicAdd(v.cache_seen+id,1u))reject(v,14);
 }
 for(u64 child=u64(blockIdx.x)*blockDim.x+threadIdx.x;child<h.states;child+=u64(blockDim.x)*gridDim.x){
  auto edge=q.wait_head[child];
  while(edge!=none){if(edge>=2*h.states){reject(v,15);break;}
   if(atomicAdd(v.edge_seen+edge,1u)){reject(v,16);break;}
   auto parent=edge/2,side=edge%2;auto p=e.arena.states[parent];
   if(p.phase!=2||(q.flags[parent]&(4u<<side))||(side?p.right:p.left)!=child){reject(v,17);break;}
   edge=q.edge_next[edge];
  }
 }
}
__global__ void check_free(EngineView e,const Control*control,Validation v){
 if(blockIdx.x||threadIdx.x)return;const auto h=*e.status;u32 id=h.free_head,count=0;
 while(id!=none){if(id>=h.states||id==control->root_state||e.arena.states[id].phase!=free_phase||atomicAdd(v.free_seen+id,1u)){reject(v,18);return;}
  ++count;id=e.arena.states[id].reserved;}
 if(count!=h.free_count)reject(v,19);
}
__global__ void check_payload(EngineView e,detail::QueueView q,const Control*control,Validation v){
 const auto h=*e.status;const auto c=*control;
 for(u64 id=u64(blockIdx.x)*blockDim.x+threadIdx.x;id<h.states;id+=u64(blockDim.x)*gridDim.x){
  const auto s=e.arena.states[id];auto flags=q.flags[id];
  if(s.phase==free_phase){
   if(v.state_seen[id]||!v.free_seen[id]||flags||q.pending[id]||q.wait_head[id]!=none||v.ready_seen[id]||v.finish_seen[id]||v.cache_seen[id])reject(v,20);continue;}
  if(s.phase!=0&&s.phase!=2&&s.phase!=3){reject(v,21);continue;}
  if(v.state_seen[id]!=1||v.free_seen[id]||!(flags&detail::ready_once)||(flags&~31u)){reject(v,22);continue;}
  if(!domain::region_valid(e.domain,region(e,u32(id)))||!domain::region_valid(e.domain,witness_region(e,u32(id)))){reject(v,23);continue;}
  const auto*words=e.arena.words+id*e.source.classes;const auto*positions=e.arena.positions+id*e.source.classes;
  const auto*roots=e.source.trees?e.arena.residual+id*e.source.trees:nullptr;
  for(u32 channel=0;channel<e.source.classes;++channel){
   u32 consumed=0;for(u32 t=0;t<e.source.trees;++t)if(u32(e.source.channels[t])==channel&&roots[t]<0)++consumed;
   if(!isfinite(__uint_as_float(words[channel]))||positions[channel]!=consumed)reject(v,24);
  }
  for(u32 t=0;t<e.source.trees;++t)if(roots[t]<-1||(roots[t]>=0&&u32(roots[t])>=e.source.nodes))reject(v,25);
  if(state_hash(e,region(e,u32(id)),words,positions,roots)!=s.hash)reject(v,26);
  if(s.phase==0){if(s.node!=none||(s.predicate!=none&&(s.predicate>=e.source.nodes||e.source.left[s.predicate]<0)))reject(v,39);if(v.ready_seen[id]!=1||v.finish_seen[id]||v.cache_seen[id]||q.pending[id]||(flags&(detail::finish_once|detail::cached_once|12u)))reject(v,27);}
  else if(s.phase==2){
   const auto resolved=u32(bool(flags&4u))+u32(bool(flags&8u));
   if(v.ready_seen[id]||v.cache_seen[id]||q.pending[id]!=2-resolved||s.predicate>=e.source.nodes||e.source.left[s.predicate]<0)reject(v,28);
   for(u32 side=0;side<2;++side){const bool done=flags&(4u<<side);const auto target=side?s.right:s.left;
    if(done){if(target>=h.nodes||v.edge_seen[2*id+side])reject(v,29);}
    else if(target>=h.states||e.arena.states[target].phase==free_phase||v.edge_seen[2*id+side]!=1)reject(v,30);}
   if(!q.pending[id]){if(v.finish_seen[id]!=1||!(flags&detail::finish_once))reject(v,31);}
   else if(v.finish_seen[id]||(flags&detail::finish_once))reject(v,32);
  }else{
   if(s.node>=h.nodes||q.pending[id]||v.ready_seen[id]||!(flags&detail::finish_once)||v.edge_seen[2*id]||v.edge_seen[2*id+1])reject(v,33);
   if(flags&detail::cached_once){if(v.cache_seen[id]!=1||v.finish_seen[id]||q.wait_head[id]!=none)reject(v,34);}
   else if(v.cache_seen[id]||(!v.finish_seen[id]&&(id!=c.root_state||!h.complete)))reject(v,35);
  }
 }
 for(u64 id=u64(blockIdx.x)*blockDim.x+threadIdx.x;id<h.nodes;id+=u64(blockDim.x)*gridDim.x){
  const auto n=e.arena.nodes[id];if(v.node_seen[id]!=1){reject(v,36);continue;}
  if(n.feature==-1){if(n.payload>=e.source.classes)reject(v,37);}
  else{const auto coordinate=n.feature<-1?-std::int64_t(n.feature)-2:std::int64_t(n.feature);
   if(coordinate<0||u64(coordinate)>=e.source.features||isnan(__uint_as_float(n.payload))||n.left>=id||n.right>=id)reject(v,38);}
 }
}
inline void validate_gpu(EngineView e,QueueStorage&q,const Buffer<Control>&control,Budget&budget,const Status&h){
 Buffer<u32> state_seen(budget,h.states),node_seen(budget,h.nodes),ready_seen(budget,h.states),finish_seen(budget,h.states),cache_seen(budget,h.states),edge_seen(budget,multiply(h.states,2)),free_seen(budget,h.states),error(budget,1);
 state_seen.zero();node_seen.zero();ready_seen.zero();finish_seen.zero();cache_seen.zero();edge_seen.zero();free_seen.zero();error.zero();
 Validation v{state_seen.data,node_seen.data,ready_seen.data,finish_seen.data,cache_seen.data,edge_seen.data,free_seen.data,error.data};
 const auto blocks=u32(std::min<u64>(1024,std::max<u64>(1,(std::max({h.states,h.nodes,u64(e.arena.state_buckets),u64(e.arena.node_buckets)})+127)/128)));
 check_tables<<<blocks,128>>>(e,v);check_queue<<<blocks,128>>>(e,q.view(),control.data,v);check_free<<<1,1>>>(e,control.data,v);synchronize();
 check_payload<<<blocks,128>>>(e,q.view(),control.data,v);synchronize();const auto code=error.download(1).front();
 if(code)throw std::runtime_error("checkpoint restored GPU structural validation failed: "+std::to_string(code));
}
inline RestoreReceipt restore(const std::filesystem::path&path,const cp::Json&identity,EngineView&e,
 std::unique_ptr<StateStorage>&states,std::unique_ptr<NodeStorage>&nodes,std::unique_ptr<QueueStorage>&queue,
 Buffer<Control>&control,Buffer<Status>&status_owner,Budget&budget,u64 max_states,u64 max_nodes){
 cp::Options options;options.maximum_payload_bytes=budget.limit;cp::Reader reader(path,identity,options);const auto&m=reader.metadata();
 require(m.at("format")=="adaptive-frontier-checkpoint-1","checkpoint frontier metadata format");
 auto number=[&](const char*key){return cp::detail::integer(m.at(key),"checkpoint frontier metadata integer");};
 for(auto pair:{std::pair<const char*,u64>{"features",e.source.features},{"classes",e.source.classes},{"source_nodes",e.source.nodes},{"trees",e.source.trees},{"numeric_features",e.domain.numeric_features},{"mask_words",e.domain.mask_words},{"support_words",e.support_words},{"state_bytes",sizeof(State)},{"node_bytes",sizeof(Node)},{"status_bytes",sizeof(Status)},{"control_bytes",sizeof(Control)}})
  require(number(pair.first)==pair.second,"checkpoint source/domain/ABI dimensions differ");
 require(m.at("native_gate_authority_serialized").is_boolean()&&!m.at("native_gate_authority_serialized").get<bool>()&&number("legacy_stack_depth")==0,"checkpoint invalid authority/stack metadata");
 const auto sc=number("state_capacity"),nc=number("node_capacity"),H=number("state_highwater"),J=number("nodes");
 require(sc&&nc&&sc<=max_states&&nc<=max_nodes&&sc<=0x3fffffffu&&nc<=0x3fffffffu&&H&&H<=sc&&J<=nc,"checkpoint saved capacity refusal");
 require(number("state_buckets")==buckets(u32(sc))&&number("node_buckets")==buckets(u32(nc)),"checkpoint hash table extent differs");
 Status expected{};expected.states=H;expected.nodes=J;Control saved_control{};
 saved_control.ready_head=number("ready_head");saved_control.ready_count=number("ready_count");
 saved_control.finish_head=number("finish_head");saved_control.finish_tail=number("finish_tail");
 saved_control.cache_head=number("cache_head");saved_control.cache_tail=number("cache_tail");
 require(saved_control.ready_head<sc&&saved_control.ready_count<=sc&&saved_control.finish_head<=saved_control.finish_tail&&saved_control.finish_tail-saved_control.finish_head<=sc&&saved_control.cache_head<=saved_control.cache_tail&&saved_control.cache_tail-saved_control.cache_head<=sc,"checkpoint metadata ring extent");
 auto fresh_states=std::make_unique<StateStorage>(budget,u32(sc),e.domain.numeric_features,e.source.classes,e.source.trees,e.domain.mask_words);
 auto fresh_nodes=std::make_unique<NodeStorage>(budget,u32(nc));auto fresh_queue=std::make_unique<QueueStorage>(budget,u32(sc));
 Buffer<Control>fresh_control(budget,1);fresh_control.zero();Buffer<Status>fresh_status(budget,1);fresh_status.zero();
 auto staged=e;bind(staged,*fresh_states,*fresh_nodes);staged.status=fresh_status.data;
 const auto wanted=sections(staged,*fresh_states,*fresh_nodes,*fresh_queue,fresh_control,expected,saved_control);
 require(wanted.size()==reader.layout().size(),"checkpoint section schema count differs");std::vector<cp::MutableSection>targets;
 for(std::size_t i=0;i<wanted.size();++i){const auto&w=wanted[i];const auto&s=reader.layout()[i];
  require(w.name==s.name&&w.bytes==s.bytes&&w.sparse==s.sparse,"checkpoint section schema/extent differs");
  auto ranges=w.ranges;if(!w.sparse&&w.bytes)ranges.push_back({0,w.bytes});require(ranges.size()==s.ranges.size(),"checkpoint sparse ring schema differs");
  for(std::size_t r=0;r<ranges.size();++r)require(ranges[r].offset==s.ranges[r].offset&&ranges[r].bytes==s.ranges[r].bytes,"checkpoint sparse range differs");
  targets.push_back({w.name,const_cast<void*>(w.data),w.bytes});}
 auto receipt=reader.restore_exact(targets);const auto h=fresh_status.download(1).front();const auto c=fresh_control.download(1).front();
 require(h.states==H&&h.nodes==J&&c.ready_head==saved_control.ready_head&&c.ready_count==saved_control.ready_count&&c.finish_head==saved_control.finish_head&&c.finish_tail==saved_control.finish_tail&&c.cache_head==saved_control.cache_head&&c.cache_tail==saved_control.cache_tail,"checkpoint restored scalar metadata differs");
 boundary(h,c,u32(sc),u32(nc));validate_gpu(staged,*fresh_queue,fresh_control,budget,h);
 RestoreReceipt result{m,std::move(receipt)};
 // All verification/allocations/copies precede these noexcept owner swaps.
 states.swap(fresh_states);nodes.swap(fresh_nodes);queue.swap(fresh_queue);control.swap(fresh_control);status_owner.swap(fresh_status);
 bind(e,*states,*nodes);e.status=status_owner.data;return result;
}
} // namespace class_conversion_adaptive::frontier::checkpoint_io