struct BypassStats {unsigned long long rewritten_edges,bypassed_nodes,left_edges,right_edges;u32 bad,max_prefix_nodes;};
struct InternStats {unsigned long long unique_leaves,unique_splits,duplicate_leaves,duplicate_splits,equal_child_collapses,total_hash_probes;u32 bad,max_hash_probes;};
// Every edge is rewritten in its own parent copy. Shared source children are immutable.
__global__ void bypass_prefixes(const Node*src,Node*dst,u32 count,BypassStats*s){
 for(u32 i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){Node p=src[i];
  if(p.feature!=-1){for(u32 branch=0;branch<2;++branch){u32 original=branch?p.right:p.left,child=original,hops=0;
    if(child>=i||child>=count){atomicAdd(&s->bad,1u);continue;}
    while(true){Node q=src[child];if(q.feature==-1||q.feature!=p.feature||q.payload!=p.payload)break;
      u32 next=branch?q.right:q.left;if(next>=child||next>=count||hops>=count){atomicAdd(&s->bad,1u);break;}child=next;++hops;
    }
    if(branch)p.right=child;else p.left=child;
    if(hops){atomicAdd(&s->rewritten_edges,1ull);atomicAdd(&s->bypassed_nodes,(unsigned long long)hops);atomicAdd(branch?&s->right_edges:&s->left_edges,1ull);atomicMax(&s->max_prefix_nodes,hops);}
  }}dst[i]=p;
 }
}
__device__ u64 integer_hash(Node v){u64 a=(u64(u32(v.feature))<<32)|v.payload,b=(u64(v.left)<<32)|v.right;a^=b+0x9e3779b97f4a7c15ull+(a<<6)+(a>>2);a^=a>>30;a*=0xbf58476d1ce4e5b9ull;a^=a>>27;a*=0x94d049bb133111ebull;return a^(a>>31);}
__device__ bool exact_node(Node a,Node b){return a.feature==b.feature&&a.payload==b.payload&&a.left==b.left&&a.right==b.right;}
// Ascending IDs make resolved children final before their parent. No predicate is evaluated.
__global__ void collapse_and_intern(const Node*src,Node*dst,u32 count,u32 K,u32*alias,u32*class_rep,u32*slots,u32 slot_count,InternStats*s){
 if(blockIdx.x||threadIdx.x)return;
 for(u32 i=0;i<count;++i){Node v=src[i];
  if(v.feature==-1){if(v.payload>=K||v.left||v.right){++s->bad;return;}u32 rep=class_rep[v.payload];if(rep==UINT32_MAX){class_rep[v.payload]=i;alias[i]=i;++s->unique_leaves;}else{alias[i]=rep;++s->duplicate_leaves;}dst[i]=v;continue;}
  if(v.left>=i||v.right>=i){++s->bad;return;}v.left=alias[v.left];v.right=alias[v.right];if(v.left>=i||v.right>=i){++s->bad;return;}
  if(v.left==v.right){alias[i]=v.left;dst[i]=v;++s->equal_child_collapses;continue;}
  u32 pos=u32(integer_hash(v))&(slot_count-1),probes=0;bool resolved=false;
  while(probes<slot_count){++probes;u32 rep=slots[pos];if(rep==UINT32_MAX){slots[pos]=i;alias[i]=i;dst[i]=v;++s->unique_splits;resolved=true;break;}
    if(rep>=i){++s->bad;return;}if(exact_node(dst[rep],v)){alias[i]=rep;dst[i]=v;++s->duplicate_splits;resolved=true;break;}pos=(pos+1)&(slot_count-1);
  }
  s->total_hash_probes+=probes;s->max_hash_probes=max(s->max_hash_probes,probes);if(!resolved){++s->bad;return;}
 }
}
// Generic compile_dag phases are unavailable for imported DAGs. A descending
// serial scan is safe for arbitrary valid older-ID edges; one parallel pass is not.
__global__ void mark_reachable_descending(const Node*n,u32 count,u32 root,u32*live,u32*bad){
 if(blockIdx.x||threadIdx.x)return;if(root>=count){++*bad;return;}live[root]=1;
 for(u32 end=count;end>0;--end){u32 i=end-1;if(!live[i])continue;Node v=n[i];if(v.feature!=-1){if(v.left>=i||v.right>=i||v.left==v.right){++*bad;return;}live[v.left]=live[v.right]=1;}}
}
template<class F>double gpu_ms(F&&f){cudaEvent_t a,b;ck(cudaEventCreate(&a),"event");ck(cudaEventCreate(&b),"event");ck(cudaEventRecord(a),"event");f();ck(cudaEventRecord(b),"event");ck(cudaEventSynchronize(b),"event");ck(cudaGetLastError(),"stage launch");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"event");ck(cudaEventDestroy(a),"event");ck(cudaEventDestroy(b),"event");return ms;}
struct Pass {Dag dag;J receipt;bool changed;};
Pass simplify_pass(const Dag&in,u32 K){
 const u32 n=u32(in.nodes.n);require(n>0&&n<=16777216&&K>=2&&K<=1024,"simplification capacity");Dev<Node>bypassed(n),interned(n);Dev<BypassStats>bs(1);bs.zero();
 double bypass_time=gpu_ms([&]{bypass_prefixes<<<blocks(n),256>>>(in.nodes.p,bypassed.p,n,bs.p);});auto b=bs.at(0);require(!b.bad&&b.left_edges+b.right_edges==b.rewritten_edges,"invalid adjacent prefix rewrite");
 u32 slot_count=1;while(u64(slot_count)<u64(n)*2)slot_count*=2;InternStats t{};u32 representative_root=0;double intern_time=0;
 {Dev<u32>alias(n),class_rep(K),slots(slot_count);Dev<InternStats>ts(1);ts.zero();intern_time=gpu_ms([&]{ck(cudaMemset(class_rep.p,0xff,K*4),"class slots");ck(cudaMemset(slots.p,0xff,u64(slot_count)*4),"node slots");collapse_and_intern<<<1,1>>>(bypassed.p,interned.p,n,K,alias.p,class_rep.p,slots.p,slot_count,ts.p);});t=ts.at(0);require(!t.bad,"invalid collapse/hashcons");representative_root=alias.at(in.root);}
 require(t.unique_leaves+t.unique_splits+t.duplicate_leaves+t.duplicate_splits+t.equal_child_collapses==n,"intern partition count differs");
 bypassed=Dev<Node>();Dev<u32>live(n),map(n),bad(1);bad.zero();double reachable_time=gpu_ms([&]{live.zero();mark_reachable_descending<<<1,1>>>(interned.p,n,representative_root,live.p,bad.p);});require(!bad.at(0),"invalid root reachability");
 double scan_time=gpu_ms([&]{thrust::exclusive_scan(thrust::device,dp(live.p),dp(live.p+n),dp(map.p));});u32 kept=map.at(n-1)+live.at(n-1);require(kept>0&&kept<=t.unique_leaves+t.unique_splits,"invalid reachable representative count");
 Dag out;out.nodes=Dev<Node>(kept);out.root=map.at(representative_root);out.built=n;double collect_time=gpu_ms([&]{collect_nodes<<<blocks(n),256>>>(interned.p,live.p,map.p,out.nodes.p,n);});
 bool changed=b.rewritten_edges||t.duplicate_leaves||t.duplicate_splits||t.equal_child_collapses||kept!=n;
 return {std::move(out),{{"raw_nodes",n},{"nodes_after_adjacent_edge_rewrites",n},{"rewritten_parent_edges",b.rewritten_edges},{"rewritten_left_edges",b.left_edges},{"rewritten_right_edges",b.right_edges},{"same_predicate_prefix_nodes_bypassed_across_parent_edges",b.bypassed_nodes},{"maximum_complete_prefix_nodes",b.max_prefix_nodes},{"prefix_walk_truncated",false},{"unique_representatives_after_dedup",t.unique_leaves+t.unique_splits},{"unique_leaves",t.unique_leaves},{"unique_splits",t.unique_splits},{"duplicate_leaves_removed",t.duplicate_leaves},{"duplicate_splits_removed",t.duplicate_splits},{"equal_child_parents_collapsed",t.equal_child_collapses},{"reachable_nodes",kept},{"unreachable_unique_representatives_removed",t.unique_leaves+t.unique_splits-kept},{"hash_slot_count",slot_count},{"hash_slot_bytes",u64(slot_count)*4},{"hash_probes",t.total_hash_probes},{"maximum_hash_probes",t.max_hash_probes},{"changed",changed},{"CUDA_event_ms",{{"adjacent_prefix_rewrite",bypass_time},{"equal_child_collapse_and_exact_hashcons",intern_time},{"root_reachability",reachable_time},{"exclusive_scan",scan_time},{"existing_collect_nodes",collect_time}}},{"counter_collection_and_metadata_host_only",true}},changed};
}
struct Simplified {Dag dag;J passes;bool fixed;};
Simplified simplify(const Dag&in,u32 K,u32 budget){require(budget>=1&&budget<=16,"outer pass budget");Dag owned_result;const Dag*current=&in;J passes=J::array();bool fixed=false;
 for(u32 p=0;p<budget;++p){auto out=simplify_pass(*current,K);out.receipt["pass"]=p+1;passes.push_back(out.receipt);bool changed=out.changed;owned_result=std::move(out.dag);current=&owned_result;if(!changed){fixed=true;break;}}
 return {std::move(owned_result),passes,fixed};
}
__global__ void fixture_inputs(float*x,u32 rows,u32 F,const u32*words,u32 nw){for(u32 r=blockIdx.x*blockDim.x+threadIdx.x;r<rows;r+=blockDim.x*gridDim.x)for(u32 f=0;f<F;++f)x[u64(r)*F+f]=__uint_as_float(words[(f?r/nw:r)%nw]);}
