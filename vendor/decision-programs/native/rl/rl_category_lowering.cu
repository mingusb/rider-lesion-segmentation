// This exact frozen source supplies unchanged default lowering and universal
// lower_audit. Link this object IN PLACE OF apply.o, never alongside apply.o.
#include "class_rank_gpu_class_apply.cu"
#include "rl_category_lowering.hpp"
#include <cstddef>

namespace rl_category_lowering {
namespace {
namespace ar = rank_gpu_online_cache;
using B = ca::sd::b::Box;

__global__ void validate_order(const unsigned* order, ca::LowerMeta* meta) {
  if (blockIdx.x || threadIdx.x) return;
  U seen = 0;
  for (unsigned k=0;k<44;++k) {
    const unsigned bit=order[k];
    if (bit>=44 || (k<4 ? bit>=4 : bit<4) || (seen&(U(1)<<bit))) {
      meta->bad=101; return;
    }
    seen |= U(1)<<bit;
  }
  if (seen!=((U(1)<<44)-1)) meta->bad=102;
}

__global__ void same_order(const unsigned* original,const unsigned* snapshot,
                          ca::LowerMeta* meta) {
  unsigned k=blockIdx.x*blockDim.x+threadIdx.x;
  if (k<44 && original[k]!=snapshot[k]) atomicCAS(&meta->bad,0,103);
}

__global__ void ordered_build(B domain,const ca::Node* nodes,U nn,
                             const ca::Arc* arcs,const unsigned* order,U version,
                             ar::Draft* drafts,U* starts,U* mapping,
                             Trace* traces,ca::LowerMeta* meta) {
  if (blockIdx.x || threadIdx.x) return;
  U next=0;
  for (U id=0;id<nn;++id) {
    const auto& n=nodes[id];
    Trace tr{};tr.source_node=id;tr.policy_version=version;
    for (unsigned k=0;k<40;++k) tr.test_order[k]=UINT_MAX;
    starts[id]=next;
    if (!n.kind) {
      ar::Draft d{};d.label=int(n.score_bits);
      drafts[next]=d;mapping[id]=next++;traces[id]=tr;continue;
    }
    U fallback=n.arc_count-1;
    if (n.dimension>=10) {
      int largest=-1;
      for (U j=0;j<n.arc_count;++j) {
        const int size=__popcll(arcs[n.first_arc+j].allowed);
        if (size>largest) {largest=size;fallback=j;}
      }
    }
    U acc=mapping[arcs[n.first_arc+fallback].child];
    if (n.dimension<10) {
      // Preserve the qualified numeric implementation and ordering verbatim.
      for (U j=n.arc_count;j-->0;) {
        if (j==fallback) continue;
        const auto& a=arcs[n.first_arc+j];
        ar::Draft d{};d.kind=0;d.feature=int(n.dimension);
        d.cut_bits=__float_as_uint(float(a.hi+1));
        d.left={mapping[a.child],2};d.right={acc,2};
        drafts[next]=d;acc=next++;
      }
    } else {
      tr.dimension=n.dimension;tr.fallback_arc=fallback;
      const U group=n.dimension==10?ca::sd::b::wilderness:ca::sd::b::soil;
      tr.selected_mask=(domain.allowed&group)&~arcs[n.first_arc+fallback].allowed;
      const unsigned begin=n.dimension==10?0:4,end=n.dimension==10?4:44;
      for (unsigned k=begin;k<end;++k)
        if (tr.selected_mask&(U(1)<<order[k])) tr.test_order[tr.count++]=order[k];
      for (unsigned k=end;k-->begin;) {
        const unsigned bit=order[k];
        if (!(tr.selected_mask&(U(1)<<bit))) continue;
        U found=ca::none,hits=0;
        for (U j=0;j<n.arc_count;++j) {
          if (arcs[n.first_arc+j].allowed&(U(1)<<bit)) {found=j;++hits;}
        }
        if (hits!=1 || found==fallback) {meta->bad=104;return;}
        ar::Draft d{};d.kind=0;d.feature=10+int(bit);d.cut_bits=0x3f000000u;
        d.left={acc,2};d.right={mapping[arcs[n.first_arc+found].child],2};
        drafts[next]=d;acc=next++;
      }
    }
    mapping[id]=acc;traces[id]=tr;
  }
  if (next!=meta->count) meta->bad=105;
}

__global__ void audit_trace(B domain,const ca::Node* nodes,U nn,const ca::Arc* arcs,
                            const unsigned* order,U version,const ar::Draft* drafts,
                            const U* starts,const U* mapping,const Trace* traces,
                            ca::LowerMeta* meta) {
  U id=U(blockIdx.x)*blockDim.x+threadIdx.x;
  if (id>=nn) return;
  const auto& n=nodes[id];const auto& tr=traces[id];
  if (tr.source_node!=id || tr.policy_version!=version) {
    atomicCAS(&meta->bad,0,106);return;
  }
  if (!n.kind || n.dimension<10) {
    if (tr.dimension!=12 || tr.fallback_arc!=ca::none || tr.selected_mask || tr.count)
      atomicCAS(&meta->bad,0,107);
    for (unsigned k=0;k<40;++k) if (tr.test_order[k]!=UINT_MAX)
      atomicCAS(&meta->bad,0,108);
    return;
  }
  // Independent fallback selection: first maximum remains fixed even on ties.
  U fallback=0;int largest=-1;
  for (U j=0;j<n.arc_count;++j) {
    int size=__popcll(arcs[n.first_arc+j].allowed);
    if (size>largest) {largest=size;fallback=j;}
  }
  const U group=n.dimension==10?ca::sd::b::wilderness:ca::sd::b::soil;
  U selected=(domain.allowed&group)&~arcs[n.first_arc+fallback].allowed;
  if (tr.dimension!=n.dimension || tr.fallback_arc!=fallback ||
      tr.selected_mask!=selected || tr.count!=unsigned(__popcll(selected))) {
    atomicCAS(&meta->bad,0,109);return;
  }
  unsigned at_order=0;U at=mapping[id],seen=0;
  const unsigned begin=n.dimension==10?0:4,end=n.dimension==10?4:44;
  for (unsigned k=begin;k<end;++k) {
    unsigned bit=order[k];if (!(selected&(U(1)<<bit))) continue;
    if (at_order>=tr.count || tr.test_order[at_order++]!=bit || at<starts[id]) {
      atomicCAS(&meta->bad,0,110);return;
    }
    const auto& d=drafts[at];
    if (d.feature!=10+int(bit)) {atomicCAS(&meta->bad,0,111);return;}
    seen|=U(1)<<bit;at=d.left.id;
  }
  if (seen!=selected || at!=mapping[arcs[n.first_arc+fallback].child] ||
      at_order!=tr.count) {atomicCAS(&meta->bad,0,112);return;}
  for (unsigned k=tr.count;k<40;++k) if (tr.test_order[k]!=UINT_MAX)
    atomicCAS(&meta->bad,0,113);
}
}

U planned_device_bytes(U nodes,U arcs,U drafts) {
  U bytes=ca::mul(nodes,sizeof(ca::Node)+2*sizeof(U)+sizeof(Trace));
  bytes=ca::add(bytes,ca::mul(arcs,sizeof(ca::Arc)));
  bytes=ca::add(bytes,sizeof(ca::Meta)+sizeof(ca::LowerMeta)+44*sizeof(unsigned));
  return ca::add(bytes,ca::mul(drafts,sizeof(ar::Draft)));
}

Result lower_binary(const ca::Result& r,DeviceOrder action,U maximum_drafts,
                    U cap,const ca::Stop& stop,int device) {
  Result out;out.policy_version=action.policy_version;
  if (!action.bits) {
    out.lowered=ca::lower_binary(r,maximum_drafts,cap,stop,device);
    return out;
  }
  ca::need(r.complete&&r.local_semantics_audited&&r.root<r.nodes.size()&&
           !r.source_binding.empty()&&!r.domain_binding.empty()&&maximum_drafts,
           "ordered lowering input/cap invalid");
  if (stop&&stop()) {out.lowered.reason="cancelled_before_lowering";return out;}
  ca::cu(cudaSetDevice(device));
  cudaPointerAttributes attributes{};
  ca::cu(cudaPointerGetAttributes(&attributes,action.bits));
  ca::need(attributes.type==cudaMemoryTypeDevice&&attributes.device==device,
           "category order must be borrowed CUDA device storage on selected device");
  ca::Budget budget{cap};
  if (planned_device_bytes(r.nodes.size(),r.arcs.size(),0)>cap) {
    out.lowered.reason="binary_device_cap";return out;
  }
  ca::Buf<ca::Node> nodes(budget,r.nodes.size());
  ca::Buf<ca::Arc> arcs(budget,r.arcs.size());
  ca::Buf<U> starts(budget,r.nodes.size()),mapping(budget,r.nodes.size());
  ca::Buf<ca::Meta> validation(budget,1);
  ca::Buf<ca::LowerMeta> meta(budget,1);
  ca::Buf<Trace> trace(budget,r.nodes.size());
  ca::Buf<unsigned> order(budget,44);
  nodes.put(r.nodes.data());arcs.put(r.arcs.data());
  ca::cu(cudaMemcpy(order.p,action.bits,44*sizeof(unsigned),cudaMemcpyDeviceToDevice));
  validate_order<<<1,1>>>(order.p,meta.p);
  ca::check_class_nodes<<<ca::blocks(r.nodes.size()),128>>>(
      r.domain,nodes.p,r.nodes.size(),arcs.p,r.arcs.size(),validation.p);
  ca::cu(cudaGetLastError());
  ca::need(!validation.get(1)[0].bad,"ordered lowering class input invalid");
  auto m=meta.get(1)[0];ca::need(!m.bad,"category order permutation invalid");
  ca::lower_count<<<1,1>>>(r.domain,nodes.p,r.nodes.size(),arcs.p,maximum_drafts,meta.p);
  ca::cu(cudaGetLastError());m=meta.get(1)[0];
  out.lowered.CUDA_executed=true;out.lowered.owned_device_peak_bytes=budget.peak;
  if (m.capped) {out.lowered.reason="binary_draft_cap";return out;}
  if (ca::mul(m.count,sizeof(ar::Draft))>cap-budget.used) {
    out.lowered.reason="binary_device_cap";return out;
  }
  if (stop&&stop()) {out.lowered.reason="cancelled_after_lowering_count";return out;}
  ca::Buf<ar::Draft> drafts(budget,m.count);
  ordered_build<<<1,1>>>(r.domain,nodes.p,r.nodes.size(),arcs.p,order.p,
                        action.policy_version,drafts.p,starts.p,mapping.p,trace.p,meta.p);
  ca::cu(cudaGetLastError());m=meta.get(1)[0];
  ca::need(!m.bad,"ordered binary lowering generation failed");
  // The original qualified universal semantics audit is unchanged.
  ca::lower_audit<<<ca::blocks(r.nodes.size()),128>>>(
      r.domain,nodes.p,r.nodes.size(),arcs.p,drafts.p,starts.p,mapping.p,meta.p);
  audit_trace<<<ca::blocks(r.nodes.size()),128>>>(
      r.domain,nodes.p,r.nodes.size(),arcs.p,order.p,action.policy_version,
      drafts.p,starts.p,mapping.p,trace.p,meta.p);
  same_order<<<1,64>>>(action.bits,order.p,meta.p);
  ca::cu(cudaGetLastError());m=meta.get(1)[0];
  if (m.bad) throw std::runtime_error("CUDA ordered lowering audit failed: "+std::to_string(m.bad));
  out.lowered.owned_device_peak_bytes=budget.peak;
  out.lowered.audited_nodes=m.nodes;out.lowered.audited_numeric_segments=m.numeric;
  out.lowered.audited_category_values=m.categories;
  ca::need(m.nodes==r.nodes.size(),"ordered lowering audit node coverage");
  if (stop&&stop()) {out.lowered.reason="cancelled_after_lowering_audit";return out;}
  auto ids=mapping.get(r.nodes.size());out.lowered.root=ids[r.root];
  out.lowered.drafts=drafts.get(m.count);out.traces=trace.get(r.nodes.size());
  auto selected=order.get(44);
  std::copy(selected.begin(),selected.end(),out.selected_order.begin());
  out.order_injected=true;out.trace_audited=true;out.lowered.complete=true;
  out.lowered.reason="GPU_universal_lowering_equal_with_audited_category_order";
  return out;
}
}
