#pragma once
// Construction-only canonical storage. Import/export numerical validation is
// delegated to Runtime; no raw model traversal is implemented here.
#include "class_model_io.cuh"
#include "class_runtime.hpp"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
struct Node {int32_t feature;u32 payload,left,right;};
static_assert(sizeof(Node)==16);
struct Dag {Dev<Node>nodes;u32 root=0,built=0;};
template<class T>auto dp(T*p){return thrust::device_pointer_cast(p);}
std::string encode(const Dag&d,u32 F,u32 K,const std::string&sha){
 std::ostringstream o(std::ios::binary);o.write("CLSGDAG1",8);for(u32 v:{1u,F,K,d.root,u32(d.nodes.n),1u})write_scalar(o,v);
 if(sha.size()!=64)throw std::runtime_error("source hash length");for(size_t i=0;i<64;i+=2){unsigned v=std::stoul(sha.substr(i,2),nullptr,16);o.put(char(v));}
 auto n=d.nodes.get();o.write(reinterpret_cast<const char*>(n.data()),n.size()*sizeof(Node));return o.str();
}
Dag decode(const std::string&bytes,u32 F,u32 K,const std::string&sha){
 auto validated=class_runtime::Runtime::load(bytes,sha256(bytes),sha,{},{},class_runtime::Residency::canonical_only);const auto&m=validated->metadata();
 if(m.features!=F||m.classes!=K)throw std::runtime_error("construction import shape differs");Dag d;d.root=m.root;d.built=m.nodes;d.nodes=Dev<Node>(m.nodes);ck(cudaMemcpy(d.nodes.p,bytes.data()+64,u64(m.nodes)*16,cudaMemcpyHostToDevice),"construction import nodes");return d;
}
__global__ void collect_nodes(const Node*n,const u32*live,const u32*map,Node*out,u32 count){
 for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=u64(blockDim.x)*gridDim.x)if(live[i]){Node v=n[i];if(v.feature!=-1){v.left=map[v.left];v.right=map[v.right];}out[map[i]]=v;}
}
__global__ void compare_labels(const u32*a,const u32*b,u32 count,u32*bad){for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=u64(blockDim.x)*gridDim.x)if(a[i]!=b[i])atomicOr(bad,1u);}
void predict_fixture(const Dag&d,const float*x,u32 rows,u32 F,u32 K,u32*out){
 auto bytes=encode(d,F,K,std::string(64,'0'));auto runtime=class_runtime::Runtime::load(bytes,sha256(bytes),std::string(64,'0'));
 runtime->predict({x,u64(rows)*F,0,rows,F},{out,rows},class_runtime::Layout::canonical16,class_runtime::Traversal::validated);
}
