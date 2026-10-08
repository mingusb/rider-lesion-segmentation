// CUDA-only regression fixture for the shared runtime. The independent oracle
// below spells out this fixture's decision function without traversing nodes.
#include "class_runtime.hpp"
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <array>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using u32=std::uint32_t;using u64=std::uint64_t;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void finish(){check(cudaGetLastError());check(cudaDeviceSynchronize());}
std::string hash(const std::string&s){unsigned char d[EVP_MAX_MD_SIZE];unsigned n=0;if(!EVP_Digest(s.data(),s.size(),d,&n,EVP_sha256(),nullptr))throw std::runtime_error("fixture hash");std::ostringstream o;o<<std::hex<<std::setfill('0');for(unsigned i=0;i<n;++i)o<<std::setw(2)<<unsigned(d[i]);return o.str();}
void append(std::string&s,u32 w){s.append(reinterpret_cast<const char*>(&w),4);}
std::string model(u32 F){std::string out="CLSGDAG1";for(u32 w:{1u,F,3u,6u,7u,1u})append(out,w);out.append(32,'\0');
  for(auto node:std::vector<std::array<u32,4>>{{UINT32_MAX,0,0,0},{UINT32_MAX,1,0,0},{UINT32_MAX,2,0,0},{0,0x80000000u,0,1},{u32(-2),0xff7fffffu,2,3},{F-1,1,4,1},{u32(-2-std::int32_t(F-1)),0x7f7fffffu,5,2}})for(u32 w:node)append(out,w);return out;}
__global__ void probes(float*x,u32 F,u64 rows,u64 stride){const u32 words[]={0xff7fffffu,0xbf800000u,0x80000001u,0x80000000u,0u,1u,0x3f800000u,0x7f7fffffu,0x7fc00001u,0xffc12345u};for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*stride;i+=u64(blockDim.x)*gridDim.x){u64 row=i/stride,f=i%stride;x[i]=f==0?__uint_as_float(words[row%10]):f==F-1?__uint_as_float(words[(row/10)%10]):0.f;}}
__global__ void independent(const float*x,u32 F,u64 first,u64 rows,u64 stride,u32*y){for(u64 r=u64(blockIdx.x)*blockDim.x+threadIdx.x;r<rows;r+=u64(blockDim.x)*gridDim.x){const float*z=x+(first+r)*stride;float b=z[F-1],a=z[0];u32 out;
 if(isnan(b))out=1;
 else if(b>=__uint_as_float(0x7f7fffffu))out=2;
 else if(b>=__uint_as_float(1u))out=1;
 else if(isnan(a)||a<__uint_as_float(0xff7fffffu))out=2;
 else out=a<0.f?0:1;y[r]=out;}}
struct Device{void*p=nullptr;explicit Device(u64 bytes){check(cudaMalloc(&p,bytes));}~Device(){if(p)cudaFree(p);}};
}
nlohmann::json class_runtime_fixture(){
 nlohmann::json cases=nlohmann::json::array();u32 rejected=0;
 for(auto spec:std::vector<std::array<u32,3>>{{2,257,0},{11,65537,7}}){const u32 F=spec[0],rows=spec[1],first=spec[2],stride=F+2,total=first+rows;
  auto bytes=model(F);auto runtime=class_runtime::Runtime::load(bytes,hash(bytes),std::string(64,'0'));auto compact=runtime->compact_bytes();auto loaded=class_runtime::Runtime::load(bytes,hash(bytes),std::string(64,'0'),compact,hash(compact));
  Device x(u64(total)*stride*4),expected(u64(rows)*4),output(u64(rows)*4);probes<<<256,256>>>(static_cast<float*>(x.p),F,total,stride);independent<<<256,256>>>(static_cast<const float*>(x.p),F,first,rows,stride,static_cast<u32*>(expected.p));finish();
  class_runtime::DenseBatch batch{static_cast<const float*>(x.p),u64(total)*stride,first,rows,stride};
  for(u32 block:{64u,128u,256u})loaded->verify(batch,static_cast<const u32*>(expected.p),rows,block);
  cases.push_back({{"features",F},{"classes",3},{"rows",rows},{"first_row",first},{"stride",stride},{"four_variants_three_blocks",true},{"canonical_sha256",runtime->metadata().canonical_sha256},{"compact_sha256",runtime->metadata().compact_sha256}});
  auto refuse=[&](auto&&call){bool no=false;try{call();}catch(const std::exception&){no=true;}if(!no)throw std::runtime_error("fixture malformed case accepted");++rejected;};
  for(u32 kind=0;kind<6;++kind){auto bad=bytes;auto set=[&](u32 at,u32 w){std::memcpy(bad.data()+at,&w,4);};if(kind==0)set(64+6*16+8,6);if(kind==1)set(64+6*16+12,UINT32_MAX);if(kind==2)set(68,3);if(kind==3)set(64+6*16,F);if(kind==4)set(64+6*16+4,0x7f800000u);if(kind==5)set(20,7);refuse([&]{auto q=class_runtime::Runtime::load(bad,hash(bad),std::string(64,'0'),compact,hash(compact));});}
  for(u32 kind=0;kind<4;++kind){auto bad=compact;auto set=[&](u32 at,u32 w){std::memcpy(bad.data()+at,&w,4);};if(kind==0)set(8,2);if(kind==1)set(32+20,7);if(kind==2)set(96+4,UINT32_MAX);if(kind==3)bad.push_back('\0');refuse([&]{auto q=class_runtime::Runtime::load(bytes,hash(bytes),std::string(64,'0'),bad,hash(bad));});}
  for(auto residency:{class_runtime::Residency::canonical_only,class_runtime::Residency::compact_only}) {
    auto selected=class_runtime::Runtime::load(bytes,hash(bytes),std::string(64,'0'),{},{},residency);selected->verify(batch,static_cast<const u32*>(expected.p),rows);
    if(selected->metadata().device_bytes>=runtime->metadata().device_bytes)throw std::runtime_error("fixture residency failed to release buffers");
  }
  for(auto keep:{class_runtime::Residency::canonical_only,class_runtime::Residency::compact_only}) {
    auto retained=class_runtime::Runtime::load(bytes,hash(bytes),std::string(64,'0'));
    const auto dual_bytes=retained->metadata().device_bytes;
    retained->retain(keep);retained->retain(keep);
    if(retained->metadata().device_bytes>=dual_bytes)throw std::runtime_error("fixture retain did not release layout");
    retained->verify(batch,static_cast<const u32*>(expected.p),rows);
    refuse([&]{retained->retain(class_runtime::Residency::dual);});
    refuse([&]{retained->retain(keep==class_runtime::Residency::canonical_only?class_runtime::Residency::compact_only:class_runtime::Residency::canonical_only);});
  }
  if(F==2) {
    auto wide=bytes;const u32 classes=UINT32_MAX;std::memcpy(wide.data()+16,&classes,4);
    for(auto residency:{class_runtime::Residency::canonical_only,class_runtime::Residency::dual}) {
      auto fallback=class_runtime::Runtime::load(wide,hash(wide),std::string(64,'0'),{},{},residency);
      if(!fallback->metadata().canonical_resident||fallback->metadata().compact_resident)throw std::runtime_error("fixture canonical fallback unavailable");
      fallback->verify(batch,static_cast<const u32*>(expected.p),rows);
    }
    refuse([&]{auto unsupported=class_runtime::Runtime::load(wide,hash(wide),std::string(64,'0'),{},{},class_runtime::Residency::compact_only);});
  }
  auto test=[&](class_runtime::DenseBatch q,class_runtime::ClassOutput y){loaded->predict(q,y,class_runtime::Layout::compact8,class_runtime::Traversal::validated);};
  auto invalid=batch;invalid.rows=0;refuse([&]{test(invalid,{static_cast<u32*>(output.p),rows});});invalid=batch;invalid.first_row=UINT64_MAX;refuse([&]{test(invalid,{static_cast<u32*>(output.p),rows});});invalid=batch;invalid.elements++;refuse([&]{test(invalid,{static_cast<u32*>(output.p),rows});});refuse([&]{test(batch,{static_cast<u32*>(output.p),rows-1});});refuse([&]{test(batch,{static_cast<u32*>(x.p),rows});});invalid=batch;invalid.device_values=reinterpret_cast<const float*>(static_cast<const char*>(x.p)+1);refuse([&]{test(invalid,{static_cast<u32*>(output.p),rows});});
 }
 return {{"complete",true},{"numerical_execution","CUDA"},{"CPU_predictions",false},{"independent_explicit_decision_function",true},{"finite_signed_zero_subnormal_extrema_both_NaN_signs",true},{"retain_without_reload_qualified",true},{"malformed_graph_and_extent_rejections",rejected},{"cases",cases},{"class_mismatches",0}};
}