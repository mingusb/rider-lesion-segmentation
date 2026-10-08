// Generic binary-array transport CLI. Dataset loading and preprocessing belong
// to callers; this entrypoint consumes already prepared FP32 rows/u32 labels.
#include "class_runtime.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <charconv>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

nlohmann::json class_runtime_fixture();

namespace {
using u32=std::uint32_t; using u64=std::uint64_t;
std::string read(const char* path) {
  std::ifstream in(path,std::ios::binary); if(!in)throw std::runtime_error("cannot read input");
  return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
void write_new(const char* path,const std::string& bytes) {
  std::ifstream existing(path,std::ios::binary); if(existing)throw std::runtime_error("output already exists");
  std::ofstream out(path,std::ios::binary); out.exceptions(std::ios::badbit|std::ios::failbit); out.write(bytes.data(),bytes.size()); out.flush();
}
u64 integer(const char* text) { u64 out=0; std::string s=text; auto parsed=std::from_chars(s.data(),s.data()+s.size(),out); if(parsed.ec!=std::errc{}||parsed.ptr!=s.data()+s.size()||s.empty())throw std::runtime_error("invalid unsigned integer");return out; }
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
struct Device { void* p=nullptr; explicit Device(const std::string& bytes) { if(bytes.empty())throw std::runtime_error("empty CUDA input"); check(cudaMalloc(&p,bytes.size()));check(cudaMemcpy(p,bytes.data(),bytes.size(),cudaMemcpyHostToDevice)); } ~Device(){if(p)cudaFree(p);} };
nlohmann::json metadata(const class_runtime::Metadata& m) {
  return {{"features",m.features},{"classes",m.classes},{"root",m.root},{"nodes",m.nodes},{"predicates",m.predicates},{"canonical_bytes",m.canonical_bytes},{"compact_bytes",m.compact_bytes},{"device_graph_bytes",m.device_bytes},{"source_sha256",m.source_sha256},{"canonical_sha256",m.canonical_sha256},{"compact_sha256",m.compact_sha256},{"whole_graph_GPU_validation",true},{"full_cross_layout_words_equal",m.cross_layout_validated},{"canonical_resident",m.canonical_resident},{"compact_resident",m.compact_resident}};
}
}
int main(int argc,char** argv) {
  try {
    if(argc==2 && std::string(argv[1])=="fixture") { std::cout<<class_runtime_fixture().dump(2)<<'\n';return 0; }
    if(argc==6 && std::string(argv[1])=="pack") {
      auto runtime=class_runtime::Runtime::load(read(argv[2]),argv[3],argv[4]); auto bytes=runtime->compact_bytes(); write_new(argv[5],bytes);
      auto result=metadata(runtime->metadata());result["complete"]=true; result["operation"]="pack";result["output"]=argv[5];std::cout<<result.dump(2)<<'\n';return 0;
    }
    if(argc==11 && std::string(argv[1])=="verify") {
      auto runtime=class_runtime::Runtime::load(read(argv[2]),argv[3],argv[4],read(argv[5]),argv[6]);
      const auto rows=integer(argv[8]),stride=integer(argv[9]); auto input=read(argv[7]),labels=read(argv[10]);
      if(!rows||stride<runtime->metadata().features||rows>UINT64_MAX/stride||rows*stride>SIZE_MAX/4||input.size()!=rows*stride*4||rows>SIZE_MAX/4||labels.size()!=rows*4)throw std::runtime_error("binary row/label file extent");
      Device x(input),y(labels); class_runtime::DenseBatch batch{static_cast<const float*>(x.p),u64(input.size()/4),0,rows,stride};
      for(u32 block:{64u,128u,256u})runtime->verify(batch,static_cast<const u32*>(y.p),rows,block);
      auto result=metadata(runtime->metadata());result["complete"]=true;result["operation"]="verify";result["rows"]=rows;result["row_stride"]=stride;result["class_mismatches"]=0;result["layouts"]=2;result["traversals"]=2;result["block_sizes"]={64,128,256};result["numerical_execution"]="CUDA";std::cout<<result.dump(2)<<'\n';return 0;
    }
    throw std::runtime_error("usage: class_runtime_cli pack CANONICAL CANONICAL_SHA SOURCE_SHA NEW_COMPACT | verify CANONICAL CANONICAL_SHA SOURCE_SHA COMPACT COMPACT_SHA FP32_ROWS ROW_COUNT ROW_STRIDE U32_LABELS");
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}