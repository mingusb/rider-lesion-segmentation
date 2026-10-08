// Generic integration oracle: original CLSTREE1 versus the shared class API.
// All numerical input conversion and model prediction execute on CUDA.
#include "class_tree_adapter.hpp"
#include "class_runtime.hpp"
#include "class_io.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

namespace {
using U=std::uint64_t;
using W=std::uint32_t;
using J=dpnative::json;
void need(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
void ck(cudaError_t e,const char* m){if(e!=cudaSuccess)throw std::runtime_error(std::string(m)+": "+cudaGetErrorString(e));}
void done(){ck(cudaGetLastError(),"kernel launch");ck(cudaDeviceSynchronize(),"synchronize");}
template<class T> struct Buffer {
  T* p=nullptr;U n;
  explicit Buffer(U count):n(count){need(n<=SIZE_MAX/sizeof(T),"device extent overflow");if(n)ck(cudaMalloc(&p,n*sizeof(T)),"allocate comparison buffer");}
  ~Buffer(){if(p)cudaFree(p);}
  Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
};
int blocks(U n){return int(std::min<U>((n+255)/256,65535));}
__device__ U word(const unsigned char* b,U at,unsigned n){U v=0;for(unsigned j=0;j<n;++j)v|=U(b[at+j])<<(8*j);return v;}
__global__ void exact_u32_input(const W* x,float* out,U count,W* bad){
  for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=U(blockDim.x)*gridDim.x){
    W v=x[i];if(v>16777216U)atomicExch(bad,1U);out[i]=float(v);
  }
}
__global__ void finite_or_NaN_input(const float* x,U count,W* bad){
  for(U i=U(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=U(blockDim.x)*gridDim.x)
    if(isinf(x[i]))atomicExch(bad,2U);
}
__global__ void original_classes(const unsigned char* b,W count,W F,W K,
                                 const float* x,U rows,W* out,W* bad){
  for(U r=U(blockIdx.x)*blockDim.x+threadIdx.x;r<rows;r+=U(blockDim.x)*gridDim.x){
    U id=0;W answer=UINT32_MAX;
    for(W step=0;step<count;++step){
      if(id>=count){atomicExch(bad,3U);break;}
      std::int64_t label=static_cast<std::int64_t>(word(b,40+25*U(count)+8*id,8));
      if(label>=0){if(U(label)>=K)atomicExch(bad,4U);else answer=W(label);break;}
      std::int32_t f=static_cast<std::int32_t>(W(word(b,40+4*id,4)));
      W cut=W(word(b,40+4*U(count)+4*id,4));
      W missing=W(word(b,40+8*U(count)+id,1));
      U left=word(b,40+9*U(count)+8*id,8),right=word(b,40+17*U(count)+8*id,8);
      if(label!=-1||f<0||W(f)>=F||missing>1||isnan(__uint_as_float(cut))||left>=count||right>=count||left==right){atomicExch(bad,5U);break;}
      float value=x[r*U(F)+W(f)];id=(isnan(value)?bool(missing):value<__uint_as_float(cut))?left:right;
    }
    if(answer==UINT32_MAX)atomicExch(bad,6U);out[r]=answer;
  }
}
std::string read(const dpnative::fs::path& p,U bound){auto n=dpnative::fs::file_size(p);need(n<=bound,"comparison file exceeds explicit byte bound");auto b=dpnative::read_text(p);need(b.size()==n,"comparison file changed during read");return b;}
W status(const Buffer<W>& b){W v=0;ck(cudaMemcpy(&v,b.p,sizeof(v),cudaMemcpyDeviceToHost),"read comparison status");return v;}
double seconds(std::chrono::steady_clock::time_point t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
}

int main(int argc,char** argv){try{
  need(argc==9&&std::string(argv[1])=="run","usage: class_tree_runtime_compare run INPUT_CONTRACT CONTRACT_SHA CANONICAL CANONICAL_SHA ORIGINAL ORIGINAL_SHA FRESH_OUTPUT");
  auto contract_bytes=read(argv[2],16*1024*1024);
  need(dpnative::sha256(contract_bytes)==argv[3],"input contract SHA256 differs");
  auto contract=J::parse(contract_bytes);
  need(contract.at("complete").get<bool>()&&contract.at("original_runtime_sha256").get<std::string>()==argv[7],"input contract model identity differs");
  auto original=read(argv[6],40+33*U(1000000));
  auto adapted=class_tree_adapter::adapt(original,argv[7],{1000000});
  auto canonical=read(argv[4],64+16*U(1000000));
  need(canonical==adapted.canonical_bytes&&dpnative::sha256(canonical)==argv[5],"canonical transport/model identity differs");
  U rows=contract.at("rows").get<U>(),F=contract.at("features").get<U>();
  need(rows>0&&rows<=UINT32_MAX&&F==adapted.features&&rows<=UINT64_MAX/F,"comparison row/feature extent differs");
  U elements=rows*F;need(elements<=268435456,"comparison input exceeds explicit 1GiB payload bound");
  auto input=read(contract.at("payload_path").get<std::string>(),4*elements);
  need(input.size()==4*elements&&dpnative::sha256(input)==contract.at("payload_sha256").get<std::string>(),"comparison input bytes/hash differ");
  std::string dtype=contract.at("stored_dtype").get<std::string>();need(dtype=="float32-le"||dtype=="uint32-le","unsupported declared raw input storage");
  need(!dpnative::fs::exists(argv[8]),"comparison output must be fresh");
  dpnative::fs::create_directories(argv[8]);dpnative::fs::path output=argv[8];
  auto total_start=std::chrono::steady_clock::now();
  Buffer<unsigned char> tree(original.size());Buffer<float> x(elements);
  Buffer<W> expected(rows),bad(1);ck(cudaMemset(bad.p,0,sizeof(W)),"clear comparison status");
  ck(cudaMemcpy(tree.p,original.data(),original.size(),cudaMemcpyHostToDevice),"upload original CLSTREE1");
  auto conversion_start=std::chrono::steady_clock::now();
  if(dtype=="uint32-le"){
    Buffer<W> raw(elements);ck(cudaMemcpy(raw.p,input.data(),input.size(),cudaMemcpyHostToDevice),"upload raw U32 features");
    exact_u32_input<<<blocks(elements),256>>>(raw.p,x.p,elements,bad.p);done();
  }else{ck(cudaMemcpy(x.p,input.data(),input.size(),cudaMemcpyHostToDevice),"upload raw FP32 features");}
  finite_or_NaN_input<<<blocks(elements),256>>>(x.p,elements,bad.p);done();
  need(status(bad)==0,"input conversion/domain check failed");double conversion_seconds=seconds(conversion_start);
  auto oracle_start=std::chrono::steady_clock::now();
  original_classes<<<blocks(rows),256>>>(tree.p,adapted.nodes,adapted.features,adapted.classes,x.p,rows,expected.p,bad.p);done();
  need(status(bad)==0,"legacy checked CUDA oracle failed");double oracle_seconds=seconds(oracle_start);
  auto load_start=std::chrono::steady_clock::now();
  auto runtime=class_runtime::Runtime::load(canonical,argv[5],argv[7]);double load_seconds=seconds(load_start);
  class_runtime::DenseBatch batch{x.p,elements,0,rows,F};
  auto verify_start=std::chrono::steady_clock::now();runtime->verify(batch,expected.p,rows,256);
  double verify_seconds=seconds(verify_start);
  auto compact=runtime->compact_bytes();dpnative::atomic_text(output/"model.clsg64b1",compact);
  std::string predicted(4*rows,'\0');ck(cudaMemcpy(predicted.data(),expected.p,predicted.size(),cudaMemcpyDeviceToHost),"download opaque legacy class words");
  dpnative::atomic_text(output/"legacy-predicted-classes.u32",predicted);
  const auto& m=runtime->metadata();
  J result={{"format","generic-CLSTREE1-shared-runtime-CUDA-comparison-1"},{"complete",true},{"passed",true},{"CUDA_executed",true},{"CPU_predictions_executed",false},{"dataset_truth_labels_read",false},{"TEST_inputs_or_labels_read",contract.value("TEST_inputs_or_labels_read",false)},{"rows",rows},{"features",m.features},{"classes",m.classes},{"nodes",m.nodes},{"root",m.root},{"predicate_count",m.predicates},{"original_sha256",argv[7]},{"canonical_sha256",argv[5]},{"input_contract_path",argv[2]},{"input_contract_sha256",argv[3]},{"input_payload_sha256",contract.at("payload_sha256")},{"stored_input_dtype",dtype},{"expected_class_source","checked CUDA evaluation of original CLSTREE1 bytes on the identical original feature matrix"},{"shared_checked_and_validated_canonical_compact_comparisons",4},{"class_disagreements",0},{"full_cross_layout_node_word_equality",true},{"compact_sha256",dpnative::sha256(compact)},{"legacy_predicted_classes_sha256",dpnative::sha256(predicted)},{"canonical_file_bytes",canonical.size()},{"compact_file_bytes",compact.size()},{"runtime_resident_device_bytes",m.device_bytes},{"comparison_extra_device_bytes_excluding_temporary_U32_upload",original.size()+4*elements+4*rows+4},{"maximum_temporary_U32_upload_bytes",dtype=="uint32-le"?4*elements:0},{"input_conversion_and_domain_check_seconds",conversion_seconds},{"legacy_checked_oracle_seconds",oracle_seconds},{"shared_load_validate_and_pack_seconds",load_seconds},{"shared_all4_verify_seconds",verify_seconds},{"total_comparison_seconds",seconds(total_start)},{"timing_scope","single integration qualification; host wall time including synchronization, not performance ranking or paired ABBA"},{"input_coordinate_contract",contract.value("model_coordinate_system",std::string("caller-declared original feature coordinates; preprocessing unchanged"))},{"preprocessing_obligations_removed",false},{"source_equivalence_proof_replayed",false},{"origin_domain_expanded",false}};
  dpnative::atomic_json(output/"result.json",result);std::cout<<result.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
