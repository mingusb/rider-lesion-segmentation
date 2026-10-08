// Public persisted-model transport/inspection frontend. All graph validation,
// prediction and exact route decisions use the single maintained Runtime.
#include "class_runtime.hpp"
#include "class_io.hpp"
#include "decision_programs_model_format.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
namespace {
namespace fs=std::filesystem;using J=nlohmann::json;using u32=std::uint32_t;using u64=std::uint64_t;
void need(bool ok,const std::string&why){if(!ok)throw std::runtime_error(why);}
void check(cudaError_t e,const char*where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
template<class T>struct Device{
  T*p=nullptr;u64 n;
  explicit Device(u64 count):n(count){need(count<=SIZE_MAX/sizeof(T),"device extent overflow");if(n)check(cudaMalloc(&p,n*sizeof(T)),"allocate");}
  ~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;Device&operator=(const Device&)=delete;
  void upload(const void*host){if(n)check(cudaMemcpy(p,host,n*sizeof(T),cudaMemcpyHostToDevice),"upload");}
  void zero(){if(n)check(cudaMemset(p,0,n*sizeof(T)),"initialize output padding");}
  std::vector<T>download()const{std::vector<T>out(n);if(n)check(cudaMemcpy(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"download");return out;}
};
u32 word(const std::string&b,std::size_t at){need(at<=b.size()&&b.size()-at>=4,"truncated canonical model");u32 value;std::memcpy(&value,b.data()+at,4);return value;}
std::string source_hash(const std::string&b){need(b.size()>=64,"truncated canonical header");std::ostringstream s;s<<std::hex<<std::setfill('0');for(std::size_t i=32;i<64;++i)s<<std::setw(2)<<unsigned(static_cast<unsigned char>(b[i]));return s.str();}
struct Model{
  std::string canonical,canonical_sha,compact,compact_sha,source;
  std::unique_ptr<class_runtime::Runtime> runtime;
  Model(const J&plan){
    canonical=dpnative::read_text(plan.at("canonical_path").get<std::string>());canonical_sha=plan.at("canonical_sha256");source=plan.value("source_sha256",source_hash(canonical));
    if(plan.contains("compact_path")){compact=dpnative::read_text(plan.at("compact_path").get<std::string>());compact_sha=plan.at("compact_sha256");}
    runtime=class_runtime::Runtime::load(canonical,canonical_sha,source,compact,compact_sha,class_runtime::Residency::dual);
  }
};
J metadata(const class_runtime::Metadata&m){return {{"features",m.features},{"classes",m.classes},{"root",m.root},{"nodes",m.nodes},{"predicates",m.predicates},{"canonical_bytes",m.canonical_bytes},{"compact_bytes",m.compact_bytes},{"retained_graph_device_bytes",m.device_bytes},{"source_sha256",m.source_sha256},{"canonical_sha256",m.canonical_sha256},{"compact_sha256",m.compact_sha256},{"canonical_resident",m.canonical_resident},{"compact_resident",m.compact_resident},{"cross_layout_validated",m.cross_layout_validated},{"input_representation","finite FP32 plus NaN; infinities excluded"},{"native_source_equivalence_scope","not established by this operation; consult the associated conversion receipt and its declared domain restrictions"},{"inference_output","class IDs"}};}
void fresh_write(const fs::path&path,const std::string&bytes){need(!fs::exists(path),"output already exists: "+path.string());dpnative::atomic_text(path,bytes);}
void emit(const J&result,const J&plan){auto bytes=result.dump(2)+"\n";if(plan.contains("out"))fresh_write(plan.at("out").get<std::string>(),bytes);std::cout<<bytes;}
__global__ void finite_input(const float*input,u64 rows,u64 stride,u32 F,u32*bad){for(u64 i=u64(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*u64(F);i+=u64(blockDim.x)*gridDim.x)if(isinf(input[(i/F)*stride+i%F]))atomicOr(bad,1u);}
struct Rows{
  std::string bytes,sha,transport;u64 count=0,stride=0;
  Rows(const J&plan,u32 F){
    if(plan.contains("input")){
      auto original=dpnative::read_text(plan.at("input").get<std::string>());sha=dpnative::sha256(original);auto document=J::parse(original);const auto&rows=document.at("rows");need(rows.is_array()&&!rows.empty(),"input requires a nonempty rows array");count=rows.size();stride=F;need(count<=SIZE_MAX/(u64(F)*4),"row extent overflow");std::vector<float>data;data.reserve(count*F);
      for(const auto&row:rows){need(row.is_array()&&row.size()==F,"JSON row feature count differs");for(const auto&v:row){need(v.is_null()||v.is_number(),"JSON features must be numbers or null (NaN)");data.push_back(v.is_null()?std::numeric_limits<float>::quiet_NaN():v.get<float>());}}
      bytes.assign(reinterpret_cast<const char*>(data.data()),data.size()*4);transport="host JSON-to-FP32 input transport; numerical model execution CUDA";
    }else{
      need(plan.contains("values")&&plan.contains("rows"),"prediction requires --input ROWS.json, --data INPUT.csv or --values FP32 --rows N");bytes=dpnative::read_text(plan.at("values").get<std::string>());sha=dpnative::sha256(bytes);count=plan.at("rows").get<u64>();stride=plan.value("row_stride",u64(F));need(count&&stride>=F&&count<=UINT64_MAX/stride&&count*stride<=SIZE_MAX/4&&bytes.size()==count*stride*4,"FP32 input extent differs");transport=plan.value("input_transport",std::string("little-endian FP32 byte transport"));
    }
  }
};
J equations(const Model&model){
  const auto&m=model.runtime->metadata();J nodes=J::array();
  for(u32 i=0;i<m.nodes;++i){std::size_t at=64+u64(i)*16;auto signed_feature=std::bit_cast<std::int32_t>(word(model.canonical,at));auto payload=word(model.canonical,at+4);J node={{"id",i}};
    if(signed_feature==-1){node["class"]=payload;node["equation"]="D"+std::to_string(i)+"(x) = "+std::to_string(payload);node["readable_equation"]=node["equation"];}
    else{
      auto feature=signed_feature>=0?u32(signed_feature):u32(-std::int64_t(signed_feature)-2);auto left=word(model.canonical,at+8),right=word(model.canonical,at+12);node.update({{"feature",feature},{"raw_threshold_bits",payload},{"missing_goes_left",signed_feature<0},{"left",left},{"right",right}});
      node["equation"]="D"+std::to_string(i)+"(x) = if (isnan(x["+std::to_string(feature)+"]) ? "+(signed_feature<0?"true":"false")+" : x["+std::to_string(feature)+"] < fp32_bits("+std::to_string(payload)+")) then D"+std::to_string(left)+"(x) else D"+std::to_string(right)+"(x)";
      node.update(decision_programs_display::threshold_fields(signed_feature,payload));
      node["readable_equation"]="D"+std::to_string(i)+"(x) = if "+decision_programs_display::condition(signed_feature,payload)+" then D"+std::to_string(left)+"(x) else D"+std::to_string(right)+"(x)";
    }nodes.push_back(std::move(node));
  }
  return {{"format","shared-hard-decision-equations-1"},{"exact_for_bound_runtime",true},{"new_source_equivalence_proved",false},{"CPU_predictions_executed",false},{"CUDA_graph_validation",true},{"model",metadata(m)},{"root_function","D"+std::to_string(m.root)},{"shared_nodes",nodes},{"comparison","strict FP32 less-than; raw threshold bits and stored NaN direction preserved"},{"fuzzy_model",nullptr}};
}
J prediction(Model&model,const J&plan,bool explain){
  const auto&m=model.runtime->metadata();Rows rows(plan,m.features);Device<float>input(rows.bytes.size()/4);input.upload(rows.bytes.data());Device<u32>invalid(1);check(cudaMemset(invalid.p,0,4),"clear");finite_input<<<std::min<u64>((rows.count*m.features+255)/256,65535),256>>>(input.p,rows.count,rows.stride,m.features,invalid.p);check(cudaGetLastError(),"input validation launch");check(cudaDeviceSynchronize(),"input validation");need(invalid.download()[0]==0,"input contains infinity outside the model domain");
  Device<u32>classes(rows.count);class_runtime::DenseBatch batch{input.p,input.n,0,rows.count,rows.stride};auto layout_name=plan.value("layout",std::string("canonical"));need(layout_name=="canonical"||layout_name=="compact","layout must be canonical or compact");auto layout=layout_name=="compact"?class_runtime::Layout::compact8:class_runtime::Layout::canonical16;auto block=plan.value("block_size",u32(256));
  J result={{"format",explain?"exact-decision-paths-1":"class-prediction-1"},{"complete",true},{"CUDA_executed",true},{"CPU_predictions_executed",false},{"model",metadata(m)},{"input_sha256",rows.sha},{"input_transport",rows.transport},{"rows",rows.count},{"row_stride",rows.stride},{"layout",layout_name}};
  if(explain){
    auto capacity=plan.value("max_path_nodes",u64(std::min<u32>(m.nodes,4096)));need(capacity>0&&capacity<=UINT32_MAX&&rows.count<=UINT64_MAX/capacity&&rows.count*capacity<=SIZE_MAX/4,"path storage extent overflow");Device<u32>paths(rows.count*capacity),lengths(rows.count);paths.zero();
    model.runtime->trace(batch,{classes.p,classes.n},{paths.p,paths.n,capacity,lengths.p,lengths.n},layout,block);auto ids=paths.download(),sizes=lengths.download(),labels=classes.download();J answers=J::array();
    for(u64 r=0;r<rows.count;++r){J steps=J::array();for(u32 s=0;s<sizes[r];++s){auto node=ids[r*capacity+s];std::size_t at=64+u64(node)*16;auto feature=std::bit_cast<std::int32_t>(word(model.canonical,at));J step={{"node",node}};if(feature==-1)step["class"]=word(model.canonical,at+4);else {step["feature"]=feature>=0?u32(feature):u32(-std::int64_t(feature)-2);step["raw_threshold_bits"]=word(model.canonical,at+4);step.update(decision_programs_display::threshold_fields(feature,word(model.canonical,at+4)));step["missing_goes_left"]=feature<0;step["followed"]=ids[r*capacity+s+1]==word(model.canonical,at+8)?"left":"right";}steps.push_back(step);}answers.push_back({{"row",r},{"class",labels[r]},{"decision_path",steps}});}result["answers"]=answers;result["scope"]="exact hard-runtime paths; no probability, causal, counterfactual or global-importance claim";
  }else{model.runtime->predict(batch,{classes.p,classes.n},layout,class_runtime::Traversal::validated,block);result["classes"]=classes.download();}
  return result;
}
__global__ void demo_rows(float*x,u32*y){if(!blockIdx.x&&!threadIdx.x){for(u32 r=0;r<8;++r){x[r*3]=r%2?-1.f:0.f;x[r*3+1]=0.f;x[r*3+2]=r%3?0.f:1.f;y[r]=r%3?(r%2?0:1):2;}x[7*3]=__uint_as_float(0x7fc00000u);y[7]=1;}}
int demo(const fs::path&out){
  need(!fs::exists(out),"demo output must be fresh");check(cudaSetDevice(0),"select device");std::string canonical="CLSGDAG1";auto append=[&](u32 v){canonical.append(reinterpret_cast<const char*>(&v),4);};for(u32 v:{1u,3u,3u,4u,5u,1u})append(v);canonical.append(32,'\0');for(auto n:std::vector<std::array<u32,4>>{{UINT32_MAX,0,0,0},{UINT32_MAX,1,0,0},{UINT32_MAX,2,0,0},{0,0,0,1},{2,0x3f000000u,3,2}})for(u32 v:n)append(v);
  auto runtime=class_runtime::Runtime::load(canonical,dpnative::sha256(canonical),std::string(64,'0'));Device<float>x(8*3);Device<u32>y(8),labels(8),paths(8*5),lengths(8);paths.zero();demo_rows<<<1,1>>>(x.p,y.p);check(cudaDeviceSynchronize(),"demo setup");class_runtime::DenseBatch batch{x.p,x.n,0,8,3};runtime->verify(batch,y.p,y.n);
  std::size_t checked_paths=0;for(auto layout:{class_runtime::Layout::canonical16,class_runtime::Layout::compact8})for(u32 block:{64u,128u,256u}){runtime->trace(batch,{labels.p,labels.n},{paths.p,paths.n,5,lengths.p,lengths.n},layout,block);need(labels.download()==y.download(),"demo traced labels differ");auto ids=paths.download(),sizes=lengths.download();for(u32 r=0;r<8;++r){need(sizes[r]==(r%3?3u:2u),"demo path length differs");need(ids[r*5]==4&&ids[r*5+sizes[r]-1]==y.download()[r],"demo path endpoints differ");++checked_paths;}}
  bool small_rejected=false;try{runtime->trace(batch,{labels.p,labels.n},{paths.p,paths.n,1,lengths.p,lengths.n},class_runtime::Layout::canonical16);}catch(const std::exception&){small_rejected=true;}need(small_rejected,"insufficient path capacity was accepted");
  bool alias_rejected=false;try{runtime->trace(batch,{labels.p,labels.n},{paths.p,paths.n,5,labels.p,labels.n},class_runtime::Layout::canonical16);}catch(const std::exception&){alias_rejected=true;}need(alias_rejected,"aliasing trace buffers were accepted");
  fs::create_directories(out);dpnative::atomic_text(out/"model.canonical",canonical);auto compact=runtime->compact_bytes();dpnative::atomic_text(out/"model.compact",compact);auto values=x.download();dpnative::atomic_text(out/"input.fp32",std::string(reinterpret_cast<const char*>(values.data()),values.size()*4));J rows=J::array();for(u32 r=0;r<8;++r){J row=J::array();for(u32 f=0;f<3;++f){float v=values[r*3+f];row.push_back(std::isnan(v)?J(nullptr):J(v));}rows.push_back(row);}dpnative::atomic_json(out/"input.json",{{"rows",rows}});
  J result={{"format","decision-programs-demo-1"},{"complete",true},{"CUDA_executed",true},{"CPU_predictions_executed",false},{"rows",8},{"class_mismatches",0},{"traced_paths_checked",checked_paths},{"insufficient_path_capacity_rejected",small_rejected},{"trace_output_alias_rejected",alias_rejected},{"model",metadata(runtime->metadata())},{"canonical_path",(out/"model.canonical").string()},{"compact_path",(out/"model.compact").string()},{"input_path",(out/"input.json").string()},{"classes",y.download()}};dpnative::atomic_json(out/"result.json",result);std::cout<<result.dump(2)<<'\n';return 0;
}
} // namespace
int main(int argc,char**argv){try{
  static_assert(sizeof(float)==4&&sizeof(u32)==4&&std::endian::native==std::endian::little);
  if(argc==2&&std::string(argv[1])=="doctor"){int count=0;auto status=cudaGetDeviceCount(&count);J devices=J::array();if(status==cudaSuccess)for(int i=0;i<count;++i){cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,i),"device properties");devices.push_back({{"index",i},{"name",p.name},{"compute_capability",std::to_string(p.major)+"."+std::to_string(p.minor)},{"global_memory_bytes",p.totalGlobalMem}});}std::cout<<J{{"CUDA_available",status==cudaSuccess&&count>0},{"devices",devices},{"error",status==cudaSuccess?J(nullptr):J(cudaGetErrorString(status))},{"kernels_executed",false}}.dump(2)<<'\n';return status==cudaSuccess&&count>0?0:1;}
  if(argc==3&&std::string(argv[1])=="demo")return demo(fs::absolute(argv[2]));
  need(argc==3,"usage: decision_programs_model_tools OPERATION PLAN.json | demo NEW_OUT | doctor");auto operation=std::string(argv[1]);auto plan=J::parse(dpnative::read_text(argv[2]));check(cudaSetDevice(0),"select device");Model model(plan);
  if(operation=="inspect"){emit({{"format","class-model-inspection-1"},{"complete",true},{"CUDA_graph_validation",true},{"model",metadata(model.runtime->metadata())}},plan);return 0;}
  if(operation=="predict"||operation=="explain"){emit(prediction(model,plan,operation=="explain"),plan);return 0;}
  if(operation=="export"){
    auto kind=plan.value("kind",std::string("equations"));need(plan.contains("out"),"export requires --out FILE");if(kind=="equations")emit(equations(model),plan);
    else if(kind=="compact"){auto bytes=model.runtime->compact_bytes();auto out=plan.at("out").get<std::string>();fresh_write(out,bytes);std::cout<<J{{"complete",true},{"kind","compact"},{"output",out},{"sha256",dpnative::sha256(bytes)},{"bytes",bytes.size()},{"CUDA_executed",true},{"model",metadata(model.runtime->metadata())}}.dump(2)<<'\n';}
    else throw std::invalid_argument("export kind must be equations or compact");return 0;
  }
  throw std::invalid_argument("unsupported model tool: "+operation);
}catch(const std::exception&e){std::cerr<<"decision-programs: "<<e.what()<<'\n';return 1;}}
