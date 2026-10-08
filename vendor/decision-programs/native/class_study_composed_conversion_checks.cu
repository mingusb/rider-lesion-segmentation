// Real-native regression fixtures for the composed adapter. All prediction,
// probability transformation, margin packing and runtime comparisons run on
// CUDA. The generated tiny JSON models are synthetic test inputs, not study
// results. No data or model files are written by this executable.
#include "class_study_convert.hpp"
#include "class_study_train.hpp"
#include "class_study_composition.hpp"
#include "class_study_composition_prune.hpp"
#include <bit>
#include <cmath>
#include "class_runtime.hpp"
#include "class_conversion/class_io.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using J=nlohmann::json;
using U=std::uint32_t;
using L=std::uint64_t;
void need(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void ck(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
void finish(){ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
template<class T> struct Device {
  T* p=nullptr;L capacity=0;
  explicit Device(L n=0){reserve(n);}
  ~Device(){if(p)cudaFree(p);}
  Device(const Device&)=delete;Device&operator=(const Device&)=delete;
  void reserve(L n){if(n<=capacity)return;need(n<=SIZE_MAX/sizeof(T),"fixture allocation overflow");if(p)ck(cudaFree(p));p=nullptr;ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));capacity=n;}
};
J tree(U id,U features,bool split,float a,float b,bool missing_left,U feature=0,float cut=0.f){
  const U n=split?3:1;
  J result={{"id",id},{"tree_param",{{"num_deleted","0"},{"num_feature",std::to_string(features)},{"num_nodes",std::to_string(n)},{"size_leaf_vector","1"}}},
    {"categories",J::array()},{"categories_nodes",J::array()},{"categories_segments",J::array()},{"categories_sizes",J::array()}};
  result["left_children"]=split?J::array({1,-1,-1}):J::array({-1});
  result["right_children"]=split?J::array({2,-1,-1}):J::array({-1});
  result["parents"]=split?J::array({2147483647,0,0}):J::array({2147483647});
  result["split_indices"]=split?J::array({feature,0,0}):J::array({0});
  result["split_conditions"]=split?J::array({cut,a,b}):J::array({a});
  result["base_weights"]=split?J::array({0.f,a,b}):J::array({a});
  result["default_left"]=split?J::array({missing_left?1:0,0,0}):J::array({0});
  result["split_type"]=std::vector<U>(n,0);
  result["loss_changes"]=std::vector<float>(n,0.f);
  result["sum_hessian"]=std::vector<float>(n,1.f);
  return result;
}
std::string model(U features,J trees,std::array<float,2> bias){
  need(trees.is_array()&&!trees.empty()&&trees.size()%2==0,"fixture requires two scalar trees per round");
  const U count=U(trees.size());
  std::vector<U> channels(count),iterations{0};
  for(U i=0;i<count;++i){channels[i]=i%2;if(i%2)iterations.push_back(i+1);}
  J result={{"version",J::array({3,4,1})},{"learner",{
    {"attributes",J::object()},{"feature_names",J::array()},{"feature_types",J::array()},
    {"learner_model_param",{{"base_score",J(bias).dump()},{"boost_from_average","0"},{"num_class","2"},{"num_feature",std::to_string(features)},{"num_target","1"}}},
    {"objective",{{"name","multi:softmax"},{"softmax_multiclass_param",{{"num_class","2"}}}}},
    {"gradient_booster",{{"name","gbtree"},{"model",{
      {"cats",{{"enc",J::array()},{"feature_segments",J::array()},{"sorted_idx",J::array()}}},
      {"gbtree_model_param",{{"num_parallel_tree","1"},{"num_trees",std::to_string(count)}}},
      {"iteration_indptr",iterations},{"tree_info",channels},{"trees",std::move(trees)}}}}}}}};
  return dp_native_json::dump(result);
}
std::string base(bool opposite,bool missing_left){
  const float a=opposite?-1.f:1.f,b=-a;
  return model(2,J::array({tree(0,2,true,a,b,missing_left,0,opposite?-0.f:0.f),tree(1,2,false,0,0,false)}),
      opposite?std::array<float,2>{-.25f,.25f}:std::array<float,2>{.25f,-.25f});
}
std::string multiround_base(bool opposite){
  // At 2^24, adding 1 rounds back to 2^24, whereas adding 4 is exact.
  // Class 1 uses 2^23 so its middle +1 survives. Thus the ordered margins
  // are (0,1) or (4,1), but consuming the cancellation before +1 changes
  // (0,1) into (1,1), changing the native class at the left-hand atom.
  const float a=opposite?4.f:1.f,b=opposite?1.f:4.f;
  return model(2,J::array({
      tree(0,2,false,0x1p24f,0,false),tree(1,2,false,0x1p23f,0,false),
      tree(2,2,true,a,b,true),tree(3,2,false,1.f,0,false),
      tree(4,2,false,-0x1p24f,0,false),tree(5,2,false,-0x1p23f,0,false)}),{0,0});
}
std::string meta(U column){
  // Probability < .5 yields class 1; probability >= .5 yields class 0.
  return model(4,J::array({tree(0,4,true,0,2,false,column,.5f),tree(1,4,false,1,0,false)}),{0,0});
}
__global__ void pack(const float* in,float* out,L rows,U column,U width){
  for(L i=L(blockIdx.x)*blockDim.x+threadIdx.x;i<rows*2;i+=L(blockDim.x)*gridDim.x)out[(i/2)*width+column+i%2]=in[i];
}
__global__ void pack_raw(const float* in,float* out,L rows,U width,U column,U raw){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x)out[row*width+column]=in[2*row+raw];
}
__global__ void raw_expected(const float* x,U* expected,L rows,U raw,float cut,bool missing_left,bool constant){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){
    float value=x[2*row+raw];expected[row]=constant?0u:U(isnan(value)?missing_left:value<cut);
  }
}
__global__ void constant_margins(const float* words,U* bad,U rows){
  for(U i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*4;i+=blockDim.x*gridDim.x)
    if(__float_as_uint(words[i])!=0u)atomicOr(bad,1u);
}
__global__ void compare_native(const float* words,const U* expected,U* bad,U rows){
  for(U i=blockIdx.x*blockDim.x+threadIdx.x;i<rows;i+=blockDim.x*gridDim.x)
    if(words[i]!=float(expected[i]))atomicOr(bad,1u);
}
__global__ void guard_arithmetic(const float* cuts,const U* missing,U predicates,const float* x,U rows,U* bad){
  for(U row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=blockDim.x*gridDim.x){
    const float value=x[2*row+1];
    for(U channel=0;channel<(predicates+23)/24;++channel){
      U mask=0;float forward=0,reverse=0;
      const U begin=24*channel,end=min(predicates,begin+24);
      for(U i=begin;i<end;++i){bool right=isnan(value)?!missing[i]:!(value<cuts[i]);const U bit=right?(1u<<(i-begin)):0u;mask|=bit;forward=__fadd_rn(forward,float(bit));}
      for(U i=end;i>begin;){--i;bool right=isnan(value)?!missing[i]:!(value<cuts[i]);reverse=__fadd_rn(reverse,right?float(1u<<(i-begin)):0.f);}
      if(__float_as_uint(forward)!=__float_as_uint(float(mask))||__float_as_uint(reverse)!=__float_as_uint(float(mask)))atomicOr(bad,1u);
    }
  }
}
__global__ void test_rows(float* out,U rows){
  const U words[]={0xff7fffffu,0xbf800000u,0x80000001u,0x80000000u,0u,1u,0x3f800000u,0x7f7fffffu,0x7fc00001u,0xffc12345u};
  for(U row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=blockDim.x*gridDim.x){out[2*row]=__uint_as_float(words[row%10]);out[2*row+1]=__uint_as_float(words[(row/10)%10]);}
}
__global__ void decode(const float* values,U* classes,U* flags,U rows){
  for(U row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=blockDim.x*gridDim.x){float x=values[row];if(x!=0.f&&x!=1.f)atomicOr(flags,4u);else{classes[row]=U(x);atomicOr(flags,1u<<U(x));}}
}
__global__ void sum_zero(const float* words,U* bad,U rows){for(U row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=blockDim.x*gridDim.x)if(words[4*row]+words[4*row+2]!=0.f||words[4*row+1]+words[4*row+3]!=0.f)atomicOr(bad,1u);}
__global__ void multiround_order(const float* x,const float* margins,U* flags,U rows){
  for(U row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=blockDim.x*gridDim.x){
    const bool left=isnan(x[2*row])||x[2*row]<0.f;
    const float middle[2]={left?1.f:4.f,left?4.f:1.f};
    for(U base=0;base<2;++base){
      const float ordered=__fadd_rn(__fadd_rn(0x1p24f,middle[base]),-0x1p24f);
      const float other=__fadd_rn(__fadd_rn(0x1p23f,1.f),-0x1p23f);
      if(__float_as_uint(margins[4*row+2*base])!=__float_as_uint(ordered)||
         __float_as_uint(margins[4*row+2*base+1])!=__float_as_uint(other))atomicOr(flags,1u);
      if(__fadd_rn(__fadd_rn(0x1p24f,-0x1p24f),middle[base])!=ordered)atomicOr(flags,2u);
    }
  }
}
__global__ void invalid_labels(float* out,L rows){for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x)out[row]=2.f;}
U scalar(Device<U>& d){U v=0;ck(cudaMemcpy(&v,d.p,sizeof(v),cudaMemcpyDeviceToHost));return v;}

struct Stack {
  std::array<std::string,2> bytes,pins;
  std::string meta_bytes,meta_pin;
  std::array<std::unique_ptr<class_study::ResidentPredictor>,2> bases;
  std::unique_ptr<class_study::ResidentPredictor> selected;
  Device<float> responses;
  std::vector<U> raw;
  U width=4;
  Stack(const J& plan,std::array<std::string,2> sources,std::string m,std::vector<U> mapping={}):bytes(std::move(sources)),meta_bytes(std::move(m)),raw(std::move(mapping)),width(4+U(raw.size())){
    for(U i=0;i<2;++i){pins[i]=class_conversion_native::sha256(bytes[i]);bases[i]=i?std::make_unique<class_study::ResidentPredictor>(plan,2,2,*bases[0]):std::make_unique<class_study::ResidentPredictor>(plan,2,2);bases[i]->restore_model(bytes[i],pins[i]);}
    meta_pin=class_conversion_native::sha256(meta_bytes);selected=std::make_unique<class_study::ResidentPredictor>(plan,width,2,*bases[0]);selected->restore_model(meta_bytes,meta_pin);
  }
  const float* columns(const float* x,L rows,bool margin){
    const U columns=margin?4:width;
    need(rows<=SIZE_MAX/(4ULL*columns),"fixture response overflow");responses.reserve(rows*columns);
    for(U i=0;i<2;++i){const float* output=margin?bases[i]->predict(x,rows,true):bases[i]->predict_probabilities(x,rows);pack<<<64,128>>>(output,responses.p,rows,i*2,columns);finish();}
    if(!margin)for(U i=0;i<raw.size();++i){pack_raw<<<64,128>>>(x,responses.p,rows,columns,4+i,raw[i]);finish();}
    return responses.p;
  }
  const float* predict(const float* x,L rows){return selected->predict(columns(x,rows,false),rows,false);}
  class_study::ComposedConversionSource source()const{
    class_study::ComposedConversionSource s;for(U i=0;i<2;++i)s.bases.push_back({bytes[i],2,2,pins[i]});s.meta={meta_bytes,width,2,meta_pin};s.meta_input_feature_indices=raw;return s;
  }
  class_study::ComposedNativeOracle oracle(const std::string& library){
    class_study::ComposedNativeOracle o;o.library_sha256=library;o.base_margins=[this](const float* x,L rows){return columns(x,rows,true);};o.predict_classes=[this](const float* x,L rows){return predict(x,rows);};return o;
  }
};
class_study::ConversionOptions bounded(){
  class_study::ConversionOptions o;o.domain.allow_nan=true;o.initial_states=64;o.initial_nodes=64;o.max_states=1024;o.max_nodes=1024;o.max_expansions=4096;o.gpu_byte_budget=64ULL<<20;o.batch_size=8;o.completion_estimate_enabled=false;o.checkpoint_on_completion=false;return o;
}
void verify_predicates(const class_study::ConvertedModel& converted){
  const auto& m=converted.runtime->metadata();need(m.features==2&&m.classes==2&&m.nodes>1,"composed artifact has wrong raw/output shape or is constant");
  const auto& bytes=converted.canonical_bytes;need(bytes.size()==64ULL+16ULL*m.nodes,"fixture canonical extent");
  for(U i=0;i<m.nodes;++i){std::int32_t f=0;U cut=0;std::memcpy(&f,bytes.data()+64+16ULL*i,4);std::memcpy(&cut,bytes.data()+68+16ULL*i,4);if(f==-1)continue;need((f==0||f==-2)&&(cut==0||cut==0x80000000u),"compiled output uses an unverified raw predicate");}
}
J qualify(Stack& stack,const class_study::ConvertedModel& converted,Device<float>& x,Device<U>& expected,U rows){
  verify_predicates(converted);need(converted.metrics.at("score_channels")==4&&converted.metrics.at("classes")==2,"converter score/public class metadata differs");need(converted.metrics.at("qualified_native_gap_pruning_enabled")==false&&converted.metrics.at("class_proof_modules_enabled")==false,"composed class pruning unexpectedly enabled");Device<U> flags(1);ck(cudaMemset(flags.p,0,4));decode<<<1,128>>>(stack.predict(x.p,rows),expected.p,flags.p,rows);finish();need(scalar(flags)==3,"fixture native composition must produce both classes");converted.runtime->verify({x.p,L(rows)*2,0,rows,2},expected.p,rows);
  return {{"nodes",converted.runtime->metadata().nodes},{"classes",converted.runtime->metadata().classes},{"source_score_channels",4},{"probe_rows",rows},{"all_three_source_atoms_covered",true},{"unused_feature_varied",true},{"canonical_bytes",converted.canonical_bytes.size()},{"native_class_mismatches",0}};
}
std::string constant_base(){return model(2,J::array({tree(0,2,false,0,0,false),tree(1,2,false,0,0,false)}),{0,0});}
std::string raw_meta(U column,float cut,bool missing_left){return model(6,J::array({tree(0,6,true,0,2,missing_left,column,cut),tree(1,6,false,1,0,false)}),{0,0});}
std::string many_raw_meta(){
  J trees=J::array();
  for(U i=0;i<26;++i){float cut=i==25?-0.f:float(int(i)-12);trees.push_back(tree(2*i,6,true,0,1,bool(i%2),4,cut));trees.push_back(tree(2*i+1,6,false,.5f,0,false));}
  return model(6,std::move(trees),{0,0});
}
J bundle(const J& plan,const std::array<std::string,2>& bases,const std::string& meta,const std::vector<U>& raw={}){
  auto entry=[](const std::string& bytes){return J{{"sha256",class_conversion_native::sha256(bytes)},{"model",J::binary(std::vector<std::uint8_t>(bytes.begin(),bytes.end()))}};};
  J teachers=J::object(),order=J::array();for(const auto& b:bases){auto item=entry(b);auto pin=item.at("sha256").get<std::string>();teachers[pin]=item;order.push_back(pin);}
  J result={{"format","native-nonlinear-deployment-bundle-1"},{"selected_kind","nonlinear_combination"},{"raw_features",2},{"classes",2},
    {"native_library_path",plan.at("native_library_path")},{"native_library_sha256",plan.at("native_library_sha256")},
    {"raw_dataset_contract",{{"features",2},{"classes",2},{"TEST_read",false}}},
    {"final_class_contract","selected native multi:softmax public class"},{"compiled_single_tree",false},
    {"teacher_probability_contract","native full-round gbtree clone; derived multi:softprob; FP32 teacher-major/class-minor; no CPU transform"},
    {"teacher_order",order},{"teacher_models",teachers},{"meta_model",entry(meta)}};
  if(!raw.empty())result["meta_input_mapping"]={{"version",1},{"raw_feature_indices",raw}};
  return result;
}

J native_json_checks(){
  using namespace class_conversion_native;
  const auto original=base(false,true);need(dp_native_json::dump(dp_native_json::parse(original))==original,"finite native JSON codec changed bytes");
  // Include a decoded marker collision expressed with a Unicode escape, literal
  // Infinity strings, escaped quotes, and keys. Only bare numeric values change.
  const auto text=R"({"plain":"Infinity","negative":"-Infinity","escaped":"quote\" -Infinity","collision":"\u001fdp-native-IEEE:+","Infinity":[Infinity,-Infinity,NaN,-0.0]})";
  auto parsed=dp_native_json::parse(text);need(parsed.at("collision")==std::string("\x1f" "dp-native-IEEE:+")&&parsed.at("plain")=="Infinity"&&parsed.at("negative")=="-Infinity","native JSON modified quoted text");
  need(std::isinf(dp_native_json::parse("+Infinity").get<double>()),"native explicit positive infinity refused");
  auto round=dp_native_json::parse(dp_native_json::dump(parsed));auto binary=J::from_ubjson(J::to_ubjson(parsed));
  for(const auto* value:{&round,&binary}){const auto& a=value->at("Infinity");need(std::isinf(a[0].get<double>())&&!std::signbit(a[0].get<double>())&&std::isinf(a[1].get<double>())&&std::signbit(a[1].get<double>())&&std::isnan(a[2].get<double>())&&std::signbit(a[3].get<double>()),"native IEEE JSON/UBJSON value words lost");need(value->at("collision")==parsed.at("collision")&&value->at("escaped")==parsed.at("escaped"),"native codec string roundtrip differs");}
  U refusals=0;auto deny=[&](auto f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}need(failed,"malformed native JSON accepted");++refusals;};
  for(auto text:{"{Infinity:1}","{ Infinity :1}","{\"x\":1Infinity}","{\"x\":Infinityx}","{\"x\":--Infinity}","{\"x\":Inf}","[Infinity,,0]"})deny([&]{dp_native_json::parse(text);});
  for(float cut:{-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()})for(bool missing:{false,true}){
    auto bytes=model(2,J::array({tree(0,2,true,0,2,missing,0,cut),tree(1,2,false,1,0,false)}),{0,0});auto forest=read_source_bytes(bytes);
    need(forest.bytes==bytes&&forest.identity==sha256(bytes)&&std::bit_cast<U>(forest.cut[0])==std::bit_cast<U>(cut)&&forest.missing_left[0]==missing,"native infinite predicate identity differs");
  }
  for(float value:{-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
    deny([&]{read_source_bytes(model(2,J::array({tree(0,2,false,value,0,false),tree(1,2,false,1,0,false)}),{0,0}));});
    auto bad=dp_native_json::parse(original);bad["learner"]["learner_model_param"]["base_score"]=J::array({value,0});deny([&]{read_source_bytes(dp_native_json::dump(bad));});
  }
  deny([&]{read_source_bytes(model(2,J::array({tree(0,2,true,0,2,true,0,std::numeric_limits<float>::quiet_NaN()),tree(1,2,false,1,0,false)}),{0,0}));});
  return {{"complete",true},{"finite_codec_bytes_unchanged",true},{"IEEE_text_and_UBJSON_roundtrip",true},{"quoted_and_escaped_marker_collision_preserved",true},{"infinite_cut_variants",4},{"invalid_token_leaf_bias_refusals",refusals}};
}

J host_checks(){
  using namespace class_conversion_native;
  const auto base_bytes=constant_base();auto b=read_source_bytes(base_bytes);
  const std::vector<U> mapping{1,0};auto m=read_source_bytes(raw_meta(4,0.f,true));
  auto source=compose_source_forests({b,b},m,mapping);
  need(source.outputs==5&&source.native_margin_outputs==4&&source.structural_guard_trees==1&&source.roots.size()==5&&source.feature.size()==7,"host guard dimensions differ");
  need(source.feature.back()==-1&&source.bias.back()==0&&source.value.back()==1,"host guard topology differs");
  auto many=read_source_bytes(many_raw_meta());auto predicates=mapped_meta_raw_predicates(many,4,mapping,2);
  auto packed=compose_source_forests({b,b},many,mapping);
  need(predicates.size()==26&&packed.outputs==6&&packed.structural_guard_trees==26,"host guard channel boundary differs");
  need(packed.channels[4+23]==4&&packed.channels[4+24]==5&&packed.value[4+3*23+2]==0x1p23f&&packed.value[4+3*24+2]==1.f,"guard bit23/channel24 mapping differs");
  bool poszero=false,negzero=false;for(auto p:predicates){poszero|=std::bit_cast<U>(p.cut)==0;negzero|=std::bit_cast<U>(p.cut)==0x80000000u;}
  need(poszero&&negzero,"raw predicate identity lost signed zero words");
  auto legacy_meta=read_source_bytes(meta(0));auto legacy=compose_source_forests({b,b},legacy_meta);
  const J expected={{"format","native-probability-stack-conversion-source-1"},{"bases",J::array({J{{"source_sha256",b.identity},{"score_channels",2}},J{{"source_sha256",b.identity},{"score_channels",2}}})},
    {"meta_sha256",legacy_meta.identity},{"features",2},{"score_channels",4},{"classes",2},{"response_order","teacher-major native full-round softprob FP32 columns"}};
  need(legacy.bytes==expected.dump()&&legacy.identity==sha256(expected.dump())&&legacy.native_margin_outputs==0,"legacy descriptor changed");
  auto changed=compose_source_forests({b,b},m,std::vector<U>{0,1});need(changed.identity!=source.identity,"raw mapping not bound to source identity");
  auto opposite=compose_source_forests({b,b},read_source_bytes(raw_meta(4,0.f,false)),mapping);need(opposite.identity!=source.identity,"raw missing direction not bound to identity");
  class_study::TrainingResult bt,mt;bt.model_json=base_bytes;bt.model_sha256=b.identity;mt.model_json=m.bytes;mt.model_sha256=m.identity;
  auto pruned=class_study::composition_prune::unused_bases({bt,bt},mt,mapping);auto pm=read_source_bytes(pruned.meta.model_json);
  need(pruned.bases.size()==1&&pruned.retained_indices==std::vector<U>{0}&&pm.features==4&&pm.feature[0]==2&&pruned.provenance.at("raw_only_meta_keeps_one_base")==true&&pruned.bases[0].model_json==base_bytes,"raw-only pruning changed carrier/mapping");
  auto constant=model(6,J::array({tree(0,6,false,0,0,false),tree(1,6,false,0,0,false)}),{1,0});mt.model_json=constant;mt.model_sha256=sha256(constant);
  auto cp=class_study::composition_prune::unused_bases({bt,bt},mt,mapping);need(cp.provenance.at("constant_meta_keeps_one_base")==true&&cp.provenance.at("raw_only_meta_keeps_one_base")==false&&read_source_bytes(cp.meta.model_json).features==4,"constant mixed pruning differs");
  J plan={{"native_library_path","/synthetic/libxgboost.so"},{"native_library_sha256",std::string(64,'a')}};
  auto declaration=bundle(plan,{base_bytes,base_bytes},m.bytes,mapping);auto declared=class_study::ResidentComposition::validate_bundle(declaration);
  need(declared.at("response_features")==6&&declared.at("native_probability_columns")==4&&declared.at("meta_input_mapping").at("raw_feature_indices")==mapping,"mixed bundle declaration width/order differs");
  auto different=declaration;different["meta_input_mapping"]["raw_feature_indices"]=J::array({0,1});need(class_study::ResidentComposition::validate_bundle(different).at("composition_identity_sha256")!=declared.at("composition_identity_sha256"),"bundle identity omitted raw mapping");
  U refusals=0;auto refuse=[&](auto&& f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}need(failed,"host mapping refusal accepted");++refusals;};
  refuse([&]{compose_source_forests({b,b},m);});
  refuse([&]{compose_source_forests({b,b},m,std::vector<U>{1,1});});
  refuse([&]{compose_source_forests({b,b},m,std::vector<U>{1,2});});
  refuse([&]{compose_source_forests({b,b},m,std::vector<U>{1});});
  // A valid 1024 native-channel bank leaves no room for even one guard.
  std::vector<SourceData> full(512,b);auto wide=read_source_bytes(model(1025,J::array({tree(0,1025,true,0,2,true,1024,0),tree(1,1025,false,1,0,false)}),{0,0}));
  refuse([&]{compose_source_forests(full,wide,std::vector<U>{1});});
  for(const J& bad:J::array({J{{"version",2},{"raw_feature_indices",mapping}},J{{"version",1},{"raw_feature_indices",J::array()}},J{{"version",1},{"raw_feature_indices",J::array({1,1})}},J{{"version",1},{"raw_feature_indices",J::array({2})}},J{{"version",1},{"raw_feature_indices",J::array({-1})}},J{{"version",1},{"raw_feature_indices",J::array({1.0})}},J{{"version",1},{"raw_feature_indices",mapping},{"ignored",true}}})){
    auto invalid=declaration;invalid["meta_input_mapping"]=bad;refuse([&]{class_study::ResidentComposition::validate_bundle(invalid);});
  }
  auto low=class_study::CompositionOptions{};low.response_gpu_byte_budget=23;refuse([&]{class_study::ResidentComposition::validate_bundle(declaration,low);});
  return {{"complete",true},{"refusals",refusals},{"legacy_descriptor_exact",true},{"signed_zero_predicate_identity",true},{"raw_only_and_constant_pruning",true},{"guard_bit23_and_channel24",true},{"mapping_order_and_identity",true},{"GPU_used",false}};
}
J mixed_checks(const J& plan,const std::string& library){
  const std::vector<U> mapping{1,0};const auto cb=constant_base();J cases=J::array();
  for(U variant=0;variant<9;++variant){
    auto m=variant==4?many_raw_meta():raw_meta(variant==2?5:4,variant==1?-0.f:0.f,variant!=3);
    if(variant>=7)m=raw_meta(4,variant==7?-std::numeric_limits<float>::infinity():std::numeric_limits<float>::infinity(),variant==7);
    if(variant==5)m=model(6,J::array({tree(0,6,false,0,0,false),tree(1,6,false,0,0,false)}),{1,0});
    if(variant==6)m=model(6,J::array({tree(0,6,true,-1,1,true,0,.5f),tree(1,6,false,0,0,false),tree(2,6,true,-2,2,false,4,0),tree(3,6,false,2,0,false)}),{0,0});
    std::array<std::string,2> bs=variant==6?std::array<std::string,2>{base(false,true),base(true,false)}:std::array<std::string,2>{cb,cb};
    Stack stack(plan,bs,m,mapping);auto source=stack.source();
    auto parsed=class_conversion_native::read_source_bytes(m);auto predicates=class_conversion_native::mapped_meta_raw_predicates(parsed,4,mapping,2);
    // Cover equality and adjacent FP32 values at every raw cut, both zero signs,
    // both quiet-NaN signs/payloads, and finite extremes on each raw column.
    std::vector<float> values={-std::numeric_limits<float>::max(),-0.f,0.f,std::numeric_limits<float>::max(),std::bit_cast<float>(0x7fc00001u),std::bit_cast<float>(0xffc12345u)};
    for(const auto& p:predicates)for(float x:{std::nextafter(p.cut,-std::numeric_limits<float>::infinity()),p.cut,std::nextafter(p.cut,std::numeric_limits<float>::infinity())})if(std::isfinite(x))values.push_back(x);
    std::vector<float> rows_host;for(float a:values)for(float b:std::array<float,3>{-1.f,1.f,std::bit_cast<float>(0x7fc00001u)}){rows_host.push_back(b);rows_host.push_back(a);rows_host.push_back(a);rows_host.push_back(b);}
    U rows=U(rows_host.size()/2);Device<float> x(rows_host.size());ck(cudaMemcpy(x.p,rows_host.data(),rows_host.size()*4,cudaMemcpyHostToDevice));Device<U> expected(rows),bad(1);ck(cudaMemset(bad.p,0,4));
    if(variant!=6){constant_margins<<<64,128>>>(stack.columns(x.p,rows,true),bad.p,rows);finish();need(!scalar(bad),"raw guard fixture base margins are not identical");}
    if(variant<4||variant==5||variant>=7){raw_expected<<<64,128>>>(x.p,expected.p,rows,variant==2?0:1,variant>=7?(variant==7?-std::numeric_limits<float>::infinity():std::numeric_limits<float>::infinity()):(variant==1?-0.f:0.f),variant>=7?variant==7:variant!=3,variant==5);finish();compare_native<<<64,128>>>(stack.predict(x.p,rows),expected.p,bad.p,rows);finish();need(!scalar(bad),"raw-only native classes differ from independent predicate oracle");}
    else {Device<U> flags(1);ck(cudaMemset(flags.p,0,4));decode<<<64,128>>>(stack.predict(x.p,rows),expected.p,flags.p,rows);finish();need(scalar(flags)==3,"mixed or packed fixture must produce both classes");}
    auto o=bounded();o.max_states=8192;o.max_nodes=8192;o.max_expansions=32768;
    auto converted=class_study::convert_composed_model(source,stack.oracle(library),o);need(converted.metrics.at("complete")==true,"mixed converter incomplete");converted.runtime->verify({x.p,L(rows)*2,0,rows,2},expected.p,rows);
    const U guard_channels=U((predicates.size()+23)/24);
    need(converted.metrics.at("score_channels")==4&&converted.metrics.at("authority_channels")==4+guard_channels&&converted.metrics.at("native_margin_audit_columns")==4&&converted.metrics.at("structural_guard_trees")==predicates.size()&&converted.metrics.at("source_trees")==4,"mixed converter authority/native counts differ");
    auto saved=J::from_cbor(J::to_cbor(bundle(plan,bs,m,mapping)));class_study::ResidentComposition replay(saved);compare_native<<<64,128>>>(replay.predict(x.p,rows),expected.p,bad.p,rows);finish();need(!scalar(bad)&&replay.response_features()==6,"mixed native bundle replay differs");
    // Pruned source remains the same native pipeline, with raw fields after its
    // smaller probability prefix and one genuine teacher for raw-only models.
    std::vector<class_study::TrainingResult> trained(2);for(U i=0;i<2;++i){trained[i].model_json=bs[i];trained[i].model_sha256=stack.pins[i];}class_study::TrainingResult mt;mt.model_json=m;mt.model_sha256=stack.meta_pin;
    auto pruned=class_study::composition_prune::unused_bases(trained,mt,mapping);
    auto pruned_bundle=saved;pruned_bundle["teacher_order"]=J::array();pruned_bundle["teacher_models"]=J::object();for(const auto& b:pruned.bases){auto pin=b.model_sha256;pruned_bundle["teacher_order"].push_back(pin);pruned_bundle["teacher_models"][pin]=saved.at("teacher_models").at(pin);}pruned_bundle["meta_model"]={{"sha256",pruned.meta.model_sha256},{"model",J::binary(std::vector<std::uint8_t>(pruned.meta.model_json.begin(),pruned.meta.model_json.end()))}};
    class_study::ResidentComposition pruned_replay(pruned_bundle);compare_native<<<64,128>>>(pruned_replay.predict(x.p,rows),expected.p,bad.p,rows);finish();need(!scalar(bad),"pruned mixed bundle replay differs");
    if(variant==0||variant==5||variant>=7){
      need(pruned.bases.size()==1,"raw-only pruning fixture must retain one native source");
      class_study::ResidentPredictor carrier(plan,2,2);carrier.restore_model(pruned.bases[0].model_json,pruned.bases[0].model_sha256);
      class_study::ComposedConversionSource reduced;reduced.bases.push_back({pruned.bases[0].model_json,2,2,pruned.bases[0].model_sha256});reduced.meta={pruned.meta.model_json,4,2,pruned.meta.model_sha256};reduced.meta_input_feature_indices=mapping;
      class_study::ComposedNativeOracle oracle;oracle.library_sha256=library;oracle.base_margins=[&](const float* values,L n){return carrier.predict(values,n,true);};oracle.predict_classes=[&](const float* values,L n){return pruned_replay.predict(values,n);};
      auto reduced_conversion=class_study::convert_composed_model(reduced,oracle,o);need(reduced_conversion.metrics.at("complete")==true&&reduced_conversion.metrics.at("native_margin_audit_columns")==2&&reduced_conversion.metrics.at("authority_channels")==2+guard_channels,"pruned native/authority channel prefix differs");reduced_conversion.runtime->verify({x.p,L(rows)*2,0,rows,2},expected.p,rows);
    }
    if(variant==4){std::vector<float> cuts;std::vector<U> missing;for(auto p:predicates){cuts.push_back(p.cut);missing.push_back(p.missing_left);}Device<float> dc(cuts.size());Device<U> dm(missing.size());ck(cudaMemcpy(dc.p,cuts.data(),cuts.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dm.p,missing.data(),missing.size()*4,cudaMemcpyHostToDevice));guard_arithmetic<<<64,128>>>(dc.p,dm.p,U(cuts.size()),x.p,rows,bad.p);finish();need(!scalar(bad),"guard sums lost exact bit signatures");}
    cases.push_back({{"variant",variant},{"rows",rows},{"guard_predicates",predicates.size()},{"authority_channels",4+guard_channels},{"native_margin_columns",4},{"meta_input_columns",6},{"nodes",converted.runtime->metadata().nodes},{"native_and_bundle_and_pruned_mismatches",0},{"base_margins_constant",variant!=6}});
  }
  return cases;
}


J infinite_cut_checks(const J& plan,const std::string& library,const std::filesystem::path& directory){
  need(!std::filesystem::exists(directory),"infinite-cut fixture directory exists");std::filesystem::create_directories(directory);
  constexpr U rows=100;Device<float> x(rows*2);Device<U> expected(rows),bad(1);test_rows<<<1,128>>>(x.p,rows);finish();J cases=J::array();
  for(U variant=0;variant<4;++variant){const float cut=variant&1?std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity();const bool missing=variant&2;
    auto bytes=model(2,J::array({tree(0,2,true,0,2,missing,0,cut),tree(1,2,false,1,0,false)}),{0,0});const auto pin=class_conversion_native::sha256(bytes);
    class_study::ResidentPredictor predictor(plan,2,2);predictor.restore_model(bytes,pin);
    raw_expected<<<1,128>>>(x.p,expected.p,rows,0,cut,missing,false);finish();ck(cudaMemset(bad.p,0,4));compare_native<<<1,128>>>(predictor.predict(x.p,rows,false),expected.p,bad.p,rows);finish();need(!scalar(bad),"native infinite cut differs from independent missing/finite predicate");
    class_study::ConversionSource source{bytes,2,2,pin};class_study::NativeOracle oracle;oracle.features=2;oracle.classes=2;oracle.library_sha256=library;oracle.source_sha256=pin;oracle.predict=[&](const float* values,L n,bool margin){return predictor.predict(values,n,margin);};
    auto options=bounded();options.checkpoint_path=(directory/("variant-"+std::to_string(variant))).string();options.checkpoint_on_completion=true;
    auto converted=class_study::convert_model(source,oracle,options);need(converted.metrics.at("complete")==true,"infinite-cut construction incomplete");converted.runtime->verify({x.p,L(rows)*2,0,rows,2},expected.p,rows);
    const auto& metadata=converted.runtime->metadata();auto loaded=class_runtime::Runtime::load(converted.canonical_bytes,metadata.canonical_sha256,pin,converted.compact_bytes,metadata.compact_sha256);loaded->verify({x.p,L(rows)*2,0,rows,2},expected.p,rows);
    auto resume=bounded();resume.resume_from=options.checkpoint_path;auto resumed=class_study::convert_model(source,oracle,resume);need(resumed.canonical_bytes==converted.canonical_bytes,"infinite checkpoint canonical words changed");resumed.runtime->verify({x.p,L(rows)*2,0,rows,2},expected.p,rows);
    cases.push_back({{"threshold_FP32_bits",std::bit_cast<U>(cut)},{"missing_left",missing},{"rows",rows},{"nodes",metadata.nodes},{"checkpoint_resume_exact",true},{"canonical_compact_reload",true},{"mismatches",0}});
  }
  return cases;
}

}

int main(int argc,char** argv){try{
  need(argc==2||(argc==4&&std::string(argv[2])=="--infinite-cuts-only"),"usage: class_study_composed_conversion_checks LIBXGBOOST|--host [--infinite-cuts-only NEW_DIRECTORY]");
  auto host=host_checks();host["native_IEEE_JSON"]=native_json_checks();
  if(std::string(argv[1])=="--host"){std::cout<<host.dump(2)<<'\n';return 0;}
  const std::string library=class_conversion_native::sha256(class_conversion_native::read_text(argv[1]));
  J plan={{"native_library_path",std::string(argv[1])},{"native_library_sha256",library}};
  if(argc==4){auto inf=infinite_cut_checks(plan,library,argv[3]);auto mixed=mixed_checks(plan,library);std::cout<<J{{"complete",true},{"host",host},{"infinite_cut_cases",inf},{"mixed_raw_cases",mixed}}.dump(2)<<'\n';return 0;}
  constexpr U rows=100;Device<float> x(rows*2);Device<U> expected(rows);test_rows<<<1,128>>>(x.p,rows);finish();
  J cases=J::array();U rejected=0;
  auto refuse=[&](auto&& operation){bool failed=false;try{operation();}catch(const std::exception&){failed=true;}need(failed,"composed fixture accepted a refusal case");++rejected;};
  for(U variant=0;variant<5;++variant){
    auto a=base(false,true),b=base(true,variant!=3);std::array<std::string,2> models={a,b};U column=0;
    if(variant==1){std::swap(models[0],models[1]);column=2;}
    if(variant==2)models[1]=models[0];
    if(variant==3)column=2;
    if(variant==4)models={multiround_base(false),multiround_base(true)};
    Stack stack(plan,std::move(models),meta(column));auto source=stack.source();auto oracle=stack.oracle(library);auto options=bounded();
    const auto started=std::chrono::steady_clock::now();options.stop_requested=[started]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>30;};
    if(variant<2){Device<U> bad(1);ck(cudaMemset(bad.p,0,4));sum_zero<<<1,128>>>(stack.columns(x.p,rows,true),bad.p,rows);finish();need(!scalar(bad),"adversarial fixture no longer has identical summed teacher margins");}
    if(variant==4){Device<U> flags(1);ck(cudaMemset(flags.p,0,4));multiround_order<<<1,128>>>(x.p,stack.columns(x.p,rows,true),flags.p,rows);finish();need(scalar(flags)==2,"three-round native margins differ from ordered FP32 additions or fixture lost order sensitivity");}
    auto converted=class_study::convert_composed_model(source,oracle,options);need(converted.metrics.at("complete")==true,"composed fixture incomplete");auto item=qualify(stack,converted,x,expected,rows);item["variant"]=variant;item["summed_teacher_margins_identical_across_regions"]=variant<2;item["boosting_rounds_per_base"]=variant==4?3:1;item["nonassociative_FP32_order_verified"]=variant==4;cases.push_back(item);
    if(variant==0||variant==4){
      // The existing first-order entrypoint must still produce native classes.
      class_study::NativeOracle first;first.features=2;first.classes=2;first.library_sha256=library;first.source_sha256=stack.pins[0];first.predict=[&](const float* values,L n,bool margin){return stack.bases[0]->predict(values,n,margin);};auto single=class_study::convert_model(source.bases[0],first,bounded());Device<U> flags(1);ck(cudaMemset(flags.p,0,4));decode<<<1,128>>>(first.predict(x.p,rows,false),expected.p,flags.p,rows);finish();need(scalar(flags)==3,"first-order fixture constant or invalid");single.runtime->verify({x.p,L(rows)*2,0,rows,2},expected.p,rows);
    }
    if(variant==0){
      // Invalid calibrated contracts fail before any native terminal callback.
      class_study::NativeOracle calibrated;calibrated.features=2;calibrated.classes=2;
      calibrated.library_sha256=library;calibrated.source_sha256=stack.pins[0];
      L calibrated_calls=0;
      calibrated.predict=[&](const float* values,L n,bool margin){++calibrated_calls;return stack.bases[0]->predict(values,n,margin);};
      for(float threshold:std::array<float,4>{-0.01f,1.01f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
        calibrated.positive_probability_threshold=threshold;
        refuse([&]{class_study::convert_model(source.bases[0],calibrated,bounded());});
      }
      need(calibrated_calls==0,"invalid calibrated threshold reached native prediction");
      Device<float> bad_classes;L invalid_calls=0;auto invalid=oracle;invalid.predict_classes=[&](const float*,L n){++invalid_calls;bad_classes.reserve(n);invalid_labels<<<64,128>>>(bad_classes.p,n);finish();return bad_classes.p;};refuse([&]{class_study::convert_composed_model(source,invalid,bounded());});need(invalid_calls>0,"class-range refusal did not reach the public callback");
      auto low=bounded();low.gpu_byte_budget=1;refuse([&]{class_study::convert_composed_model(source,oracle,low);});
      auto stopped=bounded();stopped.stop_requested=[](){return true;};refuse([&]{class_study::convert_composed_model(source,oracle,stopped);});
      auto proof=bounded();proof.proof_module_directory="unsupported-composed-proof-fixture";refuse([&]{class_study::convert_composed_model(source,oracle,proof);});
      auto resume=bounded();resume.resume_from="unsupported-composed-checkpoint-fixture";refuse([&]{class_study::convert_composed_model(source,oracle,resume);});
      auto wrong=source;wrong.bases[0].expected_source_sha256=std::string(64,'0');refuse([&]{class_study::convert_composed_model(wrong,oracle,bounded());});
    }
  }
  auto mixed=mixed_checks(plan,library);
  std::cout<<J{{"host",host},{"mixed_raw_cases",mixed},{"format","composed-converter-native-regression-1"},{"complete",true},{"real_native_base_margins_and_probability_views",true},{"real_native_meta_classes",true},{"first_order_native_equivalence",true},{"synthetic_models_not_study_outcomes",true},{"prediction_execution","CUDA"},{"CPU_predictions",false},{"cases",cases},{"refusal_cases",rejected}}.dump(2)<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
