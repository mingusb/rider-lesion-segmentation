#include "class_tree_adapter.hpp"
#include "class_io.hpp"

#include <charconv>
#include <iostream>

namespace {
using U=std::uint64_t;
using W=std::uint32_t;
using J=dpnative::json;
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void append(std::string& b,U x,unsigned n){for(unsigned i=0;i<n;++i)b.push_back(char(x>>(8*i)));}
void replace(std::string& b,U at,U x,unsigned n){for(unsigned i=0;i<n;++i)b.at(at+i)=char(x>>(8*i));}
W capacity(const std::string& s){W v=0;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),v);
  need(e==std::errc{}&&p==s.data()+s.size()&&v>0&&v<=INT32_MAX,"invalid node capacity");return v;}
std::string bounded_read(const dpnative::fs::path& p,U limit){
  auto n=dpnative::fs::file_size(p);need(n<=limit,"input exceeds explicit byte bound");
  auto b=dpnative::read_text(p);need(b.size()==n,"input changed during read");return b;}
void fresh(const dpnative::fs::path& p){need(!dpnative::fs::exists(p),"output directory must be fresh");dpnative::fs::create_directories(p);}
struct N { std::int32_t f;W cut;unsigned m;U l,r;std::int64_t label;};
std::string fixture(U f,U k,const std::vector<N>& v,U depth){std::string b="CLSTREE1";
  append(b,f,8);append(b,k,8);append(b,v.size(),8);append(b,depth,8);
  for(auto n:v)append(b,W(n.f),4);
  for(auto n:v)append(b,n.cut,4);
  for(auto n:v)append(b,n.m,1);
  for(auto n:v)append(b,n.l,8);
  for(auto n:v)append(b,n.r,8);
  for(auto n:v)append(b,U(n.label),8);
  return b;}
J self_test(){
  J cases=J::array();unsigned failed=0;
  auto check=[&](const std::string& name,auto action,bool expect){
    bool passed=false;std::string message;
    try{action();passed=expect;message=expect?"accepted":"unexpected acceptance";}
    catch(const std::exception& e){passed=!expect;message=e.what();}
    failed+=!passed;cases.push_back({{"case",name},{"passed",passed},{"expected_acceptance",expect},{"observed",message}});
  };
  const U no=UINT64_MAX;
  auto leaf=[&](std::int64_t label){return N{-1,0,0,no,no,label};};
  std::vector<N> nodes{{10,0x80000000U,1,3,1,-1},leaf(2),leaf(0),{0,0x7f7fffffU,0,2,4,-1},leaf(1)};
  auto base=fixture(11,3,nodes,2);
  auto adapt=[&](const std::string& b,W cap=1000000){return class_tree_adapter::adapt(b,dpnative::sha256(b),{cap});};
  check("dynamic_F11_K3_nonpreorder_negative_zero_and_both_NaN_defaults",[&]{auto r=adapt(base);need(r.root==4&&r.features==11&&r.classes==3&&class_tree_adapter::inverse(r,base)==base,"roundtrip differs");},true);
  for(auto [f,k]:std::vector<std::pair<U,U>>{{1,2},{784,10},{1000001,1025},{INT32_MAX,UINT32_MAX}}){
    auto v=std::vector<N>{{std::int32_t(f-1),0,1,1,2,-1},leaf(0),leaf(std::int64_t(k-1))};
    auto b=fixture(f,k,v,1);check("dynamic_shape_"+std::to_string(f)+"_"+std::to_string(k),[&]{adapt(b);},true);
  }
  auto inactive=leaf(1);inactive.f=INT32_MIN;inactive.cut=0x7fc01234U;inactive.m=255;
  auto singleton=fixture(1,2,{inactive},0);
  check("leaf_only_retains_all_inactive_raw_metadata_in_inverse",[&]{auto r=adapt(singleton);need(class_tree_adapter::inverse(r,singleton)==singleton,"inactive words differ");},true);
  check("exact_explicit_node_capacity",[&]{adapt(base,5);},true);
  check("too_small_explicit_node_capacity",[&]{adapt(base,4);},false);
  check("zero_capacity",[&]{adapt(base,0);},false);
  check("wrong_original_hash",[&]{class_tree_adapter::adapt(base,std::string(64,'0'));},false);
  check("malformed_original_hash",[&]{class_tree_adapter::adapt(base,"bad");},false);
  auto corrupt=[&](const std::string& name,U at,U value,unsigned width){auto b=base;replace(b,at,value,width);check(name,[&]{adapt(b);},false);};
  check("truncated_header",[&]{adapt(base.substr(0,39));},false);
  check("truncated_body",[&]{adapt(base.substr(0,base.size()-1));},false);
  check("trailing_byte",[&]{adapt(base+"x");},false);
  check("weighted_regional_magic_explicit_refusal",[&]{auto b=base;b.replace(0,8,"CLSRMDL1");adapt(b);},false);
  check("probability_tree_magic_explicit_refusal",[&]{auto b=base;b.replace(0,8,"DPDTREE1");adapt(b);},false);
  corrupt("zero_features",8,0,8);corrupt("feature_count_unrepresentable",8,U(INT32_MAX)+1,8);
  corrupt("one_class_unsupported_by_shared_classifier",16,1,8);corrupt("class_count_unrepresentable",16,U(UINT32_MAX)+1,8);
  corrupt("zero_nodes",24,0,8);corrupt("overflow_node_count",24,UINT64_MAX,8);
  corrupt("wrong_depth",32,1,8);corrupt("excess_depth",32,5,8);
  corrupt("feature_out_of_range",40,11,4);corrupt("negative_branch_feature",40,W(-2),4);
  for(W cut:{0x7f800000U,0xff800000U})for(W missing:{0U,1U}){auto b=base;replace(b,60,cut,4);replace(b,80,missing,1);check("infinite_cut_bit_exact_"+std::to_string(cut)+"_"+std::to_string(missing),[&]{auto a=adapt(b);need(class_tree_adapter::inverse(a,b)==b,"infinite cut or missing direction changed");},true);}
  corrupt("NaN_cut",60,0x7fc00001U,4);
  corrupt("invalid_branch_missing_flag",80,2,1);
  corrupt("child_out_of_range",85,5,8);corrupt("root_cycle",85,0,8);
  corrupt("duplicate_children",125,3,8);corrupt("leaf_has_successor",93,0,8);
  corrupt("unresolved_terminal",173,3,8);corrupt("unsupported_negative_branch_label",165,U(-2),8);
  auto r=adapt(base);
  check("inverse_duplicate_mapping_refusal",[&]{auto v=r;v.original_to_canonical[1]=v.original_to_canonical[2];class_tree_adapter::inverse(v,base);},false);
  check("inverse_metadata_refusal",[&]{auto v=r;++v.features;class_tree_adapter::inverse(v,base);},false);
  check("inverse_stale_canonical_hash_refusal",[&]{auto v=r;v.canonical_bytes[64]^=1;class_tree_adapter::inverse(v,base);},false);
  check("inverse_changed_predicate_even_with_fresh_hash",[&]{auto v=r;replace(v.canonical_bytes,64+16*v.root+4,1,4);v.canonical_sha256=dpnative::sha256(v.canonical_bytes);class_tree_adapter::inverse(v,base);},false);
  check("inverse_changed_leaf_even_with_fresh_hash",[&]{auto v=r;replace(v.canonical_bytes,68,1,4);v.canonical_sha256=dpnative::sha256(v.canonical_bytes);class_tree_adapter::inverse(v,base);},false);
  check("inverse_changed_source_identity_even_with_fresh_hash",[&]{auto v=r;v.canonical_bytes[32]^=1;v.canonical_sha256=dpnative::sha256(v.canonical_bytes);class_tree_adapter::inverse(v,base);},false);
  check("deep_iterative_topology_no_recursive_host_stack",[&]{
    constexpr W depth=10000;std::vector<N> v;v.reserve(2*depth+1);
    for(W i=0;i<depth;++i){v.push_back({0,0,i%2,U(2*i+1),U(2*i+2),-1});v.push_back(leaf(i%2));}
    v.push_back(leaf(1));auto b=fixture(1,2,v,depth);auto a=adapt(b,W(v.size()));need(a.maximum_depth==depth,"depth differs");
  },true);
  return {{"format","generic-class-tree-adapter-CPU-fixtures-1"},{"complete",failed==0},{"passed",failed==0},{"cases",cases},{"case_count",cases.size()},{"failures",failed},{"CUDA_executed",false},{"CPU_predictions_executed",false},{"scope","wire structure, exact active words, topology bijection and byte inverse; no dataset numerical inference or source-proof replay"}};
}
}  // namespace

int main(int argc,char** argv){try{
  if(argc==3&&std::string(argv[1])=="self-test"){
    auto j=self_test();fresh(argv[2]);dpnative::atomic_json(dpnative::fs::path(argv[2])/"result.json",j);
    std::cout<<J{{"passed",j.at("passed")},{"cases",j.at("case_count")},{"failures",j.at("failures")}}.dump()<<'\n';return j.at("passed").get<bool>()?0:1;
  }
  need(argc==7,"usage: class_tree_adapter INPUT INPUT_SHA ORIGIN_JSON ORIGIN_SHA FRESH_OUTPUT MAX_NODES; or self-test FRESH_OUTPUT");
  W cap=capacity(argv[6]);auto original=bounded_read(argv[1],40+33*U(cap));
  auto origin_bytes=bounded_read(argv[3],16*1024*1024);
  need(dpnative::sha256(origin_bytes)==argv[4],"origin contract SHA256 differs");
  auto origin=J::parse(origin_bytes);
  need(origin.at("runtime_sha256")==argv[2]&&origin.at("runtime_domain").is_string()
       &&!origin.at("runtime_domain").get<std::string>().empty(),"origin contract must bind the input runtime and declared domain");
  auto r=class_tree_adapter::adapt(original,argv[2],{cap});
  fresh(argv[5]);dpnative::fs::path out=argv[5];
  dpnative::atomic_text(out/"model.clsgdag1",r.canonical_bytes);
  dpnative::atomic_text(out/"original.clstree1",original);
  dpnative::atomic_text(out/"origin-contract.json",origin_bytes);
  J mapping={{"format","class-tree-postorder-bijection-1"},{"original_sha256",r.original_sha256},{"canonical_sha256",r.canonical_sha256},{"original_to_canonical",r.original_to_canonical}};
  dpnative::atomic_json(out/"inverse-mapping.json",mapping);
  auto wire=dpnative::read_text(out/"model.clsgdag1");need(wire==r.canonical_bytes,"canonical disk bytes differ");
  auto original_disk=dpnative::read_text(out/"original.clstree1");need(class_tree_adapter::inverse(r,original_disk)==original,"disk inverse differs");
  J report={{"format","generic-class-tree-canonical-adapter-1"},{"complete",true},{"CPU_structural_qualification_passed",true},{"CUDA_executed",false},{"CPU_predictions_executed",false},{"GPU_qualification_pending",true},{"source_identity_role","exact original CLSTREE1 runtime SHA256, opaque origin provenance"},{"source_sha256",r.original_sha256},{"original_sha256",r.original_sha256},{"canonical_sha256",r.canonical_sha256},{"origin_contract_path",argv[3]},{"origin_contract_sha256",argv[4]},{"runtime_domain_inherited",origin.at("runtime_domain")},{"origin_contract_replayed_or_expanded",false},{"source_equivalence_certificate_replayed",false},{"features",r.features},{"classes",r.classes},{"nodes",r.nodes},{"root",r.root},{"maximum_depth",r.maximum_depth},{"leaves",r.leaves},{"branches",r.branches},{"original_bytes",original.size()},{"canonical_bytes",r.canonical_bytes.size()},{"maximum_nodes",cap},{"node_merging_or_pruning",false},{"raw_active_predicate_words_preserved",true},{"class_labels_preserved",true},{"NaN_directions_preserved",true},{"signed_zero_cut_words_preserved",true},{"exact_original_byte_inverse",true},{"original_inactive_leaf_metadata_retained_in_transport_sidecar",true},{"deployment_needs_original_or_inverse_sidecar",false},{"inference_API_returns_classes_not_probabilities",true},{"compact_encoding","pending shared CUDA runtime encoder; no host numerical substitution"},{"scope","completed finite-cut axis-only CLSTREE1 to CLSGDAG1 v1; representable dynamic F/K; weighted/regional/rank/probability and unresolved-leaf formats explicitly unsupported"}};
  report["preprocessing_obligations_removed"]=false;
  report["input_representation_contract"]= "original feature coordinates unchanged; any original preprocessing/domain obligation remains bound to the copied origin contract";
  report["files"]=J::object();for(auto name:{"model.clsgdag1","original.clstree1","origin-contract.json","inverse-mapping.json"})report["files"][name]={{"bytes",dpnative::fs::file_size(out/name)},{"sha256",dpnative::sha256(dpnative::read_text(out/name))}};
  dpnative::atomic_json(out/"result.json",report);
  std::cout<<J{{"complete",true},{"features",r.features},{"classes",r.classes},{"nodes",r.nodes},{"canonical_sha256",r.canonical_sha256},{"GPU_qualification_pending",true}}.dump()<<'\n';
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
