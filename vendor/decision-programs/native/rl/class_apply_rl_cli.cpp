#include "class_rank_gpu_class_apply_source_receipt.hpp"
#include "class_rank_gpu_score_source_apply.hpp"
#include "class_rank_gpu_score_pool_identity.hpp"
#include <iostream>
#include "rl_category_lowering.hpp"
#include "rl_category_policy.hpp"
using U=rl_category_lowering::U;

static std::uint64_t metadata_integer(const char* value) {
 std::string text=value;
 if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)
   throw std::invalid_argument("RL metadata must be an unsigned integer");
 return std::stoull(text);
}

using namespace source_apply_receipt;namespace source_apply=rank_gpu_score_source_apply;
int main(int argc,char**argv){fs::path output;J receipt;try{need(argc==6||argc==7||argc==10||argc==11,"usage: class_apply_rl MODEL SOURCE_SHA LIB NEW_OUTPUT BUILD_TOKEN [on|off [SEED EPISODE BASELINE_BYTES [POLICY_STATE_JSON]]]");fs::path model=fs::absolute(argv[1]),lib=fs::absolute(argv[3]),out=fs::absolute(argv[4]),frozen=fs::absolute(argv[5]);auto start=std::chrono::steady_clock::now();auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};auto code=source_apply_receipt::build(frozen);auto bytes=dpnative::read_text(model);need(dpnative::sha256(bytes)==argv[2],"sequential class source SHA differs");bool rl_enabled=argc>=10;ca::Options options;if(argc>=7){need(std::string(argv[6])=="on"||std::string(argv[6])=="off","class cache mode invalid");options.state_cache=std::string(argv[6])=="on";}
 U seed=0,episode=0,baseline_bytes=0;
 std::string input_policy_bytes;
 fs::path input_policy_path;
 std::unique_ptr<rl_category_policy::Policy> policy;
 if(rl_enabled){
   seed=metadata_integer(argv[7]);episode=metadata_integer(argv[8]);baseline_bytes=metadata_integer(argv[9]);
   need(baseline_bytes>0,"RL baseline must be a completed encoded byte cost");
   policy=std::make_unique<rl_category_policy::Policy>();
   if(argc==11){
     input_policy_path=fs::absolute(argv[10]);input_policy_bytes=dpnative::read_text(input_policy_path);
     auto state=J::parse(input_policy_bytes);
     need(state.at("version").is_number_unsigned()&&state.at("logit_words").is_array()&&state.at("logit_words").size()==44,"policy word metadata shape");
     std::array<U,44> words{};
     for(unsigned k=0;k<44;++k){need(state["logit_words"][k].is_number_unsigned(),"policy word must be unsigned bits");words[k]=state["logit_words"][k].get<U>();}
     policy->upload(words,state["version"].get<U>());
   }
 }
need(!fs::exists(out)&&fs::create_directories(out/"audit")&&fs::create_directories(out/"runtime"),"sequential class output must be fresh");output=out;dpnative::atomic_text(out/"audit/source.json",bytes);receipt={{"format","source-tree-sequential-score-native-class-1"},{"passed",false},{"candidate_complete",false},{"whole_source_conversion_complete",false},{"whole_original_class_conversion_complete",false},{"whole_class_leaf_tuple_product_used",false},{"full_graph_or_state_JSON_materialized",false},{"frozen_build",code},{"state_cache",options.state_cache},{"CPU_geometry_or_predictions",false},{"independent_actual_native_grid_audit_required",true},{"timings_seconds",J::object()}};receipt["RL_ordering_enabled"]=rl_enabled;
 receipt["policy_update_performed"]=false;
 receipt["policy_actions_authorize_class_roots"]=false;
 auto save=[&](const char*phase){receipt["phase"]=phase;receipt["whole_process_seconds_including_identity_and_IO"]=seconds();dpnative::atomic_json(out/"result.json",receipt);};save("prepared");
 double before=seconds();lp::Bridge bridge(lib.string(),(out/"audit/source.json").string(),argv[2]);auto partitions=bridge.extract();receipt["bridge"]=leaf_receipt::describe(partitions);receipt["timings_seconds"]["bridge"]=seconds()-before;if(!partitions.complete){save("source_bridge_incomplete");return 2;}need(partitions.owned_device_resident_bytes<options.maximum_device_bytes,"bridge leaves no class working budget");options.maximum_device_bytes-=partitions.owned_device_resident_bytes;source_apply::Options score_options;score_options.maximum_device_bytes=options.maximum_device_bytes;before=seconds();auto folded=source_apply::fold(partitions,score_options);receipt["timings_seconds"]["source_tree_diagrams_ordered_Add_and_universal_audits"]=seconds()-before;receipt["source_score_fold"]=source_apply_receipt::describe(folded);if(!folded.complete||!folded.accepted){save("source_score_fold_incomplete");return 2;}const auto&accepted=*folded.accepted;need(accepted.matches(partitions),"accepted sequential scores source mismatch");const auto&scores=accepted.scores();receipt["sequential_scores"]={{"complete",true},{"source_digest",accepted.source_digest()},{"sequence_digest",accepted.sequence_digest()},{"score_pool_sha256",accepted.graph_digest()},{"score_nodes",scores.nodes.size()},{"score_arcs",scores.arcs.size()},{"ordered_class_roots",scores.roots},{"source_order_and_bias_preserved",true},{"same_process_opaque_acceptance",true}};std::cerr<<J{{"phase","sequential_scores_ready"},{"score_nodes",scores.nodes.size()},{"score_arcs",scores.arcs.size()}}.dump()<<'\n';
 before=seconds();auto result=ca::construct(accepted,partitions,lib.string(),(out/"audit/source.json").string(),options);receipt["timings_seconds"]["joint_class_Apply_native_queries_and_local_audit"]=seconds()-before;receipt["class_apply"]=describe(result);if(!result.complete){save("joint_class_Apply_incomplete");return 2;}before=seconds();rl_category_lowering::DeviceOrder action;
 if(rl_enabled){
   auto view=policy->sample44(seed,episode,baseline_bytes);
   action={view.order44,view.version};
   receipt["policy"]={{"version",view.version},{"seed",seed},{"episode",episode},{"baseline_bytes_frozen_before_sampling",baseline_bytes},{"logit_words",policy->logit_words()},{"owned_device_bytes",policy->owned_device_bytes()},{"credit_unit","one full44-order sample per completed conversion episode"}};
   if(!input_policy_bytes.empty())receipt["policy"]["input_state_sha256"]=dpnative::sha256(input_policy_bytes);
 }
 auto ordered=rl_category_lowering::lower_binary(result,action,1048576,options.maximum_device_bytes);
 receipt["categorical_lowering_order"]={{"injected",ordered.order_injected},{"trace_audited",ordered.trace_audited},{"policy_version",ordered.policy_version}};
 if(ordered.order_injected){
   J trace=J::array();
   for(const auto& t:ordered.traces){
     if(t.dimension<10||t.dimension>=12)continue;
     J tests=J::array();for(unsigned k=0;k<t.count;++k)tests.push_back(t.test_order[k]);
     trace.push_back(J{{"source_node",t.source_node},{"dimension",t.dimension},{"fallback_arc",t.fallback_arc},{"selected_mask",t.selected_mask},{"policy_version",t.policy_version},{"root_to_fallback_tests",tests}});
   }
   J artifact={{"format","GPU-audited-within-group-category-order-1"},{"source_sha256",partitions.source_sha256},{"source_binding",result.source_binding},{"domain_binding",result.domain_binding},{"policy_version",ordered.policy_version},{"selected_order44",ordered.selected_order},{"traces",trace},{"class_authority",false}};
   dpnative::atomic_json(out/"audit/category-order.json",artifact);
   receipt["categorical_lowering_order"]["trace_sha256"]=dpnative::sha256(dpnative::read_text(out/"audit/category-order.json"));
   receipt["categorical_lowering_order"]["selected_order44"]=ordered.selected_order;
 }
 auto lowered=std::move(ordered.lowered);receipt["binary_lowering"]=describe(lowered);if(!lowered.complete){save("binary_lowering_incomplete");return 2;}auto graph=canonical(result,partitions,std::move(lowered),out.string(),options.maximum_device_bytes);receipt["timings_seconds"]["binary_lowering_interning_and_collection"]=seconds()-before;before=seconds();auto encoded=codec::encode(graph.model);auto decoded=codec::decode(encoded);need(decoded.nodes==graph.model.nodes&&decoded.root==graph.model.root&&decoded.source_sha256==partitions.source_sha256&&decoded.rank_cut_bits==partitions.rank_cut_bits,"sequential class codec roundtrip differs");dpnative::atomic_text(out/"runtime/model.bin",encoded);auto runtime_sha=dpnative::sha256(encoded);auto readback=codec::read_model(out/"runtime/model.bin",runtime_sha);need(readback.nodes==graph.model.nodes&&readback.root==graph.model.root&&readback.rank_cut_bits==partitions.rank_cut_bits,"sequential class runtime disk readback differs");receipt["standalone_bounded_raw_quotient_audit"]=describe(ca::audit_binary_quotient(result,partitions,readback.nodes,readback.root,lib.string(),(out/"audit/source.json").string()));receipt["timings_seconds"]["codec_IO_and_bounded_native_binary_audit"]=seconds()-before;receipt["runtime_sha256"]=runtime_sha;receipt["runtime_model_bytes"]=encoded.size();receipt["runtime_nodes"]=readback.nodes.size();receipt["runtime_root"]=readback.root;receipt["unfolded_nodes"]=graph.root_module.expanded_nodes;receipt["unfolded_nodes_saturated"]=graph.root_module.nodes_saturated;receipt["arena_explicit_bytes"]=graph.arena_bytes;receipt["reported_stages_peak_excluding_collector"]=std::max(partitions.owned_device_peak_bytes,partitions.owned_device_resident_bytes+std::max({folded.owned_device_peak_bytes,result.owned_device_peak_bytes,graph.arena_bytes}));receipt["collector_explicit_device_byte_cap"]=options.maximum_device_bytes;receipt["whole_pipeline_exact_peak_available"]=false;receipt["runtime_requires_teacher_or_proof_logs"]=false;receipt["native_workspace_and_implicit_stacks_in_owned_byte_count"]=false;need(accepted.matches(partitions)&&source_apply_receipt::build(frozen)==code&&dpnative::read_text(model)==bytes&&dpnative::read_text(out/"audit/source.json")==bytes&&dpnative::read_text(out/"runtime/model.bin")==encoded,"sequential class source/authority/build/runtime changed");if(!input_policy_bytes.empty())need(dpnative::read_text(input_policy_path)==input_policy_bytes,"input policy word file changed");
 receipt["candidate_complete"]=true;receipt["passed"]=true;save("candidate_ready_for_independent_native_grid");std::cout<<receipt.dump(2)<<'\n';return 0;
 }catch(const std::exception&e){receipt["passed"]=false;receipt["candidate_complete"]=false;receipt["whole_source_conversion_complete"]=false;receipt["error"]=e.what();if(!output.empty())try{dpnative::atomic_json(output/"result.json",receipt);}catch(...){}std::cerr<<e.what()<<'\n';return 1;}}
