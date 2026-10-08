#include "class_study_grouped_segmentation.hpp"
#include "class_io.hpp"
#include "class_study_train.hpp"
#include "class_runtime.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <bit>
#include <cmath>
#include <set>
#include <algorithm>
#include <iostream>

namespace {
using J=nlohmann::json;
template<class T> void write(const std::filesystem::path& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));}
void need(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void ck(cudaError_t result){if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));}
template<class T> struct Device {T* p=nullptr;explicit Device(std::size_t n){ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;Device& operator=(const Device&)=delete;};
}
int main(int argc,char** argv){
  const bool pure_missing=argc==4&&std::string(argv[3])=="--missing-only-split";
  const bool missing_only=pure_missing||(argc==4&&std::string(argv[3])=="--missing-features-only");
  const bool mixed_only=(argc==4&&std::string(argv[3])=="--meta-inputs-only");
  if(argc<3||argc>4||(argc==4&&!mixed_only&&!missing_only&&std::string(argv[3])!="--gate-only"&&std::string(argv[3])!="--folds-only"&&std::string(argv[3])!="--reference-only"&&std::string(argv[3])!="--inverse-reference-only"&&std::string(argv[3])!="--threshold-only"&&std::string(argv[3])!="--patient-weights-only"&&std::string(argv[3])!="--threshold-no-final-only"&&std::string(argv[3])!="--threshold-histogram-only"&&std::string(argv[3])!="--threshold-histogram-benchmark"&&std::string(argv[3])!="--evaluation-batches-only"&&std::string(argv[3])!="--group-loss-only"&&std::string(argv[3])!="--rule-views-only")){std::cerr<<"usage: grouped-segmentation-integration-checks LIBXGBOOST NEW_OUTPUT_DIRECTORY [--gate-only|--folds-only|--reference-only|--inverse-reference-only|--threshold-only|--patient-weights-only|--threshold-no-final-only|--threshold-histogram-only|--threshold-histogram-benchmark|--evaluation-batches-only|--group-loss-only|--rule-views-only|--meta-inputs-only|--missing-features-only|--missing-only-split]\n";return 1;}
  const bool folds_only=argc==4&&std::string(argv[3])=="--folds-only";
  std::filesystem::path root=std::filesystem::absolute(argv[2]);need(!std::filesystem::exists(root),"new fixture directory required");std::filesystem::create_directories(root);
  std::vector<float> x;std::vector<std::uint32_t> y,pid,sid,eligible;J scans=J::array();
  // Two repeat scans per patient; the target is intentionally simple. The
  // second run changes ONLY outer labels and checks frozen model identities.
  for(unsigned p=0;p<5;++p)for(unsigned s=0;s<2;++s){scans.push_back({{"patient_id",p},{"scan_id",10*p+s},{"repeat_index",s},{"truth_positive_outside_rows",s==0?5:0}});for(unsigned i=0;i<64;++i){x.push_back(float(i));x.push_back(float((i*13+p)%17));y.push_back(i>=32);pid.push_back(p);sid.push_back(10*p+s);eligible.push_back(i%2==0);}}
  write(root/"x.fp32",x);write(root/"y.u32",y);write(root/"pid.u32",pid);write(root/"sid.u32",sid);write(root/"eligible.u32",eligible);
  J d={{"format","grouped-segmentation-dataset-1"},{"rows",y.size()},{"features",2},{"classes",2},{"train_eligible_rows",y.size()/2},{"domain_description","synthetic fixed two-scan patient leakage fixture"},{"values_path",(root/"x.fp32").string()},{"labels_path",(root/"y.u32").string()},{"patient_ids_path",(root/"pid.u32").string()},{"scan_ids_path",(root/"sid.u32").string()},{"train_eligible_path",(root/"eligible.u32").string()},{"scans",scans}};
  J hp={{"rounds",2},{"max_depth",2},{"max_bin",16},{"lambda",0},{"min_child_weight",0},{"eta",1}};
  J plan={{"native_library_path",std::filesystem::absolute(argv[1]).string()},{"native_library_sha256",dpnative::sha256(dpnative::read_text(argv[1]))},{"base_hyperparameters",J::array({hp})},{"meta_hyperparameters",J::array({hp})},{"inner_folds",3},{"oof_folds",3},{"outer_patient_ids",J::array({0})}};
  volatile std::sig_atomic_t stop=0;
  if(missing_only){
    // Both cases remain required: pure missing-vs-constant exposes native
    // infinite split tokens; the other forces a finite split with NaN routing.
    for(std::size_t row=0;row<y.size();++row){x[row*2]=y[row]?(pure_missing||row%4<2?std::bit_cast<float>(0x7fc12345u):1.f):0.f;x[row*2+1]=0.f;}
    write(root/"missing-x.f32",x);auto data=d;data["values_path"]=(root/"missing-x.f32").string();data["missing_feature_indices"]={0};
    auto p=plan;p["final_deployment"]={{"convert",false}};p["evaluation_batch_rows"]=17;
    unsigned refused=0;auto deny=[&](const J& bad,const char* name){bool failed=false;try{(void)class_study::grouped_segmentation::run(plan,bad,root/name,stop);}catch(const std::exception& e){failed=std::string(e.what()).find("invalid class, feature, eligibility or scan/patient identity")!=std::string::npos;}need(failed,"missing feature input audit did not refuse malformed values");++refused;};
    auto undeclared=data;undeclared.erase("missing_feature_indices");deny(undeclared,"undeclared");
    auto values=x;values[1]=std::numeric_limits<float>::quiet_NaN();write(root/"bad-column.f32",values);auto bad=data;bad["values_path"]=(root/"bad-column.f32").string();deny(bad,"wrong-column");
    for(unsigned sign=0;sign<2;++sign){values=x;values[0]=sign?-std::numeric_limits<float>::infinity():std::numeric_limits<float>::infinity();auto path=root/(sign?"negative-infinity.f32":"positive-infinity.f32");write(path,values);bad=data;bad["values_path"]=path.string();deny(bad,sign?"negative-infinity":"positive-infinity");}
    auto labels=y;labels[0]=2;write(root/"bad-label.u32",labels);bad=data;bad["labels_path"]=(root/"bad-label.u32").string();deny(bad,"bad-label");
    auto fit=eligible;fit[0]=2;write(root/"bad-eligibility.u32",fit);bad=data;bad["train_eligible_path"]=(root/"bad-eligibility.u32").string();deny(bad,"bad-eligibility");
    need(class_study::grouped_segmentation::run(p,data,root/"allowed",stop)==0,"declared missing study incomplete");auto result=J::parse(dpnative::read_text(root/"allowed"/"result.json"));
    need(result.at("protocol").at("missing_feature_indices")==J::array({0})&&result.at("dataset").at("missing_feature_indices")==J::array({0}),"missing declaration lost in result");
    for(const auto& metric:result.at("metrics"))for(const auto& scan:metric.at("scans"))need(scan.at("TP")==32&&scan.at("FP")==0&&scan.at("FN_in_ROI")==0&&scan.at("TN_in_ROI")==32,"native missing branch differs from independent missing-vs-zero oracle");
    const auto pipeline=result.at("final_deployment").at("native_pipeline");need(pipeline.at("missing_feature_indices")==J::array({0})&&pipeline.at("family")!="external_reference","saved missing model declaration differs");
    J replay={{"native_library_path",plan.at("native_library_path")},{"native_library_sha256",plan.at("native_library_sha256")},{"conversion",{{"qualification_mode","exhaustive"}}}};
    need(class_study::grouped_segmentation::replay(replay,pipeline,root/"replay",stop)==0,"missing-branch saved pipeline replay failed");auto converted=J::parse(dpnative::read_text(root/"replay"/"result.json"));need(converted.at("completed_native_fits")==0&&converted.at("qualified_native_class_root")==true,"missing-branch explicit qualification failed");
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"pure_missing_split",pure_missing},{"missing_feature_indices",{0}},{"input_refusals",refused},{"native_missing_branch_independent_counts",true},{"saved_pipeline_missing_declaration",true},{"replay_without_fit",true},{"explicit_fixture_exhaustive_qualification",true},{"ordinary_verification_default_unchanged",true}}.dump(2)<<'\n';return 0;
  }
  if(mixed_only){
    std::vector<float> values;std::vector<unsigned> labels,patients,scan_ids,training;J rows=J::array();
    for(unsigned g=0;g<5;++g)for(unsigned visit=0;visit<2;++visit){unsigned patient=10+g*17,scan=100+g*31+visit*7;rows.push_back({{"patient_id",patient},{"scan_id",scan},{"rows",16},{"training_rows",16},{"truth_positive_rows",12},{"truth_positive_outside_rows",0}});
      for(unsigned i=0;i<16;++i){unsigned a=i%2,b=(i/2)%2;values.insert(values.end(),{float(a),float(b)});labels.push_back(a||b);patients.push_back(patient);scan_ids.push_back(scan);training.push_back(1);}}
    write(root/"mixed-x.f32",values);write(root/"mixed-y.u32",labels);write(root/"mixed-pid.u32",patients);write(root/"mixed-sid.u32",scan_ids);write(root/"mixed-eligible.u32",training);
    J data={{"format","grouped-segmentation-dataset-1"},{"rows",labels.size()},{"features",2},{"classes",2},{"train_eligible_rows",labels.size()},{"domain_description","synthetic mixed-meta OR fixture with nonconsecutive repeated-patient IDs"},{"values_path",(root/"mixed-x.f32").string()},{"labels_path",(root/"mixed-y.u32").string()},{"patient_ids_path",(root/"mixed-pid.u32").string()},{"scan_ids_path",(root/"mixed-sid.u32").string()},{"train_eligible_path",(root/"mixed-eligible.u32").string()},{"scans",rows}};
    auto p=plan,base=hp,second=hp,meta=hp;base.update({{"rounds",1},{"max_depth",1}});second=base;second["eta"]=.1;meta.update({{"rounds",1},{"max_depth",2}});
    p["outer_patient_ids"]={10};p["base_hyperparameters"]=J::array({base,second});p["meta_hyperparameters"]=J::array({meta});p["meta_input_feature_indices"]={1,0};p["evaluation_batch_rows"]=7;
    const auto mapping=J{{"version",1},{"raw_feature_indices",{1,0}}};
    auto probability=class_study::grouped_segmentation::evaluation_probability_checks(p,data);auto odd=p;odd["meta_input_feature_indices"]={1};auto odd_probability=class_study::grouped_segmentation::evaluation_probability_checks(odd,data);
    p["native_prediction_thresholds"]={.5,.9};p["rule_search_views"]=J::array({J{{"id","all"},{"native_prediction_candidate_indices",{0,1,2}}},J{{"id","central_threshold"},{"native_prediction_candidate_indices",{1}}}});
    auto execute=[&](const J& recipe,const J& descriptor,const char* name){need(class_study::grouped_segmentation::run(recipe,descriptor,root/name,stop)==0,"mixed meta study incomplete");return J::parse(dpnative::read_text(root/name/"result.json"));};
    const auto original=execute(p,data,"mixed");const auto& outer=original.at("outer_results")[0];need(outer.at("selection").at("selected_family")=="nonlinear_combiner","OR fixture failed to select mixed combiner");
    need(original.at("protocol").at("meta_input_mapping")==mapping&&original.at("protocol").at("meta_input_columns")==6,"mixed mapping or width lost");
    need(outer.at("rule_search_views")[0].at("metrics")==outer.at("metrics")&&outer.at("rule_search_views")[0].at("baseline_model_sha256")==outer.at("baseline_model_sha256")&&outer.at("rule_search_views")[0].at("combiner_model_sha256")==outer.at("combiner_model_sha256"),"mixed paired full alias differs on shared models");
    need(outer.at("rule_search_views")[1].at("selection").at("baseline_prediction_candidate_index")==1&&outer.at("rule_search_views")[1].at("selection").at("combiner_prediction_candidate_index")==1,"mixed paired fixture failed to force threshold stride path");
    for(const auto& scan:outer.at("rule_search_views")[1].at("metrics")[0].at("scans"))need(scan.at("TP")==12&&scan.at("FP")==4&&scan.at("FN_in_ROI")==0&&scan.at("TN_in_ROI")==0,"mixed threshold base differs from independently specified stump tie counts");
    for(unsigned method:{1u,2u})for(const auto& scan:outer.at("rule_search_views")[1].at("metrics")[method].at("scans"))need(scan.at("TP")==12&&scan.at("FP")==0&&scan.at("FN_in_ROI")==0&&scan.at("TN_in_ROI")==4,"mixed threshold combiner differs from OR counts");
    for(const auto& batch:original.at("evaluation_batches"))need(batch.at("all_base_response_columns")==4&&batch.at("meta_input_columns")==6&&batch.at("meta_input_mapping")==mapping,"mixed evaluation columns differ");
    for(unsigned method:{1u,2u})for(const auto& scan:outer.at("metrics")[method].at("scans"))need(scan.at("TP")==12&&scan.at("FP")==0&&scan.at("FN_in_ROI")==0&&scan.at("TN_in_ROI")==4,"mixed fixture differs from independent OR counts");
    auto perturbed=data;auto changed=labels;for(std::size_t row=0;row<labels.size();++row)if(patients[row]==10)changed[row]=1-labels[row];write(root/"mixed-heldout-labels.u32",changed);perturbed["labels_path"]=(root/"mixed-heldout-labels.u32").string();for(auto& scan:perturbed["scans"])if(scan.at("patient_id")==10){scan["truth_positive_rows"]=4;scan["truth_positive_outside_rows"]=3;}
    const auto altered=execute(p,perturbed,"heldout-perturbed");for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256"})need(outer.at(key)==altered.at("outer_results")[0].at(key),"heldout labels changed mixed OOF selection/models");
    auto final=p;final.erase("rule_search_views");final.erase("native_prediction_thresholds");final["final_deployment"]={{"convert",false}};const auto deployed=execute(final,data,"final-native");const auto pipeline=deployed.at("final_deployment").at("native_pipeline");
    need(pipeline.at("family")=="nonlinear_combiner"&&pipeline.at("meta_input_mapping")==mapping&&pipeline.at("meta_model").at("features")==pipeline.at("base_models").size()*2+2,"mixed final pipeline mapping/pruning width differs");
    J replay={{"native_library_path",plan.at("native_library_path")},{"native_library_sha256",plan.at("native_library_sha256")},{"conversion",{{"qualification_mode","exhaustive"}}}};
    class_study::grouped_segmentation::validate_replay(replay,pipeline);need(class_study::grouped_segmentation::replay(replay,pipeline,root/"replay",stop)==0,"mixed saved pipeline conversion replay failed");
    const auto replayed=J::parse(dpnative::read_text(root/"replay"/"result.json"));need(replayed.at("completed_native_fits")==0&&replayed.at("qualified_native_class_root")==true&&replayed.at("conversion").at("meta_input_mapping")==mapping,"mixed replay lost mapping or exhaustive fixture qualification");
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"ordered_mapping",mapping},{"even_raw_suffix_probability_checks",probability},{"odd_raw_suffix_probability_checks",odd_probability},{"OOF_original_row_words_checked",true},{"heldout_labels_excluded",true},{"independent_OR_counts",true},{"paired_full_alias_on_same_mixed_models",true},{"saved_pruned_pipeline_replayed",true},{"explicit_fixture_exhaustive_qualification",true},{"ordinary_verification_default_unchanged",true}}.dump(2)<<'\n';return 0;
  }
  if(argc==4&&std::string(argv[3])=="--rule-views-only"){
    auto chooser=class_study::grouped_segmentation::rule_search_view_checks();auto p=plan;
    auto weak=hp,strong=hp;weak.update({{"rounds",1},{"max_depth",1},{"eta",.1}});strong.update({{"rounds",3},{"max_depth",1},{"eta",1.}});
    p["base_hyperparameters"]=J::array({weak,strong});p["meta_hyperparameters"]=J::array({weak,strong});p["native_prediction_thresholds"]={.5,.9};p["evaluation_batch_rows"]=17;p["inner_result_encoding"]="scan_counts_v1";
    // Interleave hard-class and probability rules for common models; the high
    // cutoff requires a stronger model than the byte-preferred central rule.
    p["rule_search_views"]=J::array({J{{"id","all"},{"native_prediction_candidate_indices",{0,1,2}}},J{{"id","threshold"},{"native_prediction_candidate_indices",{1}}},J{{"id","argmax"},{"native_prediction_candidate_indices",{0}}},J{{"id","high"},{"native_prediction_candidate_indices",{2}}}});
    auto execute=[&](const J& recipe,const J& data,const char* name){need(class_study::grouped_segmentation::run(recipe,data,root/name,stop)==0,"paired rule study incomplete");return J::parse(dpnative::read_text(root/name/"result.json"));};
    const auto original=execute(p,d,"paired");const auto& outer=original.at("outer_results")[0];const auto& selection=outer.at("selection");const auto& views=outer.at("rule_search_views");const auto& receipt=outer.at("paired_outer_execution");
    need(original.at("complete")==true&&original.at("completed_outer_patients")==1&&views.size()==4,"paired completion/coverage differs");
    need(views[0].at("metrics")==outer.at("metrics")&&views[0].at("baseline_model_sha256")==outer.at("baseline_model_sha256")&&views[0].at("combiner_model_sha256")==outer.at("combiner_model_sha256"),"full additional view differs on the same fitted models");
    need(views[1].at("baseline_model_sha256")==views[2].at("baseline_model_sha256")&&views[1].at("combiner_model_sha256")==views[2].at("combiner_model_sha256"),"central/argmax fixture failed to share both models");
    need(views[3].at("baseline_model_sha256")!=views[2].at("baseline_model_sha256")&&views[3].at("combiner_model_sha256")!=views[2].at("combiner_model_sha256"),"fixture failed to exercise distinct baseline/meta winners");
    need(receipt.at("all_base_models").size()==2&&receipt.at("union_meta_models").size()==2,"paired bank/meta union differs");
    const auto& calls=receipt.at("prediction_calls");auto tiles=original.at("evaluation_batches").back().at("batches").get<unsigned>();
    need(calls.at("base_probability")==2*tiles&&calls.at("base_native_class")==tiles&&calls.at("meta_probability")==2*tiles&&calls.at("meta_native_class")==tiles,"duplicate or missing outer native prediction passes");
    need(calls.at("registration_probability_calls")==4&&calls.at("registration_native_class_calls")==4,"registration pass accounting differs");
    unsigned inner_fits=0;for(const auto& held:selection.at("inner_validation_patient_groups")){auto fit=4u-unsigned(held.size());inner_fits+=2*(std::min(3u,fit)+1)+2;}
    need(receipt.at("inner_native_fits")==inner_fits&&receipt.at("outer_refit_native_fits")==10&&original.at("completed_native_fits")==inner_fits+10,"rule views added an extra base or duplicate meta FIT");
    for(const auto& view:views)for(const auto& metric:view.at("metrics")){need(metric.at("patients").size()==1&&metric.at("scans").size()==2,"view omitted patient or repeat scan");need(metric.at("scans")[0].at("FN_outside_ROI")==5,"view omitted outside-ROI truth");
      for(const auto& scan:metric.at("scans"))need(scan.at("TP")==32&&scan.at("FP")==0&&scan.at("FN_in_ROI")==0&&scan.at("TN_in_ROI")==32,"paired prediction differs from independently specified separable fixture counts");}
    auto default_plan=p;default_plan.erase("rule_search_views");const auto single=execute(default_plan,d,"default");
    need(!single.contains("rule_search_views")&&!single.at("protocol").contains("rule_search_views")&&!single.at("outer_results")[0].contains("paired_outer_execution"),"default acquired optional view fields");
    need(single.at("metrics")==original.at("metrics")&&single.at("outer_results")[0].at("baseline_model_sha256")==outer.at("baseline_model_sha256")&&single.at("outer_results")[0].at("combiner_model_sha256")==outer.at("combiner_model_sha256"),"optional views changed deterministic fixture primary outcome");
    auto perturbed=d;auto changed=y;for(std::size_t row=0;row<y.size();++row)if(pid[row]==0)changed[row]=1-y[row];write(root/"views-heldout-labels.u32",changed);perturbed["labels_path"]=(root/"views-heldout-labels.u32").string();for(auto& scan:perturbed["scans"])if(scan.at("patient_id")==0)scan["truth_positive_outside_rows"]=17;
    const auto altered=execute(p,perturbed,"heldout-perturbed");const auto& altered_outer=altered.at("outer_results")[0];
    for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256","paired_outer_execution"})need(outer.at(key)==altered_outer.at(key),"outer labels altered paired selection, model banks or FIT/pass schedule");
    for(unsigned v=0;v<views.size();++v)for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256"})need(views[v].at(key)==altered_outer.at("rule_search_views")[v].at(key),"outer labels altered an additional view choice");
    need(outer.at("metrics")!=altered_outer.at("metrics"),"held-out perturbation did not change paired metrics");
    std::vector<float> reference_values;for(std::size_t row=0;row<y.size();++row)reference_values.insert(reference_values.end(),{x[2*row],x[2*row+1],x[2*row]>=32?1.f:0.f});write(root/"views-reference-x.fp32",reference_values);
    auto reference_data=d;reference_data["features"]=3;reference_data["values_path"]=(root/"views-reference-x.fp32").string();auto reference_plan=p;reference_plan["reference_predictor"]={{"mask_feature_index",2}};
    const auto referenced=execute(reference_plan,reference_data,"with-reference");const auto& reference_outer=referenced.at("outer_results")[0];need(reference_outer.at("selection").at("selected_family")=="external_reference","fixture failed to exercise reference tie preference");
    for(const auto& view:reference_outer.at("rule_search_views")){need(view.at("selection").at("selected_family")=="external_reference","paired view lost reference tie preference");for(unsigned method:{2u,3u,4u})need(view.at("metrics")[method]==reference_outer.at("metrics")[method],"paired reference counts differ");}
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"chooser",chooser},{"same_model_full_view_equal",true},{"argmax_threshold_shared_outputs",true},{"distinct_base_and_meta_winners",true},{"meta_union_size",2},{"no_reference_and_reference_modes",true},{"heldout_labels_excluded",true},{"default_fixture_unchanged",true},{"prediction_pass_counts_checked",true},{"fits_per_paired_run",original.at("completed_native_fits")}}.dump(2)<<'\n';return 0;
  }
  if(argc==4&&std::string(argv[3])=="--group-loss-only"){
    auto p=plan;auto data=d;
    p["include_all_fit_positives"]=true;p["fit_group_weights"]="equal_patient";p["fit_sampling_weights"]="inverse_retention";
    p["training_loss"]={{"kind","group_mean_ce_dice"},{"version",1},{"beta",.5},{"loss_scale","FIT_rows"}};
    p["native_prediction_thresholds"]=J::array({.25,.5,.75});
    for(const char* key:{"base_hyperparameters","meta_hyperparameters"})for(auto& h:p[key])h.update({{"rounds",4},{"eta",.1},{"max_delta_step",.25}});
    for(auto& scan:data["scans"])scan.update({{"rows",64},{"training_rows",32},{"training_positive_rows",16},{"truth_positive_rows",32}});
    auto execute=[&](const J& recipe,const J& dataset,const char* name){
      need(class_study::grouped_segmentation::run(recipe,dataset,root/name,stop)==0,"group loss nested run incomplete");
      auto result=J::parse(dpnative::read_text(root/name/"result.json"));
      need(result.at("complete")==true&&result.at("conversion_performed")==false&&result.at("final_deployment_requested")==false,"group loss fixture performed unexpected final fit/conversion");
      need(result.at("protocol").at("training_loss")==recipe.at("training_loss"),"group loss provenance differs");
      const auto& effective=result.at("protocol").at("training_loss_effective_policy");
      need(effective.at("curvature_policy_version")==recipe.at("training_loss").value("curvature_policy",std::string("uniform_v1"))&&effective.at("margin_prediction_mode")=="incremental_native_proxy"&&effective.at("moment_reduction_policy")=="keyed_warp_fp64_atomic_v1","effective group loss execution policy missing");
      need(result.at("completed_group_loss_fits")==result.at("completed_native_fits")&&result.at("completed_group_loss_fits").get<unsigned>()>0,"group loss bypassed any raw, response or OOF FIT");
      return result;
    };
    auto original=execute(p,data,"hybrid");
    auto perturbed=data;auto changed=y;
    for(std::size_t row=0;row<y.size();++row)if(pid[row]==0)changed[row]=1-y[row];
    write(root/"group-loss-heldout-labels.u32",changed);perturbed["labels_path"]=(root/"group-loss-heldout-labels.u32").string();
    for(auto& scan:perturbed["scans"])if(scan.at("patient_id")==0)scan["truth_positive_outside_rows"]=17;
    auto altered=execute(p,perturbed,"heldout-perturbed");
    const auto& before=original.at("outer_results")[0];const auto& after=altered.at("outer_results")[0];
    for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256"})need(before.at(key)==after.at(key),"held-out labels or outside-positive counts altered FIT selection/models");
    need(before.at("metrics")!=after.at("metrics"),"held-out perturbation did not change evaluation");
    p["training_loss"]["beta"]=0;auto ce=execute(p,data,"custom-ce");
    p["training_loss"]["beta"]=1;auto dice=execute(p,data,"custom-dice");
    need(ce.at("completed_native_fits")==original.at("completed_native_fits")&&dice.at("completed_native_fits")==original.at("completed_native_fits"),"objective mix changed FIT schedule");
    p["training_loss"]["curvature_policy"]="class_row_sum_v2";
    p["training_loss"]["beta"]=0;auto class_ce=execute(p,data,"class-row-ce");
    for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256"})
      need(class_ce.at("outer_results")[0].at(key)==ce.at("outer_results")[0].at(key),"curvature policy changed beta-zero FIT or selection");
    p["training_loss"]["beta"]=.5;auto class_hybrid=execute(p,data,"class-row-hybrid");
    p["training_loss"]["beta"]=1;auto class_dice=execute(p,data,"class-row-dice");
    need(class_hybrid.at("completed_native_fits")==original.at("completed_native_fits")&&class_dice.at("completed_native_fits")==original.at("completed_native_fits"),"curvature policy changed FIT schedule");
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"group_loss_hybrid_and_endpoint_controls",true},
      {"class_row_sum_v2_executed",true},{"curvature_beta_zero_unchanged",true},
      {"outer_labels_and_outside_counts_excluded_from_training",true},{"all_scores_full_ROI",true},
      {"fits_per_run",original.at("completed_native_fits")},{"exhaustive_postverification_performed",false},
      {"output",root.string()}}.dump(2)<<'\n';return 0;
  }
  if(argc==4&&(std::string(argv[3])=="--threshold-histogram-only"||std::string(argv[3])=="--threshold-histogram-benchmark")){std::cout<<class_study::grouped_segmentation::threshold_histogram_checks(plan,d,std::string(argv[3])=="--threshold-histogram-benchmark"?5000000u:0u).dump(2)<<'\n';return 0;}
  if(argc==4&&std::string(argv[3])=="--evaluation-batches-only"){
    std::vector<float> values;std::vector<unsigned> labels,patients,scan_ids,training;J rows=J::array();const std::vector<unsigned> group_ids={10,20,35,50,90};unsigned eligible_rows=0;
    for(unsigned g=0;g<group_ids.size();++g)for(unsigned visit=0;visit<2;++visit){unsigned scan=11+g*41+visit*7,n=7+(g+visit)%3,positive=0,kept=0;for(unsigned i=0;i<n;++i){values.insert(values.end(),{float(i),float(g+visit)});unsigned label=unsigned(i>=3),eligible=unsigned(i%2==0);labels.push_back(label);patients.push_back(group_ids[g]);scan_ids.push_back(scan);training.push_back(eligible);positive+=label;kept+=eligible;eligible_rows+=eligible;}rows.push_back({{"patient_id",group_ids[g]},{"scan_id",scan},{"rows",n},{"training_rows",kept},{"truth_positive_rows",positive},{"truth_positive_outside_rows",visit==0?2:0}});}
    write(root/"batch-x.f32",values);write(root/"batch-y.u32",labels);write(root/"batch-pid.u32",patients);write(root/"batch-sid.u32",scan_ids);write(root/"batch-eligible.u32",training);
    J data={{"format","grouped-segmentation-dataset-1"},{"rows",labels.size()},{"features",2},{"classes",2},{"train_eligible_rows",eligible_rows},{"domain_description","synthetic batch-equivalence fixture with nonconsecutive IDs and repeated patients"},{"values_path",(root/"batch-x.f32").string()},{"labels_path",(root/"batch-y.u32").string()},{"patient_ids_path",(root/"batch-pid.u32").string()},{"scan_ids_path",(root/"batch-sid.u32").string()},{"train_eligible_path",(root/"batch-eligible.u32").string()},{"scans",rows}};
    auto p=plan;p["fit_group_weights"]="equal_patient";p["fit_sampling_weights"]="inverse_retention";p["workflow"]="native-grouped-segmentation-1";p["outer_patient_ids"]=J::array({10});auto second=hp;second["eta"]=.5;p["base_hyperparameters"]=J::array({hp,second});p["meta_hyperparameters"]=J::array({hp,second});p["native_prediction_thresholds"]=J::array({.5,0.,1.,.25,.75,std::nextafter(.5f,0.f),std::nextafter(.5f,1.f)});
    // These input/model recipes also drive the private pre-refactor executable
    // oracle outside this check; no probability or class-prediction file exists.
    dpnative::atomic_text(root/"batch-plan.json",p.dump(2));dpnative::atomic_text(root/"batch-dataset.json",data.dump(2));
    auto probability_receipt=class_study::grouped_segmentation::evaluation_probability_checks(p,data);
    J expected;for(unsigned cap:{0u,1u,5u,13u}){auto trial=p;if(cap)trial["evaluation_batch_rows"]=cap;auto out=root/("batch-"+std::to_string(cap));need(class_study::grouped_segmentation::run(trial,data,out,stop)==0,"batched native study failed");auto actual=J::parse(dpnative::read_text(out/"result.json"));if(expected.is_null())expected=actual;else{for(const char* key:{"outer_results","metrics","metrics_order","completed_native_fits","completed_outer_patients"})need(actual.at(key)==expected.at(key),"batch capacity changed FIT, frozen selection or metric counts");}for(const auto& receipt:actual.at("evaluation_batches")){need(receipt.at("all_base_response_columns")==4&&receipt.at("prediction_files_written")==false,"batch dropped base responses or cached predictions");if(cap)need(receipt.at("capacity").get<unsigned>()<=cap,"explicit row ceiling was exceeded");}}
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"row_ceilings",{1,5,13}},{"default_capacity_checked",true},{"all_inner_outer_counts_metrics_models_equal",true},{"native_probability_word_checks",probability_receipt},{"FIT_count_unchanged",true},{"nonconsecutive_patient_scan_ids",true},{"repeated_patients_and_outside_ROI",true},{"all_models_and_rows_retained",true},{"prediction_files_written",false},{"old_executable_oracle_recipe",(root/"batch-plan.json").string()}}.dump()<<'\n';return 0;
  }
  if(argc==4&&std::string(argv[3])=="--patient-weights-only"){
    // Nonconsecutive patient IDs, unequal sampled/full sizes, and both kinds
    // of single-class patient distinguish patient weighting from class totals.
    const std::vector<unsigned> patient_ids={10,20,35,50,90},full_rows={12,16,20,24,28},positives={4,4,16,0,28},keep_negative={2,3,2,6,0},keep_positive={1,2,4,0,7};
    std::vector<float> values;std::vector<unsigned> labels,patients,scan_ids,training;J group_scans=J::array();unsigned training_rows=0;
    for(unsigned g=0;g<patient_ids.size();++g){unsigned negative=full_rows[g]-positives[g];group_scans.push_back({{"patient_id",patient_ids[g]},{"scan_id",patient_ids[g]*10},{"rows",full_rows[g]},{"training_rows",keep_negative[g]+keep_positive[g]},{"training_positive_rows",keep_positive[g]},{"truth_positive_rows",positives[g]},{"truth_positive_outside_rows",g==3?1:0}});
      for(unsigned i=0;i<full_rows[g];++i){unsigned label=i>=negative,keep=label?(i-negative<keep_positive[g]):(i<keep_negative[g]);values.insert(values.end(),{float(i)/float(full_rows[g]),float(g)});labels.push_back(label);patients.push_back(patient_ids[g]);scan_ids.push_back(patient_ids[g]*10);training.push_back(keep);training_rows+=keep;}}
    write(root/"group-x.fp32",values);write(root/"group-y.u32",labels);write(root/"group-pid.u32",patients);write(root/"group-sid.u32",scan_ids);write(root/"group-eligible.u32",training);
    auto data=d;data.update({{"rows",labels.size()},{"train_eligible_rows",training_rows},{"values_path",(root/"group-x.fp32").string()},{"labels_path",(root/"group-y.u32").string()},{"patient_ids_path",(root/"group-pid.u32").string()},{"scan_ids_path",(root/"group-sid.u32").string()},{"train_eligible_path",(root/"group-eligible.u32").string()},{"scans",group_scans}});
    auto grouped=plan;grouped["outer_patient_ids"]=J::array({10});grouped["fit_group_weights"]="equal_patient";grouped["native_prediction_thresholds"]=J::array({.25,.5,.75});
    const std::vector<std::vector<double>> expected={{1.2,1.2,1.,1.,1.,0.,0.,6./7.},{1.5,.75,.6,1.2,1.,0.,0.,6./7.},{1.,1.5,1.5,.75,1.,0.,0.,6./7.}};
    auto close=[](double a,double b){return std::abs(a-b)<2e-6;};J receipts=J::array();
    for(unsigned policy=0;policy<3;++policy){auto configured=grouped;configured["fit_sampling_weights"]=policy==1?"inverse_retention":"none";configured["fit_class_balanced"]=policy==2;auto name=std::string("patient-weight-")+std::to_string(policy);
      need(class_study::grouped_segmentation::run(configured,data,root/name,stop)==0,"equal-patient study failed");auto result=J::parse(dpnative::read_text(root/name/"result.json"));need(result.at("complete")==true&&result.at("final_deployment_requested")==false&&!result.contains("final_deployment")&&!std::filesystem::exists(root/name/"final-native-pipeline.json"),"calibrated study unexpectedly performed final FIT");
      const auto& fold=result.at("outer_results").at(0);need(fold.at("outer_FIT_rows")==24&&fold.at("outer_FIT_class_counts")==J::array({11,13}),"unequal patient retained counts differ");const auto& weights=fold.at("outer_FIT_patient_weights");need(weights.size()==4&&weights==fold.at("outer_meta_FIT_patient_weights"),"meta patient identities or weights changed");
      double total=0;for(unsigned g=0;g<4;++g){const auto& row=weights.at(g);need(row.at("patient_id")==patient_ids[g+1]&&row.at("retained_class_counts")==J::array({keep_negative[g+1],keep_positive[g+1]}),"heldout patient or wrong rows entered weight counts");
        if(policy==1)need(row.at("full_class_counts")==J::array({full_rows[g+1]-positives[g+1],positives[g+1]}),"patient inverse retention used non-FIT full counts");else need(row.at("full_class_counts").is_null(),"uncorrected patient weights read full class counts");
        for(unsigned c=0;c<2;++c){double coefficient=row.at("class_weights").at(c).get<double>();need(close(coefficient,expected[policy][g*2+c]),"patient/class coefficient differs from independent fixture expectation");double count=row.at("retained_class_counts").at(c).get<double>();need(close(row.at("class_weight_sums").at(c).get<double>(),coefficient*count),"actual CUDA row weights differ from group coefficient");}
        need(close(row.at("weight_sum").get<double>(),6.),"FIT patients do not receive equal mass");total+=row.at("weight_sum").get<double>();}
      need(close(total,24.),"equal-patient row weights are not mean one");receipts.push_back(result);
      auto changed=labels;for(unsigned i=0;i<changed.size();++i)if(patients[i]==10)changed[i]=1-changed[i];write(root/"group-flipped-y.u32",changed);auto changed_data=data;changed_data["labels_path"]=(root/"group-flipped-y.u32").string();changed_data["scans"][0]["truth_positive_rows"]=8;changed_data["scans"][0]["training_positive_rows"]=2;
      auto changed_name=name+"-heldout-flipped";need(class_study::grouped_segmentation::run(configured,changed_data,root/changed_name,stop)==0,"patient-weight heldout perturbation failed");auto altered=J::parse(dpnative::read_text(root/changed_name/"result.json"));const auto& after=altered.at("outer_results").at(0);
      for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256","outer_FIT_class_counts","outer_full_FIT_class_counts","outer_FIT_patient_weights","outer_meta_FIT_patient_weights"})need(fold.at(key)==after.at(key),"heldout labels changed FIT weights, selection or models");need(fold.at("metrics")!=after.at("metrics"),"heldout perturbation did not affect scoring");
    }
    auto omitted=grouped;omitted.erase("fit_group_weights");need(class_study::grouped_segmentation::run(omitted,data,root/"patient-default-omitted",stop)==0,"default patient policy study failed");auto explicit_none=omitted;explicit_none["fit_group_weights"]="none";need(class_study::grouped_segmentation::run(explicit_none,data,root/"patient-default-none",stop)==0,"explicit none patient policy study failed");auto a=J::parse(dpnative::read_text(root/"patient-default-omitted/result.json")),b=J::parse(dpnative::read_text(root/"patient-default-none/result.json"));need(a.at("outer_results")==b.at("outer_results")&&a.at("metrics")==b.at("metrics"),"default none policy changed model selection or outcomes");need(a.at("outer_results").at(0).at("outer_FIT_patient_weights").is_null(),"default group policy fabricated weights");
    bool native_effect=false;for(const auto& result:receipts)native_effect|=result.at("outer_results").at(0).at("baseline_model_sha256")!=a.at("outer_results").at(0).at("baseline_model_sha256");need(native_effect,"patient weights did not reach the native FIT API");
    auto unsupported=training;for(unsigned i=0;i<labels.size();++i)if(patients[i]==20&&labels[i])unsupported[i]=0;write(root/"group-missing-support.u32",unsupported);auto missing=data;missing["train_eligible_path"]=(root/"group-missing-support.u32").string();missing["train_eligible_rows"]=training_rows-2;missing["scans"][1]["training_rows"]=3;missing["scans"][1]["training_positive_rows"]=0;auto inverse=grouped;inverse["fit_sampling_weights"]="inverse_retention";bool refused=false;try{class_study::grouped_segmentation::run(inverse,missing,root/"patient-missing-support",stop);}catch(const std::exception& error){refused=std::string(error.what()).find("retained support for every full FIT patient class")!=std::string::npos;}need(refused,"inverse patient weights accepted missing retained class support");
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"unequal_patient_and_class_weights_verified",true},{"single_class_patients_supported",true},{"meta_original_row_ids_preserved",true},{"heldout_labels_do_not_change_weights_or_models",true},{"default_none_unchanged",true},{"missing_retained_support_refused",true},{"calibrated_outer_only_without_final_FIT",true}}.dump()<<'\n';return 0;
  }
  if(argc==4&&(std::string(argv[3])=="--threshold-only"||std::string(argv[3])=="--threshold-no-final-only")){
    // Balanced FIT labels and gamma prohibiting splits produce equal native
    // logits. Exact native argmax chooses class 0, whereas p1>=0.5 chooses 1.
    // This fixture deliberately retains both operations as distinct candidates.
    auto threshold_plan=plan;threshold_plan["base_hyperparameters"][0]["gamma"]=1e10;threshold_plan["meta_hyperparameters"][0]["gamma"]=1e10;threshold_plan["native_prediction_thresholds"]=J::array({0.5,1.0});threshold_plan["final_deployment"]={{"convert",false}};
    need(class_study::grouped_segmentation::run(threshold_plan,d,root/"threshold-study",stop)==0,"native threshold study incomplete");auto result=J::parse(dpnative::read_text(root/"threshold-study/result.json"));const auto& first=result.at("outer_results").at(0);const auto& selection=first.at("selection");
    need(selection.at("baseline_prediction_candidate_index")==1&&selection.at("combiner_prediction_candidate_index")==1&&selection.at("selected_family")=="native_baseline","equal-logit threshold was not jointly selected");
    for(const char* key:{"inner_baseline_results","inner_combiner_results"}){const auto& scores=selection.at(key);need(scores.size()==3&&scores.at(0).at("macro_patient_Dice")==0&&scores.at(1).at("macro_patient_Dice").get<double>()>0&&scores.at(2).at("macro_patient_Dice")==0,"native argmax and >= threshold semantics were conflated");}
    for(const auto& metric:first.at("metrics"))for(const auto& scan:metric.at("scans"))need(scan.at("TP")==32&&scan.at("FP")==32&&scan.at("FN_in_ROI")==0&&scan.at("TN_in_ROI")==0,"frozen threshold was not applied to every outer ROI row");
    const auto& final=result.at("final_deployment");const auto& pipeline=final.at("native_pipeline");need(final.at("complete")==true&&final.at("native_fit_complete")==true&&final.at("conversion_attempted")==false&&pipeline.at("native_prediction_rule").at("threshold")==0.5&&pipeline.at("native_prediction_rule").at("comparison")==">="&&pipeline.at("native_prediction_rule").at("threshold_precision")=="FP32"&&pipeline.at("native_prediction_calibration").at("outer_results_used_for_selection")==false,"final calibrated native recipe is incomplete or converted");
    need(result.at("completed_native_fits")==38,"threshold candidates added model FITs");
    auto no_final_plan=threshold_plan;no_final_plan.erase("final_deployment");need(class_study::grouped_segmentation::run(no_final_plan,d,root/"threshold-no-final",stop)==0,"calibrated outer-only study failed");auto no_final=J::parse(dpnative::read_text(root/"threshold-no-final/result.json"));need(no_final.at("complete")==true&&no_final.at("outer_results")==result.at("outer_results")&&no_final.at("metrics")==result.at("metrics")&&no_final.at("final_deployment_requested")==false&&!no_final.contains("final_deployment")&&!std::filesystem::exists(root/"threshold-no-final/final-native-pipeline.json"),"omitting final FIT changed outer outcomes or created a pipeline");need(no_final.at("completed_native_fits").get<unsigned>()+final.at("completed_native_fits").get<unsigned>()==result.at("completed_native_fits").get<unsigned>(),"omitting final FIT did not remove exactly the final FIT calls");
    if(std::string(argv[3])=="--threshold-no-final-only"){std::cout<<J{{"complete",true},{"CUDA_executed",true},{"calibrated_outer_results_unchanged_without_final_FIT",true},{"only_final_FIT_calls_removed",true},{"conversion_performed",false},{"exhaustive_postverification_performed",false}}.dump()<<'\n';return 0;}
    const std::vector<float> probe={0.f,0.f,-std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::quiet_NaN(),0.f,0.f,std::numeric_limits<float>::quiet_NaN()};Device<float> probe_device(probe.size());Device<std::uint32_t> expected_device(4);ck(cudaMemcpy(probe_device.p,probe.data(),probe.size()*sizeof(float),cudaMemcpyHostToDevice));
    auto convert_fixture=[&](const J& source,const std::string& name,std::uint32_t expected,bool verify){
      auto replay_plan=plan;replay_plan["conversion"]={{"qualification_mode",verify?"exhaustive":"none"},{"maximum_seconds",0},{"batch_size",0},{"max_expansions",0},{"qualification_batch_rows",4},{"response_row_capacity",4}};
      auto metadata=class_study::grouped_segmentation::validate_replay(replay_plan,source);need(metadata.at("completed_native_fits")==0&&metadata.at("training_dataset_loaded")==false,"replay requested FIT or training data");
      need(class_study::grouped_segmentation::replay(replay_plan,source,root/name,stop)==0,"calibrated replay did not finish");auto replayed=J::parse(dpnative::read_text(root/name/"result.json"));const auto& receipt=replayed.at("conversion");need(receipt.at("complete")==true&&receipt.at("qualification_performed")==verify&&receipt.at("all_finite_FP32_and_NaN_cells_verified")==verify&&receipt.at("configuration").at("maximum_seconds")==0,"calibrated replay qualification or timeout contract differs");
      std::vector<std::uint32_t> expected_rows(4,expected);ck(cudaMemcpy(expected_device.p,expected_rows.data(),4*sizeof(std::uint32_t),cudaMemcpyHostToDevice));auto runtime=class_runtime::Runtime::load(dpnative::read_text(root/name/"replay-selected.canonical.bin"),receipt.at("canonical_sha256"),receipt.at("source_sha256"),dpnative::read_text(root/name/"replay-selected.compact.bin"),receipt.at("compact_sha256"));runtime->verify({probe_device.p,probe.size(),0,4,2},expected_device.p,4);
      if(source.contains("native_prediction_rule")){need(receipt.at("native_prediction_rule")==source.at("native_prediction_rule")&&receipt.at("calibrated_probability_threshold")==true&&receipt.at("positive_probability_threshold_bits")==std::bit_cast<std::uint32_t>(source.at("native_prediction_rule").at("threshold").get<float>()),"calibrated replay lost selected rule or threshold identity");}return receipt;
    };
    auto threshold_pipeline=[&](J source,float threshold){auto rule=source.at("native_prediction_rule");rule["threshold"]=threshold;source["native_prediction_rule"]=rule;source["native_prediction_candidates"]=J::array({source.at("native_prediction_candidates").at(0),rule});return source;};
    auto half=convert_fixture(pipeline,"threshold-half-replay",1,false);auto higher_pipeline=threshold_pipeline(pipeline,.55f);auto higher=convert_fixture(higher_pipeline,"threshold-higher-replay",0,true);need(half.at("source_sha256")==higher.at("source_sha256")&&half.at("conversion").at("decision_rule")!=higher.at("conversion").at("decision_rule"),"calibration threshold provenance or immutable source identity differs");
    // Give class 1 a small positive native margin: argmax is 1, but p1<.55.
    // These are explicit synthetic test model bytes, never a fitted artifact.
    auto biased_pipeline=threshold_pipeline(pipeline,.55f);auto model=J::parse(dpnative::read_text(pipeline.at("base_models").at(0).at("path").get<std::string>()));auto& source_model=model["learner"]["gradient_booster"]["model"];bool changed_leaf=false;for(std::size_t i=0;i<source_model.at("trees").size();++i)if(source_model.at("tree_info").at(i)==1){auto& tree=source_model["trees"][i];need(tree.at("split_conditions").size()==1,"tie fixture unexpectedly split");tree["split_conditions"][0]=.1f;tree["base_weights"][0]=.1f;changed_leaf=true;break;}need(changed_leaf,"fixture has no class-1 tree");auto model_bytes=model.dump();auto model_sha=dpnative::sha256(model_bytes);auto model_path=root/"synthetic-biased-source.json";dpnative::atomic_text(model_path,model_bytes);biased_pipeline["base_models"][0]["path"]=model_path.string();biased_pipeline["base_models"][0]["sha256"]=model_sha;
    class_study::ResidentPredictor native(plan,2,2);native.restore_model(model_bytes,model_sha);float probability[2];ck(cudaMemcpy(probability,native.predict_probabilities(probe_device.p,1),sizeof(probability),cudaMemcpyDeviceToHost));need(probability[1]>.5f&&probability[1]<.55f,"synthetic native probability is outside distinguishing interval");float native_class;ck(cudaMemcpy(&native_class,native.predict(probe_device.p,1,false),sizeof(float),cudaMemcpyDeviceToHost));need(native_class==1,"biased native argmax is not class 1");
    convert_fixture(biased_pipeline,"biased-threshold-replay",0,true);convert_fixture(threshold_pipeline(biased_pipeline,probability[1]),"probability-equality-replay",1,true);convert_fixture(threshold_pipeline(biased_pipeline,std::nextafter(probability[1],1.f)),"probability-above-replay",0,false);
    auto argmax_pipeline=biased_pipeline;for(const char* key:{"native_prediction_rule","native_prediction_candidates","native_prediction_calibration"})argmax_pipeline.erase(key);convert_fixture(argmax_pipeline,"native-argmax-replay",1,false);
    auto changed=y;for(std::size_t row=0;row<changed.size();++row)if(pid[row]==0)changed[row]=1;write(root/"threshold-flipped-y.u32",changed);auto changed_data=d;changed_data["labels_path"]=(root/"threshold-flipped-y.u32").string();need(class_study::grouped_segmentation::run(threshold_plan,changed_data,root/"threshold-flipped",stop)==0,"threshold heldout perturbation incomplete");auto altered=J::parse(dpnative::read_text(root/"threshold-flipped/result.json"));const auto& changed_first=altered.at("outer_results").at(0);for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256"})need(first.at(key)==changed_first.at(key),"outer labels changed threshold selection or fitted models");need(first.at("metrics")!=changed_first.at("metrics"),"threshold heldout perturbation did not change scoring");
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"exact_native_argmax_retained",true},{"FP32_threshold_greater_equal_verified",true},{"inner_only_threshold_selection",true},{"outer_labels_do_not_change_threshold_or_FIT",true},{"threshold_candidates_add_no_FITs",true},{"calibrated_recipe_saved",true},{"uncalibrated_conversion_replay_refused",true},{"conversion_performed",false}}.dump()<<'\n';return 0;
  }
  if(argc==4&&(std::string(argv[3])=="--reference-only"||std::string(argv[3])=="--inverse-reference-only")){
    const bool inverse=std::string(argv[3])=="--inverse-reference-only";const unsigned positive_per_scan=inverse?8:32,first_positive=64-positive_per_scan,retained_negative=first_positive/2,retained_rows=retained_negative+positive_per_scan;
    // This fixed external predictor depends only on the first input feature;
    // neither its mask nor probability feature is read from the label array.
    std::vector<unsigned> reference_labels;std::vector<float> reference_x;for(std::size_t row=0;row<y.size();++row){float intensity=x[2*row];reference_x.insert(reference_x.end(),{intensity,x[2*row+1],float(intensity>=first_positive),(intensity+.5f)/64});reference_labels.push_back(unsigned(intensity>=first_positive));}
    write(root/"reference-x.fp32",reference_x);write(root/"reference-y.u32",reference_labels);auto reference_data=d;reference_data["labels_path"]=(root/"reference-y.u32").string();reference_data["features"]=4;reference_data["values_path"]=(root/"reference-x.fp32").string();reference_data["input_provenance"]={{"fixed_external_predictor","synthetic input-only threshold fixture"}};for(auto& scan:reference_data["scans"])scan["truth_positive_rows"]=positive_per_scan;
    auto reference_plan=plan;reference_plan.erase("outer_patient_ids");reference_plan["outer_folds"]=3;reference_plan["include_all_fit_positives"]=true;reference_plan["fit_class_balanced"]=!inverse;reference_plan["fit_sampling_weights"]=inverse?"inverse_retention":"none";reference_plan["reference_predictor"]={{"mask_feature_index",2},{"probability_feature_index",3},{"thresholds",J::array({.25,.75})},{"provenance",reference_data["input_provenance"]}};reference_plan["final_deployment"]={{"convert",false},{"max_rounds",2},{"max_depth",2},{"max_trees",4},{"max_source_nodes",28}};
    need(class_study::grouped_segmentation::run(reference_plan,reference_data,root/"reference-study",stop)==0,"external reference study incomplete");auto result=J::parse(dpnative::read_text(root/"reference-study/result.json"));need(result.at("complete")==true&&result.at("completed_outer_patients")==5,"external reference omitted a patient");
    std::set<unsigned> evaluated;for(const auto& fold:result.at("outer_results")){
      for(const auto& patient:fold.at("outer_patient_ids"))need(evaluated.insert(patient.get<unsigned>()).second,"reference outer patient repeated");
      const auto& selection=fold.at("selection");need(selection.at("selected_family")=="external_reference"&&selection.at("reference_candidate_index")==0&&selection.at("reference_default_always_eligible")==true,"perfect fixed default was dropped from inner selection");need(selection.at("reference_inner_results").size()==3,"declared reference candidates were not evaluated");
      auto fit_patients=selection.at("outer_FIT_patient_ids").size();need(fold.at("outer_FIT_rows")==fit_patients*2*retained_rows,"FIT sampling failed to retain all positives plus eligible negatives");need(fold.at("outer_FIT_class_counts")==J::array({fit_patients*2*retained_negative,fit_patients*2*positive_per_scan}),"class weights were not computed from the exact FIT rows");need(fold.at("outer_full_FIT_class_counts")==J::array({fit_patients*2*first_positive,fit_patients*2*positive_per_scan}),"inverse-retention full class counts include a heldout patient");
      need(fold.at("metrics").size()==5&&fold.at("metrics").at(2)==fold.at("metrics").at(3),"selected/default reference outcomes differ");for(const auto& metric:fold.at("metrics"))for(const auto& scan:metric.at("scans"))need(scan.at("TP").get<unsigned>()+scan.at("FP").get<unsigned>()+scan.at("FN_in_ROI").get<unsigned>()+scan.at("TN_in_ROI").get<unsigned>()==64,"positive training sampling altered full-ROI evaluation");
    }
    need(evaluated==std::set<unsigned>{0,1,2,3,4},"reference evaluation missed a patient");const auto& final=result.at("final_deployment");need(final.at("complete")==true&&final.at("pipeline_complete")==true&&final.at("conversion_attempted")==false&&final.at("conversion_complete")==false&&final.at("all_patient_native_FIT_performed")==false,"external-reference pipeline fabricated native refit or conversion");need(final.at("native_pipeline").at("family")=="external_reference"&&final.at("native_pipeline").at("base_models").empty()&&final.at("native_pipeline").at("external_reference_candidate").at("kind")=="default_mask"&&final.at("native_pipeline").at("input_provenance")==reference_data.at("input_provenance"),"fixed external recipe/provenance was not preserved");
    auto altered=reference_labels;std::set<unsigned> held;const auto& first=result.at("outer_results").at(0);for(const auto& patient:first.at("outer_patient_ids"))held.insert(patient.get<unsigned>());for(std::size_t row=0;row<altered.size();++row)if(held.contains(pid[row]))altered[row]=1-altered[row];write(root/"reference-flipped-y.u32",altered);reference_data["labels_path"]=(root/"reference-flipped-y.u32").string();for(auto& scan:reference_data["scans"])if(held.contains(scan.at("patient_id").get<unsigned>()))scan["truth_positive_rows"]=first_positive;reference_plan.erase("final_deployment");need(class_study::grouped_segmentation::run(reference_plan,reference_data,root/"reference-flipped",stop)==0,"reference heldout perturbation incomplete");auto changed=J::parse(dpnative::read_text(root/"reference-flipped/result.json"));const auto& changed_first=changed.at("outer_results").at(0);for(const char* key:{"selection","baseline_model_sha256","combiner_model_sha256","outer_FIT_rows","outer_FIT_class_counts","outer_full_FIT_class_counts"})need(first.at(key)==changed_first.at(key),"heldout labels changed reference selection, sampling, weights or fitted models");need(first.at("metrics")!=changed_first.at("metrics"),"reference heldout perturbation did not change scored outcomes");
    std::cout<<J{{"complete",true},{"CUDA_executed",true},{"fixed_external_default_retained",true},{"inner_only_reference_selection",true},{"FIT_only_positive_sampling_and_weights",true},{"inverse_retention_policy",inverse},{"full_ROI_evaluation",true},{"conversion_performed",false},{"exhaustive_postverification_performed",false}}.dump()<<'\n';return 0;
  }
  if(argc!=4||folds_only){
    auto grouped=plan;grouped.erase("outer_patient_ids");grouped["outer_folds"]=3;
    grouped["final_deployment"]={{"qualification_mode","none"},{"maximum_seconds",0},{"batch_size",0},{"qualification_batch_rows",4},{"response_row_capacity",4},{"max_rounds",2},{"max_depth",2},{"max_trees",4},{"max_source_nodes",28}};
    need(class_study::grouped_segmentation::run(grouped,d,root/"grouped-folds",stop)==0,"patient-grouped outer folds incomplete");
    auto grouped_result=J::parse(dpnative::read_text(root/"grouped-folds/result.json"));
    need(grouped_result.at("complete")==true&&grouped_result.at("native_evaluation_complete")==true&&grouped_result.at("completed_outer_folds")==3&&grouped_result.at("completed_outer_patients")==5,"fold and patient completion counts differ");
    need(grouped_result.at("full_leave_one_patient_out")==false&&grouped_result.at("full_cohort_outer_evaluation")==true,"grouped evaluation completeness mislabeled");
    std::set<unsigned> covered;unsigned index=0;
    for(const auto& fold:grouped_result.at("outer_results")){
      need(fold.at("outer_fold_index")==index++,"outer fold indexes differ");
      std::set<unsigned> held;for(const auto& p:fold.at("outer_patient_ids")){unsigned id=p.get<unsigned>();need(held.insert(id).second&&covered.insert(id).second,"patient repeated in outer evaluation");}
      const auto& selection=fold.at("selection");std::set<unsigned> fit;
      for(const auto& p:selection.at("outer_FIT_patient_ids")){unsigned id=p.get<unsigned>();need(!held.contains(id)&&fit.insert(id).second,"outer patient leaked into fitting");}
      auto all=fit;all.insert(held.begin(),held.end());need(all==std::set<unsigned>{0,1,2,3,4},"outer fold drops a patient");
      for(const char* key:{"inner_validation_patient_groups","OOF_patient_groups_for_outer_refit"}){
        std::set<unsigned> nested;
        for(const auto& group:selection.at(key))for(const auto& p:group){unsigned id=p.get<unsigned>();need(fit.contains(id)&&!held.contains(id)&&nested.insert(id).second,"outer patient leaked into inner or OOF groups");}
        need(nested==fit,"inner or OOF grouping omits a fitting patient");
      }
      for(const auto& metric:fold.at("metrics")){
        std::set<unsigned> metric_patients;for(const auto& p:metric.at("patients"))metric_patients.insert(p.at("patient_id").get<unsigned>());
        need(metric_patients==held&&metric.at("scans").size()==2*held.size(),"paired patient scans split across outer folds");
        std::set<unsigned> scan_ids;
        for(const auto& scan:metric.at("scans")){unsigned patient=scan.at("patient_id").get<unsigned>(),id=scan.at("scan_id").get<unsigned>();need(held.contains(patient)&&(id==10*patient||id==10*patient+1)&&scan_ids.insert(id).second,"outer scan identity changed");
          need(scan.at("TP").get<unsigned>()+scan.at("FP").get<unsigned>()+scan.at("FN_in_ROI").get<unsigned>()+scan.at("TN_in_ROI").get<unsigned>()==64,"grouped evaluation omitted ROI rows");}
      }
    }
    need(covered==std::set<unsigned>{0,1,2,3,4},"complete grouped evaluation omitted a patient");
    for(const auto& metric:grouped_result.at("metrics"))need(metric.at("patients").size()==5&&metric.at("scans").size()==10,"grouped aggregate patient or scan count differs");
    const auto& final=grouped_result.at("final_deployment");
    need(final.at("complete")==true&&final.at("native_fit_complete")==true&&final.at("conversion_attempted")==true&&final.at("conversion_complete")==true,"single final deployment did not finish");
    need(final.at("FIT_patient_ids")==J::array({0,1,2,3,4})&&final.at("outer_results_used_for_selection")==false&&final.at("heldout_accuracy_claim")==false,"all-patient refit claims or groups differ");
    const auto& final_selection=final.at("final_selection");need(final_selection.at("FIT_patient_ids")==J::array({0,1,2,3,4})&&final_selection.at("outer_results_used_for_selection")==false,"final selection did not use independent internal selection");
    for(const char* key:{"inner_validation_patient_groups","OOF_patient_groups_for_all_patient_refit"}){
      std::set<unsigned> final_coverage;for(const auto& group:final_selection.at(key))for(const auto& p:group)need(final_coverage.insert(p.get<unsigned>()).second,"final cross-fit repeats a patient");
      need(final_coverage==std::set<unsigned>{0,1,2,3,4},"final cross-fit omits a patient");
    }
    const auto& receipt=final.at("conversion");need(receipt.at("qualification_mode")=="none"&&receipt.at("qualification_performed")==false&&receipt.at("partition_preflight_performed")==false&&receipt.at("all_finite_FP32_and_NaN_cells_verified")==false&&receipt.at("accepted_native_class_root")==false&&receipt.at("partition_cells").is_null(),"default final conversion unexpectedly ran exhaustive verification");
    need(receipt.at("configuration").at("maximum_seconds")==0&&receipt.at("configuration").at("batch_size")==0&&receipt.at("artifact_prefix")=="final-selected","final conversion default deadline, adaptive batching or artifact differs");
    auto pipeline_path=std::filesystem::path(final.at("native_pipeline_path").get<std::string>());
    need(dpnative::sha256(dpnative::read_text(pipeline_path))==final.at("native_pipeline_sha256").get<std::string>(),"final native pipeline identity differs");
    for(const char* suffix:{".canonical.bin",".compact.bin",".receipt.json"})need(std::filesystem::file_size(root/"grouped-folds"/(std::string("final-selected")+suffix))>0,"single final conversion artifact missing");
    need(grouped_result.at("deployment_conversion_attempts")==0,"final-only study converted outer deployments");
    auto changed=y;const auto& first=grouped_result.at("outer_results").at(0);std::set<unsigned> perturbed;
    for(const auto& p:first.at("outer_patient_ids"))perturbed.insert(p.get<unsigned>());
    for(std::size_t r=0;r<changed.size();++r)if(perturbed.contains(pid[r]))changed[r]=1-changed[r];
    write(root/"grouped-flipped-y.u32",changed);auto changed_data=d;changed_data["labels_path"]=(root/"grouped-flipped-y.u32").string();grouped["final_deployment"]["convert"]=false;
    need(class_study::grouped_segmentation::run(grouped,changed_data,root/"grouped-flipped",stop)==0,"grouped held-out perturbation incomplete");
    auto changed_result=J::parse(dpnative::read_text(root/"grouped-flipped/result.json"));const auto& changed_first=changed_result.at("outer_results").at(0);
    for(const char* key:{"outer_patient_ids","selection","baseline_model_sha256","combiner_model_sha256"})need(first.at(key)==changed_first.at(key),"grouped held-out labels changed selection or fitted models");
    need(first.at("metrics")!=changed_first.at("metrics"),"grouped held-out perturbation did not affect evaluation");
    const auto& native_only=changed_result.at("final_deployment");need(native_only.at("complete")==true&&native_only.at("native_fit_complete")==true&&native_only.at("native_only")==true&&native_only.at("conversion_attempted")==false&&native_only.at("conversion_complete")==false,"convert=false failed to preserve a completed native-only pipeline");
    if(folds_only){std::cout<<J{{"complete",true},{"CUDA_executed",true},{"grouped_outer_folds",3},{"evaluated_patients",5},{"paired_scans_kept_together",true},{"outer_label_perturbation_keeps_selection_and_models",true},{"single_all_patient_final_conversion",true},{"exhaustive_postverification_performed",false},{"output",root.string()}}.dump()<<'\n';return 0;}
  }
  auto gate=plan;auto tiny=hp;tiny["rounds"]=1;tiny["max_depth"]=1;gate["conversion_gate"]={{"qualification_mode","exhaustive"},{"withheld_patient_ids",J::array({0})},{"hyperparameters",J::array({tiny})}};
  need(class_study::grouped_segmentation::run(gate,d,root/"gate-success",stop)==0,"small nonconstant conversion gate failed");auto accepted=J::parse(dpnative::read_text(root/"gate-success/result.json"));
  need(accepted.at("complete")==true&&accepted.at("accepted_native_class_root")==true,"gate accepted incomplete result");const auto& model=accepted.at("conversions").at(0);
  need(model.at("nodes").get<unsigned>()>1&&model.at("partition_cells").get<unsigned>()>1,"conversion fixture was constant");need(model.at("all_finite_FP32_and_NaN_cells_verified")==true&&model.at("native_class_mismatches")==0,"finite/missing partition qualification failed");
  need(accepted.at("FIT_patient_ids")==J::array({1,2,3,4})&&accepted.at("withheld_labels_used_for_FIT_or_selection")==false,"conversion gate did not exclude outer patient");
  for(const char* suffix:{"-source.json",".canonical.bin",".compact.bin"})need(std::filesystem::file_size(root/"gate-success"/(std::string("gate-1")+suffix))>0,"accepted artifact missing");
  auto refusal=[&](const char* name,const char* field,const J& limit,const char* reason){auto bounded=gate;bounded["conversion_gate"][field]=limit;bool rejected=false;try{class_study::grouped_segmentation::run(bounded,d,root/name,stop);}catch(const std::exception&){rejected=true;}need(rejected,"conversion resource limit silently succeeded");auto result=J::parse(dpnative::read_text(root/name/"result.json"));need(result.at("complete")==false&&result.at("accepted_native_class_root")==false&&result.at("conversions").empty(),"partial conversion was accepted");need(result.at("error").get<std::string>().find(reason)!=std::string::npos,"conversion refusal had unrelated cause");for(const char* suffix:{"-source.json",".canonical.bin",".compact.bin"})need(!std::filesystem::exists(root/name/(std::string("gate-1")+suffix)),"refused conversion exported deployment artifact");};
  refusal("gate-timeout","maximum_seconds",1e-12,"stopped");refusal("gate-budget","gpu_byte_budget",1,"budget");
  auto compact=plan;auto source2=tiny,source3=tiny;source2["max_depth"]=2;source3["max_depth"]=3;
  compact["base_hyperparameters"]=J::array({source2,source3});compact["meta_hyperparameters"]=J::array({tiny,source2});
  auto composed=compact;composed["composed_conversion_gate"]={{"qualification_mode","exhaustive"},{"withheld_patient_ids",J::array({0})},{"meta_indices",J::array({0,1})}};
  need(class_study::grouped_segmentation::run(composed,d,root/"composed-success",stop)==0,"composed gate dispatch failed");
  auto stack_gate=J::parse(dpnative::read_text(root/"composed-success/result.json"));
  need(stack_gate.at("format")=="grouped-segmentation-composed-gate-1"&&stack_gate.at("complete")==true&&stack_gate.at("conversions").size()==2,"composed gate did not finish both fixed meta models");
  need(stack_gate.at("withheld_labels_used_for_FIT_or_selection")==false&&stack_gate.at("FIT_patient_ids")==J::array({1,2,3,4}),"composed gate used held-out patient for FIT");
  for(const auto& receipt:stack_gate.at("conversions")){
    need(receipt.at("classes")==2&&receipt.at("score_channels")==4&&receipt.at("nodes").get<unsigned>()>1,"composed output shape or nonconstant fixture differs");
    need(receipt.at("all_finite_FP32_and_NaN_cells_verified")==true&&receipt.at("native_class_mismatches")==0&&receipt.at("output_thresholds_are_source_thresholds")==true,"composed all-cell qualification failed");
    auto prefix=receipt.at("artifact_prefix").get<std::string>();for(const char* suffix:{"-base-1.json","-base-2.json","-meta.json",".canonical.bin",".compact.bin",".receipt.json"})need(std::filesystem::file_size(root/"composed-success"/(prefix+suffix))>0,"accepted composed artifact missing");
  }
  auto composed_refusal=composed;composed_refusal["composed_conversion_gate"]["max_partition_cells"]=1;bool composed_rejected=false;
  try{class_study::grouped_segmentation::run(composed_refusal,d,root/"composed-partition-refusal",stop);}catch(const std::exception&){composed_rejected=true;}
  need(composed_rejected,"composed partition limit silently succeeded");auto refused_stack=J::parse(dpnative::read_text(root/"composed-partition-refusal/result.json"));
  need(refused_stack.at("complete")==false&&refused_stack.at("accepted_native_class_root")==false&&refused_stack.at("conversions").empty(),"refused composed gate accepted a partial deployment");
  need(refused_stack.at("error").get<std::string>().find("partition-cell bound")!=std::string::npos&&!std::filesystem::exists(root/"composed-partition-refusal/composed-1.compact.bin"),"composed refusal exported a partial graph or had unrelated cause");
  if(argc==4){std::cout<<J{{"complete",true},{"CUDA_executed",true},{"nonconstant_exact_conversion",true},{"nonconstant_composed_exact_conversion",true},{"composed_partition_limit_refused",true},{"timeout_refused",true},{"GPU_budget_refused",true},{"partial_artifacts_exported",false},{"output",root.string()}}.dump()<<'\n';return 0;}
  compact["conversion_on_outer_deployments"]={{"qualification_mode","exhaustive"}};
  need(class_study::grouped_segmentation::run(compact,d,root/"outer-conversions",stop)==0,"outer deployment conversion study incomplete");
  auto converted_study=J::parse(dpnative::read_text(root/"outer-conversions/result.json"));
  need(converted_study.at("complete")==true&&converted_study.at("native_evaluation_complete")==true&&converted_study.at("deployment_conversion_attempts")==2&&converted_study.at("completed_deployment_conversions")==2&&converted_study.at("all_requested_deployment_conversions_complete")==true,"outer deployment completeness counters differ");
  auto limited=compact;limited["conversion_on_outer_deployments"]["max_partition_cells"]=1;
  need(class_study::grouped_segmentation::run(limited,d,root/"outer-conversion-refusals",stop)==0,"conversion refusal aborted native evaluation");
  auto refused_study=J::parse(dpnative::read_text(root/"outer-conversion-refusals/result.json"));
  need(refused_study.at("complete")==true&&refused_study.at("native_evaluation_complete")==true&&refused_study.at("deployment_conversion_attempts")==2&&refused_study.at("completed_deployment_conversions")==0&&refused_study.at("all_requested_deployment_conversions_complete")==false,"native and conversion completeness were conflated");
  const auto& completed_outer=converted_study.at("outer_results").at(0);const auto& refused_outer=refused_study.at("outer_results").at(0);
  for(const char* key:{"selection","metrics","baseline_model_sha256","combiner_model_sha256"})need(completed_outer.at(key)==refused_outer.at(key),"conversion resource refusal changed native outcomes or selection");
  for(const auto& [kind,receipt]:refused_outer.at("deployment_conversions").items()){
    need(receipt.at("complete")==false&&receipt.at("native_outer_metrics_retained_on_refusal")==true&&receipt.at("error").get<std::string>().find("partition-cell bound")!=std::string::npos,"outer conversion refusal had unexpected cause");
    need(!std::filesystem::exists(root/"outer-conversion-refusals"/(receipt.at("artifact_prefix").get<std::string>()+".compact.bin")),"failed outer conversion exported partial graph");
  }
  need(class_study::grouped_segmentation::run(plan,d,root/"original",stop)==0,"original fixture incomplete");auto original=J::parse(dpnative::read_text(root/"original/result.json"));
  for(std::size_t r=0;r<y.size();++r)if(pid[r]==0)y[r]=1-y[r];write(root/"flipped-y.u32",y);d["labels_path"]=(root/"flipped-y.u32").string();
  need(class_study::grouped_segmentation::run(plan,d,root/"flipped",stop)==0,"flipped fixture incomplete");auto flipped=J::parse(dpnative::read_text(root/"flipped/result.json"));
  const auto& a=original.at("outer_results").at(0);const auto& b=flipped.at("outer_results").at(0);
  need(a.at("selection")==b.at("selection"),"outer labels changed inner selection");need(a.at("baseline_model_sha256")==b.at("baseline_model_sha256"),"outer labels changed baseline fit");need(a.at("combiner_model_sha256")==b.at("combiner_model_sha256"),"outer labels changed combiner fit");
  need(a.at("metrics")!=b.at("metrics"),"outer label perturbation failed to change evaluation");need(original.at("full_leave_one_patient_out")==false,"pilot misreported as complete LOPO");
  for(const auto& m:a.at("metrics")){need(m.at("patients").size()==1&&m.at("patients")[0].at("patient_id")==0,"outer patient extent");need(m.at("scans").size()==2,"repeat scan extent");need(m.at("scans")[0].at("FN_outside_ROI")==5,"outside crop foreground not accounted");for(const auto& s:m.at("scans")){auto pixels=s.at("TP").get<unsigned>()+s.at("FP").get<unsigned>()+s.at("FN_in_ROI").get<unsigned>()+s.at("TN_in_ROI").get<unsigned>();need(pixels==64,"evaluation incorrectly used only sampled training rows");}need(m.at("patients")[0].at("predicted_absolute_symmetric_repeat_volume_difference").get<double>()==0,"identical scan feature target changed repeat volume");}
  auto empty_plan=plan;empty_plan["base_hyperparameters"][0]["gamma"]=1e10;empty_plan["meta_hyperparameters"][0]["gamma"]=1e10;
  need(class_study::grouped_segmentation::run(empty_plan,d,root/"empty-predictions",stop)==0,"empty prediction fixture incomplete");auto empty=J::parse(dpnative::read_text(root/"empty-predictions/result.json"));
  for(const auto& metric:empty.at("metrics")){const auto& p=metric.at("patients")[0];need(p.at("both_empty_repeat_predictions")==true,"empty prediction flag missing");need(p.at("predicted_absolute_symmetric_repeat_volume_difference").is_null(),"two empty predictions were misreported as stable segmentation");}
  for(std::size_t r=0;r<y.size();++r)if(sid[r]==0)y[r]=0;write(root/"missing-truth.u32",y);d["labels_path"]=(root/"missing-truth.u32").string();d["scans"][0]["truth_positive_outside_rows"]=0;bool refused=false;
  try{class_study::grouped_segmentation::run(plan,d,root/"missing-truth",stop);}catch(const std::exception& e){refused=std::string(e.what()).find("no reference foreground")!=std::string::npos;}
  need(refused&&!std::filesystem::exists(root/"missing-truth/result.json"),"missing lesion reference was not refused before FIT");
  std::cout<<J{{"complete",true},{"CUDA_executed",true},{"nonconstant_exact_conversion",true},{"nonconstant_composed_exact_conversion",true},{"composed_partition_limit_refused",true},{"timeout_refused",true},{"GPU_budget_refused",true},{"partial_artifacts_exported",false},{"outer_label_perturbation_keeps_selection_and_models",true},{"outer_conversion_refusal_preserves_native_outcomes",true},{"full_ROI_evaluation_verified",true},{"outside_ROI_false_negatives_verified",true},{"repeat_volume_metric_verified",true},{"both_empty_repeat_metric_null",true},{"missing_reference_refused_before_FIT",true},{"fits_per_run",original.at("completed_native_fits")},{"output",root.string()}}.dump()<<'\n';
}
