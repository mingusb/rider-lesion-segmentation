#include "class_study_grouped_segmentation.hpp"
#include "class_study_inner_metrics.hpp"
#include "class_study_model_summary.hpp"
#include "class_study_oof_aggregation.cuh"
#include "class_study_meta_bootstrap.cuh"
#include "class_study_teacher_blend.cuh"
#include "class_study_base_mean.cuh"
#include "class_study_fit_sampling.cuh"
#include "class_study_data.hpp"
#include "class_study_train.hpp"
#include "class_study_convert.hpp"
#include "class_study_composition_prune.hpp"
#include "class_conversion/class_io.hpp"
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <bit>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <tuple>
#include <stdexcept>
#include <utility>
#include <vector>

namespace class_study::grouped_segmentation {
namespace {
using J=nlohmann::json;
using U=std::uint32_t;
using L=unsigned long long;
void need(bool yes,const char* why){if(!yes)throw std::invalid_argument(why);}
void ck(cudaError_t status,const char* why){if(status!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(status));}
void sync(){ck(cudaGetLastError(),"segmentation CUDA launch");ck(cudaDeviceSynchronize(),"segmentation CUDA completion");}
U blocks(std::uint64_t n){return U(std::min<std::uint64_t>((n+255)/256,65535));}
std::uint64_t multiply(std::uint64_t a,std::uint64_t b){need(!b||a<=UINT64_MAX/b,"segmentation capacity product overflow");return a*b;}
std::uint64_t add(std::uint64_t a,std::uint64_t b){need(a<=UINT64_MAX-b,"segmentation capacity sum overflow");return a+b;}
std::uint64_t integer(const J& j,const char* key,std::uint64_t max=UINT64_MAX){
  const auto& x=j.at(key);need(x.is_number_unsigned()||(x.is_number_integer()&&x.get<std::int64_t>()>=0),key);
  auto v=x.get<std::uint64_t>();need(v<=max,key);return v;
}
thread_local L owned_CUDA_bytes=0;
template<class T> struct Device {
  T* p=nullptr;std::size_t n=0;
  Device()=default;
  explicit Device(std::size_t count):n(count){need(count&&count<=SIZE_MAX/sizeof(T),"device extent");L next=add(owned_CUDA_bytes,count*sizeof(T));ck(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),"segmentation CUDA allocation");owned_CUDA_bytes=next;}
  ~Device(){release();}
  void release(){if(p){cudaFree(p);owned_CUDA_bytes-=n*sizeof(T);p=nullptr;n=0;}}
  Device(Device&& a) noexcept:p(std::exchange(a.p,nullptr)),n(std::exchange(a.n,0)){}
  Device& operator=(Device&& a) noexcept{if(this!=&a){release();p=std::exchange(a.p,nullptr);n=std::exchange(a.n,0);}return *this;}
  Device(const Device&)=delete;Device& operator=(const Device&)=delete;
  void zero(){ck(cudaMemset(p,0,n*sizeof(T)),"segmentation CUDA zero");}
  void set(const std::vector<T>& v){need(v.size()==n,"device copy shape");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice),"segmentation CUDA upload");}
  std::vector<T> get() const{std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"segmentation CUDA download");return v;}
};
template<class T> std::vector<T> read_array(const J& d,const char* key,std::size_t n){
  std::filesystem::path p=d.at(key).get<std::string>();need(std::filesystem::file_size(p)==n*sizeof(T),"segmentation data file extent differs");
  std::vector<T> v(n);std::ifstream f(p,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));return v;
}
__device__ int locate(U x,const U* a,U n){U lo=0,hi=n;while(lo<hi){U m=lo+(hi-lo)/2;if(a[m]<x)lo=m+1;else hi=m;}return lo<n&&a[lo]==x?int(lo):-1;}
std::vector<U> missing_features(const J& d,U features,const J& plan){
  need(!plan.contains("missing_feature_indices"),"missing_feature_indices belongs in the dataset descriptor, not the plan");
  std::vector<U> result;if(!d.contains("missing_feature_indices"))return result;
  const auto& indices=d.at("missing_feature_indices");need(indices.is_array()&&!indices.empty(),"missing_feature_indices must be a nonempty array");std::set<U> seen;
  for(const auto& item:indices){need(item.is_number_unsigned()||(item.is_number_integer()&&item.get<std::int64_t>()>=0),"missing feature index must be a nonnegative integer");auto index=item.get<L>();need(index<features&&seen.insert(U(index)).second,"missing feature index duplicated or out of range");result.push_back(U(index));}
  if(plan.contains("reference_predictor")){const auto& r=plan.at("reference_predictor");for(const char* key:{"mask_feature_index","probability_feature_index"})if(r.contains(key))need(!seen.count(U(integer(r,key,features-1))),"reference mask/probability cannot declare missing values");}
  std::sort(result.begin(),result.end());return result;
}
__global__ void audit(const float* x,const U* y,const U* pid,const U* sid,const U* eligible,
                      std::uint64_t n,U f,const U* scans,const U* scan_patients,U ns,U* bad,L* eligible_count,L* scan_rows,L* scan_fit_rows,L* scan_fit_positive,L* scan_truth,const U* missing,U missing_count){
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x){
    int s=locate(sid[i],scans,ns);if(y[i]>1||eligible[i]>1||s<0||(s>=0&&scan_patients[s]!=pid[i]))atomicExch(bad,1u);
    if(s>=0){atomicAdd(scan_rows+s,1ULL);if(eligible[i]){atomicAdd(scan_fit_rows+s,1ULL);if(y[i]==1)atomicAdd(scan_fit_positive+s,1ULL);}if(y[i]==1)atomicAdd(scan_truth+s,1ULL);}if(eligible[i])atomicAdd(eligible_count,1ULL);
  }
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n*f;i+=std::uint64_t(blockDim.x)*gridDim.x)
    if(!isfinite(x[i])&&(!isnan(x[i])||locate(U(i%f),missing,missing_count)<0))atomicExch(bad,1u);
}
__global__ void audit_truth(const L* inside,const L* outside,U ns,U* bad){for(U s=blockIdx.x*blockDim.x+threadIdx.x;s<ns;s+=blockDim.x*gridDim.x)if(!inside[s]&&!outside[s])atomicExch(bad,1u);}
J reference_config(const J& plan,U features){
  if(!plan.contains("reference_predictor"))return nullptr;
  auto config=plan.at("reference_predictor");need(config.is_object(),"reference_predictor must be an object");
  U mask=U(integer(config,"mask_feature_index",features-1));J candidates=J::array({J{{"kind","default_mask"},{"feature_index",mask}}});
  if(config.contains("probability_feature_index"))integer(config,"probability_feature_index",features-1);
  if(config.contains("thresholds")){
    need(config.contains("probability_feature_index")&&config.at("thresholds").is_array()&&!config.at("thresholds").empty(),"reference thresholds require a probability feature and nonempty array");
    std::set<U> used;
    for(const auto& item:config.at("thresholds")){need(item.is_number(),"reference threshold must be numeric");double value=item.get<double>();need(std::isfinite(value)&&value>=0&&value<=1,"reference threshold must be finite and in [0,1]");float threshold=value==0?0.f:float(value);need(used.insert(std::bit_cast<U>(threshold)).second,"reference thresholds duplicate at FP32 precision");candidates.push_back({{"kind","probability_threshold"},{"feature_index",config.at("probability_feature_index")},{"threshold",threshold}});}
  }
  need(candidates.size()<=INT32_MAX,"reference candidate extent");config["candidates"]=candidates;config["input_only_fixed_predictor"]=true;return config;
}
J native_prediction_config(const J& plan){
  J rules=J::array({J{{"kind","native_argmax"},{"operation","ResidentTrainer::predict(margin=false)"},{"rounds","all"}}});
  if(!plan.contains("native_prediction_thresholds"))return rules;
  const auto& thresholds=plan.at("native_prediction_thresholds");need(thresholds.is_array()&&!thresholds.empty(),"native_prediction_thresholds must be a nonempty array");std::set<U> used;
  for(const auto& item:thresholds){need(item.is_number(),"native threshold must be numeric");double value=item.get<double>();need(std::isfinite(value)&&value>=0&&value<=1,"native threshold must be finite and in [0,1]");float threshold=value==0?0.f:float(value);need(used.insert(std::bit_cast<U>(threshold)).second,"native thresholds duplicate at FP32 precision");rules.push_back({{"kind","positive_probability_threshold"},{"probability_class",1},{"comparison",">="},{"threshold",threshold},{"threshold_precision","FP32"},{"operation","ResidentTrainer::predict_probabilities; native multi:softprob full-round view"},{"rounds","all"}});}
  need(rules.size()<=UINT32_MAX,"native prediction candidate extent");return rules;
}
J rule_search_views_config(const J& plan,U rules){
  J result=J::array();if(!plan.contains("rule_search_views"))return result;
  const auto& views=plan.at("rule_search_views");need(views.is_array()&&!views.empty()&&views.size()<=UINT32_MAX/5-1,"rule_search_views must be a nonempty bounded array");
  std::set<std::string> names={"full"};
  for(const auto& view:views){
    need(view.is_object()&&view.size()==2&&view.contains("id")&&view.at("id").is_string()&&view.contains("native_prediction_candidate_indices"),"rule view requires id and native_prediction_candidate_indices");
    const auto name=view.at("id").get<std::string>();need(!name.empty()&&name.size()<=128&&names.insert(name).second,"rule view id empty, duplicated or reserved");
    const auto& indices=view.at("native_prediction_candidate_indices");need(indices.is_array()&&!indices.empty()&&indices.size()<=rules,"rule view requires nonempty bounded candidate indices");
    U previous=0;bool first=true;for(const auto& index:indices){J item={{"index",index}};U current=U(integer(item,"index",rules-1));need(first||current>previous,"rule view indices must be distinct and in canonical increasing order");previous=current;first=false;}
    result.push_back(view);
  }
  for(const char* key:{"final_deployment","conversion_gate","composed_conversion_gate","conversion_on_outer_deployments"})need(!plan.contains(key),"rule_search_views supports outer evaluation only; final FIT and conversion are unsupported");
  return result;
}
bool oof_aggregation_config(const J& plan){
  if(!plan.contains("oof_teacher_aggregation"))return false;
  const auto& c=plan.at("oof_teacher_aggregation");
  need(c.is_object()&&c.size()==2&&c.contains("version")&&integer(c,"version")==1&&c.value("method",J(nullptr))=="patient_weighted_probability_mean","oof_teacher_aggregation requires version=1 and method=patient_weighted_probability_mean");
  for(const char* key:{"final_deployment","conversion_gate","composed_conversion_gate","conversion_on_outer_deployments"})need(!plan.contains(key),"OOF teacher aggregation supports paired native outer evaluation only; final FIT/export/conversion unsupported");
  need(plan.value("fit_group_weights",std::string("none"))=="equal_patient","OOF patient-weighted aggregation requires equal_patient FIT weighting");
  need(!plan.value("outer_model_summaries",false),"OOF aggregation structural summaries are not yet supported");
  return true;
}
J bootstrap_config(const J& plan){
  if(!plan.contains("meta_patient_bootstrap"))return nullptr;
  const auto& c=plan.at("meta_patient_bootstrap");
  need(c.is_object()&&c.size()==3&&c.contains("version")&&c.contains("replicates")&&c.contains("seed")&&integer(c,"version")==1,"meta_patient_bootstrap requires version, replicates and seed");
  need(integer(c,"replicates",UINT32_MAX)>0,"bootstrap replicas must be positive");integer(c,"seed");
  need(plan.value("fit_group_weights",std::string("none"))=="equal_patient","meta patient bootstrap requires equal_patient FIT weighting");
  need(!plan.contains("training_loss")&&!plan.contains("oof_teacher_aggregation"),"meta bootstrap supports native CE with full-refit teachers only");
  need(!plan.value("outer_model_summaries",false),"bootstrap structural summaries are not yet supported");
  for(const char* key:{"final_deployment","conversion_gate","composed_conversion_gate","conversion_on_outer_deployments"})need(!plan.contains(key),"meta bootstrap supports paired native outer evaluation only; final FIT/export/conversion unsupported");
  return c;
}
bool candidate_aggregation_config(const J& plan){
  if(!plan.contains("meta_candidate_aggregation"))return false;
  const auto& c=plan.at("meta_candidate_aggregation");need(c.is_object()&&c.size()==2&&c.contains("version")&&integer(c,"version")==1&&c.value("method",J(nullptr))=="uniform_probability_mean","meta_candidate_aggregation requires version=1 and method=uniform_probability_mean");
  for(const char* key:{"meta_patient_bootstrap","oof_teacher_aggregation","training_loss","final_deployment","conversion_gate","composed_conversion_gate","conversion_on_outer_deployments"})need(!plan.contains(key),"fixed meta mean supports ordinary full-refit teachers only; combined modes/custom loss/final/export/conversion unsupported");
  need(!plan.value("outer_model_summaries",false),"fixed meta mean structural summaries are not yet supported");return true;
}
J blend_config(const J& plan,U features){
  if(!plan.contains("teacher_probability_blend"))return nullptr;
  const auto& c=plan.at("teacher_probability_blend");need(c.is_object()&&c.size()==3&&c.contains("version")&&integer(c,"version")==1&&c.value("method",J(nullptr))=="oof_brier_convex"&&c.contains("probability_feature_index"),"teacher_probability_blend requires version=1, method=oof_brier_convex and probability_feature_index");integer(c,"probability_feature_index",features-1);
  for(const char* key:{"meta_candidate_aggregation","meta_patient_bootstrap","oof_teacher_aggregation","training_loss","meta_input_feature_indices","final_deployment","conversion_gate","composed_conversion_gate","conversion_on_outer_deployments"})need(!plan.contains(key),"teacher blend supports ordinary native full-refit teachers only; combined modes/custom loss/raw meta/final/export/conversion unsupported");
  need(!plan.value("outer_model_summaries",false),"teacher blend structural summaries unsupported");return c;
}
bool base_aggregation_config(const J& plan){
  if(!plan.contains("base_candidate_aggregation"))return false;
  const auto& c=plan.at("base_candidate_aggregation");need(c.is_object()&&c.size()==2&&c.contains("version")&&integer(c,"version")==1&&c.value("method",J(nullptr))=="uniform_probability_mean","base_candidate_aggregation requires version=1 and method=uniform_probability_mean");
  for(const char* key:{"teacher_probability_blend","meta_candidate_aggregation","meta_patient_bootstrap","oof_teacher_aggregation","training_loss","meta_input_feature_indices","final_deployment","conversion_gate","composed_conversion_gate","conversion_on_outer_deployments"})need(!plan.contains(key),"fixed base mean supports ordinary full-refit teachers only; combined modes/custom loss/raw meta/final/export/conversion unsupported");
  need(!plan.value("outer_model_summaries",false),"fixed base mean structural summaries unsupported");return true;
}
J sampling_config(const J& plan,U features){
  if(!plan.contains("fit_sampling"))return nullptr;
  const auto& c=plan.at("fit_sampling");need(c.is_object()&&c.size()==4&&integer(c,"version")==1&&c.value("method",J(nullptr))=="binary_mask_hard_negatives"&&c.value("normalization",J(nullptr))=="legacy_retained_mass","fit_sampling requires version1 binary_mask_hard_negatives legacy_retained_mass");integer(c,"mask_feature_index",features-1);
  need(plan.value("include_all_fit_positives",false)&&plan.value("fit_group_weights",std::string("none"))=="equal_patient"&&plan.value("fit_sampling_weights",std::string("none"))=="inverse_retention"&&!plan.value("fit_class_balanced",false),"hard-negative sampling requires all FIT positives, equal_patient, inverse_retention, no class balancing");
  for(const char* key:{"teacher_probability_blend","base_candidate_aggregation","meta_candidate_aggregation","meta_patient_bootstrap","oof_teacher_aggregation","training_loss","meta_input_feature_indices","final_deployment","conversion_gate","composed_conversion_gate","conversion_on_outer_deployments"})need(!plan.contains(key),"hard-negative sampling initially supports ordinary native nested evaluation only");
  need(!plan.value("outer_model_summaries",false),"hard-negative sampling structural summaries not yet supported");return c;
}
std::array<L,3> sampling_array(const J& j,const char* key,L maximum){
  const auto& a=j.at(key);need(a.is_array()&&a.size()==3,"sampling stratum array must have three counts");std::array<L,3> v{};for(U i=0;i<3;++i){need(a[i].is_number_unsigned()||(a[i].is_number_integer()&&a[i].get<std::int64_t>()>=0),"sampling count must be unsigned");v[i]=a[i].get<L>();need(v[i]<=maximum,"sampling stratum count exceeds rows");}return v;
}
J validated_sampling_counts(const J& plan,const J& d){
  const auto config=sampling_config(plan,U(integer(d,"features",INT32_MAX)));if(config.is_null())return nullptr;
  const U mask=U(integer(config,"mask_feature_index"));const auto missing=missing_features(d,U(integer(d,"features")),plan);need(!std::binary_search(missing.begin(),missing.end(),mask),"sampling binary mask cannot declare missing values");
  const auto& c=d.at("fit_sampling_counts");need(c.is_object()&&c.value("format",J(nullptr))=="grouped-binary-mask-sampling-counts-1"&&integer(c,"version")==1&&c.value("complete",J(nullptr))==true,"complete version1 sampling count extension required");
  need(c.at("method")==config.at("method")&&c.at("normalization")==config.at("normalization")&&integer(c,"mask_feature_index")==mask&&c.at("strata")==J::array({"positive","negative_mask1","negative_mask0"}),"sampling count policy differs");
  need(c.at("scans").is_array()&&c.at("scans").size()==d.at("scans").size(),"sampling scan count extent differs");std::map<U,const J*> source;for(const auto& scan:d.at("scans"))need(source.emplace(U(integer(scan,"scan_id",UINT32_MAX)),&scan).second,"duplicate sampling source scan");
  std::set<U> seen;L totals[9]={};std::map<U,L> patient_legacy;
  auto check=[&](const J& rec,L n){auto full=sampling_array(rec,"full_stratum_counts",n),kept=sampling_array(rec,"retained_stratum_counts",n);L nf=0,nk=0;for(U k=0;k<3;++k){nf=add(nf,full[k]);nk=add(nk,kept[k]);need(kept[k]<=full[k],"sampling retained exceeds full stratum");}need(nf==n&&kept[0]==full[0]&&kept[1]==full[1],"sampling census strata or full rows differ");L eligible=integer(rec,"eligible_rows",n),ep=integer(rec,"eligible_positive_rows",eligible);need(ep<=full[0]&&kept[2]<=eligible-ep&&eligible-ep-kept[2]<=full[1],"sampling eligibility/strata disagree");L old=add(eligible,full[0])-ep;need(integer(rec,"legacy_retained_rows",n)==old&&integer(rec,"retained_rows",n)==nk&&old<=nk&&integer(rec,"added_rows",n)==nk-old&&integer(rec,"rows",n)==n,"sampling retained/legacy totals disagree");return std::array<L,9>{full[0],full[1],full[2],kept[0],kept[1],kept[2],old,eligible,ep};};
  for(const auto& rec:c.at("scans")){U id=U(integer(rec,"scan_id",UINT32_MAX));need(seen.insert(id).second&&source.count(id),"sampling scan omitted/duplicated/unknown");const auto& original=*source.at(id);need(integer(rec,"patient_id",UINT32_MAX)==integer(original,"patient_id",UINT32_MAX),"sampling scan patient differs");L n=integer(original,"rows",integer(d,"rows"));auto values=check(rec,n);need(values[0]==integer(original,"truth_positive_rows",n)&&values[7]==integer(original,"training_rows",n)&&values[8]==integer(original,"training_positive_rows",n),"sampling counts differ from original scan scalars");if(original.contains("row_start"))need(integer(rec,"row_start")==integer(original,"row_start"),"sampling row start differs");for(U k=0;k<9;++k)totals[k]=add(totals[k],values[k]);auto patient=U(integer(rec,"patient_id"));patient_legacy[patient]=add(patient_legacy[patient],values[6]);}
  const auto aggregate=check(c.at("totals"),integer(d,"rows"));for(U k=0;k<9;++k)need(aggregate[k]==totals[k],"sampling aggregate differs from scans");need(totals[7]==integer(d,"train_eligible_rows"),"sampling eligibility differs from descriptor");for(const auto& [patient,legacy]:patient_legacy){(void)patient;need(legacy>0,"sampling patient lacks legacy retained support");}return c;
}
J base_mean_rule(J rule){if(rule.at("kind")=="native_argmax"){rule["kind"]="mean_probability_pair_argmax";rule["operation"]="rounded class mean p1 > p0; equality class 0";}else rule["operation"]="rounded class mean p1 >= FP32 threshold; equality class 1";rule["probability_source"]="uniform probability mean of every full-refit base in declared order";return rule;}
__global__ void choose_base_mean_family(const double* base,U b,const double* meta,U m,const double* reference,U r,U ordinary,const double* mean,U candidate,U* output){if(threadIdx.x||blockIdx.x)return;double score=ordinary==2?reference[r]:ordinary==1?meta[m]:base[b];*output=mean[candidate]>score?4:ordinary;}
__global__ void positive_blend_weight(const float* weights,U rows,U* positive){for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x)if(!weights||weights[row]>0){atomicExch(positive,1u);return;}}
J blend_rule(J rule){if(rule.at("kind")=="native_argmax"){rule["kind"]="scalar_probability_strict_half";rule["operation"]="rounded scalar blend q > FP32 0.5; equality class 0";}else rule["operation"]="rounded scalar blend q >= FP32 threshold; equality class 1";rule["probability_source"]="FIT-only OOF Brier coefficient; exact endpoint copies; alpha32 authority";return rule;}
__global__ void choose_blend_family(const double* base,U b,const double* meta,U m,const double* reference,U r,U ordinary,const double* blend,U candidate,U* output){if(threadIdx.x||blockIdx.x)return;double score=ordinary==2?reference[r]:ordinary==1?meta[m]:base[b];*output=blend[candidate]>score?3:ordinary;}
std::vector<U> meta_input_config(const J& plan,U features){
  if(!plan.contains("meta_input_feature_indices"))return {};
  const auto& raw=plan.at("meta_input_feature_indices");need(raw.is_array()&&!raw.empty()&&raw.size()<=features,"meta_input_feature_indices requires a nonempty bounded array");
  std::vector<U> result;std::set<U> used;for(const auto& item:raw){J value={{"index",item}};U index=U(integer(value,"index",features-1));need(used.insert(index).second,"meta input raw feature is duplicated");result.push_back(index);}return result;
}
J meta_input_mapping(const std::vector<U>& indices){return {{"version",1},{"raw_feature_indices",indices}};}
std::vector<U> saved_meta_input_config(const J& pipeline,U features){
  if(!pipeline.contains("meta_input_mapping"))return {};
  const auto& mapping=pipeline.at("meta_input_mapping");need(mapping.is_object()&&mapping.size()==2&&mapping.contains("version")&&integer(mapping,"version")==1&&mapping.contains("raw_feature_indices"),"unsupported saved meta input mapping");
  return meta_input_config(J{{"meta_input_feature_indices",mapping.at("raw_feature_indices")}},features);
}
J training_loss_config(const J& plan){
  if(!plan.contains("training_loss"))return nullptr;
  const auto& loss=plan.at("training_loss");
  need(loss.is_object()&&(loss.size()==4||(loss.size()==5&&loss.contains("curvature_policy")))&&loss.value("kind",J(nullptr))=="group_mean_ce_dice"&&
       loss.contains("version")&&integer(loss,"version")==1&&
       loss.value("loss_scale",J(nullptr))=="FIT_rows", "training_loss requires kind, version=1, beta and loss_scale=FIT_rows");
  need(loss.contains("beta")&&loss.at("beta").is_number(),"training_loss beta must be numeric");
  const double beta=loss.at("beta").get<double>();
  need(std::isfinite(beta)&&beta>=0&&beta<=1,"training_loss beta must be in [0,1]");
  const auto curvature=loss.value("curvature_policy",J("uniform_v1"));
  need(curvature=="uniform_v1"||curvature=="class_row_sum_v2","unknown training_loss curvature_policy");
  need(plan.value("include_all_fit_positives",false)&&
       plan.value("fit_group_weights",std::string("none"))=="equal_patient"&&
       plan.value("fit_sampling_weights",std::string("none"))=="inverse_retention"&&
       !plan.value("fit_class_balanced",false),
       "group loss requires all FIT positives, equal patient and inverse-retention weights without class balancing");
  need(!plan.contains("conversion_gate")&&!plan.contains("composed_conversion_gate"),
       "group loss requires nested evaluation, not a gate-only study");
  return loss;
}
TrainingLossSpec training_loss_spec(const J& config,L rows){
  TrainingLossSpec loss;loss.beta=config.at("beta").get<double>();
  loss.curvature_policy=config.value("curvature_policy",std::string("uniform_v1"))=="class_row_sum_v2"
    ?TrainingCurvaturePolicy::class_row_sum_v2:TrainingCurvaturePolicy::uniform_v1;
  loss.loss_scale=double(rows);loss.require_all_positives=true;return loss;
}
J training_loss_policy(const J& config,const J& hp){
  if(config.is_null())return nullptr;
  // One-row validation resolves policy defaults without assigning this
  // illustrative extent to the actual FIT or its memory accounting.
  const auto resolved=ResidentTrainer::validate_loss(hp,training_loss_spec(config,1),1,1,2);
  J policy=J::object();
  for(const char* key:{"curvature_policy_version","margin_prediction_mode","verify_incremental_margins","moment_reduction_policy"})
    policy[key]=resolved.at(key);
  policy["known_native_prediction_proxy_bytes_per_FIT_row"]=resolved.at("known_native_prediction_proxy_bytes");
  return policy;
}
// A replay rule is an executable contract, not a calibration hint. Reject
// partial markers and altered operations before reading any source models.
J validated_prediction_rule(const J& rule){
  need(rule.is_object()&&rule.contains("kind"),"calibrated native prediction rule must be an object with kind");
  if(rule.at("kind")=="native_argmax"){
    need(rule.size()==3&&rule.value("operation",J(nullptr))=="ResidentTrainer::predict(margin=false)"&&rule.value("rounds",J(nullptr))=="all","calibrated native prediction argmax schema differs");return rule;
  }
  need(rule.size()==7&&rule.at("kind")=="positive_probability_threshold"&&rule.contains("probability_class")&&integer(rule,"probability_class")==1&&rule.value("comparison",J(nullptr))==">="&&rule.value("threshold_precision",J(nullptr))=="FP32"&&rule.value("operation",J(nullptr))=="ResidentTrainer::predict_probabilities; native multi:softprob full-round view"&&rule.value("rounds",J(nullptr))=="all","calibrated native prediction threshold schema differs");
  need(rule.contains("threshold")&&rule.at("threshold").is_number(),"calibrated native prediction threshold must be numeric");double threshold=rule.at("threshold").get<double>();need(std::isfinite(threshold)&&threshold>=0&&threshold<=1,"calibrated native prediction threshold must be finite and in [0,1]");J normalized=rule;normalized["threshold"]=threshold==0?0.f:float(threshold);return normalized;
}
J replay_prediction_rule(const J& pipeline){
  bool marked=pipeline.contains("native_prediction_calibration")||pipeline.contains("native_prediction_rule")||pipeline.contains("native_prediction_candidates");
  if(!marked)return nullptr;
  need(pipeline.contains("native_prediction_calibration")&&pipeline.contains("native_prediction_rule")&&pipeline.contains("native_prediction_candidates"),"calibrated native prediction markers must all be present");
  need(pipeline.value("family",std::string())=="native_baseline","calibrated native prediction replay supports only native_baseline; calibrated composition is unsupported");
  const auto& calibration=pipeline.at("native_prediction_calibration");need(calibration.is_object(),"calibrated native prediction provenance must be an object");
  for(const char* key:{"method","evaluation_domain","source_identity"})need(calibration.contains(key)&&calibration.at(key).is_string()&&!calibration.at(key).get<std::string>().empty(),"calibrated native prediction provenance field missing");
  need(calibration.contains("outer_results_used_for_selection")&&calibration.at("outer_results_used_for_selection").is_boolean()&&calibration.at("outer_results_used_for_selection")==false&&calibration.contains("conversion_supported")&&calibration.at("conversion_supported").is_boolean(),"calibrated native prediction provenance flags differ");
  J selected=validated_prediction_rule(pipeline.at("native_prediction_rule"));const auto& candidates=pipeline.at("native_prediction_candidates");need(candidates.is_array()&&!candidates.empty()&&candidates.size()<=UINT32_MAX,"calibrated native prediction candidate extent");
  bool found=false;std::set<std::string> unique;for(std::size_t i=0;i<candidates.size();++i){auto candidate=validated_prediction_rule(candidates[i]);need(unique.insert(candidate.dump()).second,"calibrated native prediction candidates duplicate at FP32 precision");need((i==0)==(candidate.at("kind")=="native_argmax"),"calibrated native prediction candidates must start with the single native argmax rule");found|=candidate==selected;}
  need(found,"calibrated native prediction selected rule is absent from candidates");return selected;
}
__global__ void probability_classes(const float* probabilities,U rows,U width,U positive_column,float threshold,float* prediction){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){float p=probabilities[row*width+positive_column];prediction[row]=isfinite(p)&&p>=0&&p<=1?float(p>=threshold):nanf("");}
}
void threshold_prediction(const J& rule,const float* probabilities,U rows,U width,U positive_column,Device<float>& prediction){
  need(rule.at("kind")=="positive_probability_threshold","native probability rule differs");probability_classes<<<blocks(rows),256>>>(probabilities,rows,width,positive_column,rule.at("threshold").get<float>(),prediction.p);sync();
}
__global__ void audit_reference(const float* x,L rows,U features,U mask,U probability,bool has_probability,U* bad){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){float value=x[row*features+mask];if(value!=0&&value!=1)atomicOr(bad,1u);if(has_probability){float p=x[row*features+probability];if(!isfinite(p)||p<0||p>1)atomicOr(bad,2u);}}
}
__global__ void reference_classes(const float* x,U rows,U features,U feature,bool mask,float threshold,float* prediction){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x)prediction[row]=mask?x[row*features+feature]:float(x[row*features+feature]>=threshold);
}
__global__ void count_fit_classes(const U* labels,U rows,L* counts){for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x)atomicAdd(counts+labels[row],1ULL);}
__global__ void inverse_retention_weights(const U* labels,U rows,const L* retained,L full_negative,L full_positive,float* weights){
  double normalizer=double(rows)/double(full_negative+full_positive);
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){U label=labels[row];L full=label?full_positive:full_negative;weights[row]=float(double(full)/double(retained[label])*normalizer);}
}
__global__ void balanced_fit_weights(const U* labels,U rows,const L* counts,float* weights){for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x)weights[row]=counts[0]&&counts[1]?float(double(rows)/(2*double(counts[labels[row]]))):1.f;}
// Original row identities are retained by raw and meta matrices. Only the
// explicit FIT patient list can contribute counts or receive a row weight.
__global__ void count_fit_patient_classes(const U* labels,const U* ids,const U* all_patients,U source_rows,U rows,const U* patients,U groups,L* counts,U* bad){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){
    if(ids[row]>=source_rows||labels[row]>1){atomicOr(bad,1u);continue;}int group=locate(all_patients[ids[row]],patients,groups);
    if(group<0){atomicOr(bad,1u);continue;}atomicAdd(counts+L(group)*2+labels[row],1ULL);
  }
}
__global__ void dense_fit_group_ids(const U* ids,const U* all_patients,U source_rows,U rows,
                                  const U* patients,U groups,U* dense,U* bad){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){
    if(ids[row]>=source_rows){atomicOr(bad,1u);continue;}
    const int group=locate(all_patients[ids[row]],patients,groups);
    if(group<0){atomicOr(bad,1u);continue;}dense[row]=U(group);
  }
}
__global__ void patient_weight_coefficients(const L* counts,const L* full,U rows,U groups,U correction,float* coefficients,U* bad){
  for(U group=blockIdx.x*blockDim.x+threadIdx.x;group<groups;group+=blockDim.x*gridDim.x){
    L negative=counts[L(group)*2],positive=counts[L(group)*2+1],retained=negative+positive;
    if(!retained){atomicOr(bad,2u);continue;}double mass=double(rows)/double(groups);U present=U(negative>0)+U(positive>0);
    L full_rows=correction==2?full[L(group)*2]+full[L(group)*2+1]:0;
    if(correction==2&&(!full_rows||(!negative&&full[L(group)*2])||(!positive&&full[L(group)*2+1]))){atomicOr(bad,4u);continue;}
    for(U label=0;label<2;++label){L count=counts[L(group)*2+label];double weight=0;
      if(count){if(correction==2)weight=mass*(double(full[L(group)*2+label])/double(full_rows))/double(count);else if(correction==1)weight=mass/(double(present)*double(count));else weight=mass/double(retained);}
      coefficients[L(group)*2+label]=float(weight);if(!isfinite(float(weight))||weight<0)atomicOr(bad,8u);
    }
  }
}
__global__ void apply_patient_weights(const U* labels,const U* ids,const U* all_patients,U rows,const U* patients,U groups,const float* coefficients,float* weights,double* sums){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){U group=U(locate(all_patients[ids[row]],patients,groups));L index=L(group)*2+labels[row];float weight=coefficients[index];weights[row]=weight;atomicAdd(sums+index,double(weight));}
}
__global__ void flag_rows(const U* pid,const U* eligible,const U* labels,U n,const U* selected,U count,bool fit,bool include_positive,U* flags){
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x)
    flags[i]=(locate(pid[i],selected,count)>=0)&&(!fit||eligible[i]||(include_positive&&labels[i]==1));
}
__global__ void gather(const float* x,const U* y,const U* sid,const U* ids,U n,U f,float* xx,U* yy,U* ss){
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<std::uint64_t(n)*f;i+=std::uint64_t(blockDim.x)*gridDim.x)xx[i]=x[std::uint64_t(ids[i/f])*f+i%f];
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x){yy[i]=y[ids[i]];ss[i]=sid[ids[i]];}
}
__global__ void index_rows(const U* ids,U n,U* positions){for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x)positions[ids[i]]=U(i);}
__global__ void response_column(const float* prob,const U* ids,const U* positions,U n,U width,U column,float* out,U* seen,U* bad,U teachers=0){
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x){
    U row=positions?positions[ids[i]]:U(i);if(row==UINT32_MAX){atomicExch(bad,1u);continue;}
    for(U k=0;k<2;++k){float v=prob[2*i+k];if(!isfinite(v)||v<0||v>1)atomicExch(bad,1u);out[std::uint64_t(row)*width+column+k]=v;}
    if(seen&&atomicAdd(seen+std::uint64_t(row)*teachers+column/2,1u))atomicExch(bad,1u);
  }
}
__global__ void append_raw_columns(const float* raw,U rows,U features,const U* indices,U count,U response_width,U output_width,float* output){
  for(L i=L(blockIdx.x)*blockDim.x+threadIdx.x;i<L(rows)*count;i+=L(blockDim.x)*gridDim.x){U row=U(i/count),column=U(i%count);output[L(row)*output_width+response_width+column]=__uint_as_float(__float_as_uint(raw[L(row)*features+indices[column]]));}
}
__global__ void check_meta_raw_columns(const float* global,const U* ids,U rows,U features,const U* indices,U count,const float* mixed,U prefix,U width,U* bad){
  for(L i=L(blockIdx.x)*blockDim.x+threadIdx.x;i<L(rows)*count;i+=L(blockDim.x)*gridDim.x){U row=U(i/count),column=U(i%count);if(__float_as_uint(mixed[L(row)*width+prefix+column])!=__float_as_uint(global[L(ids[row])*features+indices[column]]))atomicExch(bad,1u);}
}
__global__ void check_seen(const U* seen,std::uint64_t n,U* bad){for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x)if(seen[i]!=1)atomicExch(bad,1u);}
// One representative for every atom of the source's finite/NaN partition.
// An axis starts at -FLT_MAX, then each distinct finite cut, and ends in NaN.
// Source and output use only strict less-than tests, so every test (including
// its missing direction) is constant throughout each represented atom.
__global__ void partition_rows(float* x,U rows,U features,const U* offsets,const U* widths,const L* strides,const float* representatives,L first){
  for(L i=L(blockIdx.x)*blockDim.x+threadIdx.x;i<L(rows)*features;i+=L(blockDim.x)*gridDim.x){U f=U(i%features);L cell=first+i/features;x[i]=representatives[offsets[f]+(cell/strides[f])%widths[f]];}
}
__global__ void native_classes(const float* prediction,U n,U* expected,U* bad){for(U i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){float p=prediction[i];if(p!=0&&p!=1)atomicExch(bad,1u);else expected[i]=U(p);}}
__global__ void stack_columns(const float* input,U rows,U width,U column,bool probability,float* output,U* bad){for(L i=L(blockIdx.x)*blockDim.x+threadIdx.x;i<L(rows)*2;i+=L(blockDim.x)*gridDim.x){float v=input[i];if(!isfinite(v)||(probability&&(v<0||v>1)))atomicExch(bad,1u);output[(i/2)*width+column+i%2]=v;}}
__global__ void confusion(const float* pred,const U* y,const U* sid,U n,const U* scans,U ns,L* counts,U* bad){
  for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::uint64_t(blockDim.x)*gridDim.x){
    float p=pred[i];int s=locate(sid[i],scans,ns);if((p!=0&&p!=1)||y[i]>1||s<0){atomicExch(bad,1u);continue;}
    // TP, FP, FN, TN. Complete zero-positive truth/prediction has Dice one.
    U c=p==1?(y[i]?0:1):(y[i]?2:3);atomicAdd(counts+std::uint64_t(s)*4+c,1ULL);
  }
}
// A probability contributes once to its scan/class bin. upper_bound preserves
// exact FP32 >= semantics, including equality, signed zero and both endpoints.
__global__ void probability_histogram(const float* probabilities,U rows,U width,U column,const U* labels,const U* sid,const U* scans,U ns,const float* thresholds,U cuts,L* bins,U* bad){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){
    float p=probabilities[row*width+column];int scan=locate(sid[row],scans,ns);
    if(!isfinite(p)||p<0||p>1||labels[row]>1||scan<0){atomicOr(bad,1u);continue;}
    U first=0,last=cuts;while(first<last){U middle=first+(last-first)/2;if(thresholds[middle]<=p)first=middle+1;else last=middle;}
    atomicAdd(bins+(L(scan)*2+labels[row])*(L(cuts)+1)+first,1ULL);
  }
}
__global__ void histogram_confusion(const L* bins,U ns,U cuts,const U* original_rules,L* counts,const U* bad){
  if(*bad)return;
  for(U scan=blockIdx.x*blockDim.x+threadIdx.x;scan<ns;scan+=blockDim.x*gridDim.x){
    const L* negative=bins+L(scan)*2*(L(cuts)+1);const L* positive=negative+cuts+1;
    L negatives=0,positives=0;for(U bin=0;bin<=cuts;++bin){negatives+=negative[bin];positives+=positive[bin];}
    L fp=0,tp=0;for(U bin=cuts;bin>0;--bin){fp+=negative[bin];tp+=positive[bin];L* out=counts+(L(original_rules[bin-1])*ns+scan)*4;
      out[0]+=tp;out[1]+=fp;out[2]+=positives-tp;out[3]+=negatives-fp;}
  }
}
L threshold_histogram_bytes(L rules,L scans){return rules>1?add(multiply(rules-1,8),multiply(multiply(scans,2),multiply(rules,8))):0;}
U evaluation_row_capacity(L budget,L live,L free,L reserve,L per_row,L rows,L ceiling){
  need(per_row>0&&rows>0&&rows<=UINT32_MAX,"evaluation capacity extent");need(live<budget,"no configured GPU headroom for evaluation");L available=std::min(budget-live,free);need(reserve<available,"no native registration headroom for evaluation");L capacity=(available-reserve)/per_row;if(ceiling)capacity=std::min(capacity,ceiling);capacity=std::min(capacity,rows);need(capacity>0,"GPU budget cannot hold one evaluation row");return U(capacity);
}
struct ProbabilityHistogram {
  U cuts,scans;Device<float> thresholds;Device<U> original_rules;Device<L> bins;
  ProbabilityHistogram(const J& rules,U ns):cuts(U(rules.size()-1)),scans(ns),thresholds(cuts),original_rules(cuts),bins(multiply(multiply(ns,2),L(cuts)+1)){
    need(rules.size()>1&&rules.size()<=UINT32_MAX&&ns>0,"threshold histogram extent");std::vector<std::pair<float,U>> sorted;
    for(U c=1;c<rules.size();++c)sorted.emplace_back(rules.at(c).at("threshold").get<float>(),c);
    std::sort(sorted.begin(),sorted.end());std::vector<float> values;std::vector<U> indices;
    for(const auto& [value,index]:sorted){values.push_back(value);indices.push_back(index);}thresholds.set(values);original_rules.set(indices);
  }
  void accumulate(const float* probabilities,U rows,U width,U column,const U* labels,const U* sid,const U* scan_ids,L* counts,U* bad){
    need(rows>0&&width>column,"threshold histogram probability shape");bins.zero();
    probability_histogram<<<blocks(rows),256>>>(probabilities,rows,width,column,labels,sid,scan_ids,scans,thresholds.p,cuts,bins.p,bad);
    histogram_confusion<<<blocks(scans),256>>>(bins.p,scans,cuts,original_rules.p,counts,bad);sync();
    U status=0;ck(cudaMemcpy(&status,bad,sizeof(U),cudaMemcpyDeviceToHost),"threshold histogram status");need(!status,"invalid probability, label or scan in threshold histogram");
  }
};
__global__ void summarize(const L* counts,const U* scan_patient,const L* outside,U ns,U np,U candidates,double* scores,double* patient_dice,L* patient_counts,double* scan_dice,L* scan_volumes,double* repeat_difference,U* both_empty){
  for(U c=blockIdx.x*blockDim.x+threadIdx.x;c<candidates;c+=blockDim.x*gridDim.x){
    double total=0;U patients=0;
    for(U p=0;p<np;++p){L tp=0,fp=0,fn=0,tn=0,pred_volume[2]={0,0},truth_volume[2]={0,0};bool active=false;U visits=0;
      for(U s=0;s<ns;++s)if(scan_patient[s]==p){const L* a=counts+(std::uint64_t(c)*ns+s)*4;L n=a[0]+a[1]+a[2]+a[3];auto z=std::uint64_t(c)*ns+s;scan_dice[z]=-1;scan_volumes[z*2]=scan_volumes[z*2+1]=0;if(n){active=true;tp+=a[0];fp+=a[1];fn+=a[2]+outside[s];tn+=a[3];L pv=a[0]+a[1],tv=a[0]+a[2]+outside[s];scan_volumes[z*2]=pv;scan_volumes[z*2+1]=tv;scan_dice[z]=pv+tv?2*double(a[0])/double(pv+tv):1.0;if(visits<2){pred_volume[visits]=pv;truth_volume[visits]=tv;}++visits;}}
      auto q=(std::uint64_t(c)*np+p)*4;patient_counts[q]=tp;patient_counts[q+1]=fp;patient_counts[q+2]=fn;patient_counts[q+3]=tn;
      double dice=active?(2*double(tp)+fp+fn?2*double(tp)/(2*double(tp)+fp+fn):1.0):-1;
      patient_dice[std::uint64_t(c)*np+p]=dice;if(active){total+=dice;++patients;}
      auto z=(std::uint64_t(c)*np+p)*2;repeat_difference[z]=repeat_difference[z+1]=-1;
      both_empty[std::uint64_t(c)*np+p]=visits==2&&!pred_volume[0]&&!pred_volume[1];
      if(visits==2){repeat_difference[z]=pred_volume[0]+pred_volume[1]?2*fabs(double(pred_volume[0])-double(pred_volume[1]))/double(pred_volume[0]+pred_volume[1]):-1;repeat_difference[z+1]=truth_volume[0]+truth_volume[1]?2*fabs(double(truth_volume[0])-double(truth_volume[1]))/double(truth_volume[0]+truth_volume[1]):-1;}
    }
    scores[c]=patients?total/patients:-1;
  }
}
__global__ void choose(const double* scores,const L* bytes,U n,U* best){if(threadIdx.x||blockIdx.x)return;U b=0;for(U i=1;i<n;++i)if(scores[i]>scores[b]||(scores[i]==scores[b]&&bytes[i]<bytes[b]))b=i;*best=b;}
__global__ void choose_view(const double* scores,const L* bytes,U models,U rules,const U* eligible,U count,U* best,U* bad){
  if(threadIdx.x||blockIdx.x)return;U b=eligible[0];
  for(U model=0;model<models;++model)for(U r=0;r<count;++r){U i=model*rules+eligible[r];double score=scores[i];if(!isfinite(score)||score<0||score>1){*bad=1;return;}if(score>scores[b]||(score==scores[b]&&bytes[i]<bytes[b]))b=i;}
  *best=b;
}
__global__ void choose_family(const double* base,U b,const double* meta,U m,U* result){if(!threadIdx.x&&!blockIdx.x)*result=meta[m]>base[b]?1u:0u;}
__global__ void choose_view_family(const double* base,U b,const double* meta,U m,const double* reference,U r,U* result){if(!threadIdx.x&&!blockIdx.x){U family=meta[m]>base[b]?1u:0u;double score=family?meta[m]:base[b];*result=reference&&reference[r]>=score?2u:family;}}
struct Matrix {
  U rows=0,features=0;bool response_matrix=false;J class_counts=nullptr,patient_weights=nullptr,sampling_receipt=nullptr;Device<float> x,weights;Device<U> y,scan,ids;
  Device<U> loss_group_ids;Device<FitGroupSpec> loss_group_specs;
  ResidentDataView view(const std::shared_ptr<Matrix>& self,const J& binding) const{return {x.p,y.p,rows,features,rows,0,features,2,binding,std::static_pointer_cast<void>(self),weights.p};}
};
struct Data {
  U rows,F;bool include_positive=false;L subset_workspace_budget=0;mutable L maximum_subset_workspace_bytes=0;J descriptor,sampling;std::map<U,std::array<L,3>> patient_full_strata,patient_retained_strata;std::map<U,L> patient_legacy_rows;std::map<U,L> patient_full_rows,patient_full_positive,patient_outside_positive;std::vector<U> patients,scans,scan_patient_indices;std::vector<L> outside;
  Device<float> x;Device<U> y,pid,sid,eligible,scan_ids,scan_patients,scan_indices,missing_indices;Device<L> out;
  explicit Data(const J& d,const J& plan,const J& memory):rows(U(integer(d,"rows",INT32_MAX))),F(U(integer(d,"features",INT32_MAX))),include_positive(plan.value("include_all_fit_positives",false)),subset_workspace_budget(integer(memory,"CUB_workspace_byte_budget")),descriptor(d),sampling(sampling_config(plan,F)),x(std::size_t(rows)*F),y(rows),pid(rows),sid(rows),eligible(rows){
    std::map<U,U> mapping;std::map<U,L> misses;for(const auto& s:d.at("scans")){mapping[U(integer(s,"scan_id",UINT32_MAX))]=U(integer(s,"patient_id",UINT32_MAX));misses[U(integer(s,"scan_id",UINT32_MAX))]=integer(s,"truth_positive_outside_rows");}
    for(const auto& [s,p]:mapping){scans.push_back(s);patients.push_back(p);outside.push_back(misses.at(s));}
    std::sort(patients.begin(),patients.end());patients.erase(std::unique(patients.begin(),patients.end()),patients.end());std::vector<U> sp;
    for(const auto& [s,p]:mapping){sp.push_back(p);scan_patient_indices.push_back(U(std::lower_bound(patients.begin(),patients.end(),p)-patients.begin()));}
    scan_ids=Device<U>(scans.size());scan_ids.set(scans);scan_patients=Device<U>(scans.size());scan_patients.set(sp);scan_indices=Device<U>(scans.size());scan_indices.set(scan_patient_indices);out=Device<L>(outside.size());out.set(outside);
    x.set(read_array<float>(d,"values_path",std::size_t(rows)*F));y.set(read_array<U>(d,"labels_path",rows));pid.set(read_array<U>(d,"patient_ids_path",rows));sid.set(read_array<U>(d,"scan_ids_path",rows));eligible.set(read_array<U>(d,"train_eligible_path",rows));
    auto missing=missing_features(d,F,plan);if(!missing.empty()){missing_indices=Device<U>(missing.size());missing_indices.set(missing);}
    Device<U> bad(1);Device<L> ec(1),sr(scans.size()),sf(scans.size()),sfp(scans.size()),st(scans.size());bad.zero();ec.zero();sr.zero();sf.zero();sfp.zero();st.zero();audit<<<blocks(std::uint64_t(rows)*F),256>>>(x.p,y.p,pid.p,sid.p,eligible.p,rows,F,scan_ids.p,scan_patients.p,U(scans.size()),bad.p,ec.p,sr.p,sf.p,sfp.p,st.p,missing_indices.p,U(missing_indices.n));sync();need(!bad.get()[0],"binary segmentation input has invalid class, feature, eligibility or scan/patient identity");
    need(ec.get()[0]==integer(d,"train_eligible_rows"),"declared training eligibility count differs");for(auto n:sr.get())need(n>0,"declared scan has no evaluation rows");auto row_counts=sr.get(),fit_counts=sf.get(),fit_positive_counts=sfp.get(),truth_counts=st.get();for(U i=0;i<scans.size();++i){need(fit_counts[i]>0||(include_positive&&truth_counts[i]>0),"declared scan has no training rows under requested sampling");}
    for(U i=0;i<scans.size();++i){U patient=patients[scan_patient_indices[i]];patient_full_rows[patient]=add(patient_full_rows[patient],row_counts[i]);patient_full_positive[patient]=add(patient_full_positive[patient],truth_counts[i]);patient_outside_positive[patient]=add(patient_outside_positive[patient],outside[i]);}
    for(const auto& scan:d.at("scans")){auto i=std::lower_bound(scans.begin(),scans.end(),scan.at("scan_id").get<U>())-scans.begin();
      if(scan.contains("rows"))need(row_counts[i]==integer(scan,"rows",rows),"declared scan rows differ from memory bound");
      if(scan.contains("training_rows"))need(fit_counts[i]==integer(scan,"training_rows",rows),"declared eligible scan rows differ from memory bound");
      if(scan.contains("training_positive_rows"))need(fit_positive_counts[i]==integer(scan,"training_positive_rows",rows),"declared eligible foreground differs from memory bound");
      if(scan.contains("truth_positive_rows"))need(truth_counts[i]==integer(scan,"truth_positive_rows",rows),"declared foreground differs from FIT sampling bound");
    }
    if(!sampling.is_null()){
      const auto declared=validated_sampling_counts(plan,d);Device<fit_sampling::Counts> actual(scans.size());actual.zero();fit_sampling::count_scans<<<blocks(rows),256>>>(x.p,y.p,eligible.p,sid.p,rows,F,U(integer(sampling,"mask_feature_index")),scan_ids.p,U(scans.size()),actual.p,bad.p);sync();need(!bad.get()[0],"sampling count load audit invalid");auto counts=actual.get();std::map<U,J> records;for(const auto& rec:declared.at("scans"))records.emplace(U(integer(rec,"scan_id")),rec);
      for(U i=0;i<scans.size();++i){const auto& rec=records.at(scans[i]);const auto& c=counts[i];auto full=sampling_array(rec,"full_stratum_counts",rows),retained=sampling_array(rec,"retained_stratum_counts",rows);for(U k=0;k<3;++k)need(c.full[k]==full[k]&&c.retained[k]==retained[k],"loaded sampling stratum count differs from memory metadata");need(c.baseline==integer(rec,"legacy_retained_rows")&&c.eligible==integer(rec,"eligible_rows")&&c.eligible_positive==integer(rec,"eligible_positive_rows"),"loaded sampling legacy/eligibility count differs");U patient=patients[scan_patient_indices[i]];for(U k=0;k<3;++k){patient_full_strata[patient][k]=add(patient_full_strata[patient][k],c.full[k]);patient_retained_strata[patient][k]=add(patient_retained_strata[patient][k],c.retained[k]);}patient_legacy_rows[patient]=add(patient_legacy_rows[patient],c.baseline);}
    }
    auto reference=reference_config(plan,F);if(!reference.is_null()){audit_reference<<<blocks(rows),256>>>(x.p,rows,F,U(integer(reference,"mask_feature_index")),reference.contains("probability_feature_index")?U(integer(reference,"probability_feature_index")):0,reference.contains("probability_feature_index"),bad.p);sync();need(!bad.get()[0],"reference predictor mask/probability feature violates declared semantics");}
    audit_truth<<<blocks(scans.size()),256>>>(st.p,out.p,U(scans.size()),bad.p);sync();need(!bad.get()[0],"declared lesion scan has no reference foreground inside or outside ROI");
  }
  std::vector<L> full_class_counts(const std::vector<U>& fitting) const{
    L rows=0,positive=0;std::set<U> seen;for(U patient:fitting){need(seen.insert(patient).second,"duplicate FIT patient in class-count request");rows=add(rows,patient_full_rows.at(patient));positive=add(positive,patient_full_positive.at(patient));}return {rows-positive,positive};
  }
  Device<U> select_rows(std::vector<U> selected,bool fit) const{
    need(!selected.empty(),"empty patient subset");std::sort(selected.begin(),selected.end());Device<U> chosen(selected.size());chosen.set(selected);Device<U> flags(rows),indices(rows),count(1);count.zero();
    if(sampling.is_null())flag_rows<<<blocks(rows),256>>>(pid.p,eligible.p,y.p,rows,chosen.p,U(selected.size()),fit,include_positive,flags.p);else fit_sampling::select_flags<<<blocks(rows),256>>>(x.p,pid.p,eligible.p,y.p,rows,F,U(integer(sampling,"mask_feature_index")),chosen.p,U(selected.size()),fit,flags.p);sync();
    std::size_t bytes=0;auto sequence=thrust::make_counting_iterator<U>(0);ck(cub::DeviceSelect::Flagged(nullptr,bytes,sequence,flags.p,indices.p,count.p,int(rows)),"segmentation selection workspace");need(std::max<std::size_t>(bytes,1)<=subset_workspace_budget,"CUB row-selection workspace exceeds remaining explicit CUDA budget");maximum_subset_workspace_bytes=std::max<L>(maximum_subset_workspace_bytes,std::max<std::size_t>(bytes,1));Device<unsigned char> temp(std::max<std::size_t>(bytes,1));ck(cub::DeviceSelect::Flagged(temp.p,bytes,sequence,flags.p,indices.p,count.p,int(rows)),"segmentation stable row selection");sync();
    U n=count.get()[0];need(n>0,"patient subset has no eligible rows");Device<U> result(n);ck(cudaMemcpy(result.p,indices.p,n*sizeof(U),cudaMemcpyDeviceToDevice),"segmentation row identities");return result;
  }
  std::shared_ptr<Matrix> subset(std::vector<U> selected,bool fit) const{
    auto ids=select_rows(std::move(selected),fit);U n=U(ids.n);auto m=std::make_shared<Matrix>();m->rows=n;m->features=F;m->ids=std::move(ids);m->x=Device<float>(std::size_t(n)*F);m->y=Device<U>(n);m->scan=Device<U>(n);
    gather<<<blocks(std::uint64_t(n)*F),256>>>(x.p,y.p,sid.p,m->ids.p,n,F,m->x.p,m->y.p,m->scan.p);sync();return m;
  }
};
std::vector<std::vector<U>> folds(const std::vector<U>& ids,U requested){
  U count=std::min<U>(requested,U(ids.size()));need(count>=2,"grouped cross-fitting needs at least two patients");std::vector<std::vector<U>> out(count);for(std::size_t i=0;i<ids.size();++i)out[i%count].push_back(ids[i]);return out;
}
std::vector<U> without(const std::vector<U>& all,const std::vector<U>& removed){std::set<U> omit(removed.begin(),removed.end());std::vector<U> out;for(U p:all)if(!omit.count(p))out.push_back(p);return out;}
struct Metrics {
  const Data& data;U candidates;Device<L> counts;Device<double> scores,dice;Device<L> pc;Device<U> bad,empty;Device<double> sd,repeat;Device<L> volumes;
  Metrics(const Data& d,U n):data(d),candidates(n),counts(std::size_t(n)*d.scans.size()*4),scores(n),dice(std::size_t(n)*d.patients.size()),pc(std::size_t(n)*d.patients.size()*4),bad(1),empty(std::size_t(n)*d.patients.size()),sd(std::size_t(n)*d.scans.size()),repeat(std::size_t(n)*d.patients.size()*2),volumes(std::size_t(n)*d.scans.size()*2){counts.zero();bad.zero();}
  void add(U candidate,const float* prediction,const Matrix& m){need(candidate<candidates,"metric candidate index");confusion<<<blocks(m.rows),256>>>(prediction,m.y.p,m.scan.p,m.rows,data.scan_ids.p,U(data.scans.size()),counts.p+std::size_t(candidate)*data.scans.size()*4,bad.p);sync();need(!bad.get()[0],"non-binary native class prediction");}
  void add_thresholds(U first_candidate,const float* probabilities,U width,U column,const Matrix& m,ProbabilityHistogram& histogram){need(first_candidate<candidates&&histogram.cuts<candidates-first_candidate,"threshold histogram candidate range");histogram.accumulate(probabilities,m.rows,width,column,m.y.p,m.scan.p,data.scan_ids.p,counts.p+std::size_t(first_candidate)*data.scans.size()*4,bad.p);}
  void finish(){summarize<<<blocks(candidates),256>>>(counts.p,data.scan_indices.p,data.out.p,U(data.scans.size()),U(data.patients.size()),candidates,scores.p,dice.p,pc.p,sd.p,volumes.p,repeat.p,empty.p);sync();}
  U best(const std::vector<L>& bytes){finish();Device<L> b(bytes.size());b.set(bytes);Device<U> result(1);choose<<<1,1>>>(scores.p,b.p,candidates,result.p);sync();return result.get()[0];}
  U best_view(const std::vector<L>& bytes,U rules,const std::vector<U>& eligible){
    need(bytes.size()==candidates&&rules&&candidates%rules==0&&!eligible.empty(),"rule view chooser shape");
    Device<L> b(bytes.size());b.set(bytes);Device<U> indices(eligible.size()),result(1),status(1);indices.set(eligible);status.zero();
    choose_view<<<1,1>>>(scores.p,b.p,candidates/rules,rules,indices.p,U(eligible.size()),result.p,status.p);sync();need(!status.get()[0],"rule view inner score invalid");return result.get()[0];
  }
  J compact_json(const J& axes){finish();auto sc=scores.get();auto cc=counts.get();return inner_metrics::encode(axes,data.scans,sc,cc);}
  J json(){finish();auto sc=scores.get();auto ds=dice.get();auto cc=pc.get();auto ss=counts.get();auto ss_dice=sd.get();auto vv=volumes.get();auto rr=repeat.get();auto ee=empty.get();J all=J::array();
    for(U c=0;c<candidates;++c){J patients=J::array(),scans=J::array();for(U p=0;p<data.patients.size();++p){auto q=(std::size_t(c)*data.patients.size()+p)*4;auto z=(std::size_t(c)*data.patients.size()+p)*2;if(ds[std::size_t(c)*data.patients.size()+p]>=0)patients.push_back({{"patient_id",data.patients[p]},{"Dice",ds[std::size_t(c)*data.patients.size()+p]},{"TP",cc[q]},{"FP",cc[q+1]},{"FN",cc[q+2]},{"TN_in_ROI",cc[q+3]},{"both_empty_repeat_predictions",bool(ee[std::size_t(c)*data.patients.size()+p])},{"predicted_absolute_symmetric_repeat_volume_difference",rr[z]>=0?J(rr[z]):J(nullptr)},{"reference_absolute_symmetric_repeat_volume_difference",rr[z+1]>=0?J(rr[z+1]):J(nullptr)}});}
      for(U s=0;s<data.scans.size();++s){auto z=std::size_t(c)*data.scans.size()+s,q=z*4;if(ss[q]+ss[q+1]+ss[q+2]+ss[q+3])scans.push_back({{"scan_id",data.scans[s]},{"patient_id",data.patients[data.scan_patient_indices[s]]},{"TP",ss[q]},{"FP",ss[q+1]},{"FN_in_ROI",ss[q+2]},{"FN_outside_ROI",data.outside[s]},{"TN_in_ROI",ss[q+3]},{"Dice",ss_dice[z]},{"predicted_positive_voxels",vv[z*2]},{"reference_positive_voxels",vv[z*2+1]}});}
      all.push_back({{"macro_patient_Dice",sc[c]},{"patients",patients},{"scans",scans}});
    }return all;
  }
};
struct OOFBank {std::vector<U> training,held;std::vector<TrainingResult> models;};
struct StackData {std::shared_ptr<Matrix> fit,meta_fit;std::vector<TrainingResult> models;std::vector<OOFBank> oof_banks;};
struct NoAggregationCallback {void operator()(U,const Matrix&,const float*,Device<float>&) const {}};
J bank_receipt(const std::vector<OOFBank>& banks){J result=J::array();for(U k=0;k<banks.size();++k){const auto& bank=banks[k];J models=J::array();for(U b=0;b<bank.models.size();++b)models.push_back({{"baseline_index",b},{"model_sha256",bank.models[b].model_sha256}});result.push_back({{"bank_index",k},{"FIT_patient_ids",bank.training},{"OOF_held_patient_ids",bank.held},{"integer_patient_weight",bank.held.size()},{"base_models",models}});}return result;}

struct GroupedTrainer {
  ResidentTrainer native;FitGroupView groups;
  const float* predict_probabilities(const float* values,L rows){return native.predict_probabilities(values,rows);}
};
// Consume borrowed probabilities before another native call invalidates them.
struct MeanSink {
  double* sums=nullptr;U* bad=nullptr;U rows=0,expected=0,seen=0;
  bool active() const{return sums!=nullptr;}
  void add(U member,const float* probabilities){if(!active())return;need(member==seen&&seen<expected,"fixed mean member order or extent differs");oof_aggregation::accumulate<<<blocks(L(rows)*2),256>>>(probabilities,L(rows)*2,1,sums,bad);sync();U status=0;ck(cudaMemcpy(&status,bad,sizeof(U),cudaMemcpyDeviceToHost),"fixed mean status");need(!status,"fixed mean native probability invalid");++seen;}
};
struct BootstrapModels {std::vector<std::vector<TrainingResult>> replicas;J receipts=J::array();};
struct BlendFit {std::vector<float> alpha;J receipt;};
struct Runner {
  J plan;Data& data;J reference,native_rules,training_loss,effective_loss,bootstrap,blend;J sampling_receipts=J::array();L sampling_receipt_bytes=0;std::vector<J> base,meta;std::vector<U> meta_raw;bool aggregate_oof=false,candidate_mean=false,base_candidate_mean=false;Device<U> meta_raw_indices;U oof;std::uint64_t fits=0,custom_fits=0;volatile std::sig_atomic_t& stopping;std::unique_ptr<ResidentTrainer> session;std::unique_ptr<ProbabilityHistogram> threshold_histogram;
  Runner(const J& p,Data& d,volatile std::sig_atomic_t& stop):plan(p),data(d),reference(reference_config(p,d.F)),native_rules(native_prediction_config(p)),training_loss(training_loss_config(p)),effective_loss(training_loss_policy(training_loss,p.at("base_hyperparameters")[0])),bootstrap(bootstrap_config(p)),blend(blend_config(p,d.F)),meta_raw(meta_input_config(p,d.F)),aggregate_oof(oof_aggregation_config(p)),candidate_mean(candidate_aggregation_config(p)),base_candidate_mean(base_aggregation_config(p)),oof(p.value("oof_folds",3u)),stopping(stop){plan["TEST_read"]=false;plan["VALID_read"]=false;for(auto& h:p.at("base_hyperparameters"))base.push_back(ResidentTrainer::validate_hyperparameters(h,2));for(auto& h:p.at("meta_hyperparameters"))meta.push_back(ResidentTrainer::validate_hyperparameters(h,2));if(native_rules.size()>1)threshold_histogram=std::make_unique<ProbabilityHistogram>(native_rules,U(data.scans.size()));if(!meta_raw.empty()){meta_raw_indices=Device<U>(meta_raw.size());meta_raw_indices.set(meta_raw);}}
  bool probability_mean() const{return aggregate_oof||!bootstrap.is_null()||candidate_mean;}
  bool argmax_diagnostics() const{return !bootstrap.is_null()||candidate_mean;}
  const char* mean_key() const{return aggregate_oof?"oof_teacher_aggregation":candidate_mean?"meta_candidate_aggregation":"meta_patient_bootstrap";}
  const char* mean_mode() const{return aggregate_oof?"oof_teacher_probability_mean":candidate_mean?"fixed_meta_probability_mean":"meta_patient_bootstrap_probability_mean";}
  U meta_features() const{return U(base.size()*2+meta_raw.size());}
  void append_raw(const Matrix& raw,float* output){if(!meta_raw.empty()){append_raw_columns<<<blocks(L(raw.rows)*meta_raw.size()),256>>>(raw.x.p,raw.rows,raw.features,meta_raw_indices.p,U(meta_raw.size()),U(base.size()*2),meta_features(),output);sync();}}
  void interrupted(){if(stopping)throw std::runtime_error("segmentation study interrupted at completed native FIT boundary");}
  GroupedTrainer trainer(const std::shared_ptr<Matrix>& m,const std::vector<U>& patients,const char* role,const float* override_weights=nullptr,const J& bootstrap_binding=nullptr){
    J binding={{"format","grouped-segmentation-resident-fit-1"},{"patient_ids",patients},{"rows",m->rows},{"features",m->features},{"role",role},{"training_eligibility_used",true}};
    if(data.descriptor.contains("missing_feature_indices"))binding["source_missing_feature_indices"]=data.descriptor.at("missing_feature_indices");
    const bool balanced=plan.value("fit_class_balanced",false),inverse=plan.value("fit_sampling_weights",std::string("none"))=="inverse_retention",equal_patient=plan.value("fit_group_weights",std::string("none"))=="equal_patient";
    if(!data.sampling.is_null()){
      need(!override_weights,"sampling weight override unsupported");
      if(!m->weights.p){
        auto chosen=patients;std::sort(chosen.begin(),chosen.end());need(!chosen.empty()&&std::adjacent_find(chosen.begin(),chosen.end())==chosen.end(),"sampling FIT patient IDs invalid");const U groups=U(chosen.size()),mask=U(integer(data.sampling,"mask_feature_index"));L legacy=0,expected_rows=0;std::vector<L> full_counts(std::size_t(groups)*3),expected_counts(std::size_t(groups)*3);
        for(U g=0;g<groups;++g){const U patient=chosen[g];legacy=add(legacy,data.patient_legacy_rows.at(patient));for(U k=0;k<3;++k){full_counts[L(g)*3+k]=data.patient_full_strata.at(patient)[k];expected_counts[L(g)*3+k]=data.patient_retained_strata.at(patient)[k];expected_rows=add(expected_rows,expected_counts[L(g)*3+k]);}}
        need(legacy>0&&expected_rows==m->rows,"sampling FIT rows differ from count metadata");Device<U> group_ids(groups),bad(1);Device<L> full(std::size_t(groups)*3),retained(std::size_t(groups)*3);Device<float> coefficients(std::size_t(groups)*3);Device<double> sums(std::size_t(groups)*3);group_ids.set(chosen);full.set(full_counts);retained.zero();bad.zero();sums.zero();
        fit_sampling::count_retained<<<blocks(m->rows),256>>>(data.x.p,data.y.p,data.pid.p,data.rows,data.F,mask,m->ids.p,m->y.p,m->rows,group_ids.p,groups,retained.p,bad.p);sync();need(!bad.get()[0],"sampling FIT original row IDs, labels or group membership differ");const auto actual_counts=retained.get();need(actual_counts==expected_counts,"sampling FIT retained stratum counts differ from declared patient subset");
        fit_sampling::coefficients<<<blocks(groups),256>>>(full.p,retained.p,legacy,groups,coefficients.p,bad.p);sync();need(!bad.get()[0],"sampling weight counts/support/coefficient invalid");m->weights=Device<float>(m->rows);fit_sampling::apply<<<blocks(m->rows),256>>>(data.x.p,data.pid.p,data.F,mask,m->ids.p,m->y.p,m->rows,group_ids.p,groups,coefficients.p,m->weights.p,sums.p);sync();
        const auto weights=coefficients.get();const auto masses=sums.get();m->patient_weights=J::array();L negatives=0,positives=0;double actual_mass=0;
        for(U g=0;g<groups;++g){const L i=L(g)*3;const auto& f=data.patient_full_strata.at(chosen[g]);const auto& n=data.patient_retained_strata.at(chosen[g]);const double mass=masses[i]+masses[i+1]+masses[i+2];need(std::isfinite(mass)&&mass>0,"sampling realized patient weight mass invalid");actual_mass+=mass;negatives=add(negatives,add(n[1],n[2]));positives=add(positives,n[0]);m->patient_weights.push_back({{"patient_id",chosen[g]},{"legacy_retained_rows",data.patient_legacy_rows.at(chosen[g])},{"retained_rows",add(add(n[0],n[1]),n[2])},{"full_stratum_counts",J::array({f[0],f[1],f[2]})},{"retained_stratum_counts",J::array({n[0],n[1],n[2]})},{"stratum_weights",J::array({weights[i],weights[i+1],weights[i+2]})},{"stratum_weight_sums",J::array({masses[i],masses[i+1],masses[i+2]})},{"weight_sum",mass},{"full_class_counts",J::array({add(f[1],f[2]),f[0]})},{"retained_class_counts",J::array({add(n[1],n[2]),n[0]})},{"class_weight_sums",J::array({masses[i+1]+masses[i+2],masses[i]})}});}
        need(std::isfinite(actual_mass)&&actual_mass>0,"sampling actual FIT mass invalid");m->class_counts=J::array({negatives,positives});m->sampling_receipt={{"version",1},{"method","binary_mask_hard_negatives"},{"mask_feature_index",mask},{"normalization","legacy_retained_mass"},{"arithmetic_policy",fit_sampling::arithmetic_policy},{"FIT_patient_ids",chosen},{"FIT_rows",m->rows},{"legacy_retained_rows",legacy},{"target_weight_sum",legacy},{"actual_weight_sum",actual_mass},{"mean_stored_weight",actual_mass/double(m->rows)},{"patients",m->patient_weights},{"original_row_ID_lookup",true},{"outside_ROI_labels_used",false}};
        J audit=m->sampling_receipt;audit["initialization_index"]=sampling_receipts.size();audit["role"]=role;audit["matrix_kind"]=m->response_matrix?"OOF_response":"raw";audit["features"]=m->features;sampling_receipt_bytes=add(sampling_receipt_bytes,audit.dump().size());need(sampling_receipt_bytes<=plan.value("host_byte_budget",16ULL<<30),"sampling receipts exceed host budget");sampling_receipts.push_back(std::move(audit));
      }
      need(!m->sampling_receipt.is_null(),"sampling weights lack source binding");binding["fit_sampling"]=data.sampling;binding["FIT_sampling_receipt"]=m->sampling_receipt;binding["fit_group_weights"]="equal_patient";binding["FIT_patient_weights"]=m->patient_weights;binding["FIT_class_counts"]=m->class_counts;binding["sample_weight_policy"]="three FIT-only strata; equal patient full-ROI target; preserve legacy retained n0 total mass, not mean one";
    }else if(equal_patient){
      if(!m->weights.p){
        auto chosen=patients;std::sort(chosen.begin(),chosen.end());need(!chosen.empty()&&std::adjacent_find(chosen.begin(),chosen.end())==chosen.end(),"FIT patient weights require distinct nonempty groups");U groups=U(chosen.size());
        Device<U> group_ids(groups),bad(1);Device<L> retained(std::size_t(groups)*2),full(std::size_t(groups)*2);Device<float> coefficients(std::size_t(groups)*2);Device<double> sums(std::size_t(groups)*2);group_ids.set(chosen);retained.zero();bad.zero();sums.zero();
        count_fit_patient_classes<<<blocks(m->rows),256>>>(m->y.p,m->ids.p,data.pid.p,data.rows,m->rows,group_ids.p,groups,retained.p,bad.p);sync();need(!bad.get()[0],"FIT patient weighting row identity or patient group differs");
        std::vector<L> full_counts(std::size_t(groups)*2,0);if(inverse)for(U group=0;group<groups;++group){auto patient=chosen[group];L rows=data.patient_full_rows.at(patient),positive=data.patient_full_positive.at(patient);full_counts[std::size_t(group)*2]=rows-positive;full_counts[std::size_t(group)*2+1]=positive;}full.set(full_counts);
        patient_weight_coefficients<<<blocks(groups),256>>>(retained.p,full.p,m->rows,groups,inverse?2u:balanced?1u:0u,coefficients.p,bad.p);sync();auto status=bad.get()[0];need(!(status&2u),"FIT patient weighting requires retained rows for every FIT patient");need(!(status&4u),"inverse-retention patient weights require retained support for every full FIT patient class");need(!status,"FIT patient weight coefficient is invalid");
        m->weights=Device<float>(m->rows);apply_patient_weights<<<blocks(m->rows),256>>>(m->y.p,m->ids.p,data.pid.p,m->rows,group_ids.p,groups,coefficients.p,m->weights.p,sums.p);sync();
        const auto counts=retained.get();const auto coefficient=coefficients.get();const auto masses=sums.get();std::vector<L> totals(2,0);m->patient_weights=J::array();
        for(U group=0;group<groups;++group){auto i=std::size_t(group)*2;totals[0]+=counts[i];totals[1]+=counts[i+1];m->patient_weights.push_back({{"patient_id",chosen[group]},{"retained_class_counts",J::array({counts[i],counts[i+1]})},{"full_class_counts",inverse?J::array({full_counts[i],full_counts[i+1]}):J(nullptr)},{"class_weights",J::array({coefficient[i],coefficient[i+1]})},{"class_weight_sums",J::array({masses[i],masses[i+1]})},{"weight_sum",masses[i]+masses[i+1]}});}m->class_counts=totals;
      }
      binding["fit_group_weights"]="equal_patient";binding["FIT_patient_weights"]=m->patient_weights;binding["FIT_class_counts"]=m->class_counts;
      binding["sample_weight_policy"]=inverse?"equal FIT patient mass with within-patient inverse retention; normalized to mean one":balanced?"equal FIT patient mass and equal present-class mass within each FIT patient; normalized to mean one":"equal FIT patient mass, uniform within each retained patient; normalized to mean one";
      if(inverse)binding["full_FIT_class_counts"]=data.full_class_counts(patients);
    }else if(balanced||inverse){
      if(!m->weights.p){Device<L> counts(2);counts.zero();count_fit_classes<<<blocks(m->rows),256>>>(m->y.p,m->rows,counts.p);sync();auto classes=counts.get();m->weights=Device<float>(m->rows);m->class_counts=classes;
        if(inverse){auto full=data.full_class_counts(patients);need((!full[0]||classes[0])&&(!full[1]||classes[1]),"inverse-retention weights require retained support for every full FIT class");inverse_retention_weights<<<blocks(m->rows),256>>>(m->y.p,m->rows,counts.p,full[0],full[1],m->weights.p);}
        else balanced_fit_weights<<<blocks(m->rows),256>>>(m->y.p,m->rows,counts.p,m->weights.p);sync();
      }
      binding["FIT_class_counts"]=m->class_counts;binding["sample_weight_policy"]=inverse?"inverse FIT class retention, normalized to mean one":"inverse FIT class frequency, normalized to mean one; single-class FIT weights one";
      if(inverse)binding["full_FIT_class_counts"]=data.full_class_counts(patients);
    }
    FitGroupView fit_groups;
    if(!training_loss.is_null()){
      auto chosen=patients;std::sort(chosen.begin(),chosen.end());
      need(!chosen.empty()&&std::adjacent_find(chosen.begin(),chosen.end())==chosen.end(),"group loss requires distinct FIT patients");
      // The same ordered raw row IDs survive OOF response construction. Counts
      // and outside positives are drawn only from this trainer's FIT patients.
      std::vector<FitGroupSpec> specs;J provenance=J::array();
      for(U patient:chosen){
        const auto positive=data.patient_full_positive.at(patient);
        FitGroupSpec spec;spec.full_negative=data.patient_full_rows.at(patient)-positive;
        spec.full_positive=positive;spec.outside_positive=data.patient_outside_positive.at(patient);
        specs.push_back(spec);provenance.push_back({{"patient_id",patient},{"full_negative",spec.full_negative},
          {"full_positive",positive},{"outside_positive",spec.outside_positive}});
      }
      Device<U> selected(chosen.size()),bad(1);selected.set(chosen);bad.zero();
      m->loss_group_ids=Device<U>(m->rows);m->loss_group_specs=Device<FitGroupSpec>(specs.size());m->loss_group_specs.set(specs);
      dense_fit_group_ids<<<blocks(m->rows),256>>>(m->ids.p,data.pid.p,data.rows,m->rows,selected.p,U(chosen.size()),m->loss_group_ids.p,bad.p);
      sync();need(!bad.get()[0],"group loss row identity is outside its FIT patients");
      binding["training_loss"]=training_loss;binding["FIT_loss_groups"]=provenance;
      fit_groups={m->loss_group_ids.p,m->loss_group_specs.p,m->rows,U(chosen.size()),binding,
        std::static_pointer_cast<void>(m),std::static_pointer_cast<void>(m)};
    }
    if(override_weights){need(!bootstrap_binding.is_null()&&training_loss.is_null(),"bootstrap weight override binding differs");binding["meta_patient_bootstrap"]=bootstrap_binding;}
    auto v=m->view(m,binding);if(override_weights)v.sample_weights=override_weights;if(!session){
      // Authenticate the library once without pinning the first full training
      // matrix for the entire study. This one-row owner never fits a model.
      auto probe=std::make_shared<Matrix>();probe->rows=1;probe->features=m->features;probe->x=Device<float>(m->features);probe->y=Device<U>(1);
      ck(cudaMemcpy(probe->x.p,m->x.p,m->features*sizeof(float),cudaMemcpyDeviceToDevice),"library verification row");ck(cudaMemcpy(probe->y.p,m->y.p,sizeof(U),cudaMemcpyDeviceToDevice),"library verification label");
      auto pb=binding;pb["rows"]=1;pb["role"]="LIBRARY_VERIFICATION_ONLY_NO_FIT";session=std::make_unique<ResidentTrainer>(plan,probe->view(probe,pb));
    }
    return {ResidentTrainer(plan,v,*session),std::move(fit_groups)};
  }
  TrainingResult train(GroupedTrainer& t,const J& hp){
    interrupted();auto m=training_loss.is_null()?t.native.train(hp):t.native.train(hp,training_loss_spec(training_loss,t.groups.rows),t.groups);
    if(!training_loss.is_null()){
      const auto& receipt=m.metrics.at("training_loss");
      for(const char* key:{"curvature_policy_version","margin_prediction_mode","verify_incremental_margins","moment_reduction_policy"})
        need(receipt.at(key)==effective_loss.at(key),"completed group loss execution policy differs");
      need(receipt.at("beta")==training_loss.at("beta")&&
        receipt.at("curvature_policy_version")==training_loss.value("curvature_policy",std::string("uniform_v1"))&&receipt.at("FIT_rows")==t.groups.rows&&
        receipt.at("FIT_groups")==t.groups.groups&&receipt.at("native_external_updates")==hp.at("rounds")&&
        receipt.at("fit_binding")==t.groups.fit_binding,"completed group loss receipt differs from requested FIT");
      ++custom_fits;
    }
    ++fits;std::cout<<J{{"event","grouped_segmentation_fit"},{"completed_native_fits",fits}}.dump()<<'\n'<<std::flush;return m;
  }
  std::shared_ptr<Matrix> meta_matrix(const std::shared_ptr<Matrix>& source){auto m=std::make_shared<Matrix>();m->rows=source->rows;m->response_matrix=true;m->features=meta_features();m->x=Device<float>(std::size_t(m->rows)*m->features);m->y=Device<U>(m->rows);m->scan=Device<U>(m->rows);m->ids=Device<U>(m->rows);ck(cudaMemcpy(m->y.p,source->y.p,m->rows*sizeof(U),cudaMemcpyDeviceToDevice),"meta labels");ck(cudaMemcpy(m->scan.p,source->scan.p,m->rows*sizeof(U),cudaMemcpyDeviceToDevice),"meta scans");ck(cudaMemcpy(m->ids.p,source->ids.p,m->rows*sizeof(U),cudaMemcpyDeviceToDevice),"meta row IDs");append_raw(*source,m->x.p);return m;}
  StackData prepare(const std::vector<U>& fitting){
    StackData r;L retained_bytes=0;auto retain=[&](const TrainingResult& model){retained_bytes=add(retained_bytes,model.model_json.size());need(retained_bytes<=plan.value("host_byte_budget",16ULL<<30),"retained source model bank exceeds host budget");};r.fit=data.subset(fitting,true);r.meta_fit=meta_matrix(r.fit);
    {Device<U> positions(data.rows);ck(cudaMemset(positions.p,255,positions.n*sizeof(U)),"OOF absent row map");index_rows<<<blocks(r.fit->rows),256>>>(r.fit->ids.p,r.fit->rows,positions.p);sync();Device<U> seen(std::size_t(r.fit->rows)*base.size()),bad(1);seen.zero();bad.zero();
      // Every OOF source excludes the whole held-out patient, including repeat scans.
      for(const auto& validation:folds(fitting,oof)){
        auto training=without(fitting,validation);auto fit=data.subset(training,true),valid=data.subset(validation,true);auto t=trainer(fit,training,"OOF_FIT");
        OOFBank bank;bank.training=training;bank.held=validation;
        for(U b=0;b<base.size();++b){auto model=train(t,base[b]);if(aggregate_oof){retain(model);bank.models.push_back(std::move(model));}auto pred=t.predict_probabilities(valid->x.p,valid->rows);response_column<<<blocks(valid->rows),256>>>(pred,valid->ids.p,positions.p,valid->rows,r.meta_fit->features,b*2,r.meta_fit->x.p,seen.p,bad.p,U(base.size()));sync();}
        if(aggregate_oof)r.oof_banks.push_back(std::move(bank));
      }
      check_seen<<<blocks(seen.n),256>>>(seen.p,seen.n,bad.p);sync();need(!bad.get()[0],"OOF probability coverage is not exactly once");
    }
    auto t=trainer(r.fit,fitting,"DEPLOYMENT_FIT");for(const auto& hp:base){auto model=train(t,hp);if(aggregate_oof)retain(model);r.models.push_back(std::move(model));}return r;
  }
  BlendFit fit_blends(const StackData& stack,const std::vector<U>& patients){
    BlendFit result;if(blend.is_null())return result;
    need(stack.fit&&stack.meta_fit&&stack.fit->rows==stack.meta_fit->rows,"blend aligned FIT/OOF matrix shape");const U n=stack.fit->rows,parts=U((L(n)+255)/256),teacher=U(integer(blend,"probability_feature_index",data.F-1));
    Device<teacher_blend::Pair> a(parts),b(parts);Device<double> alpha64(1);Device<float> alpha32(1);Device<U> bad(1),positive(1);J coefficients=J::array();positive.zero();positive_blend_weight<<<blocks(n),256>>>(stack.fit->weights.p,n,positive.p);sync();need(positive.get()[0]!=0,"blend FIT requires positive total sample weight");
    for(U model=0;model<base.size();++model){bad.zero();teacher_blend::partials<<<parts,256>>>(stack.fit->x.p,data.F,teacher,stack.meta_fit->x.p,stack.meta_fit->features,model*2+1,stack.fit->y.p,stack.fit->weights.p,n,a.p,bad.p);sync();need(!bad.get()[0],"invalid FIT-only blend probability/label/weight");
      U count=parts;auto* input=a.p;auto* output=b.p;while(count>1){U next=U((L(count)+255)/256);teacher_blend::reduce_pairs<<<next,256>>>(input,count,output,bad.p);sync();std::swap(input,output);count=next;}
      teacher_blend::coefficient<<<1,1>>>(input,alpha64.p,alpha32.p,bad.p);sync();need(!bad.get()[0],"invalid blend coefficient sums");teacher_blend::Pair sums;ck(cudaMemcpy(&sums,input,sizeof(sums),cudaMemcpyDeviceToHost),"blend sums");const double wide=alpha64.get()[0];const float alpha=alpha32.get()[0];result.alpha.push_back(alpha);
      coefficients.push_back({{"baseline_index",model},{"numerator",sums.numerator},{"denominator",sums.denominator},{"alpha64",wide},{"alpha32",alpha},{"alpha32_bits",std::bit_cast<U>(alpha)},{"effective_endpoint",alpha==0?"teacher":alpha==1?"native":"interior"},{"coefficient_branch",sums.denominator==0?"flat_teacher":sums.numerator<=0?"lower_endpoint":sums.numerator>=sums.denominator?"upper_endpoint":"interior"},{"full_refit_model_sha256",stack.models[model].model_sha256}});
    }
    result.receipt={{"FIT_patient_ids",patients},{"FIT_rows",n},{"OOF_patient_groups",folds(patients,oof)},{"FIT_class_counts",stack.fit->class_counts},{"FIT_patient_weights",stack.fit->patient_weights},{"weight_source","original raw FIT sample-weight words; implicit one if absent; consumed once"},{"probability_feature_index",teacher},{"coefficients",coefficients},{"arithmetic_policy",teacher_blend::arithmetic_policy},{"coefficient_authority","alpha32_bits"},{"heldout_labels_used",false},{"positive_total_FIT_weight",true},{"reduction_workspace_bytes",add(multiply(parts,2*sizeof(teacher_blend::Pair)),20)}};return result;
  }
  void blend_values(U model,const BlendFit& coefficients,const Matrix& raw,const float* responses,Device<float>& prediction,Device<U>& bad){
    need(model<coefficients.alpha.size()&&!blend.is_null(),"blend coefficient index");bad.zero();teacher_blend::apply<<<blocks(raw.rows),256>>>(raw.x.p,raw.features,U(integer(blend,"probability_feature_index")),responses,meta_features(),model*2+1,raw.rows,coefficients.alpha[model],prediction.p,bad.p);sync();need(!bad.get()[0],"invalid evaluation blend probability");
  }
  void blend_classes(U rule,const Matrix& raw,Device<float>& prediction,Device<U>& bad){
    bad.zero();teacher_blend::classify<<<blocks(raw.rows),256>>>(prediction.p,raw.rows,rule==0,rule?native_rules.at(rule).at("threshold").get<float>():.5f,prediction.p,bad.p);sync();need(!bad.get()[0],"invalid blend class probability");
  }
  BootstrapModels bootstrap_models(const std::shared_ptr<Matrix>& matrix,const std::vector<U>& patients,const std::vector<U>& indices,const char* role){
    need(!bootstrap.is_null()&&matrix->weights.p&&!matrix->patient_weights.is_null()&&!indices.empty(),"bootstrap requires initialized ordinary meta weights and settings");
    BootstrapModels result;const auto count=U(integer(bootstrap,"replicates",UINT32_MAX));const auto seed=integer(bootstrap,"seed");
    auto chosen=patients;std::sort(chosen.begin(),chosen.end());const U groups=U(chosen.size());
    Device<U> group_ids(groups),multiplicity(groups),bad(1);group_ids.set(chosen);
    Device<float> weights(matrix->rows);Device<double> masses(std::size_t(groups)*2);L retained_bytes=0;
    for(U replicate=0;replicate<count;++replicate){
      interrupted();const auto draws=meta_bootstrap::multiplicities(chosen,seed,replicate);multiplicity.set(draws);bad.zero();masses.zero();
      meta_bootstrap::apply_weights<<<blocks(matrix->rows),256>>>(matrix->weights.p,matrix->ids.p,data.pid.p,data.rows,matrix->y.p,matrix->rows,group_ids.p,multiplicity.p,groups,weights.p,masses.p,bad.p);sync();need(!bad.get()[0],"bootstrap FIT patient weight or row identity invalid");
      const auto class_masses=masses.get();J patient_receipts=J::array();U unique=0;L draw_total=0;std::vector<L> supported_rows(2,0);std::vector<double> class_totals(2,0);
      for(U g=0;g<groups;++g){const auto& original=matrix->patient_weights.at(g);need(original.at("patient_id")==chosen[g],"bootstrap original patient weights differ");draw_total=add(draw_total,draws[g]);unique+=draws[g]>0;
        for(U c=0;c<2;++c){if(draws[g])supported_rows[c]=add(supported_rows[c],original.at("retained_class_counts").at(c).get<L>());class_totals[c]+=class_masses[g*2+c];}
        patient_receipts.push_back({{"patient_id",chosen[g]},{"multiplicity",draws[g]},{"original_weight_sum",original.at("weight_sum")},{"retained_class_counts",original.at("retained_class_counts")},{"bootstrap_class_weight_sums",J::array({class_masses[g*2],class_masses[g*2+1]})}});
      }
      need(draw_total==groups&&unique>0&&std::isfinite(class_totals[0])&&std::isfinite(class_totals[1])&&class_totals[0]+class_totals[1]>0,"bootstrap draw or actual weight mass invalid");
      J receipt={{"replicate_index",replicate},{"seed",seed},{"draw_policy",meta_bootstrap::draw_policy},{"weight_policy",meta_bootstrap::weight_policy},{"FIT_patient_ids",chosen},{"FIT_rows",matrix->rows},{"draw_count",draw_total},{"unique_patient_count",unique},{"patients",patient_receipts},{"positive_weight_class_row_counts",supported_rows},{"class_weight_sums",class_totals},{"total_weight_mass",class_totals[0]+class_totals[1]},{"single_class_positive_weight_support",!supported_rows[0]||!supported_rows[1]},{"zero_weight_rows_retained",true},{"fresh_immutable_weighted_DMatrix",true},{"OOF_feature_words_reused",true},{"meta_hyperparameter_indices",indices}};
      std::vector<TrainingResult> models;J pins=J::array();
      {auto owner=trainer(matrix,patients,role,weights.p,receipt);for(U index:indices){auto model=train(owner,meta.at(index));retained_bytes=add(retained_bytes,model.model_json.size());need(retained_bytes<=plan.value("host_byte_budget",16ULL<<30),"bootstrap retained models exceed host budget");pins.push_back({{"combiner_index",index},{"model_sha256",model.model_sha256},{"native_model_bytes",model.model_json.size()}});models.push_back(std::move(model));}}
      receipt["models"]=std::move(pins);result.receipts.push_back(std::move(receipt));result.replicas.push_back(std::move(models));
    }
    return result;
  }
  J evaluation_batches=J::array();
  template<class BaseCallback,class BatchCallback,class AggregationCallback=NoAggregationCallback>
  void evaluate(const std::vector<U>& patients,const std::vector<TrainingResult>& models,const std::vector<TrainingResult>& metas,BaseCallback base_callback,BatchCallback batch_callback,const std::vector<OOFBank>& banks={},AggregationCallback aggregation_callback={},const BootstrapModels& bootstrapped={}){
    need(models.size()==base.size()&&!metas.empty(),"evaluation model bank differs");auto ids=data.select_rows(patients,false);U width=meta_features();
    L total_weight=0;for(const auto& bank:banks){need(bank.models.size()==base.size()&&!bank.held.empty(),"OOF evaluation bank shape/weight differs");total_weight=add(total_weight,bank.held.size());}need(total_weight<=(1ULL<<53),"OOF patient weight exceeds exact integer range");
    const U bag_models=bootstrapped.replicas.empty()?0:U(bootstrapped.replicas.front().size());
    need(banks.empty()||bootstrapped.replicas.empty(),"teacher and meta bootstrap aggregation cannot be combined");
    for(const auto& replica:bootstrapped.replicas)need(!replica.empty()&&replica.size()==bag_models,"bootstrap evaluation model extents differ");
    need(!candidate_mean||(banks.empty()&&bootstrapped.replicas.empty()&&metas.size()==meta.size()),"fixed mean requires exactly every ordinary meta candidate");
    const L mean_models=candidate_mean?1:banks.empty()?bag_models:metas.size();
    L model_bytes=0;for(const auto& replica:bootstrapped.replicas)for(const auto& model:replica)model_bytes=add(model_bytes,model.model_json.size());for(const auto& bank:banks)for(const auto& model:bank.models)model_bytes=add(model_bytes,model.model_json.size());for(const auto* bank:{&models,&metas})for(const auto& model:*bank)model_bytes=add(model_bytes,model.model_json.size());need(model_bytes<=plan.value("host_byte_budget",16ULL<<30),"retained evaluation models exceed host budget");
    ResidentPredictor predictor(plan,data.F,2,*session),meta_predictor(plan,width,2,*session);Device<U> bad(1);bad.zero();
    std::size_t free_before=0,total=0;ck(cudaMemGetInfo(&free_before,&total),"evaluation free memory before registrations");std::size_t minimum_free=free_before;
    // Warm every immutable native model with one row before sizing buffers.
    // Source byte length is provenance, never an invented GPU residency bound.
    {Device<float> raw(data.F),responses(width);Device<U> label(1),scan(1);responses.zero();gather<<<blocks(data.F),256>>>(data.x.p,data.y.p,data.sid.p,ids.p,1,data.F,raw.p,label.p,scan.p);sync();
      auto warm=[&](ResidentPredictor& owner,const TrainingResult& model,const float* values){owner=ResidentPredictor(plan,owner.features(),2,*session);owner.restore_model(model.model_json,model.model_sha256);owner.predict(values,1,false);owner.predict_probabilities(values,1);sync();std::size_t available=0;ck(cudaMemGetInfo(&available,&total),"evaluation registered native memory");minimum_free=std::min(minimum_free,available);};
      for(const auto& model:models)warm(predictor,model,raw.p);for(const auto& bank:banks)for(const auto& model:bank.models)warm(predictor,model,raw.p);for(const auto& model:metas)warm(meta_predictor,model,responses.p);for(const auto& replica:bootstrapped.replicas)for(const auto& model:replica)warm(meta_predictor,model,responses.p);
    }
    std::size_t available=0;ck(cudaMemGetInfo(&available,&total),"evaluation registered native free memory");L reserve=free_before>minimum_free?free_before-minimum_free:0;
    L budget=plan.value("gpu_byte_budget",12ULL<<30),per_row=add(add(multiply(data.F,4),multiply(width,4)),add(48,base_candidate_mean?8:mean_models?add(multiply(mean_models,16),8):0));
    // 16 owned bytes/row: IDs, labels, scans and shared prediction. 32 more
    // reserve both predictors' original/derived two-class output buffers.
    U capacity=evaluation_row_capacity(budget,owned_CUDA_bytes,std::min<L>(available,minimum_free),reserve,per_row,ids.n,plan.value("evaluation_batch_rows",0ULL));
    Matrix raw;raw.features=data.F;raw.x=Device<float>(multiply(capacity,data.F));raw.y=Device<U>(capacity);raw.scan=Device<U>(capacity);raw.ids=Device<U>(capacity);Device<float> responses(multiply(capacity,width)),prediction(capacity);
    Device<double> sums;Device<float> averaged;if(base_candidate_mean)averaged=Device<float>(multiply(capacity,2));if(mean_models){sums=Device<double>(multiply(multiply(capacity,2),mean_models));averaged=Device<float>(multiply(capacity,2));}
    U tiles=0;L consumed=0;auto registered=[&](ResidentPredictor& owner,const TrainingResult& model){owner=ResidentPredictor(plan,owner.features(),2,*session);std::size_t free=0;ck(cudaMemGetInfo(&free,&total),"evaluation native registration headroom");need(free>=reserve,"native evaluation residency grew beyond reserved headroom");owner.restore_model(model.model_json,model.model_sha256);};
    for(L first=0;first<ids.n;first+=capacity){interrupted();raw.rows=U(std::min<L>(capacity,ids.n-first));ck(cudaMemcpy(raw.ids.p,ids.p+first,raw.rows*sizeof(U),cudaMemcpyDeviceToDevice),"evaluation batch row identities");
      gather<<<blocks(L(raw.rows)*data.F),256>>>(data.x.p,data.y.p,data.sid.p,raw.ids.p,raw.rows,data.F,raw.x.p,raw.y.p,raw.scan.p);sync();
      append_raw(raw,responses.p);
      for(U b=0;b<models.size();++b){registered(predictor,models[b]);auto probabilities=predictor.predict_probabilities(raw.x.p,raw.rows);response_column<<<blocks(raw.rows),256>>>(probabilities,nullptr,nullptr,raw.rows,width,b*2,responses.p,nullptr,bad.p);sync();need(!bad.get()[0],"evaluation base probabilities invalid");base_callback(b,predictor,raw,responses.p,prediction);}
      if(base_candidate_mean){base_mean::probabilities<<<blocks(raw.rows),256>>>(responses.p,raw.rows,width,U(models.size()),averaged.p,bad.p);sync();need(!bad.get()[0],"fixed base mean output invalid");aggregation_callback(0,raw,averaged.p,prediction);}
      MeanSink mean;if(candidate_mean){sums.zero();mean={sums.p,bad.p,raw.rows,U(metas.size()),0};}
      if constexpr(requires{batch_callback(raw,responses.p,meta_predictor,prediction,registered,mean);})batch_callback(raw,responses.p,meta_predictor,prediction,registered,mean);
      else{need(!candidate_mean,"fixed mean callback does not consume member probabilities");batch_callback(raw,responses.p,meta_predictor,prediction,registered);}
      if(candidate_mean){need(mean.seen==mean.expected,"fixed mean omitted a meta candidate");oof_aggregation::finish<<<blocks(L(raw.rows)*2),256>>>(sums.p,L(raw.rows)*2,metas.size(),averaged.p,bad.p);sync();need(!bad.get()[0],"fixed mean output invalid");aggregation_callback(0,raw,averaged.p,prediction);}
      if(!banks.empty()){
        sums.zero();
        for(const auto& bank:banks){
          for(U b=0;b<bank.models.size();++b){registered(predictor,bank.models[b]);auto probabilities=predictor.predict_probabilities(raw.x.p,raw.rows);response_column<<<blocks(raw.rows),256>>>(probabilities,nullptr,nullptr,raw.rows,width,b*2,responses.p,nullptr,bad.p);sync();need(!bad.get()[0],"OOF bank probabilities invalid");}
          for(U m=0;m<metas.size();++m){registered(meta_predictor,metas[m]);auto probabilities=meta_predictor.predict_probabilities(responses.p,raw.rows);oof_aggregation::accumulate<<<blocks(L(raw.rows)*2),256>>>(probabilities,L(raw.rows)*2,bank.held.size(),sums.p+L(m)*capacity*2,bad.p);sync();need(!bad.get()[0],"OOF meta probability accumulation invalid");}
        }
        for(U m=0;m<metas.size();++m){oof_aggregation::finish<<<blocks(L(raw.rows)*2),256>>>(sums.p+L(m)*capacity*2,L(raw.rows)*2,total_weight,averaged.p,bad.p);sync();need(!bad.get()[0],"OOF averaged probabilities invalid");aggregation_callback(m,raw,averaged.p,prediction);}
      }
      if(!bootstrapped.replicas.empty()){
        sums.zero();
        for(const auto& replica:bootstrapped.replicas)for(U m=0;m<replica.size();++m){registered(meta_predictor,replica[m]);auto probabilities=meta_predictor.predict_probabilities(responses.p,raw.rows);oof_aggregation::accumulate<<<blocks(L(raw.rows)*2),256>>>(probabilities,L(raw.rows)*2,1,sums.p+L(m)*capacity*2,bad.p);sync();need(!bad.get()[0],"bootstrap meta probability accumulation invalid");}
        for(U m=0;m<bag_models;++m){oof_aggregation::finish<<<blocks(L(raw.rows)*2),256>>>(sums.p+L(m)*capacity*2,L(raw.rows)*2,bootstrapped.replicas.size(),averaged.p,bad.p);sync();need(!bad.get()[0],"bootstrap mean probabilities invalid");aggregation_callback(m,raw,averaged.p,prediction);}
      }
      consumed+=raw.rows;++tiles;
    }
    need(consumed==ids.n,"evaluation batches omitted rows");evaluation_batches.push_back({{"rows",ids.n},{"capacity",capacity},{"batches",tiles},{"base_models",models.size()},{"meta_models",metas.size()},{"all_base_response_columns",models.size()*2},{"serialized_model_bytes",model_bytes},{"native_registration_reserve_bytes",reserve},{"bytes_per_row_including_output_reserve",per_row},{"native_opaque_allocations_additional",true},{"prediction_files_written",false}});if(!banks.empty()){evaluation_batches.back()["OOF_banks"]=banks.size();evaluation_batches.back()["OOF_integer_weight_total"]=total_weight;evaluation_batches.back()["OOF_accumulator_bytes_per_row"]=add(multiply(metas.size(),16),8);evaluation_batches.back()["OOF_arithmetic_policy"]=oof_aggregation::arithmetic_policy;}if(candidate_mean){evaluation_batches.back()["fixed_mean_members"]=metas.size();evaluation_batches.back()["fixed_mean_accumulator_bytes_per_row"]=24;evaluation_batches.back()["fixed_mean_arithmetic_policy"]=oof_aggregation::arithmetic_policy;evaluation_batches.back()["fixed_mean_reuses_ordinary_probability_calls"]=true;}if(!bootstrapped.replicas.empty()){evaluation_batches.back()["bootstrap_replicates"]=bootstrapped.replicas.size();evaluation_batches.back()["bootstrap_meta_settings"]=bag_models;evaluation_batches.back()["bootstrap_accumulator_bytes_per_row"]=add(multiply(bag_models,16),8);evaluation_batches.back()["bootstrap_arithmetic_policy"]=oof_aggregation::arithmetic_policy;}if(base_candidate_mean){evaluation_batches.back()["fixed_base_mean_members"]=models.size();evaluation_batches.back()["fixed_base_mean_pair_bytes_per_row"]=8;evaluation_batches.back()["fixed_base_mean_arithmetic_policy"]=base_mean::arithmetic_policy;evaluation_batches.back()["fixed_base_mean_reuses_ordinary_probability_calls"]=true;}if(!meta_raw.empty()){evaluation_batches.back()["meta_input_columns"]=width;evaluation_batches.back()["meta_input_mapping"]=meta_input_mapping(meta_raw);}
  }
};
struct BaseMeanChoice {U rule,family;J details;};
struct BlendChoice {U baseline,rule,family;J details;};
struct Selection {U baseline,meta,family,reference=0,baseline_rule=0,meta_rule=0;J details;std::vector<Selection> views,aggregated;std::vector<BlendChoice> blended;std::vector<BaseMeanChoice> base_means;};
void reference_prediction(const J& config,U candidate,const Matrix& m,Device<float>& output){
  const auto& setting=config.at("candidates").at(candidate);reference_classes<<<blocks(m.rows),256>>>(m.x.p,m.rows,m.features,U(integer(setting,"feature_index")),setting.at("kind")=="default_mask",setting.value("threshold",0.f),output.p);sync();
}
J metric_order(const Runner& runner){J order={"native_baseline","nonlinear_combiner","inner_selected"};if(!runner.reference.is_null()){order.push_back("reference_default");order.push_back("reference_inner_selected");}return order;}
const char* family_name(U family){return family==4?"fixed_base_probability_mean":family==3?"teacher_probability_blend":family==2?"external_reference":family==1?"nonlinear_combiner":"native_baseline";}
J aggregation_rule(J rule,bool bootstrap=false,bool fixed_mean=false){rule["operation"]=rule.at("kind")=="native_argmax"?"strict FP32 averaged p1>p0; exact tie class 0":fixed_mean?"FP32 uniform ordinary-meta probability mean; p1>=threshold":bootstrap?"FP32 equal bootstrap-meta probability mean; p1>=threshold":"FP32 patient-weighted OOF mean probability; p1>=threshold";rule["probability_source"]=fixed_mean?"same full-refit teacher response through every ordinary meta candidate; uniform mean after nonlinear meta":bootstrap?"same full-refit teacher response through each bootstrap-fitted meta; equal mean after nonlinear meta":"same fitted meta applied separately to each OOF teacher bank; mean after nonlinear meta";return rule;}
void averaged_prediction(const J& rule,const float* probabilities,const Matrix& raw,Device<float>& prediction){
  if(rule.at("kind")=="native_argmax"){/* finish already audited every probability before this callback. */oof_aggregation::binary_argmax<<<blocks(raw.rows),256>>>(probabilities,raw.rows,prediction.p,nullptr);sync();}
  else threshold_prediction(rule,probabilities,raw.rows,2,1,prediction);
}
Selection select_models(const J& plan,Data& data,Runner& runner,const std::vector<U>& fitting,const char* role){
  U rules=U(runner.native_rules.size());Metrics bs(data,U(runner.base.size())*rules),ms(data,U(runner.meta.size())*rules);std::vector<L> bb(runner.base.size()*rules,0),mb(runner.meta.size()*rules,0);
  std::unique_ptr<Metrics> aggregated_scores;if(runner.probability_mean())aggregated_scores=std::make_unique<Metrics>(data,runner.candidate_mean?rules:U(runner.meta.size())*rules);J inner_banks=J::array(),inner_bootstraps=J::array(),inner_meta_fits=J::array();std::vector<L> ab(runner.meta.size()*rules,0);Device<L> argmax_differences;Device<U> diagnostic_bad;if(runner.argmax_diagnostics()){argmax_differences=Device<L>(runner.meta.size());argmax_differences.zero();diagnostic_bad=Device<U>(1);diagnostic_bad.zero();}
  std::unique_ptr<Metrics> reference_scores;if(!runner.reference.is_null())reference_scores=std::make_unique<Metrics>(data,U(runner.reference.at("candidates").size()));
  std::unique_ptr<Metrics> blend_scores;Device<U> blend_bad;J inner_blends=J::array();if(!runner.blend.is_null()){blend_scores=std::make_unique<Metrics>(data,U(runner.base.size())*rules);blend_bad=Device<U>(1);}
  std::unique_ptr<Metrics> base_mean_scores;J inner_base_mean_fits=J::array();if(runner.base_candidate_mean)base_mean_scores=std::make_unique<Metrics>(data,rules);
  auto inner=folds(fitting,plan.value("inner_folds",3u));
  for(const auto& held:inner){auto fit=without(fitting,held);need(fit.size()>=2,"inner FIT has too few patient groups for OOF");auto prepared=runner.prepare(fit);std::vector<TrainingResult> meta_models;
    {auto t=runner.trainer(prepared.meta_fit,fit,role);for(U m=0;m<runner.meta.size();++m){auto model=runner.train(t,runner.meta[m]);for(U c=0;c<rules;++c)mb[m*rules+c]+=model.model_json.size();meta_models.push_back(std::move(model));}}
    if(runner.base_candidate_mean){J members=J::array();for(U b=0;b<prepared.models.size();++b)members.push_back({{"baseline_index",b},{"model_sha256",prepared.models[b].model_sha256},{"native_model_bytes",prepared.models[b].model_json.size()}});inner_base_mean_fits.push_back({{"inner_validation_patient_ids",held},{"enclosing_FIT_patient_ids",fit},{"models",members}});}
    if(runner.candidate_mean){J members=J::array();for(U m=0;m<meta_models.size();++m)members.push_back({{"combiner_index",m},{"model_sha256",meta_models[m].model_sha256},{"native_model_bytes",meta_models[m].model_json.size()}});inner_meta_fits.push_back({{"inner_validation_patient_ids",held},{"enclosing_FIT_patient_ids",fit},{"models",members}});}
    BootstrapModels bagged;if(!runner.bootstrap.is_null()){std::vector<U> indices(runner.meta.size());std::iota(indices.begin(),indices.end(),0);bagged=runner.bootstrap_models(prepared.meta_fit,fit,indices,"INNER_BOOTSTRAP_META_FIT");inner_bootstraps.push_back({{"inner_validation_patient_ids",held},{"enclosing_FIT_patient_ids",fit},{"replicates",bagged.receipts}});for(const auto& replica:bagged.replicas)for(U m=0;m<replica.size();++m)for(U c=0;c<rules;++c)ab[m*rules+c]=add(ab[m*rules+c],replica[m].model_json.size());}
    for(U b=0;b<prepared.models.size();++b)for(U c=0;c<rules;++c)bb[b*rules+c]+=prepared.models[b].model_json.size();
    if(runner.aggregate_oof)inner_banks.push_back({{"inner_validation_patient_ids",held},{"enclosing_FIT_patient_ids",fit},{"banks",bank_receipt(prepared.oof_banks)}});
    const auto blend_fit=runner.fit_blends(prepared,fit);if(blend_scores){J receipt=blend_fit.receipt;receipt["inner_validation_patient_ids"]=held;inner_blends.push_back(std::move(receipt));}
    prepared.fit.reset();prepared.meta_fit.reset();
    runner.evaluate(held,prepared.models,meta_models,
      [&](U b,ResidentPredictor& predictor,const Matrix& raw,const float* responses,Device<float>& prediction){bs.add(b*rules,predictor.predict(raw.x.p,raw.rows,false),raw);if(rules>1)bs.add_thresholds(b*rules,responses,runner.meta_features(),b*2+1,raw,*runner.threshold_histogram);if(blend_scores){runner.blend_values(b,blend_fit,raw,responses,prediction,blend_bad);if(rules>1)blend_scores->add_thresholds(b*rules,prediction.p,1,0,raw,*runner.threshold_histogram);runner.blend_classes(0,raw,prediction,blend_bad);blend_scores->add(b*rules,prediction.p,raw);}},
      [&](const Matrix& raw,const float* responses,ResidentPredictor& predictor,Device<float>& prediction,auto& registered,MeanSink& mean){
        for(U m=0;m<meta_models.size();++m){registered(predictor,meta_models[m]);const auto classes=predictor.predict(responses,raw.rows,false);ms.add(m*rules,classes,raw);if(runner.argmax_diagnostics())ck(cudaMemcpy(prediction.p,classes,raw.rows*sizeof(float),cudaMemcpyDeviceToDevice),"preserve ordinary inner argmax diagnostic");if(rules>1||runner.argmax_diagnostics()){auto probabilities=predictor.predict_probabilities(responses,raw.rows);if(rules>1)ms.add_thresholds(m*rules,probabilities,2,1,raw,*runner.threshold_histogram);mean.add(m,probabilities);if(runner.argmax_diagnostics()){meta_bootstrap::argmax_disagreements<<<blocks(raw.rows),256>>>(prediction.p,probabilities,raw.rows,argmax_differences.p+m,diagnostic_bad.p);sync();need(!diagnostic_bad.get()[0],"ordinary inner argmax diagnostic invalid");}}}
        if(reference_scores)for(U candidate=0;candidate<runner.reference.at("candidates").size();++candidate){reference_prediction(runner.reference,candidate,raw,prediction);reference_scores->add(candidate,prediction.p,raw);}
      },prepared.oof_banks,[&](U m,const Matrix& raw,const float* probabilities,Device<float>& prediction){averaged_prediction(runner.native_rules.at(0),probabilities,raw,prediction);if(base_mean_scores){base_mean_scores->add(0,prediction.p,raw);if(rules>1)base_mean_scores->add_thresholds(0,probabilities,2,1,raw,*runner.threshold_histogram);}else{aggregated_scores->add(m*rules,prediction.p,raw);if(rules>1)aggregated_scores->add_thresholds(m*rules,probabilities,2,1,raw,*runner.threshold_histogram);}},bagged);
  }
  U best_base=bs.best(bb),best_meta=ms.best(mb);Device<U> family(1);choose_family<<<1,1>>>(bs.scores.p,best_base,ms.scores.p,best_meta,family.p);sync();U selected=family.get()[0],best_reference=0;
  if(reference_scores){std::vector<L> cost(runner.reference.at("candidates").size(),0);best_reference=reference_scores->best(cost);auto r=reference_scores->scores.get(),b=bs.scores.get(),m=ms.scores.get();double native=selected?m[best_meta]:b[best_base];if(r[best_reference]>=native)selected=2;}
  U baseline=best_base/rules,meta=best_meta/rules,baseline_rule=best_base%rules,meta_rule=best_meta%rules;
  const bool compact=inner_metrics::encoding(plan)=="scan_counts_v1";J axes;
  if(compact){std::vector<U> scan_patients;for(auto index:data.scan_patient_indices)scan_patients.push_back(data.patients[index]);axes=inner_metrics::make_axes(fitting,data.scans,scan_patients,data.outside);}
  J details={{"baseline_index",baseline},{"combiner_index",meta},{"selected_family",family_name(selected)},{"baseline_hyperparameters",runner.base[baseline]},{"combiner_hyperparameters",runner.meta[meta]},{"inner_validation_patient_groups",inner},{"inner_baseline_results",compact?bs.compact_json(axes):bs.json()},{"inner_combiner_results",compact?ms.compact_json(axes):ms.json()}};
  if(!runner.meta_raw.empty()){details["meta_input_mapping"]=meta_input_mapping(runner.meta_raw);details["meta_input_columns"]=runner.meta_features();}
  if(compact)details["inner_metric_encoding"]=axes;
  if(rules>1){details["baseline_prediction_candidate_index"]=baseline_rule;details["combiner_prediction_candidate_index"]=meta_rule;details["baseline_prediction_rule"]=runner.native_rules.at(baseline_rule);details["combiner_prediction_rule"]=runner.native_rules.at(meta_rule);details["native_prediction_candidates"]=runner.native_rules;details["native_inner_result_layout"]="model-major, prediction-candidate-minor; exact native argmax is candidate 0";details["native_threshold_selection_scope"]="inner held-out full ROI patient macro Dice only; outside-ROI truth contributes false negatives";}
  if(reference_scores){details["reference_candidate_index"]=best_reference;details["reference_candidate"]=runner.reference.at("candidates").at(best_reference);details["reference_inner_results"]=compact?reference_scores->compact_json(axes):reference_scores->json();details["reference_default_always_eligible"]=true;details["reference_tie_policy"]="prefer fixed reference over a fitted family on equal inner Dice";}
  Selection result{baseline,meta,selected,best_reference,baseline_rule,meta_rule,std::move(details)};
  const auto views=rule_search_views_config(plan,rules);
  if(runner.base_candidate_mean&&views.empty()){std::vector<L> base_bytes,meta_bytes;for(U b=0;b<runner.base.size();++b)base_bytes.push_back(bb[b*rules]);for(U m=0;m<runner.meta.size();++m)meta_bytes.push_back(mb[m*rules]);result.details["inner_baseline_model_bytes"]=base_bytes;result.details["inner_combiner_model_bytes"]=meta_bytes;}
  if(!views.empty()){
    result.details["rule_search_views"]=J::array();const auto base_scores=bs.scores.get(),meta_scores=ms.scores.get();
    std::vector<L> base_bytes,meta_bytes;for(U b=0;b<runner.base.size();++b)base_bytes.push_back(bb[b*rules]);for(U m=0;m<runner.meta.size();++m)meta_bytes.push_back(mb[m*rules]);
    result.details["inner_baseline_model_bytes"]=base_bytes;result.details["inner_combiner_model_bytes"]=meta_bytes;
    for(const auto& view:views){
      const auto indices=view.at("native_prediction_candidate_indices").get<std::vector<U>>();U b=bs.best_view(bb,rules,indices),m=ms.best_view(mb,rules,indices);
      choose_view_family<<<1,1>>>(bs.scores.p,b,ms.scores.p,m,reference_scores?reference_scores->scores.p:nullptr,best_reference,family.p);sync();U f=family.get()[0];
      J item={{"id",view.at("id")},{"baseline_index",b/rules},{"combiner_index",m/rules},{"selected_family",family_name(f)},{"baseline_hyperparameters",runner.base[b/rules]},{"combiner_hyperparameters",runner.meta[m/rules]},
        {"baseline_prediction_candidate_index",b%rules},{"combiner_prediction_candidate_index",m%rules},{"baseline_prediction_rule",runner.native_rules.at(b%rules)},{"combiner_prediction_rule",runner.native_rules.at(m%rules)},
        {"inner_baseline_candidate_index",b},{"inner_combiner_candidate_index",m},{"inner_baseline_macro_patient_Dice",base_scores[b]},{"inner_combiner_macro_patient_Dice",meta_scores[m]},
        {"inner_metric_source","primary selection shared inner arrays; canonical model-major, rule-minor indices"}};
      if(reference_scores){item["reference_candidate_index"]=best_reference;item["reference_candidate"]=runner.reference.at("candidates").at(best_reference);}
      result.details["rule_search_views"].push_back(item);result.views.push_back({b/rules,m/rules,f,best_reference,b%rules,m%rules,std::move(item)});
    }
  }
  if(base_mean_scores){
    auto& scores=*base_mean_scores;scores.finish();const auto values=scores.scores.get();L bytes=0;J indices=J::array(),rules_json=J::array();for(U b=0;b<runner.base.size();++b){bytes=add(bytes,bb[b*rules]);indices.push_back(b);}for(const auto& rule:runner.native_rules)rules_json.push_back(base_mean_rule(rule));std::vector<L> costs(rules,bytes);
    J saved={{"version",1},{"method","uniform_probability_mean"},{"inference_mode","fixed_base_probability_mean"},{"arithmetic_policy",base_mean::arithmetic_policy},{"base_recipe","uniform_all_declared_base_candidates_v1"},{"base_indices",indices},{"base_hyperparameter_bank",runner.base},{"inner_base_fits",inner_base_mean_fits},{"inner_base_mean_results",compact?scores.compact_json(axes):scores.json()},{"mean_prediction_candidates",rules_json},{"inner_result_layout","one fixed base ensemble; prediction-rule index only"},{"inner_base_mean_model_bytes",J::array({bytes})},{"byte_tie_policy","sum every required serialized base model across inner FITs; no deduplication; canonical rule order"},{"family_tie_policy","all ordinary candidates retained; base mean wins only on strict inner Dice improvement"},{"rule_search_views",J::array()}};
    if(compact)saved["inner_metric_encoding"]=axes;
    std::vector<const Selection*> ordinary={&result};for(const auto& view:result.views)ordinary.push_back(&view);
    for(U v=0;v<ordinary.size();++v){const auto& control=*ordinary[v];std::vector<U> eligible;if(v)eligible=views[v-1].at("native_prediction_candidate_indices").get<std::vector<U>>();else{eligible.resize(rules);std::iota(eligible.begin(),eligible.end(),0);}U winner=scores.best_view(costs,rules,eligible);
      choose_base_mean_family<<<1,1>>>(bs.scores.p,control.baseline*rules+control.baseline_rule,ms.scores.p,control.meta*rules+control.meta_rule,reference_scores?reference_scores->scores.p:nullptr,best_reference,control.family,scores.scores.p,winner,family.p);sync();U selected=family.get()[0];
      J item={{"id",v?views[v-1].at("id"):J("full")},{"base_recipe","uniform_all_declared_base_candidates_v1"},{"base_indices",indices},{"mean_prediction_candidate_index",winner},{"mean_prediction_rule",rules_json.at(winner)},{"inner_mean_candidate_index",winner},{"inner_mean_macro_patient_Dice",values[winner]},{"ordinary_selected_family",family_name(control.family)},{"augmented_selected_family",family_name(selected)},{"ordinary_baseline_index",control.baseline},{"ordinary_combiner_index",control.meta},{"ordinary_baseline_prediction_candidate_index",control.baseline_rule},{"ordinary_combiner_prediction_candidate_index",control.meta_rule},{"ordinary_reference_candidate_index",control.reference}};
      saved["rule_search_views"].push_back(item);result.base_means.push_back({winner,selected,std::move(item)});
    }
    saved["selection"]=result.base_means[0].details;result.details["base_candidate_aggregation"]=std::move(saved);
  }
  if(blend_scores){
    auto& scores=*blend_scores;scores.finish();const auto values=scores.scores.get();J rules_json=J::array();for(const auto& rule:runner.native_rules)rules_json.push_back(blend_rule(rule));
    J saved={{"version",1},{"method","oof_brier_convex"},{"inference_mode","oof_brier_convex_blend"},{"arithmetic_policy",teacher_blend::arithmetic_policy},{"coefficient_authority","alpha32_bits"},{"inner_coefficient_fits",inner_blends},{"inner_blend_results",compact?scores.compact_json(axes):scores.json()},{"blend_prediction_candidates",rules_json},{"inner_result_layout","base-model-major, rule-minor; scalar strict-half is candidate0"},{"byte_tie_policy","summed corresponding native base model bytes across inner FITs, then canonical model/rule order; not total blend deployment bytes"},{"family_tie_policy","all ordinary candidates retained; blend wins only on strict inner Dice improvement"},{"rule_search_views",J::array()}};
    if(compact)saved["inner_metric_encoding"]=axes;J bytes=J::array();for(U b=0;b<runner.base.size();++b)bytes.push_back(bb[b*rules]);saved["inner_blend_model_bytes"]=bytes;
    std::vector<const Selection*> ordinary={&result};for(const auto& view:result.views)ordinary.push_back(&view);
    for(U v=0;v<ordinary.size();++v){const auto& control=*ordinary[v];std::vector<U> eligible;if(v)eligible=views[v-1].at("native_prediction_candidate_indices").get<std::vector<U>>();else{eligible.resize(rules);std::iota(eligible.begin(),eligible.end(),0);}const U winner=scores.best_view(bb,rules,eligible);
      choose_blend_family<<<1,1>>>(bs.scores.p,control.baseline*rules+control.baseline_rule,ms.scores.p,control.meta*rules+control.meta_rule,reference_scores?reference_scores->scores.p:nullptr,best_reference,control.family,scores.scores.p,winner,family.p);sync();U selected=family.get()[0];
      J item={{"id",v?views[v-1].at("id"):J("full")},{"blend_baseline_index",winner/rules},{"blend_hyperparameters",runner.base[winner/rules]},{"blend_prediction_candidate_index",winner%rules},{"blend_prediction_rule",rules_json.at(winner%rules)},{"inner_blend_candidate_index",winner},{"inner_blend_macro_patient_Dice",values[winner]},{"ordinary_selected_family",family_name(control.family)},{"augmented_selected_family",family_name(selected)},{"ordinary_baseline_index",control.baseline},{"ordinary_combiner_index",control.meta},{"ordinary_baseline_prediction_candidate_index",control.baseline_rule},{"ordinary_combiner_prediction_candidate_index",control.meta_rule},{"ordinary_reference_candidate_index",control.reference}};
      saved["rule_search_views"].push_back(item);result.blended.push_back({winner/rules,winner%rules,selected,std::move(item)});
    }
    saved["selection"]=result.blended[0].details;result.details["teacher_probability_blend"]=std::move(saved);
  }
  if(aggregated_scores&&!runner.candidate_mean){
    const auto& mean_bytes=runner.bootstrap.is_null()?mb:ab;auto& am=*aggregated_scores;const U m=am.best(mean_bytes);choose_view_family<<<1,1>>>(bs.scores.p,best_base,am.scores.p,m,reference_scores?reference_scores->scores.p:nullptr,best_reference,family.p);sync();const U f=family.get()[0];
    J details=result.details;details.erase("rule_search_views");details["combiner_index"]=m/rules;details["combiner_hyperparameters"]=runner.meta[m/rules];details["selected_family"]=family_name(f);details["inner_combiner_results"]=compact?am.compact_json(axes):am.json();
    details["combiner_prediction_candidate_index"]=m%rules;details["combiner_prediction_rule"]=aggregation_rule(runner.native_rules.at(m%rules),!runner.bootstrap.is_null());details["combiner_prediction_candidates"]=J::array();for(const auto& rule:runner.native_rules)details["combiner_prediction_candidates"].push_back(aggregation_rule(rule,!runner.bootstrap.is_null()));details["inference_mode"]=runner.mean_mode();details["arithmetic_policy"]=oof_aggregation::arithmetic_policy;if(runner.aggregate_oof)details["inner_teacher_banks"]=inner_banks;else{details["bootstrap_config"]=runner.bootstrap;details["inner_bootstrap_fits"]=inner_bootstraps;details["ordinary_native_vs_probability_argmax_disagreements"]=argmax_differences.get();details["inner_combiner_model_bytes"]=J::array();for(U m=0;m<runner.meta.size();++m)details["inner_combiner_model_bytes"].push_back(ab[m*rules]);details["byte_tie_policy"]="sum all required serialized bootstrap member bytes across inner FITs; no deduplication";}details["control_baseline_and_reference_counts_shared"]=true;details["argmax_semantics"]="FP32 p1 > p0; exact tie class 0";details["native_inner_result_layout"]="model-major, prediction-candidate-minor; base candidate0 is native argmax; combiner candidate0 is strict rounded mean-probability argmax";details["native_prediction_candidates_scope"]="baseline only; combiner_prediction_candidates declares the mode-specific operations";
    result.aggregated.push_back({baseline,m/rules,f,best_reference,baseline_rule,m%rules,details});
    const auto base_scores=bs.scores.get(),meta_scores=am.scores.get();details["rule_search_views"]=J::array();
    for(const auto& view:views){const auto indices=view.at("native_prediction_candidate_indices").get<std::vector<U>>();U b=bs.best_view(bb,rules,indices),m=am.best_view(mean_bytes,rules,indices);
      choose_view_family<<<1,1>>>(bs.scores.p,b,am.scores.p,m,reference_scores?reference_scores->scores.p:nullptr,best_reference,family.p);sync();U selected=family.get()[0];
      J item={{"id",view.at("id")},{"baseline_index",b/rules},{"combiner_index",m/rules},{"selected_family",family_name(selected)},{"baseline_hyperparameters",runner.base[b/rules]},{"combiner_hyperparameters",runner.meta[m/rules]},
        {"baseline_prediction_candidate_index",b%rules},{"combiner_prediction_candidate_index",m%rules},{"baseline_prediction_rule",runner.native_rules.at(b%rules)},{"combiner_prediction_rule",aggregation_rule(runner.native_rules.at(m%rules),!runner.bootstrap.is_null())},
        {"inner_baseline_candidate_index",b},{"inner_combiner_candidate_index",m},{"inner_baseline_macro_patient_Dice",base_scores[b]},{"inner_combiner_macro_patient_Dice",meta_scores[m]},
        {"inner_metric_source","probability-mean selection shared arrays; canonical model-major, rule-minor indices"},{"inference_mode",runner.mean_mode()}};
      if(reference_scores){item["reference_candidate_index"]=best_reference;item["reference_candidate"]=runner.reference.at("candidates").at(best_reference);}
      details["rule_search_views"].push_back(item);result.aggregated.push_back({b/rules,m/rules,selected,best_reference,b%rules,m%rules,std::move(item)});
    }
    result.aggregated[0].details=details;result.details[runner.mean_key()]=std::move(details);
  }
  if(runner.candidate_mean){
    L all_bytes=0;std::vector<L> ordinary_bytes;std::vector<U> indices;J hp=J::array();for(U m=0;m<runner.meta.size();++m){indices.push_back(m);hp.push_back(runner.meta[m]);ordinary_bytes.push_back(mb[m*rules]);all_bytes=add(all_bytes,mb[m*rules]);}
    result.details["inner_combiner_model_bytes"]=ordinary_bytes;if(!result.details.contains("inner_baseline_model_bytes")){result.details["inner_baseline_model_bytes"]=J::array();for(U b=0;b<runner.base.size();++b)result.details["inner_baseline_model_bytes"].push_back(bb[b*rules]);}
    auto& am=*aggregated_scores;std::vector<L> costs(rules,all_bytes);const U best=am.best(costs);choose_view_family<<<1,1>>>(bs.scores.p,best_base,am.scores.p,best,reference_scores?reference_scores->scores.p:nullptr,best_reference,family.p);sync();const U f=family.get()[0];
    J details=result.details;details.erase("combiner_index");details.erase("combiner_hyperparameters");details.erase("rule_search_views");
    details["combiner_recipe"]="uniform_all_declared_meta_candidates_v1";details["combiner_indices"]=indices;details["combiner_hyperparameter_bank"]=hp;details["selected_family"]=family_name(f);details["inner_combiner_results"]=compact?am.compact_json(axes):am.json();details["inner_combiner_model_bytes"]=J::array({all_bytes});details["inner_meta_fits"]=inner_meta_fits;
    details["combiner_prediction_candidate_index"]=best;details["combiner_prediction_rule"]=aggregation_rule(runner.native_rules.at(best),false,true);details["combiner_prediction_candidates"]=J::array();for(const auto& rule:runner.native_rules)details["combiner_prediction_candidates"].push_back(aggregation_rule(rule,false,true));details["inference_mode"]=runner.mean_mode();details["arithmetic_policy"]=oof_aggregation::arithmetic_policy;details["argmax_semantics"]="FP32 p1 > p0; exact tie class 0";details["inner_combiner_result_layout"]="one fixed ensemble; prediction-candidate-minor; no meta HP selection";details["native_inner_result_layout"]="baseline model-major rule-minor; combiner one fixed ensemble with rule-minor scores";details["native_prediction_candidates_scope"]="baseline only; combiner_prediction_candidates declares the mode-specific operations";details["control_baseline_and_reference_counts_shared"]=true;details["ordinary_native_vs_probability_argmax_disagreements"]=argmax_differences.get();details["byte_tie_policy"]="sum every ordinary meta member across inner FITs; no deduplication";
    result.aggregated.push_back({baseline,UINT32_MAX,f,best_reference,baseline_rule,best,details});details["rule_search_views"]=J::array();const auto bvalues=bs.scores.get(),mvalues=am.scores.get();
    for(const auto& view:views){const auto eligible=view.at("native_prediction_candidate_indices").get<std::vector<U>>();const U b=bs.best_view(bb,rules,eligible),m=am.best_view(costs,rules,eligible);choose_view_family<<<1,1>>>(bs.scores.p,b,am.scores.p,m,reference_scores?reference_scores->scores.p:nullptr,best_reference,family.p);sync();U chosen=family.get()[0];
      J item={{"id",view.at("id")},{"baseline_index",b/rules},{"baseline_hyperparameters",runner.base[b/rules]},{"combiner_recipe","uniform_all_declared_meta_candidates_v1"},{"combiner_indices",indices},{"combiner_hyperparameter_bank",hp},{"selected_family",family_name(chosen)},
        {"baseline_prediction_candidate_index",b%rules},{"combiner_prediction_candidate_index",m},{"baseline_prediction_rule",runner.native_rules.at(b%rules)},{"combiner_prediction_rule",aggregation_rule(runner.native_rules.at(m),false,true)},
        {"inner_baseline_candidate_index",b},{"inner_combiner_candidate_index",m},{"inner_baseline_macro_patient_Dice",bvalues[b]},{"inner_combiner_macro_patient_Dice",mvalues[m]},{"inner_metric_source","fixed-ensemble shared inner scores; rule index only"},{"inference_mode",runner.mean_mode()}};
      if(reference_scores){item["reference_candidate_index"]=best_reference;item["reference_candidate"]=runner.reference.at("candidates").at(best_reference);}details["rule_search_views"].push_back(item);result.aggregated.push_back({b/rules,UINT32_MAX,chosen,best_reference,b%rules,m,std::move(item)});
    }
    result.aggregated[0].details=details;result.details["meta_candidate_aggregation"]=std::move(details);
  }
  return result;
}
std::vector<std::vector<U>> outer_groups(const J& plan,const std::vector<U>& patients){
  if(plan.contains("outer_folds"))return folds(patients,U(integer(plan,"outer_folds",UINT32_MAX)));
  auto selected=plan.contains("outer_patient_ids")?plan.at("outer_patient_ids").get<std::vector<U>>():patients;
  std::vector<std::vector<U>> result;for(U patient:selected)result.push_back({patient});return result;
}
void publish(const std::filesystem::path& output,const J& report){std::filesystem::create_directories(output);auto tmp=output/"result.json.tmp";{std::ofstream f(tmp);f.exceptions(std::ios::badbit|std::ios::failbit);f<<report.dump(2)<<'\n';}std::filesystem::rename(tmp,output/"result.json");}
J conversion_limits(J g,bool composed=false){
  need(g.is_object(),"conversion limits must be an object");if(g.contains("convert"))need(g.at("convert").is_boolean(),"convert must be boolean");
  const J defaults={{"qualification_mode","none"},{"max_rounds",1},{"max_depth",composed?3:2},{"max_trees",composed?4:2},{"max_source_nodes",composed?44:14},{"max_predicates",composed?20:6},{"max_partition_cells",composed?4096:729},{"max_states",32768},{"max_nodes",32768},{"max_expansions",65536},{"gpu_byte_budget",256ULL<<20},{"maximum_seconds",0.0},{"batch_size",64},{"qualification_batch_rows",1024},{"response_gpu_byte_budget",16ULL<<20}};
  for(const auto& [k,v]:defaults.items())if(!g.contains(k))g[k]=v;
  need(g.at("qualification_mode").is_string()&&(g.at("qualification_mode")=="exhaustive"||g.at("qualification_mode")=="none"),"conversion qualification_mode must be exhaustive or none");
  for(const auto* k:{"max_rounds","max_depth","max_trees","max_source_nodes","max_predicates","max_partition_cells","max_states","max_nodes","gpu_byte_budget","qualification_batch_rows"})need(integer(g,k)>0,"conversion gate limits must be positive integers");
  integer(g,"max_expansions");
  need(integer(g,"max_states")<=0x3fffffffu&&integer(g,"max_nodes")<=0x3fffffffu,"conversion gate graph index limit");
  need(integer(g,"batch_size")<=65536&&integer(g,"qualification_batch_rows")<=INT32_MAX,"conversion gate batch index limit");
  need(g.at("maximum_seconds").is_number(),"conversion gate time limit must be numeric");double seconds=g.at("maximum_seconds").get<double>();need(std::isfinite(seconds)&&seconds>=0,"conversion gate time limit must be nonnegative and finite; zero disables the deadline");
  if(!g.contains("response_row_capacity"))g["response_row_capacity"]=std::max(integer(g,"batch_size"),integer(g,"qualification_batch_rows"));need(integer(g,"response_row_capacity",INT32_MAX)>=std::max(integer(g,"batch_size"),integer(g,"qualification_batch_rows")),"native response capacity must cover conversion and qualification batches");need(integer(g,"response_gpu_byte_budget")>0,"native response budget must be positive");return g;
}
void gate_patients(const J& g,const std::set<U>& patients){
  need(g.at("withheld_patient_ids").is_array()&&!g.at("withheld_patient_ids").empty(),"conversion gate needs withheld patient IDs");std::set<U> withheld;
  for(const auto& id:g.at("withheld_patient_ids")){J item={{"id",id}};U p=U(integer(item,"id",UINT32_MAX));need(patients.count(p)&&withheld.insert(p).second,"conversion gate withheld patient unknown or duplicated");}need(withheld.size()<patients.size(),"conversion gate needs training patients");
}
void check_hyperparameter_limit(const J& hp,const J& g){need(integer(hp,"rounds")<=integer(g,"max_rounds")&&integer(hp,"max_depth")>0&&integer(hp,"max_depth")<=integer(g,"max_depth"),"conversion gate hyperparameters exceed round/depth bounds");}
J gate_config(const J& plan,const std::set<U>& patients){
  J g=conversion_limits(plan.at("conversion_gate"));gate_patients(g,patients);
  need(g.at("hyperparameters").is_array()&&!g.at("hyperparameters").empty(),"conversion gate needs explicit hyperparameters");
  for(auto& hp:g["hyperparameters"]){hp=ResidentTrainer::validate_hyperparameters(hp,2);check_hyperparameter_limit(hp,g);}
  return g;
}
J stack_config(const J& plan,const J& raw,const std::set<U>& patients,bool standalone){J g=conversion_limits(raw,true);if(standalone){gate_patients(g,patients);need(patients.size()-g.at("withheld_patient_ids").size()>=2,"composed gate needs at least two FIT patient groups for OOF");if(!g.contains("meta_indices")){g["meta_indices"]=J::array();for(U i=0;i<plan.at("meta_hyperparameters").size();++i)g["meta_indices"].push_back(i);}need(g.at("meta_indices").is_array()&&!g.at("meta_indices").empty(),"composed gate needs fixed meta indices");std::set<U> selected;for(const auto& i:g.at("meta_indices")){J index={{"i",i}};U k=U(integer(index,"i",UINT32_MAX));need(k<plan.at("meta_hyperparameters").size()&&selected.insert(k).second,"composed gate meta index unknown or duplicated");}}
  for(const auto* key:{"base_hyperparameters","meta_hyperparameters"})for(const auto& hp:plan.at(key))check_hyperparameter_limit(ResidentTrainer::validate_hyperparameters(hp,2),g);return g;
}
J final_config(const J& plan,const std::set<U>& patients){
  const auto& config=plan.at("final_deployment");need(config.is_object(),"final_deployment must be an object");if(config.contains("convert"))need(config.at("convert").is_boolean(),"convert must be boolean");
  if(!config.value("convert",true))return config;
  return stack_config(plan,config,patients,false);
}
struct MemoryBound {L bytes=0,cub_live_bytes=0,fit_rows=0,evaluation_rows=0,evaluation_batch_rows=0;bool detailed=false;std::string stage="whole-dataset conservative fallback";};
MemoryBound grouped_memory_bound(const J& plan,const J& descriptor,const std::vector<U>& patients,const std::vector<std::vector<U>>& outer,L rows,L features,L bank,L meta,L reference,L fit_bound,L prediction_rules){
  MemoryBound result;result.fit_rows=fit_bound;const auto sampling=validated_sampling_counts(plan,descriptor);std::map<U,L> sampled_fit;const L sampling_audit_bytes=sampling.is_null()?0:multiply(descriptor.at("scans").size(),sizeof(fit_sampling::Counts));if(!sampling.is_null())for(const auto& scan:sampling.at("scans"))sampled_fit.emplace(U(integer(scan,"scan_id")),integer(scan,"retained_rows"));
  const L raw_columns=meta_input_config(plan,U(features)).size(),response_bytes=multiply(add(multiply(bank,2),raw_columns),4);
  const auto views=rule_search_views_config(plan,U(prediction_rules));L view_scratch=0;for(const auto& view:views)view_scratch=std::max<L>(view_scratch,add(multiply(view.at("native_prediction_candidate_indices").size(),4),8));
  const bool base_mean=base_aggregation_config(plan);const L base_mean_metrics=base_mean?multiply(add(prediction_rules,multiply(4,add(views.size(),1))),56*descriptor.at("scans").size()+60*patients.size()+16):0;
  const bool blending=!blend_config(plan,U(features)).is_null();const L blend_metrics=blending?multiply(add(multiply(bank,prediction_rules),multiply(4,add(views.size(),1))),56*descriptor.at("scans").size()+60*patients.size()+16):0;
  const bool fixed_mean=candidate_aggregation_config(plan);const bool aggregation=oof_aggregation_config(plan)||!bootstrap_config(plan).is_null()||fixed_mean;const bool bagging=!bootstrap_config(plan).is_null();
  const L aggregation_row_bytes=base_mean?8:fixed_mean?24:aggregation?add(multiply(meta,16),8):0;
  const L aggregation_inner_metrics=aggregation?multiply(multiply(fixed_mean?1:meta,prediction_rules),56*descriptor.at("scans").size()+60*patients.size()+16):0;
  const L view_metrics=multiply(multiply(add(multiply(2,views.size()),aggregation?multiply(2,add(views.size(),1)):0),reference?5:3),56*descriptor.at("scans").size()+60*patients.size()+16);
  const auto loss=training_loss_config(plan);
  auto loss_workspace=[&](L count,L groups){
    if(loss.is_null())return L(0);
    const auto shape=ResidentTrainer::validate_loss(plan.at("base_hyperparameters")[0],
      training_loss_spec(loss,count),count,U(std::min<L>(groups,count)),2);
    return L(add(integer(shape,"owned_objective_CUDA_bytes"),integer(shape,"known_native_total_additional_bytes")));
  };
  auto loss_group_workspace=[&](L count,L groups){return loss.is_null()?L(0):L(add(multiply(count,4),multiply(groups,sizeof(FitGroupSpec))));};
  const bool group_weighted=plan.value("fit_group_weights",std::string("none"))=="equal_patient";
  const bool weighted=group_weighted||plan.value("fit_class_balanced",false)||plan.value("fit_sampling_weights",std::string("none"))!="none";
  // FIT group IDs, two retained/full counts, two coefficients/sums, and status.
  const L group_workspace=group_weighted?add(multiply(patients.size(),sampling.is_null()?60:88),4):0;
  const L bootstrap_workspace=bagging?add(multiply(patients.size(),24),add(multiply(meta,8),8)):fixed_mean?add(multiply(meta,8),4):0;
  result.bytes=add(add(multiply(rows,4*features+16),multiply(rows,4*features+48+response_bytes)),multiply(fit_bound,24*features+96+2*response_bytes+(weighted?12:0)));
  if(prediction_rules>1)result.bytes=add(result.bytes,add(multiply(rows,4),multiply(multiply(add(bank,meta),prediction_rules-1),56*descriptor.at("scans").size()+60*patients.size()+16)));
  result.bytes=add(result.bytes,add(add(group_workspace,sampling_audit_bytes),threshold_histogram_bytes(prediction_rules,descriptor.at("scans").size())));
  result.bytes=add(result.bytes,add(loss_workspace(fit_bound,patients.size()),multiply(3,loss_group_workspace(fit_bound,patients.size()))));
  result.bytes=add(result.bytes,add(add(view_metrics,view_scratch),multiply(raw_columns,4)));
  result.bytes=add(result.bytes,add(aggregation_inner_metrics,multiply(rows,aggregation_row_bytes)));if(bagging||fixed_mean)result.bytes=add(result.bytes,add(bagging?multiply(fit_bound,4):0,bootstrap_workspace));
  if(base_mean)result.bytes=add(result.bytes,base_mean_metrics);
  if(blending)result.bytes=add(result.bytes,add(blend_metrics,add(multiply((fit_bound+255)/256,32),28)));
  result.cub_live_bytes=result.bytes;
  for(const auto& scan:descriptor.at("scans"))if(!scan.contains("rows")||!scan.contains("training_rows"))return result;
  std::map<U,L> full,fit;L total_rows=0,total_eligible=0;
  for(const auto& scan:descriptor.at("scans")){
    U patient=U(integer(scan,"patient_id",UINT32_MAX));L n=integer(scan,"rows",rows),eligible=integer(scan,"training_rows",n);need(n>0,"scan memory accounting requires positive rows");L kept=eligible;
    if(scan.contains("truth_positive_rows"))integer(scan,"truth_positive_rows",n);
    if(scan.contains("training_positive_rows")){L positive=integer(scan,"training_positive_rows",eligible);need(scan.contains("truth_positive_rows")&&positive<=integer(scan,"truth_positive_rows",n),"training foreground accounting exceeds scan foreground");}
    if(!sampling.is_null())kept=sampled_fit.at(U(integer(scan,"scan_id")));else if(plan.value("include_all_fit_positives",false)){L positive=integer(scan,"truth_positive_rows",n),intersection=scan.contains("training_positive_rows")?integer(scan,"training_positive_rows",eligible):0;kept=std::min<L>(n,add(eligible,positive)-intersection);}
    full[patient]=add(full[patient],n);fit[patient]=add(fit[patient],kept);total_rows=add(total_rows,n);total_eligible=add(total_eligible,eligible);
  }
  need(total_rows==rows&&total_eligible==integer(descriptor,"train_eligible_rows",rows),"per-scan memory accounting differs from dataset totals");
  auto sum=[&](const std::vector<U>& group,const std::map<U,L>& counts){L total=0;for(U patient:group)total=add(total,counts.at(patient));return total;};
  L scans=descriptor.at("scans").size(),groups=patients.size(),outer_candidates=reference?5:3;
  // Exact owned metric arrays plus choose scratch. Native XGBoost allocations
  // are separately reported and remain outside this explicit-allocation bound.
  L candidate_arrays=outer_candidates+std::max<L>(outer_candidates,multiply(add(bank,meta),prediction_rules)+reference);
  L metrics=add(add(multiply(candidate_arrays,56*scans+60*groups+16),32),add(add(view_metrics,view_scratch),aggregation_inner_metrics));
  L metadata=add(add(add(add(add(multiply(scans,20),add(multiply(features,8),64)),group_workspace),threshold_histogram_bytes(prediction_rules,scans)),multiply(raw_columns,4)),sampling_audit_bytes);
  L resident=add(multiply(rows,4*features+16),add(add(metadata,metrics),bootstrap_workspace));if(blending)resident=add(resident,add(blend_metrics,8));if(base_mean)resident=add(resident,base_mean_metrics);
  result.bytes=add(multiply(rows,4*features+16),add(add(multiply(scans,52),32),sampling_audit_bytes));result.detailed=true;result.fit_rows=0;result.evaluation_rows=0;result.stage="initial dataset audit";
  result.cub_live_bytes=result.bytes;auto peak=[&](L bytes,const char* stage,bool cub_live=true){bytes=add(resident,bytes);if(cub_live)result.cub_live_bytes=std::max(result.cub_live_bytes,bytes);if(bytes>result.bytes){result.bytes=bytes;result.stage=stage;}};
  auto prepare=[&](const std::vector<U>& fitting,const std::vector<U>& evaluation){
    L training=sum(fitting,fit),evaluating=sum(evaluation,full);result.fit_rows=std::max(result.fit_rows,training);result.evaluation_rows=std::max(result.evaluation_rows,evaluating);
    L persistent=add(multiply(training,4*features+24+response_bytes),add(multiply(rows,4),multiply(training,4*bank+(weighted?8:0))));
    // Raw and response FIT owners can both retain dense IDs and group counts.
    persistent=add(persistent,multiply(2,loss_group_workspace(training,fitting.size())));
    L selection=add(multiply(rows,8),4*(patients.size()+1));
    // A selected OOF FIT and validation matrix coexist; together their row
    // bound is the enclosing FIT group. CUB workspace is queried and checked
    // against the remaining budget at runtime before its allocation.
    peak(add(persistent,add(multiply(training,4*features+12),selection)),"OOF row selection and gathered matrices");
    L maximum_oof_fit=0;for(const auto& held:folds(fitting,plan.value("oof_folds",3u)))maximum_oof_fit=std::max(maximum_oof_fit,sum(without(fitting,held),fit));
    peak(add(persistent,add(multiply(training,4*features+12),multiply(maximum_oof_fit,4*features+4+(weighted?8:0)))),"OOF native FIT staging");
    peak(add(persistent,multiply(training,4*features+4+(weighted?8:0))),"full native base FIT staging");
    peak(add(persistent,multiply(training,response_bytes+4+(weighted?8:0)+(bagging?4:0))),"native meta FIT staging");
    if(blending)peak(add(persistent,add(multiply((training+255)/256,32),20)),"FIT-only blend reduction after native staging released",false);
    if(!loss.is_null()){
      const auto objective=loss_workspace(training,fitting.size());
      const auto oof_objective=add(loss_workspace(maximum_oof_fit,fitting.size()),loss_group_workspace(maximum_oof_fit,fitting.size()));
      peak(add(add(persistent,add(multiply(training,4*features+12),multiply(maximum_oof_fit,4*features+4+(weighted?8:0)))),oof_objective),"OOF group-loss update and known native buffers",false);
      peak(add(add(persistent,multiply(training,4*features+4+(weighted?8:0))),objective),"base group-loss update and known native buffers",false);
      peak(add(add(persistent,multiply(training,response_bytes+4+(weighted?8:0))),objective),"meta group-loss update and known native buffers",false);
    }
    if(evaluating){
      peak(add(add(multiply(rows,8),multiply(evaluating,4)),4*(patients.size()+1)),"evaluation row-ID selection");
      L probe=add(add(multiply(features,4),response_bytes),16);peak(add(multiply(evaluating,4),probe),"evaluation native one-row registration",false);
      L live=add(resident,add(multiply(evaluating,4),4)),per_row=add(4*features+response_bytes+48,aggregation_row_bytes);
      U capacity=evaluation_row_capacity(plan.value("gpu_byte_budget",12ULL<<30),live,UINT64_MAX,0,per_row,evaluating,plan.value("evaluation_batch_rows",0ULL));
      result.evaluation_batch_rows=std::max<L>(result.evaluation_batch_rows,capacity);peak(add(add(multiply(evaluating,4),4),multiply(capacity,per_row)),"evaluation row batches including native output reserve",false);
      if(capacity<evaluating&&!plan.contains("evaluation_batch_rows")){result.bytes=std::max<L>(result.bytes,plan.value("gpu_byte_budget",12ULL<<30));result.stage="budget-limited evaluation row batches including native output reserve";}
    }
  };
  auto selected_work=[&](const std::vector<U>& fitting,const std::vector<U>& evaluation){for(const auto& held:folds(fitting,plan.value("inner_folds",3u)))prepare(without(fitting,held),held);prepare(fitting,evaluation);};
  for(const auto& held:outer)selected_work(without(patients,held),held);
  if(plan.contains("composed_conversion_gate")){auto held=plan.at("composed_conversion_gate").at("withheld_patient_ids").get<std::vector<U>>();prepare(without(patients,held),held);}
  if(plan.contains("final_deployment"))selected_work(patients,{});
  return result;
}
struct Partition {
  std::vector<U> offsets,widths;std::vector<L> strides;std::vector<float> representatives;
  std::set<std::pair<U,U>> thresholds;std::set<std::tuple<U,U,bool>> branch_predicates;L cells=1,source_trees=0,source_nodes=0;U features=0;
  Partition(const class_conversion_native::SourceData& source,const J& limits):Partition(std::span(&source,1),limits){}
  Partition(std::span<const class_conversion_native::SourceData> sources,const J& limits,std::span<const class_conversion_native::MappedRawPredicate> raw={}){
    need(!sources.empty(),"conversion partition needs source models");features=U(sources[0].features);std::vector<std::vector<float>> cuts(features);std::vector<bool> tested(features,false);
    for(const auto& source:sources){source_trees+=source.roots.size();source_nodes+=source.feature.size();need(U(source.features)==features&&source.outputs==2,"conversion base shape differs");need(source_trees<=integer(limits,"max_trees")&&source_nodes<=integer(limits,"max_source_nodes")&&U(source.depth)<=integer(limits,"max_depth"),"trained conversion source exceeds tree/node/depth bounds");
      for(std::size_t i=0;i<source.feature.size();++i)if(source.feature[i]>=0){need(!std::isnan(source.cut[i]),"conversion gate refuses NaN source thresholds");U f=U(source.feature[i]),cut=std::bit_cast<U>(source.cut[i]==0.0f?0.0f:source.cut[i]);thresholds.insert({f,cut});branch_predicates.insert({f,cut,bool(source.missing_left[i])});tested[f]=true;if(std::isfinite(source.cut[i]))cuts[f].push_back(source.cut[i]);}}
    for(const auto& predicate:raw){need(predicate.feature<features&&!std::isnan(predicate.cut),"mapped meta raw predicate differs from input domain");U bits=std::bit_cast<U>(predicate.cut==0.f?0.f:predicate.cut);thresholds.insert({predicate.feature,bits});branch_predicates.insert({predicate.feature,bits,predicate.missing_left});tested[predicate.feature]=true;if(std::isfinite(predicate.cut))cuts[predicate.feature].push_back(predicate.cut);}
    need(branch_predicates.size()<=integer(limits,"max_predicates"),"trained conversion source exceeds branch-predicate bound including missing direction");
    // Infinite thresholds do not divide finite FP32 inputs, but their native
    // missing direction may differ: retain a NaN atom on every tested axis.
    for(auto& axis:cuts){offsets.push_back(U(representatives.size()));strides.push_back(cells);if(axis.empty()){representatives.push_back(0);if(tested[widths.size()])representatives.push_back(std::numeric_limits<float>::quiet_NaN());}else{
        std::sort(axis.begin(),axis.end());axis.erase(std::unique(axis.begin(),axis.end()),axis.end());representatives.push_back(-std::numeric_limits<float>::max());
        for(float cut:axis)if(cut>-std::numeric_limits<float>::max())representatives.push_back(cut);
        representatives.push_back(std::numeric_limits<float>::quiet_NaN());
      }U width=U(representatives.size()-offsets.back());widths.push_back(width);need(cells<=integer(limits,"max_partition_cells")/width,"trained conversion source exceeds partition-cell bound");cells*=width;
    }
  }
  void check_program(const ConvertedModel& model) const{
    // Runtime already checks topology and canonical/compact equality on CUDA.
    // This independent metadata check proves its predicates cannot subdivide
    // an atom that this verifier did not enumerate.
    const auto& bytes=model.canonical_bytes;const auto& m=model.runtime->metadata();need(bytes.size()==64+16ULL*m.nodes,"conversion gate canonical extent differs");
    for(U i=0;i<m.nodes;++i){std::int32_t feature;U cut;std::memcpy(&feature,bytes.data()+64+16ULL*i,4);std::memcpy(&cut,bytes.data()+68+16ULL*i,4);if(feature==-1)continue;if((cut&0x7fffffffu)==0)cut=0;std::int64_t f=feature>=0?feature:-std::int64_t(feature)-2;need(f>=0&&f<INT32_MAX&&thresholds.count({U(f),cut}),"emitted threshold absent from exhaustively verified source partition");}
  }
};
struct NativeDeployment {
  std::vector<std::unique_ptr<ResidentPredictor>> bases;std::unique_ptr<ResidentPredictor> meta;U width=0,meta_width=0,raw_features=0,capacity=0;L scratch_bytes=0;
  Device<float> margins,probabilities,calibrated;Device<U> bad,raw_indices;J prediction_rule=nullptr;
  template<class Verified> NativeDeployment(const J& plan,const std::vector<TrainingResult>& models,const TrainingResult* combined,U features,const Verified& verified,const J& config,const J& rule,const std::vector<U>& raw={}){
    prediction_rule=rule;need(!combined||rule.is_null(),"calibrated native prediction composition is unsupported");
    need(!models.empty()&&models.size()<=UINT32_MAX/2,"deployment base extent");width=U(models.size()*2);need(add(width,raw.size())<=INT32_MAX,"deployment mixed input extent");meta_width=U(width+raw.size());raw_features=features;capacity=U(integer(config,"response_row_capacity",INT32_MAX));
    if(combined){scratch_bytes=add(add(multiply(capacity,4ULL*(width+meta_width)),4),multiply(raw.size(),4));need(scratch_bytes<=integer(config,"response_gpu_byte_budget")&&scratch_bytes<integer(config,"gpu_byte_budget"),"composed native response scratch exceeds budget");margins=Device<float>(std::size_t(capacity)*width);probabilities=Device<float>(std::size_t(capacity)*meta_width);bad=Device<U>(1);bad.zero();if(!raw.empty()){raw_indices=Device<U>(raw.size());raw_indices.set(raw);}}
    else {need(models.size()==1,"baseline deployment must have one source");if(!rule.is_null()&&rule.at("kind")=="positive_probability_threshold"){scratch_bytes=multiply(capacity,4);need(scratch_bytes<=integer(config,"response_gpu_byte_budget")&&scratch_bytes<integer(config,"gpu_byte_budget"),"calibrated native response scratch exceeds budget");calibrated=Device<float>(capacity);}}
    for(const auto& model:models){auto owner=std::make_unique<ResidentPredictor>(plan,features,2,verified);owner->restore_model(model.model_json,model.model_sha256);bases.push_back(std::move(owner));}
    if(combined){meta=std::make_unique<ResidentPredictor>(plan,meta_width,2,verified);meta->restore_model(combined->model_json,combined->model_sha256);}
  }
  const float* response(const float* x,std::uint64_t rows,bool probability){need(rows>0&&rows<=capacity,"native composition callback exceeds response row capacity");float* out=probability?probabilities.p:margins.p;U stride=probability?meta_width:width;for(U b=0;b<bases.size();++b){const float* p=probability?bases[b]->predict_probabilities(x,rows):bases[b]->predict(x,rows,true);stack_columns<<<blocks(rows*2),256>>>(p,U(rows),stride,2*b,probability,out,bad.p);sync();}if(probability&&raw_indices.p){append_raw_columns<<<blocks(rows*raw_indices.n),256>>>(x,U(rows),raw_features,raw_indices.p,U(raw_indices.n),width,meta_width,out);sync();}need(!bad.get()[0],"native composition returned invalid response word");return out;}
  const float* classes(const float* x,std::uint64_t rows){if(calibrated.p){need(rows>0&&rows<=capacity,"calibrated native callback exceeds response row capacity");threshold_prediction(prediction_rule,bases[0]->predict_probabilities(x,rows),U(rows),2,1,calibrated);return calibrated.p;}return meta?meta->predict(response(x,rows,true),rows,false):bases[0]->predict(x,rows,false);}
};
J qualify_partition(ConvertedModel& converted,const Partition& partition,const J& config,const std::function<bool()>& stopped,const std::function<const float*(const float*,std::uint64_t)>& predict,L response_scratch){
  need(converted.metrics.at("complete")==true,"incomplete conversion cannot pass gate");need(converted.runtime->metadata().features==partition.features&&converted.runtime->metadata().classes==2,"compiled deployment header shape differs");partition.check_program(converted);
  U batch=U(std::min<L>(partition.cells,integer(config,"qualification_batch_rows")));auto bytes=multiply(batch,4ULL*partition.features+8);bytes=add(bytes,multiply(partition.features,16));bytes=add(bytes,multiply(partition.representatives.size(),4));bytes=add(bytes,8);bytes=add(bytes,converted.runtime->metadata().device_bytes);bytes=add(bytes,response_scratch);need(bytes<=integer(config,"gpu_byte_budget"),"conversion qualification and native response scratch exceed GPU budget");
  Device<U> offsets(partition.offsets.size()),widths(partition.widths.size()),expected(batch),bad(1);Device<L> strides(partition.strides.size());Device<float> reps(partition.representatives.size()),x(std::size_t(batch)*partition.features);offsets.set(partition.offsets);widths.set(partition.widths);strides.set(partition.strides);reps.set(partition.representatives);bad.zero();
  for(L first=0;first<partition.cells;first+=batch){need(!stopped(),"conversion timed out or interrupted before exhaustive qualification finished");U n=U(std::min<L>(batch,partition.cells-first));partition_rows<<<blocks(L(n)*partition.features),256>>>(x.p,n,partition.features,offsets.p,widths.p,strides.p,reps.p,first);sync();auto native=predict(x.p,n);native_classes<<<blocks(n),256>>>(native,n,expected.p,bad.p);sync();need(!bad.get()[0],"native deployment output is not a class ID");converted.runtime->verify({x.p,L(n)*partition.features,0,n,partition.features},expected.p,n);}
  need(!stopped(),"conversion timed out or interrupted at final qualification boundary");return {{"source_trees",partition.source_trees},{"source_nodes",partition.source_nodes},{"source_unique_thresholds",partition.thresholds.size()},{"source_branch_predicates",partition.branch_predicates.size()},{"branch_predicate_identity","feature, threshold bits, missing direction"},{"partition_cells",partition.cells},{"output_thresholds_are_source_thresholds",true},{"all_finite_FP32_and_NaN_cells_verified",true},{"native_class_mismatches",0},{"qualification_owned_peak_bytes",bytes},{"native_response_scratch_bytes",response_scratch},{"qualification_memory_scope","compiled graph, qualification arrays and native response scratch; excludes dataset/FIT and opaque native XGBoost allocations"}};
}
template<class Verified> J accept_deployment(const J& plan,const std::vector<TrainingResult>& bases,const TrainingResult* meta,const Verified& verified,const J& config,const std::filesystem::path& output,const std::string& prefix,volatile std::sig_atomic_t& stopping,const J& rule=nullptr){
  const bool exhaustive=config.at("qualification_mode")=="exhaustive";
  std::vector<class_conversion_native::SourceData> parsed;J identities=J::array();for(const auto& base:bases){parsed.push_back(class_conversion_native::read_source_bytes(base.model_json));identities.push_back(base.model_sha256);}
  need(!parsed.empty(),"conversion deployment needs source models");U raw_features=U(parsed[0].features);L source_trees=0,source_nodes=0;
  std::set<std::pair<U,U>> thresholds;std::set<std::tuple<U,U,bool>> predicates;
  for(const auto& source:parsed){source_trees+=source.roots.size();source_nodes+=source.feature.size();need(U(source.features)==raw_features&&source.outputs==2,"conversion base shape differs");need(source_trees<=integer(config,"max_trees")&&source_nodes<=integer(config,"max_source_nodes")&&U(source.depth)<=integer(config,"max_depth"),"trained conversion source exceeds tree/node/depth bounds");
    for(std::size_t i=0;i<source.feature.size();++i)if(source.feature[i]>=0){need(!std::isnan(source.cut[i]),"conversion gate refuses NaN source thresholds");U f=U(source.feature[i]),cut=std::bit_cast<U>(source.cut[i]==0.f?0.f:source.cut[i]);thresholds.insert({f,cut});predicates.insert({f,cut,bool(source.missing_left[i])});}}
  need(predicates.size()<=integer(config,"max_predicates"),"trained conversion source exceeds branch-predicate bound including missing direction");
  std::unique_ptr<Partition> partition;std::vector<class_conversion_native::MappedRawPredicate> meta_predicates;
  const auto meta_raw=meta?meta_input_config(plan,raw_features):std::vector<U>{};
  if(meta){auto m=class_conversion_native::read_source_bytes(meta->model_json);need(m.outputs==2&&U(m.features)==bases.size()*2+meta_raw.size(),"composed meta response shape differs");need(m.roots.size()<=integer(config,"max_trees")&&m.feature.size()<=integer(config,"max_source_nodes")&&U(m.depth)<=integer(config,"max_depth"),"composed meta source exceeds resource bounds");if(!meta_raw.empty())meta_predicates=class_conversion_native::mapped_meta_raw_predicates(m,U(bases.size()*2),meta_raw,raw_features);}
  for(const auto& predicate:meta_predicates){U cut=std::bit_cast<U>(predicate.cut==0.f?0.f:predicate.cut);thresholds.insert({predicate.feature,cut});predicates.insert({predicate.feature,cut,predicate.missing_left});}
  need(predicates.size()<=integer(config,"max_predicates"),"composed raw predicate count exceeds branch-predicate bound");if(exhaustive)partition=std::make_unique<Partition>(std::span<const class_conversion_native::SourceData>(parsed),config,meta_predicates);
  NativeDeployment native(plan,bases,meta,raw_features,verified,config,rule,meta_raw);ConversionOptions options;options.domain.allow_nan=true;options.gpu_byte_budget=integer(config,"gpu_byte_budget")-native.scratch_bytes;options.max_states=integer(config,"max_states");options.max_nodes=integer(config,"max_nodes");options.initial_states=std::min<L>(1024,options.max_states);options.initial_nodes=std::min<L>(1024,options.max_nodes);options.max_expansions=integer(config,"max_expansions");options.batch_size=U(integer(config,"batch_size"));if(meta||native.calibrated.p)options.max_batch_size=std::min<U>(65536,native.capacity);options.completion_estimate_enabled=false;options.checkpoint_on_completion=false;options.residency=class_runtime::Residency::dual;
  auto started=std::chrono::steady_clock::now();double maximum_seconds=config.at("maximum_seconds").get<double>();auto stopped=[&]{return stopping||(maximum_seconds>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>=maximum_seconds);};options.stop_requested=stopped;options.progress=[&](const J& state){std::cout<<J{{"event","grouped_segmentation_deployment_conversion_progress"},{"deployment",prefix},{"statistics",state}}.dump()<<'\n'<<std::flush;};
  ConvertedModel converted;
  if(meta){ComposedConversionSource source;for(const auto& base:bases)source.bases.push_back({base.model_json,raw_features,2,base.model_sha256});source.meta={meta->model_json,native.meta_width,2,meta->model_sha256};source.meta_input_feature_indices=meta_raw;ComposedNativeOracle oracle;oracle.library_sha256=verified.metadata().at("native_library_sha256");oracle.base_margins=[&](const float* x,std::uint64_t n){return native.response(x,n,false);};oracle.predict_classes=[&](const float* x,std::uint64_t n){return native.classes(x,n);};converted=convert_composed_model(source,oracle,options);}
  else{NativeOracle oracle;oracle.features=raw_features;oracle.classes=2;oracle.library_sha256=verified.metadata().at("native_library_sha256");oracle.source_sha256=bases[0].model_sha256;if(native.calibrated.p)oracle.positive_probability_threshold=rule.at("threshold").get<float>();oracle.predict=[&](const float* x,std::uint64_t n,bool margin){return margin?native.bases[0]->predict(x,n,true):native.classes(x,n);};converted=convert_model({bases[0].model_json,raw_features,2,bases[0].model_sha256},oracle,options);}
  need(converted.metrics.at("complete")==true,"incomplete conversion cannot finish deployment");
  double construction_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();auto qualification_started=std::chrono::steady_clock::now();
  J receipt={{"source_trees",source_trees},{"source_nodes",source_nodes},{"source_unique_thresholds",thresholds.size()},{"source_branch_predicates",predicates.size()},{"branch_predicate_identity","feature, threshold bits, missing direction"},{"partition_cells",nullptr},{"output_thresholds_are_source_thresholds",nullptr},{"all_finite_FP32_and_NaN_cells_verified",false},{"native_class_mismatches",nullptr},{"qualification_owned_peak_bytes",0},{"native_response_scratch_bytes",native.scratch_bytes}};
  if(exhaustive)receipt=qualify_partition(converted,*partition,config,stopped,[&](const float* x,std::uint64_t n){return native.classes(x,n);},native.scratch_bytes);const auto& m=converted.runtime->metadata();
  receipt.update({{"complete",true},{"construction_complete",true},{"qualification_mode",config.at("qualification_mode")},{"qualification_performed",exhaustive},{"qualification_complete",exhaustive},{"partition_preflight_performed",exhaustive},{"builtin_converter_checks_performed",true},{"artifact_qualification",exhaustive?"exhaustive_native_class_qualified":"unqualified_conversion_artifact"},{"accepted_native_class_root",exhaustive},{"kind",meta?"native_probability_composition":"native_baseline"},{"raw_features",raw_features},{"classes",2},{"score_channels",meta?native.width:2u},{"base_source_sha256",identities},{"meta_source_sha256",meta?J(meta->model_sha256):J(nullptr)},{"source_sha256",m.source_sha256},{"canonical_sha256",m.canonical_sha256},{"compact_sha256",m.compact_sha256},{"nodes",m.nodes},{"canonical_bytes",converted.canonical_bytes.size()},{"compact_bytes",converted.compact_bytes.size()},{"conversion",converted.metrics},{"construction_seconds",construction_seconds},{"qualification_seconds",exhaustive?std::chrono::duration<double>(std::chrono::steady_clock::now()-qualification_started).count():0.},{"conversion_and_qualification_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()},{"configuration",config},{"artifact_prefix",prefix},{"scope",exhaustive?"classes over all finite FP32 raw feature vectors and NaNs; excludes infinities; no probability-preservation or accuracy-improvement claim":"Complete maintained converter construction with builtin checks; independent exhaustive native-class qualification not requested"}});
  receipt["native_prediction_rule"]=rule.is_null()?native_prediction_config(J::object()).at(0):rule;receipt["calibrated_probability_threshold"]=bool(native.calibrated.p);receipt["input_scope"]="declared input feature vectors; upstream feature generation is outside this conversion";if(!meta_raw.empty()){receipt["meta_input_mapping"]=meta_input_mapping(meta_raw);receipt["meta_input_columns"]=native.meta_width;}if(native.calibrated.p){receipt["positive_probability_threshold_bits"]=std::bit_cast<U>(rule.at("threshold").get<float>());receipt["prediction_semantics"]="native full-round FP32 probability of class 1 >= FP32 threshold; equality selects class 1";}
  std::filesystem::create_directories(output);for(std::size_t i=0;i<bases.size();++i)class_conversion_native::atomic_text(output/(prefix+"-base-"+std::to_string(i+1)+".json"),bases[i].model_json);if(meta)class_conversion_native::atomic_text(output/(prefix+"-meta.json"),meta->model_json);class_conversion_native::atomic_text(output/(prefix+".canonical.bin"),converted.canonical_bytes);class_conversion_native::atomic_text(output/(prefix+".compact.bin"),converted.compact_bytes);class_conversion_native::atomic_json(output/(prefix+".receipt.json"),receipt);return receipt;
}
template<class Verified> J attempt_deployment(const J& plan,const std::vector<TrainingResult>& bases,const TrainingResult* meta,const Verified& verified,const J& config,const std::filesystem::path& output,const std::string& prefix,volatile std::sig_atomic_t& stopping,const J& rule=nullptr){try{return accept_deployment(plan,bases,meta,verified,config,output,prefix,stopping,rule);}catch(const std::exception& e){J refusal={{"complete",false},{"accepted_native_class_root",false},{"kind",meta?"native_probability_composition":"native_baseline"},{"artifact_prefix",prefix},{"native_prediction_rule",rule},{"error",e.what()},{"configuration",config},{"interrupted",bool(stopping)}};if(auto* failure=dynamic_cast<const ConversionFailure*>(&e))refusal["partial_statistics"]=failure->partial_statistics;return refusal;}}
int composed_gate(const J& plan,Data& data,Runner& runner,const std::filesystem::path& output,volatile std::sig_atomic_t& stopping){
  auto config=stack_config(plan,plan.at("composed_conversion_gate"),std::set<U>(data.patients.begin(),data.patients.end()),true);auto withheld=config.at("withheld_patient_ids").get<std::vector<U>>();auto fitting=without(data.patients,withheld);J report={{"format","grouped-segmentation-composed-gate-1"},{"complete",false},{"accepted_native_class_root",false},{"configuration",config},{"FIT_patient_ids",fitting},{"withheld_patient_ids",withheld},{"OOF_patient_groups",folds(fitting,runner.oof)},{"withheld_labels_used_for_FIT_or_selection",false},{"hyperparameter_selection_performed",false},{"shared_converter","class_study::convert_composed_model"},{"conversions",J::array()}};
  const bool exhaustive=config.at("qualification_mode")=="exhaustive";report["qualification_mode"]=config.at("qualification_mode");report["qualification_performed"]=false;report["all_finite_FP32_and_NaN_cells_verified"]=false;
  try{auto stack=runner.prepare(fitting);auto trainer=runner.trainer(stack.meta_fit,fitting,"FIXED_COMPOSED_GATE_META_FIT");for(const auto& index:config.at("meta_indices")){U m=index.get<U>();auto meta=runner.train(trainer,runner.meta[m]);auto receipt=accept_deployment(plan,stack.models,&meta,*runner.session,config,output,"composed-"+std::to_string(m+1),stopping);receipt["meta_index"]=m;receipt["base_hyperparameters"]=runner.base;receipt["meta_hyperparameters"]=runner.meta[m];report["conversions"].push_back(receipt);report["completed_native_fits"]=runner.fits;if(!runner.training_loss.is_null())report["completed_group_loss_fits"]=runner.custom_fits;publish(output,report);std::cout<<J{{"event",exhaustive?"grouped_segmentation_composed_gate_accepted":"grouped_segmentation_composed_gate_constructed_unqualified"},{"result",receipt}}.dump()<<'\n'<<std::flush;}report["complete"]=true;report["accepted_native_class_root"]=exhaustive;report["qualification_performed"]=exhaustive;report["all_finite_FP32_and_NaN_cells_verified"]=exhaustive;publish(output,report);return 0;}catch(const std::exception& e){report["error"]=e.what();report["completed_native_fits"]=runner.fits;if(!runner.training_loss.is_null())report["completed_group_loss_fits"]=runner.custom_fits;report["interrupted"]=bool(stopping);if(auto* failure=dynamic_cast<const ConversionFailure*>(&e))report["partial_statistics"]=failure->partial_statistics;publish(output,report);if(stopping)return 2;throw;}
}
int conversion_gate(const J& plan,Data& data,const std::filesystem::path& output,volatile std::sig_atomic_t& stopping){
  const auto config=gate_config(plan,std::set<U>(data.patients.begin(),data.patients.end()));const auto withheld=config.at("withheld_patient_ids").get<std::vector<U>>();const auto fitting=without(data.patients,withheld);
  const bool exhaustive=config.at("qualification_mode")=="exhaustive";
  auto fit=data.subset(fitting,true);J p=plan;p["TEST_read"]=false;p["VALID_read"]=false;J binding={{"format","grouped-segmentation-conversion-fit-1"},{"patient_ids",fitting},{"rows",fit->rows},{"features",fit->features},{"training_eligibility_used",true}};ResidentTrainer trainer(p,fit->view(fit,binding));
  J report={{"format","grouped-segmentation-conversion-gate-1"},{"complete",false},{"accepted_native_class_root",false},{"configuration",config},{"FIT_patient_ids",fitting},{"withheld_patient_ids",withheld},{"FIT_rows",fit->rows},{"withheld_labels_used_for_FIT_or_selection",false},{"hyperparameter_selection_performed",false},{"shared_converter","class_study::convert_model"},{"conversions",J::array()},{"per_trial_intermediate_file_writes",0},{"time_limit_scope","per conversion and exhaustive qualification, checked at batch boundaries; native FIT excluded"}};
  report["qualification_mode"]=config.at("qualification_mode");report["qualification_performed"]=false;
  report["all_finite_FP32_and_NaN_cells_verified"]=false;
  std::string stage;auto stage_started=std::chrono::steady_clock::now();
  try{for(const auto& hp:config.at("hyperparameters")){
      report["pending_trial"]={{"trial",report["conversions"].size()+1},{"hyperparameters",hp},{"complete",false},{"accepted_native_class_root",false},{"construction_complete",false},{"qualification_complete",false},{"qualified_cells",0}};
      auto& pending=report["pending_trial"];stage="training";stage_started=std::chrono::steady_clock::now();pending["stage"]=stage;
      need(!stopping,"conversion gate interrupted before FIT");auto source=trainer.train(hp);
      pending["training_seconds"]=source.metrics.at("seconds_including_buffer_export");pending["training_metrics"]=source.metrics;
      auto parsed=class_conversion_native::read_source_bytes(source.model_json);
      pending.update({{"source_trees",parsed.roots.size()},{"source_nodes",parsed.feature.size()},{"source_depth",parsed.depth},{"source_sha256",source.model_sha256},{"source_bytes",source.model_json.size()}});
      stage="source_preflight";stage_started=std::chrono::steady_clock::now();pending["stage"]=stage;
      std::unique_ptr<Partition> partition_owner;
      std::set<std::pair<U,U>> source_thresholds;std::set<std::tuple<U,U,bool>> source_predicates;
      need(parsed.roots.size()<=integer(config,"max_trees")&&parsed.feature.size()<=integer(config,"max_source_nodes")&&U(parsed.depth)<=integer(config,"max_depth"),"trained conversion source exceeds tree/node/depth bounds");
      for(std::size_t i=0;i<parsed.feature.size();++i)if(parsed.feature[i]>=0){need(!std::isnan(parsed.cut[i]),"conversion gate refuses NaN source thresholds");U f=U(parsed.feature[i]),cut=std::bit_cast<U>(parsed.cut[i]==0.f?0.f:parsed.cut[i]);source_thresholds.insert({f,cut});source_predicates.insert({f,cut,bool(parsed.missing_left[i])});}
      need(source_predicates.size()<=integer(config,"max_predicates"),"trained conversion source exceeds branch-predicate bound including missing direction");
      if(exhaustive){stage="partition_preflight";pending["stage"]=stage;partition_owner=std::make_unique<Partition>(parsed,config);}
      pending.update({{"source_unique_thresholds",source_thresholds.size()},{"source_branch_predicates",source_predicates.size()},{"partition_cells",exhaustive?J(partition_owner->cells):J(nullptr)},{"partition_preflight_performed",exhaustive},{"partition_preflight_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-stage_started).count()},{"qualification_mode",config.at("qualification_mode")},{"qualification_performed",false}});
      ConversionOptions options;options.domain.allow_nan=true;options.gpu_byte_budget=integer(config,"gpu_byte_budget");options.max_states=integer(config,"max_states");options.max_nodes=integer(config,"max_nodes");options.initial_states=std::min<L>(1024,options.max_states);options.initial_nodes=std::min<L>(1024,options.max_nodes);options.max_expansions=integer(config,"max_expansions");options.batch_size=U(integer(config,"batch_size"));options.completion_estimate_enabled=false;options.checkpoint_on_completion=false;options.residency=class_runtime::Residency::dual;
      auto started=std::chrono::steady_clock::now();double maximum_seconds=config.at("maximum_seconds").get<double>();auto stopped=[&]{return stopping||(maximum_seconds>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>=maximum_seconds);};options.stop_requested=stopped;
      stage="construction";stage_started=started;pending["stage"]=stage;
      options.progress=[&](const J& state){std::cout<<J{{"event","grouped_segmentation_conversion_progress"},{"trial",report["conversions"].size()+1},{"statistics",state}}.dump()<<'\n'<<std::flush;};
      NativeOracle oracle;oracle.features=trainer.features();oracle.classes=trainer.classes();oracle.library_sha256=trainer.metadata().at("native_library_sha256");oracle.source_sha256=source.model_sha256;oracle.predict=[&](const float* x,std::uint64_t n,bool margin){return trainer.predict(x,n,margin);};
      auto converted=convert_model({source.model_json,data.F,2,source.model_sha256},oracle,options);need(converted.metrics.at("complete")==true,"incomplete conversion cannot pass gate");
      const auto& built=converted.runtime->metadata();
      pending.update({{"construction_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-stage_started).count()},{"construction_complete",true},{"nodes",built.nodes},{"canonical_bytes",converted.canonical_bytes.size()},{"compact_bytes",converted.compact_bytes.size()},{"canonical_sha256",built.canonical_sha256},{"compact_sha256",built.compact_sha256},{"conversion",converted.metrics}});
      L bytes=0;pending["qualification_seconds"]=0.;
      if(exhaustive){const auto& partition=*partition_owner;
      stage="qualification";stage_started=std::chrono::steady_clock::now();pending["stage"]=stage;pending["qualification_performed"]=true;partition.check_program(converted);
      // Qualification retains the compiled graph and owns representative
      // inputs, expected classes, Runtime::verify output, and both bad flags.
      // Dataset/FIT and opaque native prediction allocations remain separate.
      U batch=U(std::min<L>(partition.cells,integer(config,"qualification_batch_rows")));bytes=multiply(batch,4ULL*data.F+8);bytes=add(bytes,multiply(data.F,16));bytes=add(bytes,multiply(partition.representatives.size(),4));bytes=add(bytes,8);bytes=add(bytes,converted.runtime->metadata().device_bytes);need(bytes<=integer(config,"gpu_byte_budget"),"conversion gate qualification exceeds GPU budget");
      Device<U> offsets(partition.offsets.size()),widths(partition.widths.size()),expected(batch),bad(1);Device<L> strides(partition.strides.size());Device<float> reps(partition.representatives.size()),x(std::size_t(batch)*data.F);offsets.set(partition.offsets);widths.set(partition.widths);strides.set(partition.strides);reps.set(partition.representatives);bad.zero();
      auto last_qualification_progress=stage_started;
      for(L first=0;first<partition.cells;first+=batch){need(!stopped(),"conversion gate timed out or interrupted before exhaustive qualification finished");U n=U(std::min<L>(batch,partition.cells-first));partition_rows<<<blocks(L(n)*data.F),256>>>(x.p,n,data.F,offsets.p,widths.p,strides.p,reps.p,first);sync();auto native=trainer.predict(x.p,n,false);native_classes<<<blocks(n),256>>>(native,n,expected.p,bad.p);sync();need(!bad.get()[0],"conversion gate native output is not a class ID");converted.runtime->verify({x.p,L(n)*data.F,0,n,data.F},expected.p,n);
        pending["qualified_cells"]=first+n;
        auto now=std::chrono::steady_clock::now();double elapsed=std::chrono::duration<double>(now-stage_started).count();
        if(std::chrono::duration<double>(now-last_qualification_progress).count()>=1.){
          std::cout<<J{{"event","grouped_segmentation_qualification_progress"},{"trial",pending.at("trial")},{"qualified_cells",first+n},{"partition_cells",partition.cells},{"qualification_seconds",elapsed},{"qualified_cells_per_second",elapsed>0?(first+n)/elapsed:0.}}.dump()<<'\n'<<std::flush;last_qualification_progress=now;
        }
      }
      pending["qualification_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-stage_started).count();
      need(!stopped(),"conversion gate timed out or interrupted at final qualification boundary");
      pending["qualification_complete"]=true;
      }
      stage="publication";stage_started=std::chrono::steady_clock::now();pending["stage"]=stage;
      const auto& m=converted.runtime->metadata();J item={{"hyperparameters",hp},{"source_trees",parsed.roots.size()},{"source_nodes",parsed.feature.size()},{"source_unique_thresholds",source_thresholds.size()},{"source_branch_predicates",source_predicates.size()},{"branch_predicate_identity","feature, threshold bits, missing direction"},{"partition_cells",pending.at("partition_cells")},{"output_thresholds_are_source_thresholds",exhaustive?J(true):J(nullptr)},{"all_finite_FP32_and_NaN_cells_verified",exhaustive},{"qualification_owned_peak_bytes",bytes},{"qualification_memory_scope","includes resident compiled graph, inputs, expected classes, runtime verification output, axis metadata and two flags; excludes dataset/FIT/native XGBoost allocations"},{"native_class_mismatches",exhaustive?J(0):J(nullptr)},{"complete",true},{"accepted_native_class_root",exhaustive},{"source_sha256",source.model_sha256},{"canonical_sha256",m.canonical_sha256},{"compact_sha256",m.compact_sha256},{"nodes",m.nodes},{"canonical_bytes",converted.canonical_bytes.size()},{"compact_bytes",converted.compact_bytes.size()},{"conversion",converted.metrics},{"conversion_and_qualification_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()},{"scope",exhaustive?"classes for all finite FP32 feature vectors and NaNs; excludes infinities; no probability preservation or accuracy-improvement claim":"Completed converter construction with builtin checks only; no independent exhaustive native-class qualification"}};
      item.update({{"source_depth",parsed.depth},{"source_bytes",source.model_json.size()},{"training_seconds",pending.at("training_seconds")},{"training_metrics",pending.at("training_metrics")},{"partition_preflight_seconds",pending.at("partition_preflight_seconds")},{"partition_preflight_performed",exhaustive},{"construction_seconds",pending.at("construction_seconds")},{"qualification_seconds",pending.at("qualification_seconds")},{"qualified_cells",pending.at("qualified_cells")},{"construction_complete",true},{"qualification_complete",exhaustive},{"qualification_performed",exhaustive},{"qualification_mode",config.at("qualification_mode")},{"builtin_converter_checks_performed",true},{"artifact_qualification",exhaustive?"exhaustive_native_class_qualified":"unqualified_capacity_artifact"}});
      // Default artifacts retain builtin checks without exhaustive acceptance.
      // Explicit exhaustive mode publishes only independently qualified output.
      std::filesystem::create_directories(output);std::string name="gate-"+std::to_string(report["conversions"].size()+1);class_conversion_native::atomic_text(output/(name+"-source.json"),source.model_json);class_conversion_native::atomic_text(output/(name+".canonical.bin"),converted.canonical_bytes);class_conversion_native::atomic_text(output/(name+".compact.bin"),converted.compact_bytes);item["artifact_prefix"]=name;report["conversions"].push_back(item);report.erase("pending_trial");publish(output,report);std::cout<<J{{"event",exhaustive?"grouped_segmentation_conversion_accepted":"grouped_segmentation_conversion_constructed_unqualified"},{"result",item}}.dump()<<'\n'<<std::flush;
    }report["complete"]=true;report["accepted_native_class_root"]=exhaustive;report["qualification_performed"]=exhaustive;report["all_finite_FP32_and_NaN_cells_verified"]=exhaustive;publish(output,report);return 0;
  }catch(const std::exception& e){report["error"]=e.what();report["interrupted"]=bool(stopping);if(report.contains("pending_trial")){
      auto& pending=report["pending_trial"];pending["stage"]=stage;pending["error"]=e.what();double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-stage_started).count();
      pending[stage+"_seconds"]=elapsed;pending["interrupted"]=bool(stopping);
      if(stage=="construction")pending["construction_seconds"]=elapsed;
      if(stage=="qualification")pending["qualification_seconds"]=elapsed;
      if(auto* failure=dynamic_cast<const ConversionFailure*>(&e))pending["partial_statistics"]=failure->partial_statistics;
      report["failed_trial"]=pending;
    }if(auto* failure=dynamic_cast<const ConversionFailure*>(&e))report["partial_statistics"]=failure->partial_statistics;publish(output,report);if(stopping)return 2;throw;}
}
void final_deployment(const J& plan,Data& data,Runner& runner,const std::filesystem::path& output,J& report,volatile std::sig_atomic_t& stopping){
  auto limits=final_config(plan,std::set<U>(data.patients.begin(),data.patients.end()));
  const bool requested_conversion=limits.value("convert",true);auto started=std::chrono::steady_clock::now();L fits_before=runner.fits;
  J final={{"complete",false},{"attempt_complete",false},{"native_fit_complete",false},{"conversion_attempted",false},{"conversion_requested_by_plan",requested_conversion},{"FIT_patient_ids",data.patients},{"heldout_accuracy_claim",false},{"outer_results_used_for_selection",false},{"selection_method","fresh patient-grouped inner cross-validation over all patients"},{"configuration",limits}};
  report["final_deployment"]=final;publish(output,report);
  try{
    auto selection_started=std::chrono::steady_clock::now();auto selected=select_models(plan,data,runner,data.patients,"FINAL_INNER_META_FIT");
    selected.details["FIT_patient_ids"]=data.patients;selected.details["all_patient_refit_required"]=selected.family!=2;selected.details["OOF_patient_groups_for_all_patient_refit"]=selected.family==2?J(nullptr):J(folds(data.patients,runner.oof));selected.details["outer_results_used_for_selection"]=false;
    final["final_selection"]=selected.details;final["selection_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-selection_started).count();report["final_deployment"]=final;publish(output,report);
    std::vector<TrainingResult> fitted,sources;std::vector<U> source_indices;TrainingResult meta,original_meta;J pruning=nullptr;auto refit_started=std::chrono::steady_clock::now();
    if(selected.family!=2){
      {auto stack=runner.prepare(data.patients);fitted=std::move(stack.models);if(selected.family==1){auto trainer=runner.trainer(stack.meta_fit,data.patients,"FINAL_ALL_PATIENT_META_FIT");meta=runner.train(trainer,runner.meta[selected.meta]);}}
      original_meta=meta;
      if(selected.family==1){auto reduced=composition_prune::unused_bases(fitted,meta,runner.meta_raw);sources=std::move(reduced.bases);meta=std::move(reduced.meta);source_indices=std::move(reduced.retained_indices);pruning=std::move(reduced.provenance);}
      else{sources.push_back(fitted[selected.baseline]);source_indices.push_back(selected.baseline);}
    }
    final["refit_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-refit_started).count();final["fit_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();final["completed_native_fits"]=runner.fits-fits_before;final["all_patient_native_FIT_performed"]=selected.family!=2;
    J pipeline={{"format","grouped-segmentation-final-native-pipeline-1"},{"complete",true},{"classes",2},{"raw_features",data.F},{"FIT_patient_ids",data.patients},{"training_eligibility_used",true},{"include_all_fit_positives",data.include_positive},{"fit_class_balanced",plan.value("fit_class_balanced",false)},{"fit_sampling_weights",plan.value("fit_sampling_weights",std::string("none"))},{"fit_group_weights",plan.value("fit_group_weights",std::string("none"))},{"family",family_name(selected.family)},{"selection",selected.details},{"heldout_accuracy_claim",false},{"outer_results_used_for_selection",false},{"native_library_path",plan.at("native_library_path")},{"native_library_sha256",plan.at("native_library_sha256")},{"base_models",J::array()},{"meta_model",nullptr},{"source_pruning",pruning},{"all_patient_native_FIT_performed",selected.family!=2}};
    if(runner.native_rules.size()>1){pipeline["native_prediction_candidates"]=runner.native_rules;pipeline["native_prediction_rule"]=selected.family==2?J(nullptr):runner.native_rules.at(selected.family==1?selected.meta_rule:selected.baseline_rule);pipeline["native_prediction_calibration"]={{"method","fresh inner patient-grouped selection on all patients"},{"evaluation_domain","complete declared ROI; outside-ROI positives counted as false negatives"},{"outer_results_used_for_selection",false},{"conversion_supported",false},{"source_identity","base_models/meta_model SHA256 and native_library_sha256 recorded in this pipeline"}};}
    if(!runner.reference.is_null())pipeline["reference_predictor"]=runner.reference;
    for(const char* key:{"feature_schema","input_provenance","preparation","external_predictor","missing_feature_indices"})if(data.descriptor.contains(key))pipeline[key]=data.descriptor.at(key);
    if(selected.family==2){pipeline["external_reference_candidate"]=runner.reference.at("candidates").at(selected.reference);pipeline["upstream_predictor_required"]=true;pipeline["scope"]="Frozen external predictor output followed by the inner-selected default-mask or probability-threshold rule; external predictor weights are not converted";}
    for(U i=0;i<sources.size();++i){U bank=source_indices[i];auto path=output/("final-native-base-"+std::to_string(bank+1)+".json");class_conversion_native::atomic_text(path,sources[i].model_json);pipeline["base_models"].push_back({{"bank_index",bank},{"path",path.string()},{"sha256",sources[i].model_sha256},{"hyperparameters",runner.base[bank]},{"training_metrics",sources[i].metrics}});}
    if(selected.family==1){auto original_path=output/"final-native-meta.json";class_conversion_native::atomic_text(original_path,original_meta.model_json);auto path=original_path;if(pruning.at("applied").get<bool>()){path=output/"final-native-meta-pruned.json";class_conversion_native::atomic_text(path,meta.model_json);pipeline["original_meta_model"]={{"path",original_path.string()},{"sha256",original_meta.model_sha256}};}
      pipeline["meta_model"]={{"path",path.string()},{"sha256",meta.model_sha256},{"hyperparameters",runner.meta[selected.meta]},{"features",sources.size()*2+runner.meta_raw.size()},{"base_response","ordered native full-round probability channels"},{"training_metrics",meta.metrics}};if(!runner.meta_raw.empty())pipeline["meta_input_mapping"]=meta_input_mapping(runner.meta_raw);final["source_pruning"]=pruning;}
    auto pipeline_path=output/"final-native-pipeline.json";class_conversion_native::atomic_json(pipeline_path,pipeline);final["native_fit_complete"]=selected.family!=2;final["pipeline_complete"]=true;final["native_pipeline_path"]=pipeline_path.string();final["native_pipeline_sha256"]=class_conversion_native::sha256(pipeline.dump(2)+'\n');final["native_pipeline"]=pipeline;
    const bool convert=requested_conversion&&selected.family!=2;final["conversion_requested"]=convert;final["conversion_deferred"]=!convert;final["native_only"]=!convert&&selected.family!=2;final["deployment_status"]=selected.family==2?"external_reference_recipe":convert?"conversion_requested":"native_only";
    report["final_deployment"]=final;report["completed_native_fits"]=runner.fits;if(!runner.training_loss.is_null())report["completed_group_loss_fits"]=runner.custom_fits;publish(output,report);
    if(convert){auto conversion_started=std::chrono::steady_clock::now();final["conversion_attempted"]=true;report["conversion_performed"]=true;report["final_deployment"]=final;publish(output,report);final["conversion"]=attempt_deployment(runner.plan,sources,selected.family==1?&meta:nullptr,*runner.session,limits,output,"final-selected",stopping);final["conversion_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-conversion_started).count();final["conversion_complete"]=final.at("conversion").at("complete");final["qualified_native_class_root"]=final.at("conversion").at("accepted_native_class_root");final["complete"]=final.at("conversion_complete");}
    else{final["conversion"]=nullptr;final["conversion_seconds"]=0.;final["conversion_complete"]=false;final["qualified_native_class_root"]=false;final["conversion_deferred_reason"]=selected.family==2?"External predictor passthrough retained as a pipeline recipe; external predictor weights are not converted":runner.native_rules.size()>1?"convert=false; calibrated native recipe saved; exact calibrated conversion is not supported":"convert=false; completed native pipeline retained for later replay";final["complete"]=true;}
    final["attempt_complete"]=true;final["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();report["final_deployment"]=final;report["completed_native_fits"]=runner.fits;if(!runner.training_loss.is_null())report["completed_group_loss_fits"]=runner.custom_fits;publish(output,report);std::cout<<J{{"event","grouped_segmentation_final_deployment_complete"},{"result",final}}.dump()<<'\n'<<std::flush;
  }catch(const std::exception& e){final["error"]=e.what();final["interrupted"]=bool(stopping);final["completed_native_fits"]=runner.fits-fits_before;final["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();report["final_deployment"]=final;report["completed_native_fits"]=runner.fits;if(!runner.training_loss.is_null())report["completed_group_loss_fits"]=runner.custom_fits;publish(output,report);throw;}
}
struct SavedPipeline {
  std::vector<TrainingResult> bases;TrainingResult meta,original_meta;std::vector<U> retained_indices,meta_raw;J pruning=nullptr,limits,rule=nullptr;bool combined=false;U features=0;
};
SavedPipeline read_pipeline(const J& plan,const J& pipeline){
  need(!plan.contains("base_candidate_aggregation")&&!pipeline.contains("base_candidate_aggregation"),"fixed base mean has no supported native pipeline export/conversion contract");
  need(!plan.contains("teacher_probability_blend")&&!pipeline.contains("teacher_probability_blend"),"teacher blend has no supported native pipeline export/conversion contract");
  need(!plan.contains("meta_candidate_aggregation")&&!pipeline.contains("meta_candidate_aggregation"),"fixed meta mean has no supported native pipeline export/conversion contract");
  need(!plan.contains("meta_patient_bootstrap")&&!pipeline.contains("meta_patient_bootstrap"),"meta bootstrap has no supported native pipeline export/conversion contract");
  need(!plan.contains("oof_teacher_aggregation")&&!pipeline.contains("oof_teacher_aggregation"),"OOF teacher aggregation has no supported native pipeline export/conversion contract");
  need(pipeline.at("format")=="grouped-segmentation-final-native-pipeline-1"&&pipeline.at("complete")==true,"replay requires a completed final native pipeline");
  SavedPipeline saved;saved.rule=replay_prediction_rule(pipeline);need(integer(pipeline,"classes")==2,"replay requires binary native pipeline");saved.features=U(integer(pipeline,"raw_features",INT32_MAX));need(saved.features>0,"replay raw feature extent");
  need(plan.at("native_library_sha256")==pipeline.at("native_library_sha256"),"replay native library SHA256 differs from pipeline");saved.limits=conversion_limits(plan.value("conversion",J::object()),true);
  saved.combined=pipeline.at("family")=="nonlinear_combiner";need(saved.combined||pipeline.at("family")=="native_baseline","replay native pipeline family unsupported");saved.meta_raw=saved_meta_input_config(pipeline,saved.features);need(saved.combined||saved.meta_raw.empty(),"baseline replay cannot carry a meta input mapping");if(plan.contains("meta_input_feature_indices"))need(meta_input_config(plan,saved.features)==saved.meta_raw,"replay plan meta input mapping differs from saved pipeline");
  auto load=[&](const J& item){std::filesystem::path path=item.at("path").get<std::string>();need(path.is_absolute(),"replay model path must be absolute");TrainingResult model{class_conversion_native::read_text(path),item.at("sha256").get<std::string>(),item.value("training_metrics",J::object())};need(class_conversion_native::sha256(model.model_json)==model.model_sha256,"replay native model SHA256 differs");return model;};
  const auto& entries=pipeline.at("base_models");need(entries.is_array()&&!entries.empty()&&entries.size()<=UINT32_MAX/2,"replay base model extent");
  for(const auto& item:entries){auto model=load(item);auto parsed=class_conversion_native::read_source_bytes(model.model_json);need(parsed.features==std::int32_t(saved.features)&&parsed.outputs==2,"replay base model shape differs");saved.bases.push_back(std::move(model));}
  if(saved.combined){saved.original_meta=load(pipeline.at("meta_model"));auto reduced=composition_prune::unused_bases(saved.bases,saved.original_meta,saved.meta_raw);saved.bases=std::move(reduced.bases);saved.meta=std::move(reduced.meta);saved.retained_indices=std::move(reduced.retained_indices);saved.pruning=std::move(reduced.provenance);}
  else{need(saved.bases.size()==1&&pipeline.at("meta_model").is_null(),"replay baseline pipeline contains composition models");saved.retained_indices={0};}
  // Source shape is already validated from immutable model bytes. Replay has
  // no training search, so absent FIT-oriented gate bounds follow this source.
  L trees=0,nodes=0,depth=1;std::set<std::tuple<U,U,bool>> predicates;
  for(const auto& base:saved.bases){auto source=class_conversion_native::read_source_bytes(base.model_json);trees+=source.roots.size();nodes+=source.feature.size();depth=std::max<L>(depth,source.depth);for(std::size_t i=0;i<source.feature.size();++i)if(source.feature[i]>=0)predicates.insert({U(source.feature[i]),std::bit_cast<U>(source.cut[i]==0?0.f:source.cut[i]),bool(source.missing_left[i])});}
  if(saved.combined){auto source=class_conversion_native::read_source_bytes(saved.meta.model_json);trees=std::max<L>(trees,source.roots.size());nodes=std::max<L>(nodes,source.feature.size());depth=std::max<L>(depth,source.depth);if(!saved.meta_raw.empty())for(const auto& predicate:class_conversion_native::mapped_meta_raw_predicates(source,U(saved.bases.size()*2),saved.meta_raw,saved.features))predicates.insert({predicate.feature,std::bit_cast<U>(predicate.cut==0.f?0.f:predicate.cut),predicate.missing_left});}
  const auto requested=plan.value("conversion",J::object());const J source_limits={{"max_trees",trees},{"max_source_nodes",nodes},{"max_depth",depth},{"max_predicates",std::max<L>(1,predicates.size())},{"max_partition_cells",UINT64_MAX}};for(const auto& [key,value]:source_limits.items())if(!requested.contains(key))saved.limits[key]=value;
  return saved;
}

// Synthetic scorer checks use the prior per-cutoff kernels as a differential
// oracle only. No models, training results, or prediction files are generated.
__global__ void compare_response_words(const float* actual,U width,U column,const float* expected,U expected_width,U expected_column,const U* ids,const U* positions,U rows,U* bad){
  for(L i=L(blockIdx.x)*blockDim.x+threadIdx.x;i<L(rows)*2;i+=L(blockDim.x)*gridDim.x){U row=U(i/2),channel=U(i%2),position=positions[ids[row]];if(position==UINT32_MAX){atomicOr(bad,1u);continue;}if(__float_as_uint(actual[L(row)*width+column+channel])!=__float_as_uint(expected[L(position)*expected_width+expected_column+channel]))atomicOr(bad,2u);}
}
__global__ void fill_histogram_fixture(const U* ids,U rows,const float* cuts,U ncuts,U width,U column,float* probability,float* argmax,U* labels,U* sid,const U* scans,U ns){
  for(L row=L(blockIdx.x)*blockDim.x+threadIdx.x;row<rows;row+=L(blockDim.x)*gridDim.x){
    U id=ids?ids[row]:U(row);float p=cuts[id%ncuts];U variant=(id/ncuts)%3;
    if(variant==0&&p>0)p=nextafterf(p,0.f);if(variant==2&&p<1)p=nextafterf(p,1.f);
    for(U column_index=0;column_index<width;++column_index)probability[row*width+column_index]=column_index==column?p:1.f-p;
    if(argmax)argmax[row]=float(p>.5f);
    if(labels)labels[row]=U((id%101)==0);if(sid)sid[row]=scans[row*ns/rows];
  }
}
}

J threshold_histogram_checks(const J& plan,const J& descriptor,std::uint32_t benchmark_rows){
  J p=plan;p.erase("final_deployment");std::vector<float> cuts;std::set<U> bits;
  auto retain=[&](float value){if(bits.insert(std::bit_cast<U>(value)).second)cuts.push_back(value);};
  for(U i=1;i<20;++i)retain(float(double(i)/20));std::vector<float> extra={0,1};
  for(U i=1;i<40;++i)extra.push_back(float(double(i)/40));for(U k=1;k<=24;++k){float v=std::ldexp(1.f,-int(k));extra.push_back(v);extra.push_back(1.f-v);}std::sort(extra.begin(),extra.end());for(float v:extra)retain(v);
  need(cuts.size()==83,"fixed-grid fixture threshold family differs");p["native_prediction_thresholds"]=cuts;J rules=native_prediction_config(p);U nr=U(rules.size());
  auto metadata=validate(p,descriptor);Data data(descriptor,p,metadata);need(data.patients.size()>1&&data.scans.size()>data.patients.size(),"threshold fixture requires repeated scans and several patients");
  ProbabilityHistogram histogram(rules,U(data.scans.size()));Metrics old(data,nr*2),fast(data,nr*2);
  auto middle=data.patients.begin()+data.patients.size()/2;std::vector<std::vector<U>> partitions={{data.patients.begin(),middle},{middle,data.patients.end()}};
  for(const auto& patients:partitions){auto m=data.subset(patients,false);Device<float> probabilities(std::size_t(m->rows)*5),argmax(m->rows),prediction(m->rows);
    fill_histogram_fixture<<<blocks(m->rows),256>>>(m->ids.p,m->rows,histogram.thresholds.p,histogram.cuts,5,3,probabilities.p,argmax.p,nullptr,nullptr,nullptr,0);sync();
    for(U model=0;model<2;++model){U first=model*nr;old.add(first,argmax.p,*m);fast.add(first,argmax.p,*m);
      fast.add_thresholds(first,probabilities.p,5,3,*m,histogram);
      for(U c=1;c<nr;++c){threshold_prediction(rules.at(c),probabilities.p,m->rows,5,3,prediction);old.add(first+c,prediction.p,*m);}}
  }
  need(old.counts.get()==fast.counts.get(),"histogram and rescan integer confusion counts differ");auto expected=old.json(),actual=fast.json();need(expected==actual,"histogram patient/scan Dice or outside-ROI accounting differs");
  std::vector<L> cost(nr*2,1);U winner=old.best(cost);need(fast.best(cost)==winner&&winner<nr,"histogram changed original candidate or model tie priority");
  std::vector<U> codec_scan_patients;for(auto index:data.scan_patient_indices)codec_scan_patients.push_back(data.patients[index]);
  const auto codec_axes=inner_metrics::make_axes(data.patients,data.scans,codec_scan_patients,data.outside);const auto compact=fast.compact_json(codec_axes);
  need(compact.size()==actual.size(),"compact inner candidate extent differs");for(U c=0;c<compact.size();++c)need(inner_metrics::decode(compact.at(c),codec_axes)==actual.at(c),"compact inner metric roundtrip differs");
  need(fast.best(cost)==winner,"compact serialization changed frozen candidate choice");
  U half=0;for(U c=1;c<nr;++c)if(rules.at(c).at("threshold")==.5f)half=c;
  need(half&&expected.at(0)!=expected.at(half),"fixture conflated exact native argmax with >=.5");
  bool outside=false;for(const auto& scan:actual.at(half).at("scans"))outside|=scan.at("FN_outside_ROI").get<L>()>0;need(outside,"fixture omitted outside-ROI false negatives");
  Device<float> one_probability(1),one_prediction(1);Device<U> one_label(1),one_scan(1),bad(1);Device<L> counts(std::size_t(nr)*data.scans.size()*4);unsigned refused=0;
  auto reject=[&](float probability,U label,U scan){one_probability.set({probability});one_label.set({label});one_scan.set({scan});bad.zero();counts.zero();bool denied=false;
    try{histogram.accumulate(one_probability.p,1,1,0,one_label.p,one_scan.p,data.scan_ids.p,counts.p,bad.p);}catch(const std::invalid_argument&){denied=true;}need(denied,"histogram accepted invalid probability/label/scan");
    bad.zero();probability_classes<<<1,256>>>(one_probability.p,1,1,0,.5f,one_prediction.p);confusion<<<1,256>>>(one_prediction.p,one_label.p,one_scan.p,1,data.scan_ids.p,U(data.scans.size()),counts.p,bad.p);sync();need(bad.get()[0]!=0,"rescan oracle did not reject invalid fixture");++refused;};
  for(float v:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::denorm_min(),std::nextafter(1.f,2.f)})reject(v,0,data.scans[0]);reject(.5f,2,data.scans[0]);U missing=0;while(std::binary_search(data.scans.begin(),data.scans.end(),missing))++missing;reject(.5f,0,missing);
  J result={{"complete",true},{"CUDA_executed",true},{"thresholds",cuts.size()},{"native_candidates",nr},{"all_integer_counts_equal",true},{"patient_and_scan_metrics_equal",true},{"outside_ROI_false_negatives_preserved",true},{"FP32_equality_nextafter_endpoints_verified",true},{"strided_probability_columns_verified",true},{"original_order_and_model_offsets_preserved",true},{"candidate_tie_priority_preserved",true},{"compact_inner_roundtrip_equal",true},{"compact_serialization_preserves_choice",true},{"native_argmax_remains_distinct",true},{"invalid_cases_refused",refused},{"threshold_histogram_CUDA_bytes",threshold_histogram_bytes(nr,data.scans.size())},{"native_FITs",0},{"prediction_files_written",false}};
  if(benchmark_rows){need(benchmark_rows>=1000,"threshold benchmark row extent");Matrix m;m.rows=benchmark_rows;m.y=Device<U>(m.rows);m.scan=Device<U>(m.rows);Device<float> probabilities(m.rows),prediction(m.rows);Metrics scan_score(data,nr),histogram_score(data,nr);
    fill_histogram_fixture<<<blocks(m.rows),256>>>(nullptr,m.rows,histogram.thresholds.p,histogram.cuts,1,0,probabilities.p,nullptr,m.y.p,m.scan.p,data.scan_ids.p,U(data.scans.size()));sync();
    auto started=std::chrono::steady_clock::now();histogram_score.add_thresholds(0,probabilities.p,1,0,m,histogram);double histogram_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    started=std::chrono::steady_clock::now();for(U c=1;c<nr;++c){threshold_prediction(rules.at(c),probabilities.p,m.rows,1,0,prediction);scan_score.add(c,prediction.p,m);}double rescan_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    need(scan_score.counts.get()==histogram_score.counts.get(),"benchmark histogram/rescan counts differ");result["benchmark"]={{"rows",m.rows},{"thresholds",cuts.size()},{"scans",data.scans.size()},{"histogram_seconds",histogram_seconds},{"rescan_seconds",rescan_seconds},{"speedup",rescan_seconds/histogram_seconds},{"integer_counts_equal",true},{"repetitions",1},{"workload","synthetic contiguous scan blocks, 1/101 positives, repeated threshold equality and neighboring FP32 values; not real-study predictions"},{"scope","warm scorer calls including synchronization/status reads; excludes allocation, input generation, FIT and probability prediction"}};
  }return result;
}

J rule_search_view_checks(){
  const U models=3,rules=4;Device<double> scores(models*rules),reference(1);Device<L> bytes(models*rules);Device<U> indices(rules),result(1),bad(1);
  std::vector<double> values={.5,.5,.1,.2,.5,.4,.1,.5,.5,.3,.1,.7};std::vector<L> costs={100,100,100,100,50,50,50,50,50,50,50,50};scores.set(values);bytes.set(costs);unsigned checks=0;
  auto pick=[&](const std::vector<U>& eligible){auto padded=eligible;padded.resize(rules);indices.set(padded);bad.zero();choose_view<<<1,1>>>(scores.p,bytes.p,models,rules,indices.p,U(eligible.size()),result.p,bad.p);sync();need(!bad.get()[0],"valid chooser fixture refused");++checks;return result.get()[0];};
  need(pick({0,1})==4,"view byte/canonical model tie changed");need(pick({1,3})==11,"noncontiguous rule subset lost model-major index");
  auto whole=pick({0,1,2,3});choose<<<1,1>>>(scores.p,bytes.p,models*rules,result.p);sync();need(whole==result.get()[0],"full rule view differs from original chooser");
  values[1]=std::nextafter(.5,1.);scores.set(values);need(pick({0,1})==1,"rule chooser treated one ULP score difference as a tie");
  values[1]=.5;costs[0]=costs[1]=25;bytes.set(costs);scores.set(values);need(pick({0,1})==0,"same-model rule tie changed canonical priority");
  for(double invalid:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-.01,1.01}){values[0]=invalid;scores.set(values);indices.set({0,0,0,0});bad.zero();choose_view<<<1,1>>>(scores.p,bytes.p,models,rules,indices.p,1,result.p,bad.p);sync();need(bad.get()[0]!=0,"view accepted nonfinite/out-of-range inner score");++checks;}
  values[0]=.5;scores.set(values);reference.set({.5});
  choose_view_family<<<1,1>>>(scores.p,0,scores.p,4,nullptr,0,result.p);sync();need(result.get()[0]==0,"base/meta tie changed");++checks;
  choose_view_family<<<1,1>>>(scores.p,0,scores.p,4,reference.p,0,result.p);sync();need(result.get()[0]==2,"fixed reference tie changed");++checks;
  values[4]=std::nextafter(.5,1.);scores.set(values);choose_view_family<<<1,1>>>(scores.p,0,scores.p,4,reference.p,0,result.p);sync();need(result.get()[0]==1,"family one ULP difference changed");++checks;
  return {{"complete",true},{"CUDA_executed",true},{"checks",checks},{"native_FITs",0},{"exact_score_byte_canonical_order",true},{"full_view_matches_original_chooser",true}};
}
J evaluation_probability_checks(const J& plan,const J& descriptor){
  auto metadata=validate(plan,descriptor);Data data(descriptor,plan,metadata);volatile std::sig_atomic_t stop=0;Runner runner(plan,data,stop);std::vector<U> held={data.patients.front()};auto fitting=without(data.patients,held);auto prepared=runner.prepare(fitting);std::vector<TrainingResult> metas;
  {auto trainer=runner.trainer(prepared.meta_fit,fitting,"BATCH_PROBABILITY_FIXTURE_META_FIT");for(const auto& hp:runner.meta)metas.push_back(runner.train(trainer,hp));}
  if(!runner.meta_raw.empty()){Device<U> fit_bad(1);fit_bad.zero();check_meta_raw_columns<<<blocks(L(prepared.meta_fit->rows)*runner.meta_raw.size()),256>>>(data.x.p,prepared.meta_fit->ids.p,prepared.meta_fit->rows,data.F,runner.meta_raw_indices.p,U(runner.meta_raw.size()),prepared.meta_fit->x.p,U(runner.base.size()*2),runner.meta_features(),fit_bad.p);sync();need(!fit_bad.get()[0],"OOF FIT raw suffix differs from original row identity");}prepared.fit.reset();prepared.meta_fit.reset();
  auto full=data.subset(held,false);U base_width=runner.meta_features(),meta_width=U(metas.size()*2);Device<float> base_expected(multiply(full->rows,base_width)),meta_expected(multiply(full->rows,meta_width));runner.append_raw(*full,base_expected.p);Device<U> positions(data.rows),bad(1);ck(cudaMemset(positions.p,255,positions.n*sizeof(U)),"batch fixture absent rows");index_rows<<<blocks(full->rows),256>>>(full->ids.p,full->rows,positions.p);bad.zero();sync();
  {ResidentPredictor base(plan,data.F,2,*runner.session),meta(plan,base_width,2,*runner.session);
    for(U b=0;b<prepared.models.size();++b){base.restore_model(prepared.models[b].model_json,prepared.models[b].model_sha256);auto values=base.predict_probabilities(full->x.p,full->rows);response_column<<<blocks(full->rows),256>>>(values,nullptr,nullptr,full->rows,base_width,b*2,base_expected.p,nullptr,bad.p);sync();}
    for(U m=0;m<metas.size();++m){meta.restore_model(metas[m].model_json,metas[m].model_sha256);auto values=meta.predict_probabilities(base_expected.p,full->rows);response_column<<<blocks(full->rows),256>>>(values,nullptr,nullptr,full->rows,meta_width,m*2,meta_expected.p,nullptr,bad.p);sync();}
  }
  auto fits=runner.fits;for(U capacity:{1u,5u,13u}){runner.plan["evaluation_batch_rows"]=capacity;runner.evaluate(held,prepared.models,metas,
    [&](U b,ResidentPredictor&,const Matrix& raw,const float* responses,Device<float>&){compare_response_words<<<blocks(L(raw.rows)*2),256>>>(responses,base_width,b*2,base_expected.p,base_width,b*2,raw.ids.p,positions.p,raw.rows,bad.p);sync();},
    [&](const Matrix& raw,const float* responses,ResidentPredictor& predictor,Device<float>&,auto& registered){if(!runner.meta_raw.empty()){check_meta_raw_columns<<<blocks(L(raw.rows)*runner.meta_raw.size()),256>>>(data.x.p,raw.ids.p,raw.rows,data.F,runner.meta_raw_indices.p,U(runner.meta_raw.size()),responses,U(runner.base.size()*2),base_width,bad.p);sync();}for(U m=0;m<metas.size();++m){registered(predictor,metas[m]);auto values=predictor.predict_probabilities(responses,raw.rows);compare_response_words<<<blocks(L(raw.rows)*2),256>>>(values,2,0,meta_expected.p,meta_width,m*2,raw.ids.p,positions.p,raw.rows,bad.p);sync();}}
  );need(!bad.get()[0],"batched/full native probability FP32 words differ");}
  need(runner.fits==fits,"batch probability checks added FIT calls");return {{"complete",true},{"native_base_and_meta_probability_words_identical",true},{"ceilings",{1,5,13}},{"source_ID_mapping_checked",true},{"prediction_files_written",false},{"FIT_calls_for_fixture_models",fits}};
}

J validate_replay(const J& plan,const J& pipeline){
  auto saved=read_pipeline(plan,pipeline);return {{"raw_features",saved.features},{"classes",2},{"original_base_count",pipeline.at("base_models").size()},{"retained_base_count",saved.bases.size()},{"source_pruning",saved.pruning},{"native_prediction_rule",saved.rule.is_null()?native_prediction_config(J::object()).at(0):saved.rule},{"native_prediction_calibration",pipeline.value("native_prediction_calibration",J(nullptr))},{"configuration",saved.limits},{"completed_native_fits",0},{"training_dataset_loaded",false},{"shared_converter",saved.combined?"class_study::convert_composed_model":"class_study::convert_model"}};
}
int replay(const J& plan,const J& pipeline,const std::filesystem::path& output,volatile std::sig_atomic_t& stopping){
  need(!std::filesystem::exists(output/"result.json"),"refusing to overwrite a replay result");auto started=std::chrono::steady_clock::now();auto saved=read_pipeline(plan,pipeline);ck(cudaSetDevice(0),"replay device");
  J report={{"format","grouped-segmentation-native-replay-result-1"},{"complete",false},{"attempt_complete",false},{"native_fit_complete",true},{"completed_native_fits",0},{"training_dataset_loaded",false},{"heldout_accuracy_claim",false},{"selection_performed",false},{"original_pipeline",pipeline},{"source_pruning",saved.pruning},{"native_prediction_rule",saved.rule.is_null()?native_prediction_config(J::object()).at(0):saved.rule},{"native_prediction_calibration",pipeline.value("native_prediction_calibration",J(nullptr))},{"configuration",saved.limits},{"conversion_attempted",false}};
  std::filesystem::create_directories(output);J reduced=pipeline;reduced["base_models"]=J::array();reduced["source_pruning"]=saved.pruning;reduced["replay_without_FIT"]=true;
  for(U i=0;i<saved.bases.size();++i){auto item=pipeline.at("base_models").at(saved.retained_indices[i]);auto path=output/("replay-native-base-"+std::to_string(i+1)+".json");class_conversion_native::atomic_text(path,saved.bases[i].model_json);item["source_path"]=item.at("path");item["path"]=path.string();reduced["base_models"].push_back(std::move(item));}
  if(saved.combined){auto original_path=output/"replay-native-meta-original.json";class_conversion_native::atomic_text(original_path,saved.original_meta.model_json);auto path=output/"replay-native-meta.json";class_conversion_native::atomic_text(path,saved.meta.model_json);auto item=pipeline.at("meta_model");item["source_path"]=item.at("path");item["path"]=path.string();item["sha256"]=saved.meta.model_sha256;item["features"]=saved.bases.size()*2+saved.meta_raw.size();item["training_metrics"]=saved.meta.metrics;reduced["meta_model"]=std::move(item);reduced["original_meta_model"]={{"path",original_path.string()},{"sha256",saved.original_meta.model_sha256}};}
  auto native_path=output/"final-native-pipeline.json";class_conversion_native::atomic_json(native_path,reduced);report["native_pipeline_path"]=native_path.string();report["native_pipeline_sha256"]=class_conversion_native::sha256(reduced.dump(2)+'\n');publish(output,report);
  try{
    ResidentPredictor verified(plan,saved.features,2);report["conversion_attempted"]=true;report["conversion_performed"]=true;publish(output,report);auto conversion_started=std::chrono::steady_clock::now();
    auto replay_plan=plan;if(!saved.meta_raw.empty())replay_plan["meta_input_feature_indices"]=saved.meta_raw;
    report["conversion"]=attempt_deployment(replay_plan,saved.bases,saved.combined?&saved.meta:nullptr,verified,saved.limits,output,"replay-selected",stopping,saved.rule);report["conversion_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-conversion_started).count();
    report["conversion"]["native_prediction_calibration"]=pipeline.value("native_prediction_calibration",J(nullptr));if(report["conversion"].at("complete")==true)class_conversion_native::atomic_json(output/"replay-selected.receipt.json",report["conversion"]);report["conversion_complete"]=report.at("conversion").at("complete");report["qualified_native_class_root"]=report.at("conversion").at("accepted_native_class_root");report["attempt_complete"]=true;report["complete"]=report.at("conversion_complete");report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();publish(output,report);
    std::cout<<J{{"event","grouped_segmentation_native_replay_complete"},{"result",report}}.dump()<<'\n'<<std::flush;return stopping?2:(report.at("complete").get<bool>()?0:1);
  }catch(const std::exception& e){report["error"]=e.what();report["interrupted"]=bool(stopping);report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();publish(output,report);if(stopping)return 2;throw;}
}

J validate(const J& plan,const J& d){
  const auto inner_encoding=inner_metrics::encoding(plan);const bool aggregation=oof_aggregation_config(plan);const auto bootstrap=bootstrap_config(plan);const bool fixed_mean=candidate_aggregation_config(plan);const bool fixed_base_mean=base_aggregation_config(plan);
  need(d.at("format")=="grouped-segmentation-dataset-1","segmentation descriptor format");auto n=integer(d,"rows",INT32_MAX),f=integer(d,"features",INT32_MAX);need(n&&f&&integer(d,"classes")==2,"binary segmentation shape");need(n<=SIZE_MAX/4/f,"segmentation shape overflow");
  need(d.at("domain_description").is_string()&&!d.at("domain_description").get<std::string>().empty(),"segmentation domain description required");
  for(auto k:{"values_path","labels_path","patient_ids_path","scan_ids_path","train_eligible_path"})need(d.at(k).is_string()&&!d.at(k).get<std::string>().empty(),"segmentation input path required");
  need(d.at("scans").is_array()&&!d.at("scans").empty()&&d.at("scans").size()<=n,"scan metadata required");std::set<U> patients,scans;for(const auto& s:d.at("scans")){patients.insert(U(integer(s,"patient_id",UINT32_MAX)));need(scans.insert(U(integer(s,"scan_id",UINT32_MAX))).second,"duplicate scan ID");integer(s,"truth_positive_outside_rows",UINT64_MAX/d.at("scans").size()/8);}
  need(patients.size()>=5,"nested grouped segmentation requires at least five patients");if(!bootstrap.is_null())need(patients.size()<=(1u<<24),"bootstrap patient multiplicity must be exactly representable in FP32");
  for(const char* key:{"include_all_fit_positives","fit_class_balanced","outer_model_summaries"})if(plan.contains(key))need(plan.at(key).is_boolean(),"FIT sampling/weight options must be boolean");
  if(plan.contains("evaluation_batch_rows"))need(integer(plan,"evaluation_batch_rows",UINT32_MAX)>0,"evaluation_batch_rows must be positive");
  if(plan.contains("fit_group_weights"))need(plan.at("fit_group_weights").is_string()&&(plan.at("fit_group_weights")=="none"||plan.at("fit_group_weights")=="equal_patient"),"fit_group_weights must be none or equal_patient");
  if(plan.contains("fit_sampling_weights"))need(plan.at("fit_sampling_weights").is_string()&&(plan.at("fit_sampling_weights")=="none"||plan.at("fit_sampling_weights")=="inverse_retention"),"fit_sampling_weights must be none or inverse_retention");
  need(!plan.value("fit_class_balanced",false)||plan.value("fit_sampling_weights",std::string("none"))=="none","class balancing and inverse-retention weighting are mutually exclusive");
  const auto sampling=validated_sampling_counts(plan,d);
  auto reference=reference_config(plan,U(f));auto native_rules=native_prediction_config(plan);auto rule_views=rule_search_views_config(plan,U(native_rules.size()));const auto raw_meta=meta_input_config(plan,U(f));
  const auto blend=blend_config(plan,U(f));const auto missing=missing_features(d,U(f),plan);if(!blend.is_null())need(!std::binary_search(missing.begin(),missing.end(),U(integer(blend,"probability_feature_index"))),"blend teacher probability cannot declare missing values");const L missing_bytes=multiply(missing.size(),sizeof(U));
  const auto training_loss=training_loss_config(plan);
  const bool source_gate=plan.contains("conversion_gate"),composition_gate=plan.contains("composed_conversion_gate"),outer_conversion=plan.contains("conversion_on_outer_deployments"),final_conversion=plan.contains("final_deployment");
  need(!(source_gate&&composition_gate)&&!((source_gate||composition_gate)&&outer_conversion),"conversion gate and nested-study conversion modes are mutually exclusive");
  need(!((source_gate||composition_gate)&&final_conversion),"final deployment requires nested outer evaluation");
  need(reference.is_null()||(!source_gate&&!composition_gate),"reference predictor selection requires nested evaluation");
  if(native_rules.size()>1){need(!source_gate&&!composition_gate&&!outer_conversion,"native prediction threshold selection does not support conversion gates or outer conversions");need(!final_conversion||(plan.at("final_deployment").is_object()&&plan.at("final_deployment").contains("convert")&&plan.at("final_deployment").at("convert").is_boolean()&&!plan.at("final_deployment").at("convert").get<bool>()),"native prediction threshold selection requires omitted final_deployment or explicit final_deployment.convert=false");}
  need(!source_gate||(!plan.value("fit_class_balanced",false)&&plan.value("fit_sampling_weights",std::string("none"))=="none"&&plan.value("fit_group_weights",std::string("none"))=="none"),"FIT weighting requires nested or composed evaluation");
  if(source_gate)(void)gate_config(plan,patients);
  for(auto k:{"base_hyperparameters","meta_hyperparameters"}){need(plan.at(k).is_array()&&!plan.at(k).empty()&&plan.at(k).size()<=UINT32_MAX/2,"nonempty bounded hyperparameter list required");for(const auto& hp:plan.at(k))ResidentTrainer::validate_hyperparameters(hp,2);}
  need(add(multiply(plan.at("base_hyperparameters").size(),2),raw_meta.size())<=INT32_MAX,"mixed meta feature extent exceeds supported native shape");need(raw_meta.empty()||!source_gate,"raw meta inputs require a combiner workflow");
  need(add(plan.at("base_hyperparameters").size(),plan.at("meta_hyperparameters").size())<=UINT32_MAX/native_rules.size(),"native model/prediction candidate product exceeds index extent");
  if(composition_gate)(void)stack_config(plan,plan.at("composed_conversion_gate"),patients,true);
  if(outer_conversion)(void)stack_config(plan,plan.at("conversion_on_outer_deployments"),patients,false);
  if(final_conversion)(void)final_config(plan,patients);
  U inner=plan.value("inner_folds",3u),oof=plan.value("oof_folds",3u);need(inner>=2&&oof>=2&&inner<=patients.size()-1&&oof<=patients.size()-1,"grouped fold extent");
  if(plan.contains("outer_patient_ids")){need(plan.at("outer_patient_ids").is_array()&&!plan.at("outer_patient_ids").empty(),"outer subset must be nonempty");std::set<U> used;for(const auto& x:plan.at("outer_patient_ids")){need(x.is_number_unsigned()||(x.is_number_integer()&&x.get<std::int64_t>()>=0),"outer patient identifier");auto p=x.get<std::uint64_t>();need(p<=UINT32_MAX&&patients.count(U(p))&&used.insert(U(p)).second,"outer patient unknown or duplicated");}}
  need(!(plan.contains("outer_folds")&&plan.contains("outer_patient_ids")),"outer_folds and outer_patient_ids are mutually exclusive");
  if(plan.contains("outer_folds")){auto count=integer(plan,"outer_folds",patients.size());need(count>=2,"outer_folds must be at least two");need(!source_gate&&!composition_gate,"outer_folds requires nested outer evaluation");}
  std::vector<U> patient_list(patients.begin(),patients.end());auto groups=outer_groups(plan,patient_list);
  if(!source_gate&&!composition_gate)for(const auto& held:groups){auto fitting=without(patient_list,held);for(const auto& inner_held:folds(fitting,inner))need(fitting.size()-inner_held.size()>=2,"outer folds leave too few inner FIT patient groups for OOF");}
  // Conservative bound for explicit CUDA arrays, gathered matrices and native
  // FIT copies. Opaque XGBoost allocations are additional; retain free headroom.
  auto bank=plan.at("base_hyperparameters").size();need(bank<=SIZE_MAX/16/n,"response size overflow");
  auto eligible_fit=integer(d,"train_eligible_rows",n),fit=eligible_fit;
  if(!sampling.is_null())fit=integer(sampling.at("totals"),"retained_rows",n);else if(plan.value("include_all_fit_positives",false)){L positive=0;for(const auto& scan:d.at("scans"))positive=add(positive,integer(scan,"truth_positive_rows",n));need(positive<=n,"declared foreground exceeds dataset rows");fit=std::min<L>(n,add(fit,positive));}
  need(fit>0,"nonempty declared training rows required");
  if(!training_loss.is_null())for(const auto* key:{"base_hyperparameters","meta_hyperparameters"})
    for(const auto& hp:plan.at(key))ResidentTrainer::validate_loss(hp,training_loss_spec(training_loss,fit),fit,U(patients.size()),2);
  std::uint64_t resident=add(multiply(n,4*f+16),missing_bytes);
  auto memory=grouped_memory_bound(plan,d,patient_list,groups,n,f,bank,plan.at("meta_hyperparameters").size(),reference.is_null()?0:reference.at("candidates").size(),fit,native_rules.size());memory.bytes=add(memory.bytes,missing_bytes);memory.cub_live_bytes=add(memory.cub_live_bytes,missing_bytes);auto estimate=memory.bytes;
  auto budget=plan.value("gpu_byte_budget",12ULL<<30);need(resident<=budget,"input CUDA storage exceeds declared budget");
  need(estimate<=budget,"conservative explicit CUDA peak exceeds budget; reduce declared ROI/features or use a larger budget");
  need(memory.cub_live_bytes<budget,"no remaining CUB selection workspace budget");
  need(n*4*f<=plan.value("host_byte_budget",16ULL<<30),"host source read exceeds budget");
  J metadata={{"patients",patients.size()},{"scans",scans.size()},{"rows",n},{"features",f},{"resident_CUDA_bytes",resident},{"conservative_explicit_CUDA_bytes",estimate},{"explicit_CUDA_bound_method",memory.detailed?"live grouped matrices from per-scan counts":"whole-dataset fallback"},{"peak_bound_stage",memory.stage},{"largest_prepare_FIT_rows",memory.fit_rows},{"largest_prepare_evaluation_rows",memory.evaluation_rows},{"CUB_workspace_additional",true},{"CUB_workspace_byte_budget",budget-memory.cub_live_bytes},{"CUB_live_explicit_CUDA_bound",memory.cub_live_bytes},{"evaluation_batch_rows_limit",plan.value("evaluation_batch_rows",0ULL)},{"preflight_evaluation_batch_rows_estimate",memory.evaluation_batch_rows},{"evaluation_batch_capacity","remaining configured budget and actual free CUDA memory after native registrations"},{"evaluation_native_output_reserve_bytes_per_row",32},{"opaque_native_allocations_additional",true},{"inner_folds",inner},{"OOF_folds",oof},{"outer_fold_count",groups.size()},{"outer_patient_groups",groups},{"outer_evaluation_method",plan.contains("outer_folds")?"patient_grouped_k_fold":"leave_one_patient_out"},{"final_deployment_requested",final_conversion},{"reference_predictor",reference},{"native_prediction_candidates",native_rules},{"native_threshold_scoring",native_rules.size()>1?"fixed-grid CUDA scan/class histogram":"native argmax only"},{"native_threshold_histogram_CUDA_bytes",threshold_histogram_bytes(native_rules.size(),scans.size())},{"include_all_fit_positives",plan.value("include_all_fit_positives",false)},{"fit_class_balanced",plan.value("fit_class_balanced",false)},{"fit_sampling_weights",plan.value("fit_sampling_weights",std::string("none"))},{"fit_group_weights",plan.value("fit_group_weights",std::string("none"))},{"conservative_FIT_rows",fit},{"full_ROI_evaluation",true},{"patient_grouping_at_every_level",true},{"training_loss",training_loss},{"training_loss_effective_policy",training_loss_policy(training_loss,plan.at("base_hyperparameters")[0])},{"training_loss_fit_only",!training_loss.is_null()}};
  if(!sampling.is_null())metadata["fit_sampling"]={{"version",1},{"method","binary_mask_hard_negatives"},{"mask_feature_index",sampling.at("mask_feature_index")},{"normalization","legacy_retained_mass"},{"arithmetic_policy",fit_sampling::arithmetic_policy},{"strata",sampling.at("strata")},{"full_ROI_evaluation_unchanged",true},{"original_array_paths_unchanged",true},{"count_extension_validated_at_load",true},{"coefficient_scope","only explicit FIT patient IDs; original row IDs used in raw and OOF response matrices"},{"weight_target","equal FIT patient full-ROI class/stratum prevalence; old enclosing FIT retained n0 total mass"},{"mean_one_normalization",false},{"audit_scope","one receipt per FIT matrix weight initialization, not per hyperparameter model"},{"count_audit_CUDA_bytes",multiply(scans.size(),sizeof(fit_sampling::Counts))},{"weight_workspace_CUDA_bytes_upper_bound",add(multiply(patients.size(),88),4)},{"legacy_retained_rows",sampling.at("totals").at("legacy_retained_rows")},{"retained_rows",sampling.at("totals").at("retained_rows")},{"added_rows",sampling.at("totals").at("added_rows")}};
  if(plan.contains("inner_result_encoding"))metadata["inner_result_encoding"]=inner_encoding;
  if(!missing.empty()){metadata["missing_feature_indices"]=d.at("missing_feature_indices");metadata["missing_feature_policy"]="NaN permitted only in declared source columns; all infinities refused; native missing-value branch semantics";}
  if(!rule_views.empty())metadata["rule_search_views"]={{"version",1},{"primary_view","full"},{"additional_views",rule_views},{"shared_inner_counts",true},{"shared_outer_refits",true},{"tie_order","exact score, smaller summed native model bytes, canonical model-major rule-minor order; base before meta, fixed reference before fitted family"}};
  if(aggregation)metadata["oof_teacher_aggregation"]={{"version",1},{"method","patient_weighted_probability_mean"},{"control_mode","full_refit"},{"additional_mode","oof_teacher_probability_mean"},{"weights","integer OOF-held patient counts, divided by enclosing FIT patient count"},{"arithmetic_policy",oof_aggregation::arithmetic_policy},{"accumulator_bytes_per_row_upper_bound",add(multiply(plan.at("meta_hyperparameters").size(),16),8)},{"additional_teacher_FITs",0},{"selection_scope","independent inner-only choices per inference mode and rule view"},{"shared_meta_FIT_union",true},{"conversion_supported",false}};
  if(fixed_mean)metadata["meta_candidate_aggregation"]={{"version",1},{"method","uniform_probability_mean"},{"control_mode","full_refit"},{"additional_mode","fixed_meta_probability_mean"},{"member_order","all declared meta_hyperparameters in original order"},{"members",plan.at("meta_hyperparameters").size()},{"arithmetic_policy",oof_aggregation::arithmetic_policy},{"accumulator_bytes_per_row_upper_bound",24},{"additional_inner_FITs",0},{"additional_teacher_FITs",0},{"outer_meta_refits","all declared ordinary meta candidates once; shared by both policies"},{"selection_scope","ordinary HP/cut/family; fixed mean cut/family only; inner data exclusively"},{"conversion_supported",false}};
  if(!bootstrap.is_null()){const L replicas=integer(bootstrap,"replicates"),settings=plan.at("meta_hyperparameters").size();need(multiply(replicas,settings)<=UINT32_MAX,"bootstrap model product exceeds index range");metadata["meta_patient_bootstrap"]={{"version",1},{"replicates",replicas},{"seed",bootstrap.at("seed")},{"control_mode","full_refit"},{"additional_mode","meta_patient_bootstrap_probability_mean"},{"draw_policy",meta_bootstrap::draw_policy},{"weight_policy",meta_bootstrap::weight_policy},{"arithmetic_policy",oof_aggregation::arithmetic_policy},{"additional_teacher_FITs",0},{"fresh_weighted_DMatrix_per_replicate",true},{"OOF_feature_words_reused",true},{"zero_weight_rows_retained",true},{"OOB_scoring",false},{"selection_scope","independent inner-only choices per mode and rule view"},{"conversion_supported",false},{"accumulator_bytes_per_row_upper_bound",add(multiply(settings,16),8)},{"weight_workspace_bytes_upper_bound",add(multiply(fit,4),add(multiply(patients.size(),24),8))}};}
  if(!blend.is_null())metadata["teacher_probability_blend"]={{"version",1},{"method","oof_brier_convex"},{"probability_feature_index",blend.at("probability_feature_index")},{"control_mode","ordinary"},{"additional_outputs",J::array({"blend_only","augmented_inner_selected"})},{"arithmetic_policy",teacher_blend::arithmetic_policy},{"coefficient_authority","alpha32_bits"},{"additional_native_FITs",0},{"additional_native_prediction_calls",0},{"evaluation_scratch","reuse ordinary prediction buffer; histogram before in-place classification"},{"coefficient_reduction_workspace_bytes_upper_bound",add(multiply((fit+255)/256,32),20)},{"selection_scope","FIT-only patient OOF Brier coefficient; independent inner-heldout rule and family selection; all ordinary candidates retained"},{"family_tie_policy","blend last after unchanged ordinary family selection"},{"conversion_supported",false}};
  if(fixed_base_mean)metadata["base_candidate_aggregation"]={{"version",1},{"method","uniform_probability_mean"},{"control_mode","ordinary"},{"additional_outputs",J::array({"base_mean_only","augmented_inner_selected"})},{"inference_mode","fixed_base_probability_mean"},{"base_recipe","uniform_all_declared_base_candidates_v1"},{"members",plan.at("base_hyperparameters").size()},{"member_order","all declared base_hyperparameters in original order; no deduplication"},{"arithmetic_policy",base_mean::arithmetic_policy},{"pair_bytes_per_row",8},{"additional_native_FITs",0},{"additional_native_prediction_calls",0},{"coefficient_fits",0},{"selection_scope","fixed mean cutoff and augmented family from inner counts only; all ordinary candidates retained"},{"family_tie_policy","base mean last after unchanged ordinary family selection"},{"conversion_supported",false}};
  if(!raw_meta.empty()){metadata["meta_input_mapping"]=meta_input_mapping(raw_meta);metadata["meta_input_columns"]=plan.at("base_hyperparameters").size()*2+raw_meta.size();}
  return metadata;
}

int run(const J& plan,const J& descriptor,const std::filesystem::path& output,volatile std::sig_atomic_t& stopping){
  auto metadata=validate(plan,descriptor);need(!std::filesystem::exists(output/"result.json"),"refusing to overwrite a segmentation result");auto started=std::chrono::steady_clock::now();ck(cudaSetDevice(0),"segmentation device");Data data(descriptor,plan,metadata);Runner runner(plan,data,stopping);
  if(plan.contains("conversion_gate"))return conversion_gate(plan,data,output,stopping);
  if(plan.contains("composed_conversion_gate"))return composed_gate(plan,data,runner,output,stopping);
  const bool convert_outer=plan.contains("conversion_on_outer_deployments");
  const J deployment_limits=convert_outer?stack_config(plan,plan.at("conversion_on_outer_deployments"),std::set<U>(data.patients.begin(),data.patients.end()),false):J(nullptr);
  auto groups=outer_groups(plan,data.patients);std::vector<U> outer;for(const auto& group:groups)outer.insert(outer.end(),group.begin(),group.end());
  L completed_patients=0;double outer_conversion_seconds=0;
  J results=J::array();Metrics total(data,runner.reference.is_null()?3:5);J report={{"format","grouped-segmentation-result-1"},{"complete",false},{"native_evaluation_complete",false},{"protocol",metadata},{"dataset",descriptor},{"plan",plan},{"outer_patient_ids",outer},{"outer_patient_groups",groups},{"outer_fold_count",groups.size()},{"completed_outer_folds",0},{"completed_outer_patients",0},{"full_cohort_outer_evaluation",false},{"full_cohort_outer_evaluation_requested",outer.size()==data.patients.size()},{"full_leave_one_patient_out",!plan.contains("outer_folds")&&outer.size()==data.patients.size()},{"outer_evaluation_method",plan.contains("outer_folds")?"patient_grouped_k_fold":"leave_one_patient_out"},{"selection_metric",runner.reference.is_null()?"macro_patient_Dice; ties smaller native bytes then declared order":"macro_patient_Dice; fixed reference preferred on family ties; native ties smaller bytes then declared order"},{"patient_Dice_definition","pool both scans per patient; include all truth-positive voxels outside ROI as false negatives"},{"training_eligibility_applies_to_evaluation",false},{"per_trial_file_writes",0},{"upstream_XGBoost_modified",false},{"conversion_performed",false},{"deployment_conversion_requested",convert_outer},{"deployment_conversion_attempts",0},{"completed_deployment_conversions",0},{"all_requested_deployment_conversions_complete",false},{"conversion_failure_changes_native_selection",false},{"final_deployment_requested",plan.contains("final_deployment")}};
  report["completed_qualified_deployment_conversions"]=0;report["all_requested_deployment_conversions_qualified"]=false;
  const auto rule_views=rule_search_views_config(plan,U(runner.native_rules.size()));std::vector<std::unique_ptr<Metrics>> view_totals;
  for(const auto& view:rule_views){(void)view;view_totals.push_back(std::make_unique<Metrics>(data,runner.reference.is_null()?3:5));}
  std::vector<std::unique_ptr<Metrics>> aggregation_totals;
  if(runner.probability_mean())for(U v=0;v<=rule_views.size();++v)aggregation_totals.push_back(std::make_unique<Metrics>(data,runner.reference.is_null()?3:5));
  std::vector<std::unique_ptr<Metrics>> blend_totals;if(!runner.blend.is_null())for(U v=0;v<=rule_views.size();++v)blend_totals.push_back(std::make_unique<Metrics>(data,2));
  std::vector<std::unique_ptr<Metrics>> base_mean_totals;if(runner.base_candidate_mean)for(U v=0;v<=rule_views.size();++v)base_mean_totals.push_back(std::make_unique<Metrics>(data,2));
  if(convert_outer)report["deployment_qualification_mode"]=deployment_limits.at("qualification_mode");
  try{
    for(U outer_fold=0;outer_fold<groups.size();++outer_fold){const auto& heldout=groups[outer_fold];runner.interrupted();auto fitting=without(data.patients,heldout);
      const auto before_inner_fits=runner.fits;auto choice=select_models(plan,data,runner,fitting,"INNER_META_FIT");U best_base=choice.baseline,best_meta=choice.meta,selected=choice.family;const auto before_outer_fits=runner.fits;
      // Freeze all settings and family before opening outer labels to scoring.
      J selection=choice.details;selection["outer_FIT_patient_ids"]=fitting;selection["OOF_patient_groups_for_outer_refit"]=folds(fitting,runner.oof);
      J identity={{"outer_fold_index",outer_fold},{"outer_patient_ids",heldout}};if(heldout.size()==1)identity["outer_patient_id"]=heldout[0];
      J frozen=identity;frozen["event"]="grouped_segmentation_selection_frozen";frozen["selection"]=selection;std::cout<<frozen.dump()<<'\n'<<std::flush;
      auto s=runner.prepare(fitting);std::vector<U> meta_indices={best_meta};for(const auto& view:choice.views)meta_indices.push_back(view.meta);if(runner.aggregate_oof)for(const auto& view:choice.aggregated)meta_indices.push_back(view.meta);std::sort(meta_indices.begin(),meta_indices.end());meta_indices.erase(std::unique(meta_indices.begin(),meta_indices.end()),meta_indices.end());
      const auto ordinary_meta_indices=meta_indices;if(runner.candidate_mean){meta_indices.resize(runner.meta.size());std::iota(meta_indices.begin(),meta_indices.end(),0);}
      std::vector<TrainingResult> meta_models;meta_models.reserve(meta_indices.size());
      {auto combiner=runner.trainer(s.meta_fit,fitting,"FROZEN_META_FIT");for(U m:meta_indices)meta_models.push_back(runner.train(combiner,runner.meta[m]));}
      std::vector<U> bag_indices;BootstrapModels bagged;
      if(!runner.bootstrap.is_null()){for(const auto& view:choice.aggregated)bag_indices.push_back(view.meta);std::sort(bag_indices.begin(),bag_indices.end());bag_indices.erase(std::unique(bag_indices.begin(),bag_indices.end()),bag_indices.end());bagged=runner.bootstrap_models(s.meta_fit,fitting,bag_indices,"FROZEN_BOOTSTRAP_META_FIT");}
      const auto& model=meta_models[std::lower_bound(meta_indices.begin(),meta_indices.end(),best_meta)-meta_indices.begin()];
      J fit_receipt={{"outer_FIT_rows",s.fit->rows},{"outer_FIT_class_counts",s.fit->class_counts},{"outer_full_FIT_class_counts",data.full_class_counts(fitting)},{"outer_FIT_patient_weights",s.fit->patient_weights},{"outer_meta_FIT_patient_weights",s.meta_fit->patient_weights}};
      const auto blend_fit=runner.fit_blends(s,fitting);
      s.fit.reset();s.meta_fit.reset();Metrics fold(data,runner.reference.is_null()?3:5);std::vector<std::unique_ptr<Metrics>> view_folds,aggregation_folds,blend_folds,base_mean_folds;J paired_calls;Device<U> blend_bad;if(!choice.blended.empty()){blend_bad=Device<U>(1);for(U v=0;v<choice.blended.size();++v)blend_folds.push_back(std::make_unique<Metrics>(data,2));}
      if(!choice.base_means.empty())for(U v=0;v<choice.base_means.size();++v)base_mean_folds.push_back(std::make_unique<Metrics>(data,2));
      if(choice.views.empty()&&choice.aggregated.empty()&&choice.blended.empty()&&choice.base_means.empty())runner.evaluate(heldout,s.models,meta_models,
        [&](U b,ResidentPredictor& predictor,const Matrix& raw,const float* responses,Device<float>& prediction){if(b!=best_base)return;const float* bp;
          if(choice.baseline_rule){threshold_prediction(runner.native_rules.at(choice.baseline_rule),responses,raw.rows,runner.meta_features(),b*2+1,prediction);bp=prediction.p;}else bp=predictor.predict(raw.x.p,raw.rows,false);
          fold.add(0,bp,raw);total.add(0,bp,raw);if(!selected){fold.add(2,bp,raw);total.add(2,bp,raw);}},
        [&](const Matrix& raw,const float* responses,ResidentPredictor& predictor,Device<float>& prediction,auto& registered){registered(predictor,model);const float* mp;
          if(choice.meta_rule){auto probabilities=predictor.predict_probabilities(responses,raw.rows);threshold_prediction(runner.native_rules.at(choice.meta_rule),probabilities,raw.rows,2,1,prediction);mp=prediction.p;}else mp=predictor.predict(responses,raw.rows,false);
          fold.add(1,mp,raw);total.add(1,mp,raw);if(selected==1){fold.add(2,mp,raw);total.add(2,mp,raw);}
          if(!runner.reference.is_null()){reference_prediction(runner.reference,0,raw,prediction);fold.add(3,prediction.p,raw);total.add(3,prediction.p,raw);reference_prediction(runner.reference,choice.reference,raw,prediction);fold.add(4,prediction.p,raw);total.add(4,prediction.p,raw);if(selected==2){fold.add(2,prediction.p,raw);total.add(2,prediction.p,raw);}}
        });
      else{
        std::vector<const Selection*> choices={&choice};std::vector<Metrics*> fold_metrics={&fold},total_metrics={&total};
        for(U v=0;v<choice.views.size();++v){choices.push_back(&choice.views[v]);view_folds.push_back(std::make_unique<Metrics>(data,runner.reference.is_null()?3:5));fold_metrics.push_back(view_folds.back().get());total_metrics.push_back(view_totals[v].get());}
        const U control_views=U(choices.size());
        for(U v=0;v<choice.aggregated.size();++v){choices.push_back(&choice.aggregated[v]);aggregation_folds.push_back(std::make_unique<Metrics>(data,runner.reference.is_null()?3:5));fold_metrics.push_back(aggregation_folds.back().get());total_metrics.push_back(aggregation_totals[v].get());}
        L base_probability_calls=0,base_class_calls=0,meta_probability_calls=0,meta_class_calls=0;
        Device<L> argmax_differences;Device<U> diagnostic_bad;if(runner.argmax_diagnostics()){argmax_differences=Device<L>(runner.meta.size());argmax_differences.zero();diagnostic_bad=Device<U>(1);diagnostic_bad.zero();}
        auto record=[&](U view,U method,const float* prediction,const Matrix& raw){fold_metrics[view]->add(method,prediction,raw);total_metrics[view]->add(method,prediction,raw);if(!choice.blended.empty()&&method==2&&choice.blended.at(view).family!=3){blend_folds[view]->add(1,prediction,raw);blend_totals[view]->add(1,prediction,raw);}if(!choice.base_means.empty()&&method==2&&choice.base_means.at(view).family!=4){base_mean_folds[view]->add(1,prediction,raw);base_mean_totals[view]->add(1,prediction,raw);}};
        runner.evaluate(heldout,s.models,meta_models,
          [&](U b,ResidentPredictor& predictor,const Matrix& raw,const float* responses,Device<float>& prediction){
            ++base_probability_calls;const float* classes=nullptr;
            for(U v=0;v<choices.size();++v){const auto& c=*choices[v];if(c.baseline!=b)continue;const float* values;
              if(c.baseline_rule){threshold_prediction(runner.native_rules.at(c.baseline_rule),responses,raw.rows,runner.meta_features(),b*2+1,prediction);values=prediction.p;}
              else{if(!classes){classes=predictor.predict(raw.x.p,raw.rows,false);++base_class_calls;}values=classes;}
              record(v,0,values,raw);if(c.family==0)record(v,2,values,raw);
            }
            for(U v=0;v<choice.blended.size();++v){const auto& c=choice.blended[v];if(c.baseline!=b)continue;runner.blend_values(b,blend_fit,raw,responses,prediction,blend_bad);runner.blend_classes(c.rule,raw,prediction,blend_bad);blend_folds[v]->add(0,prediction.p,raw);blend_totals[v]->add(0,prediction.p,raw);if(c.family==3){blend_folds[v]->add(1,prediction.p,raw);blend_totals[v]->add(1,prediction.p,raw);}}
          },
          [&](const Matrix& raw,const float* responses,ResidentPredictor& predictor,Device<float>& prediction,auto& registered,MeanSink& mean){
            for(U m=0;m<meta_models.size();++m){bool probability=false,classes=false;for(U v=0;v<control_views;++v){const auto* c=choices[v];if(c->meta==meta_indices[m]){probability|=c->meta_rule!=0;classes|=c->meta_rule==0;}}
              registered(predictor,meta_models[m]);
              // Consume every threshold before any native call invalidates the borrowed probabilities.
              if(probability||runner.candidate_mean){const auto* values=predictor.predict_probabilities(responses,raw.rows);++meta_probability_calls;
                for(U v=0;v<control_views;++v){const auto& c=*choices[v];if(c.meta!=meta_indices[m]||!c.meta_rule)continue;threshold_prediction(runner.native_rules.at(c.meta_rule),values,raw.rows,2,1,prediction);record(v,1,prediction.p,raw);if(c.family==1)record(v,2,prediction.p,raw);}mean.add(m,values);}
              if(classes){const auto* values=predictor.predict(responses,raw.rows,false);++meta_class_calls;for(U v=0;v<control_views;++v){const auto& c=*choices[v];if(c.meta!=meta_indices[m]||c.meta_rule)continue;record(v,1,values,raw);if(c.family==1)record(v,2,values,raw);}}
              if(runner.argmax_diagnostics()&&std::binary_search(ordinary_meta_indices.begin(),ordinary_meta_indices.end(),meta_indices[m])){const auto* native=predictor.predict(responses,raw.rows,false);++meta_class_calls;ck(cudaMemcpy(prediction.p,native,raw.rows*sizeof(float),cudaMemcpyDeviceToDevice),"preserve ordinary outer argmax diagnostic");const auto* probabilities=predictor.predict_probabilities(responses,raw.rows);++meta_probability_calls;meta_bootstrap::argmax_disagreements<<<blocks(raw.rows),256>>>(prediction.p,probabilities,raw.rows,argmax_differences.p+meta_indices[m],diagnostic_bad.p);sync();need(!diagnostic_bad.get()[0],"ordinary outer argmax diagnostic invalid");}
            }
            if(!runner.reference.is_null()){
              reference_prediction(runner.reference,0,raw,prediction);for(U v=0;v<choices.size();++v)record(v,3,prediction.p,raw);
              reference_prediction(runner.reference,choice.reference,raw,prediction);for(U v=0;v<choices.size();++v){need(choices[v]->reference==choice.reference,"rule views must share reference selection");record(v,4,prediction.p,raw);if(choices[v]->family==2)record(v,2,prediction.p,raw);}
            }
          },s.oof_banks,[&](U m,const Matrix& raw,const float* probabilities,Device<float>& prediction){
            if(runner.base_candidate_mean){for(U v=0;v<choice.base_means.size();++v){const auto& c=choice.base_means[v];averaged_prediction(runner.native_rules.at(c.rule),probabilities,raw,prediction);base_mean_folds[v]->add(0,prediction.p,raw);base_mean_totals[v]->add(0,prediction.p,raw);if(c.family==4){base_mean_folds[v]->add(1,prediction.p,raw);base_mean_totals[v]->add(1,prediction.p,raw);}}return;}
            for(U v=control_views;v<choices.size();++v){const auto& c=*choices[v];if(!runner.candidate_mean&&c.meta!=(runner.bootstrap.is_null()?meta_indices[m]:bag_indices[m]))continue;averaged_prediction(runner.native_rules.at(c.meta_rule),probabilities,raw,prediction);record(v,1,prediction.p,raw);if(c.family==1)record(v,2,prediction.p,raw);}
          },bagged);
        paired_calls={{"base_probability",base_probability_calls},{"base_native_class",base_class_calls},{"meta_probability",meta_probability_calls},{"meta_native_class",meta_class_calls},{"scope","outer scoring calls; each call handles one evaluation tile"},
          {"registration_probability_calls",s.models.size()+meta_models.size()},{"registration_native_class_calls",s.models.size()+meta_models.size()},{"registration_rows_per_call",1}};
        if(runner.aggregate_oof){const auto tiles=runner.evaluation_batches.back().at("batches").get<L>();paired_calls["OOF_base_probability"]=multiply(multiply(tiles,s.oof_banks.size()),s.models.size());paired_calls["OOF_meta_probability"]=multiply(multiply(tiles,s.oof_banks.size()),meta_models.size());paired_calls["registration_probability_calls"]=add(s.models.size()+meta_models.size(),multiply(s.oof_banks.size(),s.models.size()));paired_calls["registration_native_class_calls"]=paired_calls.at("registration_probability_calls");}
        if(runner.candidate_mean){const auto tiles=runner.evaluation_batches.back().at("batches").get<L>();paired_calls["fixed_mean_members"]=meta_indices.size();paired_calls["fixed_mean_probability_accumulations"]=multiply(tiles,meta_indices.size());paired_calls["fixed_mean_duplicate_probability_calls"]=0;paired_calls["fixed_mean_probability_call_scope"]="no duplicate calls for mean accumulation; all-member and diagnostic calls are counted in meta_probability";paired_calls["ordinary_native_vs_probability_argmax_disagreements"]=argmax_differences.get();paired_calls["argmax_diagnostic_combiner_indices"]=ordinary_meta_indices;paired_calls["argmax_diagnostic_scope"]="ordinary selected union models; canonical HP indices; diagnostics do not change selection";}
        if(!runner.bootstrap.is_null()){const auto tiles=runner.evaluation_batches.back().at("batches").get<L>();const auto members=multiply(bagged.replicas.size(),bag_indices.size());paired_calls["bootstrap_meta_probability"]=multiply(tiles,members);paired_calls["registration_probability_calls"]=add(s.models.size()+meta_models.size(),members);paired_calls["registration_native_class_calls"]=paired_calls.at("registration_probability_calls");paired_calls["ordinary_native_vs_probability_argmax_disagreements"]=argmax_differences.get();paired_calls["argmax_diagnostic_combiner_indices"]=meta_indices;paired_calls["argmax_diagnostic_scope"]="ordinary control union models; canonical HP indices; diagnostics do not change selection";}

      }
      J result=identity;result.update(fit_receipt);result.update({{"selection",selection},{"baseline_model_sha256",s.models[best_base].model_sha256},{"combiner_model_sha256",model.model_sha256},{"metrics_order",metric_order(runner)},{"metrics",fold.json()}});results.push_back(result);completed_patients+=heldout.size();
      if(!choice.views.empty()||!choice.aggregated.empty()||!choice.blended.empty()||!choice.base_means.empty()){
        J bases=J::array(),metas=J::array();for(U b=0;b<s.models.size();++b)bases.push_back({{"baseline_index",b},{"model_sha256",s.models[b].model_sha256}});for(U m=0;m<meta_models.size();++m)metas.push_back({{"combiner_index",meta_indices[m]},{"model_sha256",meta_models[m].model_sha256}});
        auto& saved=results.back();saved["paired_outer_execution"]={{"all_base_models",bases},{"union_meta_models",metas},{"inner_native_fits",before_outer_fits-before_inner_fits},{"outer_refit_native_fits",runner.fits-before_outer_fits},{"prediction_calls",paired_calls},{"selection_frozen_before_outer_scoring",true},{"all_views_use_same_model_objects",true}};if(!runner.bootstrap.is_null()){J members=J::array();for(U r=0;r<bagged.replicas.size();++r)for(U m=0;m<bag_indices.size();++m)members.push_back({{"replicate_index",r},{"combiner_index",bag_indices[m]},{"model_sha256",bagged.replicas[r][m].model_sha256},{"native_model_bytes",bagged.replicas[r][m].model_json.size()}});saved["paired_outer_execution"]["union_bootstrap_meta_models"]=members;saved["paired_outer_execution"]["shared_full_refit_teacher_bank"]=true;saved["paired_outer_execution"]["model_sharing_scope"]="same method and HP across rule views; ordinary and bootstrap meta fits are distinct";}
        if(runner.candidate_mean){saved["paired_outer_execution"]["ordinary_selected_meta_indices"]=ordinary_meta_indices;saved["paired_outer_execution"]["fixed_mean_meta_indices"]=meta_indices;saved["paired_outer_execution"]["shared_ordinary_meta_models"]=true;}
        saved["rule_search_views"]=J::array();J aggregate=J::array();
        for(U v=0;v<choice.views.size();++v){const auto& c=choice.views[v];const auto m=std::lower_bound(meta_indices.begin(),meta_indices.end(),c.meta)-meta_indices.begin();saved["rule_search_views"].push_back({{"id",rule_views[v].at("id")},{"selection",c.details},{"baseline_model_sha256",s.models[c.baseline].model_sha256},{"combiner_model_sha256",meta_models[m].model_sha256},{"metrics_order",metric_order(runner)},{"metrics",view_folds[v]->json()}});aggregate.push_back({{"id",rule_views[v].at("id")},{"metrics_order",metric_order(runner)},{"metrics",view_totals[v]->json()}});}
        report["rule_search_views"]={{"version",1},{"primary_view","full"},{"additional_views",aggregate},{"outer_selection_of_view",false}};
      }
      if(!choice.base_means.empty()){
        const J order=J::array({"base_mean_only","augmented_inner_selected"});J members=J::array();L bytes=0;for(U b=0;b<s.models.size();++b){bytes=add(bytes,s.models[b].model_json.size());members.push_back({{"baseline_index",b},{"model_sha256",s.models[b].model_sha256},{"native_model_bytes",s.models[b].model_json.size()}});}
        J mode={{"version",1},{"method","uniform_probability_mean"},{"inference_mode","fixed_base_probability_mean"},{"selection",choice.base_means[0].details},{"base_models",members},{"native_model_bytes",bytes},{"arithmetic_policy",base_mean::arithmetic_policy},{"metrics_order",order},{"metrics",base_mean_folds[0]->json()},{"rule_search_views",J::array()},{"conversion_supported",false}};
        J aggregate_views=J::array();for(U v=1;v<choice.base_means.size();++v){mode["rule_search_views"].push_back({{"id",rule_views[v-1].at("id")},{"selection",choice.base_means[v].details},{"base_models",members},{"native_model_bytes",bytes},{"metrics_order",order},{"metrics",base_mean_folds[v]->json()}});aggregate_views.push_back({{"id",rule_views[v-1].at("id")},{"metrics_order",order},{"metrics",base_mean_totals[v]->json()}});}
        results.back()["base_candidate_aggregation"]=std::move(mode);auto& execution=results.back()["paired_outer_execution"];execution["prediction_calls"]["fixed_base_mean_additional_native_calls"]=0;execution["prediction_calls"]["fixed_base_mean_probability_reductions"]=runner.evaluation_batches.back().at("batches");execution["fixed_base_mean_uses_shared_base_probability_words"]=true;
        report["base_candidate_aggregation"]={{"version",1},{"method","uniform_probability_mean"},{"inference_mode","fixed_base_probability_mean"},{"metrics_order",order},{"metrics",base_mean_totals[0]->json()},{"rule_search_views",{{"version",1},{"primary_view","full"},{"additional_views",aggregate_views}}},{"outer_selection_of_mode",false},{"conversion_supported",false}};
      }
      if(!choice.blended.empty()){
        const J order=J::array({"blend_only","augmented_inner_selected"});const auto& primary=choice.blended[0];J mode={{"version",1},{"method","oof_brier_convex"},{"inference_mode","oof_brier_convex_blend"},{"selection",primary.details},{"coefficient_fit",blend_fit.receipt},{"blend_model_sha256",s.models[primary.baseline].model_sha256},{"selected_blend_coefficient",blend_fit.receipt.at("coefficients").at(primary.baseline)},{"arithmetic_policy",teacher_blend::arithmetic_policy},{"metrics_order",order},{"metrics",blend_folds[0]->json()},{"rule_search_views",J::array()},{"conversion_supported",false}};
        J aggregate_views=J::array();for(U v=1;v<choice.blended.size();++v){const auto& c=choice.blended[v];mode["rule_search_views"].push_back({{"id",rule_views[v-1].at("id")},{"selection",c.details},{"blend_model_sha256",s.models[c.baseline].model_sha256},{"selected_blend_coefficient",blend_fit.receipt.at("coefficients").at(c.baseline)},{"metrics_order",order},{"metrics",blend_folds[v]->json()}});aggregate_views.push_back({{"id",rule_views[v-1].at("id")},{"metrics_order",order},{"metrics",blend_totals[v]->json()}});}
        results.back()["teacher_probability_blend"]=std::move(mode);results.back()["paired_outer_execution"]["prediction_calls"]["blend_additional_native_calls"]=0;results.back()["paired_outer_execution"]["blend_uses_shared_base_probability_words"]=true;
        report["teacher_probability_blend"]={{"version",1},{"method","oof_brier_convex"},{"inference_mode","oof_brier_convex_blend"},{"metrics_order",order},{"metrics",blend_totals[0]->json()},{"rule_search_views",{{"version",1},{"primary_view","full"},{"additional_views",aggregate_views}}},{"outer_selection_of_mode",false},{"conversion_supported",false}};
      }
      if(runner.aggregate_oof){
        auto& saved=results.back();const auto& primary=choice.aggregated[0];auto meta_pin=[&](U m){return meta_models[std::lower_bound(meta_indices.begin(),meta_indices.end(),m)-meta_indices.begin()].model_sha256;};
        J mode={{"version",1},{"method","patient_weighted_probability_mean"},{"selection",primary.details},{"baseline_model_sha256",s.models[primary.baseline].model_sha256},{"combiner_model_sha256",meta_pin(primary.meta)},{"teacher_banks",bank_receipt(s.oof_banks)},{"arithmetic_policy",oof_aggregation::arithmetic_policy},{"metrics_order",metric_order(runner)},{"metrics",aggregation_folds[0]->json()},{"rule_search_views",J::array()},{"conversion_supported",false}};
        J aggregate_views=J::array();for(U v=1;v<choice.aggregated.size();++v){const auto& c=choice.aggregated[v];mode["rule_search_views"].push_back({{"id",rule_views[v-1].at("id")},{"selection",c.details},{"baseline_model_sha256",s.models[c.baseline].model_sha256},{"combiner_model_sha256",meta_pin(c.meta)},{"metrics_order",metric_order(runner)},{"metrics",aggregation_folds[v]->json()}});aggregate_views.push_back({{"id",rule_views[v-1].at("id")},{"metrics_order",metric_order(runner)},{"metrics",aggregation_totals[v]->json()}});}
        saved["oof_teacher_aggregation"]=std::move(mode);report["oof_teacher_aggregation"]={{"version",1},{"method","patient_weighted_probability_mean"},{"metrics_order",metric_order(runner)},{"metrics",aggregation_totals[0]->json()},{"rule_search_views",{{"version",1},{"primary_view","full"},{"additional_views",aggregate_views}}},{"outer_selection_of_mode",false},{"conversion_supported",false}};
      }
      if(!runner.bootstrap.is_null()){
        auto& saved=results.back();const auto& primary=choice.aggregated[0];
        auto members=[&](U index){J pins=J::array();const auto m=std::lower_bound(bag_indices.begin(),bag_indices.end(),index)-bag_indices.begin();need(m<bag_indices.size()&&bag_indices[m]==index,"bootstrap selected meta absent from union");for(U r=0;r<bagged.replicas.size();++r)pins.push_back({{"replicate_index",r},{"model_sha256",bagged.replicas[r][m].model_sha256},{"native_model_bytes",bagged.replicas[r][m].model_json.size()}});return pins;};
        J mode={{"version",1},{"replicates",runner.bootstrap.at("replicates")},{"seed",runner.bootstrap.at("seed")},{"selection",primary.details},{"baseline_model_sha256",s.models[primary.baseline].model_sha256},{"combiner_models",members(primary.meta)},{"bootstrap_fits",bagged.receipts},{"arithmetic_policy",oof_aggregation::arithmetic_policy},{"metrics_order",metric_order(runner)},{"metrics",aggregation_folds[0]->json()},{"rule_search_views",J::array()},{"conversion_supported",false}};
        J aggregate_views=J::array();for(U v=1;v<choice.aggregated.size();++v){const auto& c=choice.aggregated[v];mode["rule_search_views"].push_back({{"id",rule_views[v-1].at("id")},{"selection",c.details},{"baseline_model_sha256",s.models[c.baseline].model_sha256},{"combiner_models",members(c.meta)},{"metrics_order",metric_order(runner)},{"metrics",aggregation_folds[v]->json()}});aggregate_views.push_back({{"id",rule_views[v-1].at("id")},{"metrics_order",metric_order(runner)},{"metrics",aggregation_totals[v]->json()}});}
        saved["meta_patient_bootstrap"]=std::move(mode);report["meta_patient_bootstrap"]={{"version",1},{"replicates",runner.bootstrap.at("replicates")},{"seed",runner.bootstrap.at("seed")},{"metrics_order",metric_order(runner)},{"metrics",aggregation_totals[0]->json()},{"rule_search_views",{{"version",1},{"primary_view","full"},{"additional_views",aggregate_views}}},{"outer_selection_of_mode",false},{"conversion_supported",false}};
      }
      if(runner.candidate_mean){
        auto& saved=results.back();const auto& primary=choice.aggregated[0];J members=J::array();for(U m=0;m<meta_models.size();++m)members.push_back({{"combiner_index",meta_indices[m]},{"model_sha256",meta_models[m].model_sha256},{"native_model_bytes",meta_models[m].model_json.size()}});
        J mode={{"version",1},{"method","uniform_probability_mean"},{"combiner_indices",meta_indices},{"selection",primary.details},{"baseline_model_sha256",s.models[primary.baseline].model_sha256},{"combiner_models",members},{"arithmetic_policy",oof_aggregation::arithmetic_policy},{"metrics_order",metric_order(runner)},{"metrics",aggregation_folds[0]->json()},{"rule_search_views",J::array()},{"conversion_supported",false}};
        J aggregate_views=J::array();for(U v=1;v<choice.aggregated.size();++v){const auto& c=choice.aggregated[v];mode["rule_search_views"].push_back({{"id",rule_views[v-1].at("id")},{"selection",c.details},{"baseline_model_sha256",s.models[c.baseline].model_sha256},{"combiner_models",members},{"metrics_order",metric_order(runner)},{"metrics",aggregation_folds[v]->json()}});aggregate_views.push_back({{"id",rule_views[v-1].at("id")},{"metrics_order",metric_order(runner)},{"metrics",aggregation_totals[v]->json()}});}
        saved["meta_candidate_aggregation"]=std::move(mode);report["meta_candidate_aggregation"]={{"version",1},{"method","uniform_probability_mean"},{"combiner_indices",meta_indices},{"metrics_order",metric_order(runner)},{"metrics",aggregation_totals[0]->json()},{"rule_search_views",{{"version",1},{"primary_view","full"},{"additional_views",aggregate_views}}},{"outer_selection_of_mode",false},{"conversion_supported",false}};
      }
      if(plan.value("outer_model_summaries",false)){
        model_summary::Collector summaries;const auto raw_mapping=model_summary::raw_mapping(data.descriptor,data.F);std::vector<std::string> teacher_pins;for(const auto& teacher:s.models)teacher_pins.push_back(teacher.model_sha256);
        const auto meta_mapping=model_summary::meta_mapping(raw_mapping,teacher_pins,2,runner.meta_raw);
        auto add_base=[&](U b,J use){use["input_mapping"]="raw";use["hyperparameter_index"]=b;summaries.add(s.models[b].model_json,s.models[b].model_sha256,std::move(use));};
        add_base(best_base,{{"role","baseline_winner"},{"view","full"}});
        for(U v=0;v<choice.views.size();++v)add_base(choice.views[v].baseline,{{"role","baseline_winner"},{"view",rule_views[v].at("id")}});
        for(U m=0;m<meta_models.size();++m){const auto& current=meta_models[m];J views=J::array();if(meta_indices[m]==best_meta)views.push_back("full");for(U v=0;v<choice.views.size();++v)if(choice.views[v].meta==meta_indices[m])views.push_back(rule_views[v].at("id"));
          summaries.add(current.model_json,current.model_sha256,{{"role","combiner_winner"},{"views",views},{"input_mapping","meta"},{"hyperparameter_index",meta_indices[m]}});
          std::set<U> dependencies;for(const auto& feature:summaries.models.at(current.model_sha256).at("features")){const U f=feature.at("feature_index");if(f<s.models.size()*2)dependencies.insert(f/2);}
          for(U b:dependencies)add_base(b,{{"role","combiner_teacher_dependency"},{"used_by_meta_sha256",current.model_sha256}});
        }
        auto diagnostics=summaries.finish({{"raw",raw_mapping},{"meta",meta_mapping}});
        need(diagnostics.dump().size()<=plan.value("host_byte_budget",16ULL<<30),"outer model summaries exceed host byte budget");results.back()["model_summaries"]=std::move(diagnostics);
      }
      report["actual_CUB_workspace_bytes"]=data.maximum_subset_workspace_bytes;report["evaluation_batches"]=runner.evaluation_batches;if(!data.sampling.is_null())report["fit_sampling_receipts"]=runner.sampling_receipts;report["outer_results"]=results;report["completed_outer_folds"]=results.size();report["completed_outer_patients"]=completed_patients;report["completed_native_fits"]=runner.fits;if(!runner.training_loss.is_null())report["completed_group_loss_fits"]=runner.custom_fits;report["metrics_order"]=metric_order(runner);report["metrics"]=total.json();report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();publish(output,report);
      // Native outcomes are durable before conversion. Resource refusals never
      // remove an outer patient or retrospectively change its selected family.
      if(convert_outer){
        auto& result=results.back();result["deployment_conversions"]=J::object();
        const std::string prefix=plan.contains("outer_folds")?"outer-fold-"+std::to_string(outer_fold):"outer-"+std::to_string(heldout[0]);
        const std::vector<TrainingResult> selected_base={s.models[best_base]};
        for(bool composed:{false,true}){
          if(stopping)break;
          const auto& sources=composed?s.models:selected_base;
          auto conversion_started=std::chrono::steady_clock::now();auto receipt=attempt_deployment(runner.plan,sources,composed?&model:nullptr,*runner.session,deployment_limits,output,prefix+(composed?"-composed":"-baseline"),stopping);outer_conversion_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-conversion_started).count();
          receipt.update(identity);receipt["selection_frozen_before_outer_scoring"]=true;receipt["native_outer_metrics_retained_on_refusal"]=true;
          result["deployment_conversions"][composed?"nonlinear_combiner":"native_baseline"]=receipt;
          report["conversion_performed"]=true;
          report["deployment_conversion_attempts"]=report.at("deployment_conversion_attempts").get<L>()+1;
          if(receipt.at("complete").get<bool>())report["completed_deployment_conversions"]=report.at("completed_deployment_conversions").get<L>()+1;
          if(receipt.at("accepted_native_class_root").get<bool>())report["completed_qualified_deployment_conversions"]=report.at("completed_qualified_deployment_conversions").get<L>()+1;
          report["outer_results"]=results;publish(output,report);
          J progress=identity;progress["event"]="grouped_segmentation_deployment_conversion_complete";progress["result"]=receipt;std::cout<<progress.dump()<<'\n'<<std::flush;
        }
      }
      J progress=identity;progress.update({{"event","grouped_segmentation_outer_complete"},{"completed_outer_folds",results.size()},{"requested_outer_folds",groups.size()},{"completed_outer_patients",completed_patients},{"requested_outer_patients",outer.size()},{"metrics",fold.json()}});if(heldout.size()==1)progress["patient_id"]=heldout[0];std::cout<<progress.dump()<<'\n'<<std::flush;
    }
    runner.interrupted();report["native_evaluation_complete"]=true;report["full_cohort_outer_evaluation"]=completed_patients==data.patients.size();report["native_evaluation_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()-outer_conversion_seconds;report["outer_deployment_conversion_seconds"]=outer_conversion_seconds;
    report["all_requested_deployment_conversions_complete"]=convert_outer&&report.at("completed_deployment_conversions").get<L>()==2ULL*groups.size();report["all_requested_deployment_conversions_qualified"]=convert_outer&&report.at("completed_qualified_deployment_conversions").get<L>()==2ULL*groups.size();publish(output,report);
    if(plan.contains("final_deployment"))final_deployment(plan,data,runner,output,report,stopping);
    report["complete"]=true;report["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();publish(output,report);return stopping?2:0;
  }catch(const std::exception& e){report["actual_CUB_workspace_bytes"]=data.maximum_subset_workspace_bytes;report["evaluation_batches"]=runner.evaluation_batches;if(!data.sampling.is_null())report["fit_sampling_receipts"]=runner.sampling_receipts;report["outer_results"]=results;report["completed_outer_folds"]=results.size();report["completed_outer_patients"]=completed_patients;report["completed_native_fits"]=runner.fits;if(!runner.training_loss.is_null())report["completed_group_loss_fits"]=runner.custom_fits;report["error"]=e.what();report["interrupted"]=bool(stopping);publish(output,report);if(stopping)return 2;throw;}
}
} // namespace class_study::grouped_segmentation
