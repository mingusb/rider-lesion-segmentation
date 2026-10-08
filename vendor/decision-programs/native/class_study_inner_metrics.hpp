#pragma once
// Host-only lossless serialization of already-scored inner validation counts.
// Training, prediction, candidate selection and outer metrics do not use this codec.
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace class_study::inner_metrics {
using J=nlohmann::json;
using Count=unsigned long long;
using Id=std::uint32_t;
inline constexpr const char* format="grouped-segmentation-inner-counts-1";
inline void require(bool b,const char* s){if(!b)throw std::invalid_argument(s);}
inline Count integer(const J& j,Count maximum=std::numeric_limits<Count>::max()){
  require(j.is_number_unsigned()||(j.is_number_integer()&&j.get<std::int64_t>()>=0),"inner metric integer type");
  const auto v=j.get<Count>();require(v<=maximum,"inner metric integer range");return v;
}
inline Count add(Count a,Count b){require(a<=std::numeric_limits<Count>::max()-b,"inner metric count overflow");return a+b;}
inline std::string encoding(const J& plan){
  if(!plan.contains("inner_result_encoding"))return "full";
  const auto& v=plan.at("inner_result_encoding");
  require(v.is_string()&&(v=="full"||v=="scan_counts_v1"),"inner_result_encoding must be full or scan_counts_v1");
  return v.get<std::string>();
}
struct Scan {Id id,patient;Count outside;};
struct Axes {std::vector<Id> patients;std::vector<Scan> scans;};
inline Axes axes(const J& j){
  require(j.is_object()&&j.at("format")==format&&j.at("integer_encoding")=="json-u64", "unknown inner metric encoding");
  require(j.at("count_order")==J::array({"TP","FP","FN_in_ROI","TN_in_ROI"}),"inner metric count order");
  require(j.at("patient_ids").is_array()&&!j.at("patient_ids").empty()&&j.at("scans").is_array()&&!j.at("scans").empty(),"empty inner metric axes");
  Axes a;std::set<Id> patients,scans,seen;
  for(const auto& value:j.at("patient_ids")){auto id=Id(integer(value,UINT32_MAX));require(patients.insert(id).second,"duplicate inner patient");a.patients.push_back(id);}
  for(const auto& s:j.at("scans")){
    Scan v{Id(integer(s.at("scan_id"),UINT32_MAX)),Id(integer(s.at("patient_id"),UINT32_MAX)),integer(s.at("FN_outside_ROI"))};
    require(scans.insert(v.id).second&&patients.contains(v.patient),"invalid inner scan identity");seen.insert(v.patient);a.scans.push_back(v);
  }
  require(seen==patients,"inner patient without scans");return a;
}
inline J make_axes(const std::vector<Id>& patients,const std::vector<Id>& scan_ids,
                   const std::vector<Id>& scan_patient_ids,const std::vector<Count>& outside){
  require(scan_ids.size()==scan_patient_ids.size()&&scan_ids.size()==outside.size(),"inner scan axis shape");
  const std::set<Id> active(patients.begin(),patients.end());J scans=J::array();
  for(std::size_t i=0;i<scan_ids.size();++i)if(active.contains(scan_patient_ids[i]))
    scans.push_back({{"scan_id",scan_ids[i]},{"patient_id",scan_patient_ids[i]},{"FN_outside_ROI",outside[i]}});
  J j={{"format",format},{"integer_encoding","json-u64"},{"count_order",J::array({"TP","FP","FN_in_ROI","TN_in_ROI"})},{"patient_ids",patients},{"scans",std::move(scans)}};
  axes(j);return j;
}
inline J encode(const J& metadata,std::span<const Id> all_scan_ids,
                std::span<const double> scores,std::span<const Count> counts){
  const auto a=axes(metadata);require(!scores.empty()&&!all_scan_ids.empty(),"empty inner count matrix");
  require(all_scan_ids.size()<=SIZE_MAX/4&&scores.size()<=SIZE_MAX/(4*all_scan_ids.size())&&counts.size()==scores.size()*all_scan_ids.size()*4,"inner count matrix shape");
  std::map<Id,std::size_t> positions;std::set<Id> active;
  for(std::size_t i=0;i<all_scan_ids.size();++i)require(positions.emplace(all_scan_ids[i],i).second,"duplicate dense scan ID");
  for(const auto& s:a.scans){require(positions.contains(s.id),"inner scan missing from dense counts");active.insert(s.id);}
  std::vector<Count> positive(all_scan_ids.size()),negative(all_scan_ids.size());J result=J::array();
  for(std::size_t c=0;c<scores.size();++c){
    require(std::isfinite(scores[c])&&scores[c]>=0&&scores[c]<=1,"invalid saved inner score");
    for(std::size_t s=0;s<all_scan_ids.size();++s){const auto q=(c*all_scan_ids.size()+s)*4;
      const Count p=add(counts[q],counts[q+2]),n=add(counts[q+1],counts[q+3]),total=add(p,n);
      require(bool(total)==active.contains(all_scan_ids[s]),"inner candidate scan support differs");
      if(c)require(p==positive[s]&&n==negative[s],"inner candidate truth support differs");else{positive[s]=p;negative[s]=n;}
    }
    J flat=J::array();flat.get_ref<J::array_t&>().reserve(a.scans.size()*4);
    for(const auto& s:a.scans){const auto q=(c*all_scan_ids.size()+positions.at(s.id))*4;for(unsigned k=0;k<4;++k)flat.push_back(counts[q+k]);}
    result.push_back({{"macro_patient_Dice",scores[c]},{"scan_counts",std::move(flat)}});
  }
  return result;
}
inline J decode(const J& metric,const J& metadata){
  const auto a=axes(metadata);require(metric.is_object()&&!metric.contains("patients")&&!metric.contains("scans"),"mixed full/compact inner metric");
  const auto& score=metric.at("macro_patient_Dice");require(score.is_number()&&std::isfinite(score.get<double>())&&score.get<double>()>=0&&score.get<double>()<=1,"invalid saved inner score");
  const auto& flat=metric.at("scan_counts");require(flat.is_array()&&a.scans.size()<=SIZE_MAX/4&&flat.size()==a.scans.size()*4,"inner count vector shape");
  J scans=J::array(),patients=J::array();
  struct Pooled {Count tp=0,fp=0,fn=0,tn=0;std::vector<Count> predicted,reference;};std::map<Id,Pooled> pooled;
  for(std::size_t i=0;i<a.scans.size();++i){
    const auto& s=a.scans[i];const Count tp=integer(flat[4*i]),fp=integer(flat[4*i+1]),fn=integer(flat[4*i+2]),tn=integer(flat[4*i+3]);
    require(add(add(tp,fp),add(fn,tn))>0,"empty active inner scan");
    const Count pv=add(tp,fp),tv=add(add(tp,fn),s.outside),den=add(pv,tv);
    scans.push_back({{"scan_id",s.id},{"patient_id",s.patient},{"TP",tp},{"FP",fp},{"FN_in_ROI",fn},{"FN_outside_ROI",s.outside},{"TN_in_ROI",tn},{"Dice",den?2*double(tp)/double(den):1.0},{"predicted_positive_voxels",pv},{"reference_positive_voxels",tv}});
    auto& v=pooled[s.patient];v.tp=add(v.tp,tp);v.fp=add(v.fp,fp);v.fn=add(v.fn,add(fn,s.outside));v.tn=add(v.tn,tn);v.predicted.push_back(pv);v.reference.push_back(tv);
  }
  double total=0;
  for(auto id:a.patients){const auto& p=pooled.at(id);const double den=2*double(p.tp)+double(p.fp)+double(p.fn),dice=den?2*double(p.tp)/den:1.0;total+=dice;
    auto repeat=[](const std::vector<Count>& v)->J{if(v.size()!=2)return nullptr;const auto den=add(v[0],v[1]);return den?J(2*std::fabs(double(v[0])-double(v[1]))/double(den)):J(nullptr);};
    patients.push_back({{"patient_id",id},{"Dice",dice},{"TP",p.tp},{"FP",p.fp},{"FN",p.fn},{"TN_in_ROI",p.tn},{"both_empty_repeat_predictions",p.predicted.size()==2&&!p.predicted[0]&&!p.predicted[1]},{"predicted_absolute_symmetric_repeat_volume_difference",repeat(p.predicted)},{"reference_absolute_symmetric_repeat_volume_difference",repeat(p.reference)}});
  }
  require(std::fabs(total/double(a.patients.size())-score.get<double>())<=1e-10,"inner macro score/count mismatch");
  return {{"macro_patient_Dice",score},{"patients",std::move(patients)},{"scans",std::move(scans)}};
}
inline J decode_selection(const J& selection){
  if(!selection.contains("inner_metric_encoding")){
    for(const char* key:{"inner_baseline_results","inner_combiner_results","reference_inner_results"})if(selection.contains(key))
      for(const auto& metric:selection.at(key))require(!metric.contains("scan_counts"),"compact inner metric missing shared encoding");
    return selection;
  }
  J result=selection;const auto& meta=selection.at("inner_metric_encoding");axes(meta);
  for(const char* key:{"inner_baseline_results","inner_combiner_results","reference_inner_results"})if(result.contains(key)){
    auto& rows=result.at(key);require(rows.is_array()&&!rows.empty(),"empty inner candidate family");for(auto& metric:rows)metric=decode(metric,meta);
  }
  result.erase("inner_metric_encoding");return result;
}
} // namespace class_study::inner_metrics
