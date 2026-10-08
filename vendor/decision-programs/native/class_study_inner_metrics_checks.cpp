#include "class_study_inner_metrics.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <functional>
using namespace class_study::inner_metrics;
namespace {
unsigned assertions=0,refusals=0,candidates=0,derived_float_differences=0;
double max_float_difference=0;
void check(bool value,const char* message){++assertions;if(!value)throw std::runtime_error(message);}
void deny(const std::function<void()>& fn){try{fn();}catch(const std::exception&){++refusals;return;}throw std::runtime_error("malformed compact payload accepted");}
void equal(const J& a,const J& b,const std::string& key=""){
  if(a.is_number_float()||b.is_number_float()){
    check(a.is_number()&&b.is_number(),"decoded numeric type differs");double x=a.get<double>(),y=b.get<double>();
    if(key=="macro_patient_Dice")check(std::bit_cast<std::uint64_t>(x)==std::bit_cast<std::uint64_t>(y),"saved macro score words changed");
    else{const auto error=std::fabs(x-y);check(error<=1e-12,"derived metric differs");if(error)++derived_float_differences;max_float_difference=std::max(max_float_difference,error);}return;
  }
  check(a.type()==b.type()||(a.is_number_integer()&&b.is_number_integer()),"decoded value type differs");
  if(a.is_object()){check(a.size()==b.size(),"decoded object extent differs");for(const auto& [k,v]:a.items()){check(b.contains(k),"decoded object key missing");equal(v,b.at(k),k);}}
  else if(a.is_array()){check(a.size()==b.size(),"decoded array extent differs");for(std::size_t i=0;i<a.size();++i)equal(a[i],b[i],key);}
  else check(a==b,"decoded integer/identity/order differs");
}
void synthetic(){
  check(encoding(J::object())=="full"&&encoding({{"inner_result_encoding","full"}})=="full","legacy default changed");
  check(encoding({{"inner_result_encoding","scan_counts_v1"}})=="scan_counts_v1","compact enum refused");
  for(const auto& v:std::vector<J>{J(),true,7,"future"})deny([&]{encoding({{"inner_result_encoding",v}});});
  std::vector<Id> ids{5,7,11,18,24,45,60},pids{2,9,9,77,77,77,999};
  auto meta=make_axes({2,9,77},ids,pids,{0,1,0,0,4,0,6});
  std::vector<Count> counts{1,1,0,8, 0,0,0,10, 0,0,0,7, 2,0,1,4, 0,3,2,5, 1,0,0,10, 0,0,0,0};
  auto twice=counts;twice.insert(twice.end(),counts.begin(),counts.end());const double score=(2.0/3+6.0/16)/3;
  auto rows=encode(meta,ids,std::vector<double>{score,score},twice);check(rows.size()==2&&rows[0]==rows[1],"equal candidate order changed");
  auto metric=decode(rows[0],meta);check(metric["patients"][1]["both_empty_repeat_predictions"]==true&&metric["patients"][1]["predicted_absolute_symmetric_repeat_volume_difference"].is_null(),"empty repeated scans differ");
  check(metric["patients"][0]["predicted_absolute_symmetric_repeat_volume_difference"].is_null()&&metric["patients"][2]["predicted_absolute_symmetric_repeat_volume_difference"].is_null(),"one/many visit repeat metric differs");
  check(metric["patients"][2]["FN"]==7&&metric["scans"][4]["FN_outside_ROI"]==4,"outside FN lost");
  J selection={{"inner_metric_encoding",meta},{"inner_baseline_results",rows},{"inner_combiner_results",rows},{"reference_inner_results",rows},{"baseline_index",1},{"selected_family","native_baseline"}};
  auto missing_axes=selection;missing_axes.erase("inner_metric_encoding");deny([&]{decode_selection(missing_axes);});
  const auto expanded=decode_selection(selection);check(expanded["baseline_index"]==1&&expanded["selected_family"]=="native_baseline","frozen selection changed");check(!expanded.contains("inner_metric_encoding"),"decoder retained encoding marker");equal(expanded,decode_selection(expanded));
  for(const auto& key:{"inner_baseline_results","inner_combiner_results","reference_inner_results"})for(const auto& m:expanded[key]){equal(m,metric);++candidates;}
  auto bad=meta;bad["format"]="future";deny([&]{decode(rows[0],bad);});bad=meta;bad["count_order"][0]="FP";deny([&]{decode(rows[0],bad);});
  bad=meta;bad["patient_ids"].push_back(2);deny([&]{decode(rows[0],bad);});bad=meta;bad["scans"][1]["scan_id"]=5;deny([&]{decode(rows[0],bad);});bad=meta;bad["scans"][0]["patient_id"]=42;deny([&]{decode(rows[0],bad);});
  for(const auto& v:std::vector<J>{true,-1,1.5,"2"}){bad=meta;bad["scans"][0]["FN_outside_ROI"]=v;deny([&]{decode(rows[0],bad);});}
  for(const auto& v:std::vector<J>{true,-1,1.5,"2"}){bad=rows[0];bad["scan_counts"][0]=v;deny([&]{decode(bad,meta);});}
  bad=rows[0];bad["scan_counts"].erase(bad["scan_counts"].begin());deny([&]{decode(bad,meta);});bad=rows[0];bad["patients"]=J::array();deny([&]{decode(bad,meta);});bad=rows[0];bad["macro_patient_Dice"]=.99;deny([&]{decode(bad,meta);});
  bad=rows[0];for(unsigned i=0;i<4;++i)bad["scan_counts"][i]=0;deny([&]{decode(bad,meta);});
  auto changed=twice;changed.back()=1;deny([&]{encode(meta,ids,std::vector<double>{score,score},changed);});changed=twice;changed[counts.size()]+=1;deny([&]{encode(meta,ids,std::vector<double>{score,score},changed);});
  const Count large=(Count(1)<<53)+3;const auto single=make_axes({9},{8},{9},{0});auto large_row=encode(single,std::vector<Id>{8},std::vector<double>{1},std::vector<Count>{0,0,0,large});check(decode(large_row[0],single)["scans"][0]["TN_in_ROI"]==large,"uint64 rounded through double");
  bad=large_row[0];bad["scan_counts"][0]=std::numeric_limits<Count>::max();deny([&]{decode(bad,single);});
}
void roundtrip(const char* file){
  std::ifstream in(file);check(bool(in),"roundtrip source missing");J report;in>>report;check(report.at("complete")==true,"roundtrip requires completed result");
  for(const auto& fold:report.at("outer_results")){
    const auto& before=fold.at("selection");check(!before.contains("inner_metric_encoding"),"roundtrip fixture must start full");const auto& first=before.at("inner_baseline_results")[0];
    std::vector<Id> patients,ids,pids;std::vector<Count> outside;
    for(const auto& row:first.at("patients"))patients.push_back(Id(integer(row.at("patient_id"),UINT32_MAX)));
    for(const auto& row:first.at("scans")){ids.push_back(Id(integer(row.at("scan_id"),UINT32_MAX)));pids.push_back(Id(integer(row.at("patient_id"),UINT32_MAX)));outside.push_back(integer(row.at("FN_outside_ROI")));}
    const auto meta=make_axes(patients,ids,pids,outside);J compact=before;compact["inner_metric_encoding"]=meta;
    for(const char* key:{"inner_baseline_results","inner_combiner_results","reference_inner_results"})if(before.contains(key)){
      std::vector<double> scores;std::vector<Count> counts;
      for(const auto& metric:before.at(key)){scores.push_back(metric.at("macro_patient_Dice").get<double>());check(metric.at("scans").size()==ids.size(),"fixture scan support differs");std::size_t i=0;
        for(const auto& row:metric.at("scans")){check(row.at("scan_id")==ids[i++],"fixture scan order differs");for(const char* k:{"TP","FP","FN_in_ROI","TN_in_ROI"})counts.push_back(integer(row.at(k)));}++candidates;}
      compact[key]=encode(meta,ids,scores,counts);
    }
    equal(before,decode_selection(compact));
  }
}
}
int main(int argc,char** argv)try{
  synthetic();if(argc==3&&std::string(argv[1])=="--roundtrip")roundtrip(argv[2]);else if(argc!=1)throw std::invalid_argument("usage: checks [--roundtrip completed-result.json]");
  std::cout<<J{{"complete",true},{"GPU_used",false},{"assertions",assertions},{"malformed_payload_refusals",refusals},{"candidate_roundtrips",candidates},{"derived_float_differences",derived_float_differences},{"maximum_derived_float_abs_difference",max_float_difference}}.dump()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
