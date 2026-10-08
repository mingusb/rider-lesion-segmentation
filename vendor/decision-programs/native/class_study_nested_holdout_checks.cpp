#include "class_study_nested_holdout.hpp"
#include <cassert>
#include <iostream>
using namespace class_study::nested_holdout;
template<class F> void rejects(F f) { bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}assert(rejected); }
int main(){
  J dev={{"features",2},{"classes",3},{"rows",30},{"values_sha256","development"}};
  J fresh={{"features",2},{"classes",3},{"rows",7},{"values_sha256","fresh"}};
  J option={{"depth",3},{"dataset",fresh}};
  const auto config=configuration(option,dev);
  assert(config.at("partitions")==J::array({J{{"offset",0},{"rows",3}},J{{"offset",3},{"rows",2}},J{{"offset",5},{"rows",2}}}));
  rejects([&]{auto o=option;o["depth"]=0;configuration(o,dev);});
  rejects([&]{auto o=option;o["depth"]=8;configuration(o,dev);});
  rejects([&]{auto o=option;o["depth"]=-1;configuration(o,dev);});
  rejects([&]{auto o=option;o["dataset"]["values_sha256"]="development";configuration(o,dev);});
  rejects([&]{auto o=option;o["dataset"]["features"]=3;configuration(o,dev);});
  {auto grouped=option;grouped["partition_rows"]=J::array({1,4,2});
   assert(configuration(grouped,dev).at("partitions").at(2).at("offset")==5);}
  rejects([&]{auto o=option;o["partition_rows"]=J::array({1,4,1});configuration(o,dev);});
  rejects([&]{auto o=option;o["partition_rows"]=J::array({1,4,3});configuration(o,dev);});
  rejects([&]{auto o=option;o["partition_rows"]=J::array({0,5,2});configuration(o,dev);});
  rejects([&]{auto o=option;o["partition_rows"]=J::array({1,6});configuration(o,dev);});
  const J evaluation={{"native_library_sha256","qualified-library"}};
  const J winner={{"model_sha256","frozen-challenger"},{"evaluation",evaluation}},
          baseline={{"model_sha256","declared-baseline"},{"evaluation",evaluation}};
  auto state=frozen(config,2,winner,0,baseline);
  auto score=[&](unsigned offset,unsigned rows,unsigned a,unsigned b){return J{
    {"offset",offset},{"rows",rows},{"candidate_errors",a},{"baseline_errors",b},
    {"accepted",a<b},{"CUDA_computed",true},
    {"candidate_source_sha256",winner.at("model_sha256")},
    {"baseline_source_sha256",baseline.at("model_sha256")},
    {"candidate_native_library_sha256",evaluation.at("native_library_sha256")},
    {"baseline_native_library_sha256",evaluation.at("native_library_sha256")}};};
  assert(state.at("gate_data_used_for_selection")==false);
  record(state,score(0,3,0,1));
  assert(state.at("gate_data_used_for_selection")==true);
  // A persisted successful prefix resumes only with the same candidate and data.
  const auto restored=J::parse(state.dump());
  validate_state(restored,config,2,winner,0,baseline);
  rejects([&]{auto changed=winner;changed["model_sha256"]="runner-up";validate_state(restored,config,1,changed,0,baseline);});
  rejects([&]{auto changed=config;changed["dataset"]["values_sha256"]="changed";validate_state(restored,changed,2,winner,0,baseline);});
  // A tie is no confirmed improvement. No runner-up or later gate may be tried.
  record(state,score(3,2,1,1));
  assert(state.at("status")=="baseline_retained"&&state.at("final_selected_index")==0&&state.at("gates").size()==2);
  rejects([&]{record(state,score(5,2,0,2));});
  validate_state(state,config,2,winner,0,baseline);
  rejects([&]{auto altered=state;altered["final_selected_index"]=2;validate_state(altered,config,2,winner,0,baseline);});
  rejects([&]{auto altered=state;altered["gates"][0]["offset"]=5;validate_state(altered,config,2,winner,0,baseline);});
  auto passed=frozen(config,2,winner,0,baseline);
  record(passed,score(0,3,0,1));record(passed,score(3,2,0,1));record(passed,score(5,2,0,1));
  assert(passed.at("status")=="candidate_accepted"&&passed.at("final_selected_index")==2);
  validate_state(passed,config,2,winner,0,baseline);
  auto unchanged=frozen(config,0,baseline,0,baseline);
  assert(unchanged.at("status")=="baseline_already_selected"&&unchanged.at("gates").empty()&&
         unchanged.at("gate_data_used_for_selection")==false);
  rejects([&]{record(unchanged,score(0,3,0,1));});
  J plan={{"nested_holdout",option}},donor={{"nested_holdout",passed}};
  rejects([&]{reject_spent_donor(plan,donor);});
  plan["nested_holdout"]["dataset"]["values_sha256"]="new-independent-data";
  reject_spent_donor(plan,donor);
  // A device result must be consistent; host replay cannot bless a forged gain.
  auto invalid=frozen(config,2,winner,0,baseline);auto bad=score(0,3,1,1);bad["accepted"]=true;
  rejects([&]{record(invalid,bad);});
  // A valid-looking gain from another interval/model/library cannot confirm the
  // frozen pair. Refusal leaves the gate unconsumed and its cursor unchanged.
  for(const char* field:{"offset","candidate_source_sha256","baseline_source_sha256",
                        "candidate_native_library_sha256","baseline_native_library_sha256"}){
    auto wrong=score(0,3,0,1);
    if(std::string(field)=="offset")wrong[field]=3;else wrong[field]="different-source";
    const auto before=invalid;
    rejects([&]{record(invalid,wrong);});
    assert(invalid==before);
  }
  rejects([&]{auto altered=restored;altered["gates"][0]["candidate_source_sha256"]="runner-up";validate_state(altered,config,2,winner,0,baseline);});
  std::cout<<"nested holdout partition, frozen pair, fallback, resume and reuse checks passed\n";
}
