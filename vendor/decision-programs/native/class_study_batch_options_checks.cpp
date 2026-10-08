#include "class_study_options.hpp"
#include <iostream>

int main() {
  try {
    using J=nlohmann::json;
    unsigned checks=0;
    auto expect=[&](bool ok){++checks;if(!ok)throw std::runtime_error("batch option contract failure");};
    for (auto features : {1u,13u,54u,784u}) {
      auto automatic=class_study::parse_conversion_options(J::object(),features);
      expect(automatic.batch_size==0&&automatic.max_batch_size==0);
      expect(automatic.admission_threads==0);
      expect(automatic.draft_threads==0);
      expect(automatic.split_policy=="source_order");
      expect(automatic.oldest_ready_jobs==0);
      for(std::uint32_t count:{0u,1u,256u,UINT32_MAX}) {
        auto choice=class_study::parse_conversion_options(J{{"oldest_ready_jobs",count}},features);
        expect(choice.oldest_ready_jobs==count);
        auto restored=class_study::parse_conversion_options(class_study::describe_conversion_options(choice),features);
        expect(restored.oldest_ready_jobs==count);
      }
      for(const char* name:{"source_order","widest_residual","aggregate_residual","contracting_residual"}) {
        auto choice=class_study::parse_conversion_options(J{{"split_policy",name}},features);
        expect(choice.split_policy==name);
        auto restored=class_study::parse_conversion_options(class_study::describe_conversion_options(choice),features);
        expect(restored.split_policy==name);
      }
      expect(!automatic.completed_cache_limit&&!automatic.refinement_visit_budget);
      const auto explicit_auto=class_study::parse_conversion_options(J{{"completed_cache_limit",nullptr},{"refinement_visit_budget",nullptr}},features);
      expect(!explicit_auto.completed_cache_limit&&!explicit_auto.refinement_visit_budget);
      const auto auto_description=class_study::describe_conversion_options(explicit_auto);
      expect(auto_description["completed_cache_limit"].is_null()&&auto_description["refinement_visit_budget"].is_null());
      for(auto limit:{0u,1u,32u,UINT32_MAX}){
        const auto policies=class_study::parse_conversion_options(J{{"completed_cache_limit",limit},{"refinement_visit_budget",limit}},features);
        expect(policies.completed_cache_limit&&policies.refinement_visit_budget&&*policies.completed_cache_limit==limit&&*policies.refinement_visit_budget==limit);
        const auto policies_roundtrip=class_study::parse_conversion_options(class_study::describe_conversion_options(policies),features);
        expect(policies_roundtrip.completed_cache_limit==policies.completed_cache_limit&&policies_roundtrip.refinement_visit_budget==policies.refinement_visit_budget);
      }
      auto bounded=class_study::parse_conversion_options(J{{"max_batch_size",4096}},features);
      expect(bounded.batch_size==0&&bounded.max_batch_size==4096);
      auto fixed=class_study::parse_conversion_options(J{{"batch_size",1024},{"max_batch_size",4096}},features);
      expect(fixed.batch_size==1024&&fixed.max_batch_size==4096);
      auto roundtrip=class_study::parse_conversion_options(class_study::describe_conversion_options(fixed),features);
      expect(roundtrip.batch_size==fixed.batch_size&&roundtrip.max_batch_size==fixed.max_batch_size);
      auto fixed_threads=class_study::parse_conversion_options(J{{"admission_threads",128}},features);
      expect(fixed_threads.admission_threads==128);
      auto thread_roundtrip=class_study::parse_conversion_options(class_study::describe_conversion_options(fixed_threads),features);
      expect(thread_roundtrip.admission_threads==128);
      auto fixed_drafts=class_study::parse_conversion_options(J{{"draft_threads",256}},features);
      expect(fixed_drafts.draft_threads==256&&fixed_drafts.admission_threads==0);
      auto draft_roundtrip=class_study::parse_conversion_options(class_study::describe_conversion_options(fixed_drafts),features);
      expect(draft_roundtrip.draft_threads==256&&draft_roundtrip.batch_size==0);
      auto independent=class_study::parse_conversion_options(J{{"draft_threads",64},{"admission_threads",128},{"batch_size",2}},features);
      expect(independent.draft_threads==64&&independent.admission_threads==128&&independent.batch_size==2);
    }
    for (const auto& invalid : {
        J{{"batch_size",-1}},J{{"batch_size",1.5}},J{{"batch_size",true}},
        J{{"batch_size","auto"}},J{{"batch_size",65537}},
        J{{"max_batch_size",-1}},J{{"max_batch_size",1.5}},J{{"max_batch_size",true}},
        J{{"max_batch_size",65537}},J{{"batch_size",1024},{"max_batch_size",512}},
        J{{"admission_threads",-1}},J{{"admission_threads",1.5}},J{{"admission_threads",true}},
        J{{"admission_threads","auto"}},J{{"admission_threads",3}},J{{"admission_threads",4294967296ull}},
        J{{"draft_threads",-1}},J{{"draft_threads",1.5}},J{{"draft_threads",true}},
        J{{"draft_threads","auto"}},J{{"draft_threads",3}},J{{"draft_threads",4294967296ull}}}) {
      bool refused=false;
      try{(void)class_study::parse_conversion_options(invalid,13);}catch(const std::invalid_argument&){refused=true;}
      expect(refused);
    }
    for(const char*key:{"completed_cache_limit","refinement_visit_budget","oldest_ready_jobs"})for(const auto&bad:{J(-1),J(1.5),J(true),J("auto"),J(4294967296ull)}){
      bool refused=false;try{(void)class_study::parse_conversion_options(J{{key,bad}},13);}catch(const std::invalid_argument&){refused=true;}expect(refused);
    }
    auto one=class_study::parse_conversion_options(J{{"max_batch_size",1}},13);
    for(const auto& bad:{J(nullptr),J(false),J(2),J("unknown")}) {
      bool refused=false;try{(void)class_study::parse_conversion_options(J{{"split_policy",bad}},13);}catch(const std::invalid_argument&){refused=true;}expect(refused);
    }
    expect(one.batch_size==0&&one.max_batch_size==1);
    auto fixed_max=class_study::parse_conversion_options(J{{"batch_size",65536}},13);
    expect(fixed_max.batch_size==65536);
    std::cout<<J{{"complete",true},{"metadata_checks",checks},{"CUDA_executed",false}}.dump()<<'\n';
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
