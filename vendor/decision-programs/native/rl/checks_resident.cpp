#include "resident_session.hpp"
#include "class_io.hpp"
#include <iostream>
#include <type_traits>
static_assert(!std::is_default_constructible_v<rl_qualified_session::QualifiedCostTable>);
static_assert(!std::is_default_constructible_v<rl_qualified_session::ResidentTraining>);
static_assert(!std::is_copy_constructible_v<rl_qualified_session::ResidentTraining>);
int main(int argc,char**argv){try{
  if(argc==2&&std::string(argv[1])=="--cpu"){
    using namespace rl_qualified_session;ResidentWord assertions=0,rejections=0;
    auto check=[&](bool value){++assertions;if(!value)throw std::runtime_error("resident metadata check failed");};
    check(!std::is_default_constructible_v<QualifiedCostTable>);check(!std::is_default_constructible_v<ResidentTraining>);check(!std::is_copy_constructible_v<ResidentTraining>);
    check(sizeof(ResidentEpisodeRecord)==624);check(rl_category_policy::sampler_schema_version==1);
    check(resident_device_plan(77,4,2,32,false)==32304);check(resident_device_plan(77,4,2,32,true)==109616);
    for(unsigned k=0;k<8;++k){bool failed=false;try{resident_device_plan(k==0?0:k==5?UINT64_MAX:77,k==1?0:k==4?UINT64_MAX:4,k==2?0:k==6?UINT64_MAX:2,k==3?0:k==7?UINT64_MAX:32,false);}catch(const std::exception&){failed=true;}
      check(failed);++rejections;}
    std::cout<<dpnative::json{{"passed",true},{"CPU_metadata_only",true},{"CUDA_executed",false},{"assertions",assertions},{"rejections",rejections},{"record_bytes",sizeof(ResidentEpisodeRecord)},{"sampler_schema_version",1},{"private_table_constructible",false},{"CPU_model_math",false}}.dump(2)<<'\n';return 0;
  }
  if(argc!=2||std::string(argv[1])!="--gpu")throw std::runtime_error("usage checks_resident --cpu | --gpu");
  auto r=rl_qualified_session::testing::resident_gpu_checks();std::cout<<dpnative::json{{"passed",r.passed},{"CUDA_executed",r.CUDA_executed},{"assertions",r.assertions},{"rejections",r.rejections},{"episode_word_checks",r.episode_word_checks},{"sampler_schema_version",1},{"source_class_authority",false},{"resident_policy1_body",true},{"whole_source_conversion_complete",false},{"CPU_model_math",false}}.dump(2)<<'\n';return r.passed?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
