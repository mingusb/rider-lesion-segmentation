#include "decision_programs_model_format.hpp"
#include <iostream>

int main(){try{
  namespace d=decision_programs_display;unsigned checks=0;
  auto require=[&](bool ok,const char*message){if(!ok)throw std::runtime_error(message);++checks;};
  constexpr std::uint32_t three_bits=1077936128u;
  auto fields=d::threshold_fields(0,three_bits);
  require(fields.at("threshold")==3,"stored FP32 three must display numeric 3");
  require(fields.at("threshold_text")=="3","stored FP32 three must display text 3");
  require(fields.at("decision_rule")=="NaN -> right; x[0] < 3 -> left; otherwise -> right","readable decision rule differs");
  require(d::condition(0,three_bits)=="(isnan(x[0]) ? false : x[0] < 3)","readable condition differs");
  require(d::condition(-2,three_bits)=="(isnan(x[0]) ? true : x[0] < 3)","missing-left display differs");
  require(d::threshold_text(0x80000000u)=="-0","signed-zero display lost its sign");
  for(auto bits:{0u,0x80000000u,0x3eaaaaabu,0x00000001u,0x7f7fffffu,0xff7fffffu}){
    auto text=d::threshold_text(bits);float parsed=0;auto result=std::from_chars(text.data(),text.data()+text.size(),parsed);
    require(result.ec==std::errc{}&&result.ptr==text.data()+text.size()&&std::bit_cast<std::uint32_t>(parsed)==bits,"readable threshold is not an exact FP32 round trip");
  }
  std::cout<<nlohmann::json{{"passed",true},{"checks",checks},{"CUDA_executed",false},{"CPU_predicates_evaluated",false},{"threshold_display",fields}}.dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
