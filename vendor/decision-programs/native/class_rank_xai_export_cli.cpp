#include "class_rank_xai_export.hpp"
#include "class_io.hpp"
#include <iostream>
int main(int argc,char**argv){try{
 namespace x=rank_xai_export;
 if(argc==3&&std::string(argv[1])=="--verify"){x::verify(argv[2]);std::cout<<"{\"verified\":true,\"CPU_predictions_evaluated\":false,\"GPU_executed\":false}\n";return 0;}
 if(argc!=4&&argc!=6)throw std::runtime_error("usage: xai_export RUNTIME EXPECTED_SHA256 NEW_OUTPUT [--smooth-numeric 0|1]; or --verify OUTPUT");
 x::Options o;if(argc==6){if(std::string(argv[4])!="--smooth-numeric"||(std::string(argv[5])!="0"&&std::string(argv[5])!="1"))throw std::runtime_error("invalid smooth option");o.smooth_numeric=std::string(argv[5])=="1";}
 auto r=x::write(argv[1],argv[2],argv[3],o);
 std::cout<<dpnative::json{{"complete",true},{"output",argv[3]},{"runtime_sha256",r.runtime_sha256},{"result_sha256",r.result_sha256},{"nodes",r.nodes},{"runtime_bytes",r.runtime_bytes},{"structural_roundtrip",r.structural_roundtrip},{"GPU_executed",false}}.dump(2)<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

