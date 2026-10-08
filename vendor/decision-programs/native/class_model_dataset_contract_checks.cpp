#include "class_model_dataset_contract.hpp"
#include <iostream>
#include <vector>
using J=nlohmann::json;
int main(){try{
 unsigned checks=0,rejections=0;auto need=[&](bool b,const char*m){++checks;if(!b)throw std::runtime_error(m);};
 J valid={{"format","dense-fp32-u32-class-labels-1"},{"features",3},{"classes",3},{"rows",257},{"row_stride",5},{"FIT_rows",128},{"VALID_rows",129},{"values_path","/mnt/c/example/values.fp32"},{"values_sha256",std::string(64,'a')},{"labels_path","/mnt/c/example/labels.u32"},{"labels_sha256",std::string(64,'b')},{"preprocessing","Declared model input"},{"TEST_read",false}};
 auto d=class_model_contract::dense(valid);need(d.features==3&&d.classes==3&&d.rows==257&&d.stride==5&&d.fit_rows==128&&d.valid_rows==129,"valid descriptor changed");
 auto fit=valid;fit["FIT_rows"]=257;fit["VALID_rows"]=0;auto f=class_model_contract::dense(fit,true);need(f.fit_rows==f.rows&&f.valid_rows==0,"FIT-only contract changed");
 std::vector<std::pair<std::string,J>>bad;
 for(const char*field:{"features","classes","rows","row_stride","FIT_rows","VALID_rows"}){bad.emplace_back(field,-1);bad.emplace_back(field,3.0);bad.emplace_back(field,true);bad.emplace_back(field,"3");}
 bad.emplace_back("features",std::uint64_t(4294967299ULL));bad.emplace_back("classes",std::uint64_t(4294967298ULL));bad.emplace_back("features",0);bad.emplace_back("classes",1);bad.emplace_back("rows",0);bad.emplace_back("row_stride",2);bad.emplace_back("FIT_rows",258);bad.emplace_back("VALID_rows",128);bad.emplace_back("rows",UINT64_MAX);bad.emplace_back("preprocessing","");bad.emplace_back("TEST_read",true);bad.emplace_back("values_sha256",std::string(64,'A'));bad.emplace_back("labels_sha256",std::string(63,'b'));
 for(const auto&[field,value]:bad){auto q=valid;q[field]=value;bool refused=false;try{(void)class_model_contract::dense(q);}catch(const std::exception&){refused=true;}need(refused,"invalid descriptor accepted");++rejections;}
 for(auto pair:{std::pair<J,bool>{valid,true},std::pair<J,bool>{fit,false}}){bool refused=false;try{(void)class_model_contract::dense(pair.first,pair.second);}catch(const std::exception&){refused=true;}need(refused,"wrong role accepted");++rejections;}
 std::cout<<J{{"complete",true},{"CPU_metadata_only",true},{"rows_read",false},{"CPU_predictions",false},{"CUDA_executed",false},{"checks",checks},{"invalid_descriptors_rejected",rejections},{"integer_narrowing_before_validation",false},{"FIT_and_evaluation_role_boundaries_checked",true}}.dump(2)<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}