#include "class_study_train.hpp"
#include <iostream>
#include <type_traits>
int main(){
  static_assert(std::is_move_constructible_v<class_study::ResidentTrainer>);
  static_assert(!std::is_copy_constructible_v<class_study::ResidentTrainer>);
  using J=nlohmann::json;
  J d={{"format","dense-fp32-u32-class-labels-1"},{"features",5},{"classes",3},{"rows",17},{"row_stride",7},{"FIT_rows",17},{"VALID_rows",0},{"values_path","/var/caller-data/interface-only.fp32"},{"values_sha256",std::string(64,'0')},{"labels_path","/var/caller-data/interface-only.u32"},{"labels_sha256",std::string(64,'1')},{"preprocessing","caller declared generic FP32 input"},{"TEST_read",false}};
  J p={{"dataset",d},{"TEST_read",false},{"VALID_read",false},{"native_library_path","/opt/caller-dependencies/libxgboost.so"},{"native_library_sha256",std::string(64,'a')}};
  auto metadata=class_study::ResidentTrainer::validate_metadata(p);if(metadata.at("features")!=5||metadata.at("classes")!=3)throw std::runtime_error("generic metadata differs");
  unsigned refused=0;for(auto key:{"features","classes","rows","row_stride"}){auto bad=p;bad["dataset"][key]=-1;try{(void)class_study::ResidentTrainer::validate_metadata(bad);}catch(...){++refused;}}
  if(metadata.at("native_library_sha256")!=p.at("native_library_sha256"))throw std::runtime_error("caller library identity lost");
  for(const auto& offered:std::vector<J>{J{{"native_library_path","relative/libxgboost.so"}},J{{"native_library_path",std::string("/opt/lib\0xgboost.so",18)}},J{{"native_library_sha256",std::string(64,'A')}},J{{"native_library_sha256",std::string(63,'a')}},J{{"native_library_sha256",false}}}){auto bad=p;bad.update(offered);try{(void)class_study::ResidentTrainer::validate_metadata(bad);}catch(...){++refused;}}
  auto valid=class_study::ResidentTrainer::validate_hyperparameters(J{{"rounds",2},{"max_depth",3}},3);if(valid.at("rounds")!=2)throw std::runtime_error("parameter metadata differs");
  for(auto bad:std::vector<J>{{{"rounds",0},{"max_depth",2}},{{"rounds",2},{"max_depth",2},{"unknown",1}},{{"rounds",2},{"max_depth",2},{"eta",0.0}}})try{(void)class_study::ResidentTrainer::validate_hyperparameters(bad,3);}catch(...){++refused;}
  if(refused!=12)throw std::runtime_error("malformed metadata accepted");
  std::cout<<J{{"complete",true},{"CUDA_executed",false},{"dataset_rows_read",false},{"CPU_predictions",false},{"interface_and_metadata_only",true},{"malformed_metadata_refusals",refused}}.dump()<<'\n';
}
