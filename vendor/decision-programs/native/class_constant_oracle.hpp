#pragma once

// Native XGBoost 3.4.1 objective-transform adapter. This creates no learned
// trees and performs no local softmax. CUDA base_margin supplies the exact
// ordered-FP32 scores already established by the converter's constant region.
//
// Pinned source contract:
// https://github.com/dmlc/xgboost/blob/v3.4.1/src/predictor/predictor.cc
//   Predictor::InitOutPredictions copies base_margin without arithmetic.
// https://github.com/dmlc/xgboost/blob/v3.4.1/src/predictor/gpu_predictor.cu
//   The multiclass zero-tree loop performs no score additions.
// https://github.com/dmlc/xgboost/blob/v3.4.1/src/learner.cc
//   InplacePredict(type=value) calls the native objective's PredTransform.
// https://github.com/dmlc/xgboost/blob/v3.4.1/src/objective/multiclass_obj.cu
//   multi:softprob uses the same native CUDA Transform for every model.

#include "class_cuda.hpp"

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dpnative {

class XGBoostConstantOracle {
 public:
  using Size=std::uint64_t;
  using Handle=void*;

  explicit XGBoostConstantOracle(const std::string& library_path,Size classes,int device=0)
      :classes_(classes),device_(device) {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
    static_assert(std::endian::native==std::endian::little);
    if(classes<2 || classes>7 || device<0)
      throw std::invalid_argument("invalid constant oracle dimensions/device");
    library_=dlopen(library_path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!library_)throw std::runtime_error(std::string("dlopen constant oracle: ")+dlerror());
    try {
      last_error_=symbol<LastError>("XGBGetLastError");
      symbol<Version>("XGBoostVersion")(&version_[0],&version_[1],&version_[2]);
      if(version_!=std::array<int,3>{3,4,1})
        throw std::runtime_error("constant oracle requires XGBoost 3.4.1 exactly");
      free_=symbol<Free>("XGBoosterFree");
      proxy_free_=symbol<Free>("XGDMatrixFree");
      set_parameter_=symbol<Parameter>("XGBoosterSetParam");
      set_data_=symbol<SetData>("XGProxyDMatrixSetDataCudaArrayInterface");
      set_info_=symbol<SetInfo>("XGDMatrixSetInfoFromInterface");
      predict_=symbol<Predict>("XGBoosterPredictFromCudaArray");
      const char* build=nullptr;
      check(symbol<Info>("XGBuildInfo")(&build),"XGBuildInfo");
      if(!build || compact_json(build).find("\"USE_CUDA\":true")==std::string::npos)
        throw std::runtime_error("constant oracle library has no CUDA support");
      build_info_=build;
      cuda_check(cudaSetDevice(device_),"select constant oracle CUDA device");
      check(symbol<Create>("XGBoosterCreate")(nullptr,0,&booster_),"create zero-tree booster");
      parameter("booster","gbtree");
      parameter("objective","multi:softprob");
      parameter("num_class",std::to_string(classes_));
      // The supplied margin matrix doubles as ignored dummy feature data.
      // No learned tree can inspect it, and only K rather than54 columns move.
      parameter("num_feature",std::to_string(classes_));
      parameter("device","cuda:"+std::to_string(device_));
      parameter("nthread","4");
      parameter("boost_from_average","0");
      parameter("base_score","0");
      Size length=0;
      const char* configuration=nullptr;
      check(symbol<SaveConfig>("XGBoosterSaveJsonConfig")(
        booster_,&length,&configuration),"configure zero-tree native objective");
      if(!configuration || !length)throw std::runtime_error("missing constant oracle configuration");
      configuration_=configuration;
      const auto compact=compact_json(configuration_);
      if(compact.find("\"device\":\"cuda:"+std::to_string(device_)+"\"")==std::string::npos
         || compact.find("\"name\":\"multi:softprob\"")==std::string::npos
         || compact.find("\"name\":\"gbtree\"")==std::string::npos
         || compact.find("\"num_class\":\""+std::to_string(classes_)+"\"")==std::string::npos)
        throw std::runtime_error("constant oracle is not the requested CUDA multiclass gbtree");
      int rounds=-1;
      check(symbol<Rounds>("XGBoosterBoostedRounds")(booster_,&rounds),"check zero boosted rounds");
      if(rounds!=0)throw std::runtime_error("constant oracle unexpectedly contains trained rounds");
      Size actual_features=0;
      check(symbol<NumFeature>("XGBoosterGetNumFeature")(booster_,&actual_features),"check dummy feature count");
      if(actual_features!=classes_)throw std::runtime_error("constant oracle feature count mismatch");
      check(symbol<ProxyCreate>("XGProxyDMatrixCreate")(&proxy_),"create constant-margin proxy");
    } catch(...) {release();throw;}
  }

  ~XGBoostConstantOracle(){release();}
  XGBoostConstantOracle(const XGBoostConstantOracle&)=delete;
  XGBoostConstantOracle& operator=(const XGBoostConstantOracle&)=delete;
  XGBoostConstantOracle(XGBoostConstantOracle&&)=delete;
  XGBoostConstantOracle& operator=(XGBoostConstantOracle&&)=delete;

  // A borrowed CUDA [rows,K] output, valid until this handle's next prediction.
  // margins must contain finite source scores; the converter establishes this
  // precondition. No fit or learned-model prediction occurs. margin=true is a
  // qualification-only identity path that must preserve every input FP32 bit.
  const float* predict_values(const float* margins,Size rows,bool margin=false) {
    if(!margins || !rows || rows>std::numeric_limits<Size>::max()/classes_
       || rows>std::numeric_limits<std::size_t>::max()/sizeof(float)/classes_)
      throw std::invalid_argument("invalid constant oracle margin shape");
    int current_device=-1;
    cuda_check(cudaGetDevice(&current_device),"read constant oracle CUDA device");
    if(current_device!=device_)throw std::runtime_error("constant oracle CUDA device changed");
    require_device_pointer(margins,"constant oracle margins");
    cuda_check(cudaDeviceSynchronize(),"synchronize constant oracle margins");
    const std::string array="{\"data\":["+
      std::to_string(reinterpret_cast<std::uintptr_t>(margins))+",true],\"shape\":["+
      std::to_string(rows)+","+std::to_string(classes_)+
      "],\"strides\":null,\"typestr\":\"<f4\",\"version\":3,\"stream\":1}";
    // Set proxy dimensions/device before attaching multidimensional metadata.
    check(set_data_(proxy_,array.c_str()),"set constant oracle dummy CUDA data");
    check(set_info_(proxy_,"base_margin",array.c_str()),"set constant oracle CUDA base_margin");
    const std::string configuration=std::string("{\"type\":")+(margin?"1":"0")+
      ",\"training\":false,\"iteration_begin\":0,\"iteration_end\":0,"
      "\"missing\":NaN,\"strict_shape\":true,\"cache_id\":0}";
    const Size* shape=nullptr;Size dimensions=0;const float* prediction=nullptr;
    check(predict_(booster_,array.c_str(),configuration.c_str(),proxy_,&shape,&dimensions,&prediction),
      "native zero-tree CUDA prediction");
    if(!shape || dimensions!=2 || shape[0]!=rows || shape[1]!=classes_ || !prediction)
      throw std::runtime_error("constant oracle prediction shape mismatch");
    require_device_pointer(prediction,"constant oracle output");
    cuda_check(cudaDeviceSynchronize(),"synchronize constant oracle native transform");
    return prediction;
  }

  std::vector<std::int32_t> predict_classes(const float* margins,Size rows) {
    // first_argmax rounds the launch up with (rows +255) in int arithmetic.
    if(rows>static_cast<Size>(std::numeric_limits<int>::max()-255)
       || rows>std::numeric_limits<std::size_t>::max()/sizeof(std::int32_t))
      throw std::invalid_argument("constant oracle rows exceed CUDA class reduction range");
    const float* probabilities=predict_values(margins,rows);
    std::vector<std::int32_t> output(static_cast<std::size_t>(rows));
    if(rows>label_capacity_) {
      std::int32_t* replacement=nullptr;
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&replacement),
                            static_cast<std::size_t>(rows)*sizeof(std::int32_t)),
                 "allocate constant oracle class buffer");
      // Allocate before retiring the old buffer, so allocation failure leaves
      // the previous capacity usable. No pending reduction can use it here:
      // predict_values and every class download complete synchronously.
      if(labels_) {
        const auto status=cudaFree(labels_);
        if(status!=cudaSuccess) {
          cudaFree(replacement);
          cuda_check(status,"retire constant oracle class buffer");
        }
      }
      labels_=replacement;label_capacity_=rows;
    }
    native_cuda::first_argmax(probabilities,static_cast<int>(rows),static_cast<int>(classes_),labels_);
    cuda_check(cudaMemcpy(output.data(),labels_,output.size()*sizeof(std::int32_t),cudaMemcpyDeviceToHost),
               "download constant oracle classes");
    for(auto label:output)
      if(label<0 || static_cast<Size>(label)>=classes_)
        throw std::invalid_argument("argmax input contains nonfinite values or invalid class output");
    return output;
  }
  const std::array<int,3>& version()const noexcept{return version_;}
  const std::string& configuration_json()const noexcept{return configuration_;}
  const std::string& build_info()const noexcept{return build_info_;}
  Size classes()const noexcept{return classes_;}
  int device()const noexcept{return device_;}

 private:
  using LastError=const char*(*)();
  using Version=void(*)(int*,int*,int*);
  using Info=int(*)(const char**);
  using Create=int(*)(const Handle[],Size,Handle*);
  using ProxyCreate=int(*)(Handle*);
  using Free=int(*)(Handle);
  using Parameter=int(*)(Handle,const char*,const char*);
  using SetData=int(*)(Handle,const char*);
  using SetInfo=int(*)(Handle,const char*,const char*);
  using SaveConfig=int(*)(Handle,Size*,const char**);
  using Rounds=int(*)(Handle,int*);
  using NumFeature=int(*)(Handle,Size*);
  using Predict=int(*)(Handle,const char*,const char*,Handle,const Size**,Size*,const float**);

  template<class Function>Function symbol(const char* name) {
    dlerror();void* address=dlsym(library_,name);const char* error=dlerror();
    if(error || !address)throw std::runtime_error(std::string("missing constant oracle C API symbol ")+name);
    return reinterpret_cast<Function>(address);
  }
  void check(int status,const char* operation)const {
    if(status) {
      const char* error=last_error_?last_error_():nullptr;
      throw std::runtime_error(std::string(operation)+": "+(error?error:"unknown XGBoost error"));
    }
  }
  static void cuda_check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
  }
  void parameter(const char* name,const std::string& value) {
    check(set_parameter_(booster_,name,value.c_str()),name);
  }
  void require_device_pointer(const void* pointer,const char* role)const {
    cudaPointerAttributes attributes{};
    cuda_check(cudaPointerGetAttributes(&attributes,pointer),role);
    if(attributes.type!=cudaMemoryTypeDevice || attributes.device!=device_)
      throw std::runtime_error(std::string(role)+" must be allocated on constant oracle CUDA device");
  }
  static std::string compact_json(const std::string& text) {
    std::string compact;compact.reserve(text.size());bool quoted=false,escaped=false;
    for(char character:text) {
      if(quoted || (character!=' ' && character!='\t' && character!='\n' && character!='\r'))compact.push_back(character);
      if(!escaped && character=='"')quoted=!quoted;
      if(quoted && character=='\\')escaped=!escaped;else escaped=false;
    }
    return compact;
  }
  void release()noexcept {
    if(labels_) {
      int previous_device=-1;
      const bool restore=cudaGetDevice(&previous_device)==cudaSuccess && previous_device!=device_;
      if(restore)cudaSetDevice(device_);
      cudaFree(std::exchange(labels_,nullptr));label_capacity_=0;
      if(restore)cudaSetDevice(previous_device);
    }
    if(proxy_ && proxy_free_)proxy_free_(std::exchange(proxy_,nullptr));
    if(booster_ && free_)free_(std::exchange(booster_,nullptr));
    if(library_)dlclose(std::exchange(library_,nullptr));
  }
  void* library_=nullptr;
  Handle booster_=nullptr,proxy_=nullptr;
  LastError last_error_=nullptr;
  Free free_=nullptr,proxy_free_=nullptr;
  Parameter set_parameter_=nullptr;
  SetData set_data_=nullptr;
  SetInfo set_info_=nullptr;
  Predict predict_=nullptr;
  std::array<int,3> version_{};
  std::string configuration_,build_info_;
  std::int32_t* labels_=nullptr;
  Size label_capacity_=0;
  Size classes_;
  int device_;
};

} // namespace dpnative
