#pragma once

// Native prediction-only binding to the pinned XGBoost 3.4.1 C ABI.
// Signatures: https://github.com/dmlc/xgboost/blob/v3.4.1/include/xgboost/c_api.h
// CUDA output ownership: src/c_api/c_api.cu, InplacePreidctCUDA.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include "class_cuda.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace dpnative {

class XGBoostOracle {
 public:
  using Size = std::uint64_t;  // XGBoost's bst_ulong, including on LLP64 hosts.
  using Handle = void*;

  XGBoostOracle(const std::string& library_path, const std::string& model_path,
                Size features, Size classes, int device = 0)
      : features_(features), classes_(classes), device_(device) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(std::endian::native == std::endian::little);
    if (features == 0 || classes < 2 || device < 0) {
      throw std::invalid_argument("invalid native XGBoost oracle dimensions/device");
    }
    library_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
      throw std::runtime_error(std::string("dlopen XGBoost: ") + dlerror());
    }
    try {
      last_error_ = symbol<LastError>("XGBGetLastError");
      auto get_version = symbol<Version>("XGBoostVersion");
      get_version(&version_[0], &version_[1], &version_[2]);
      if (version_ != std::array<int, 3>{3, 4, 1}) {
        throw std::runtime_error("native oracle requires XGBoost 3.4.1 exactly");
      }
      create_ = symbol<Create>("XGBoosterCreate");
      free_ = symbol<Free>("XGBoosterFree");
      load_ = symbol<Load>("XGBoosterLoadModel");
      parameter_ = symbol<Parameter>("XGBoosterSetParam");
      predict_ = symbol<Predict>("XGBoosterPredictFromCudaArray");
      auto info = symbol<Info>("XGBuildInfo");
      const char* build = nullptr;
      check(info(&build), "XGBuildInfo");
      if (!build) throw std::runtime_error("missing XGBoost build information");
      build_info_ = build;
      if (compact_json(build_info_).find("\"USE_CUDA\":true") == std::string::npos) {
        throw std::runtime_error("XGBoost oracle library has no CUDA support");
      }
      cuda_check(cudaSetDevice(device_), "select oracle CUDA device");
      check(create_(nullptr, 0, &booster_), "XGBoosterCreate");
      check(load_(booster_, model_path.c_str()), "XGBoosterLoadModel");
      check(parameter_(booster_, "device", ("cuda:" + std::to_string(device_)).c_str()),
            "set XGBoost device");
      check(parameter_(booster_, "nthread", "4"), "set XGBoost threads");
      Size actual_features = 0;
      check(symbol<NumFeature>("XGBoosterGetNumFeature")(booster_, &actual_features),
            "XGBoosterGetNumFeature");
      if (actual_features != features_) {
        throw std::runtime_error("oracle/source feature count mismatch");
      }
      Size length = 0;
      const char* configuration = nullptr;
      check(symbol<SaveConfig>("XGBoosterSaveJsonConfig")(
                booster_, &length, &configuration), "XGBoosterSaveJsonConfig");
      if (!configuration || length == 0) {
        throw std::runtime_error("missing native oracle configuration");
      }
      configuration_ = configuration;
      const auto compact = compact_json(configuration_);
      if (compact.find("\"device\":\"cuda:" + std::to_string(device_) + "\"") ==
              std::string::npos ||
          compact.find("\"name\":\"multi:softprob\"") == std::string::npos ||
          compact.find("\"name\":\"gbtree\"") == std::string::npos ||
          compact.find("\"num_class\":\"" + std::to_string(classes_) + "\"") ==
              std::string::npos) {
        throw std::runtime_error("oracle is not the requested CUDA multiclass gbtree");
      }
    } catch (...) {
      release();
      throw;
    }
  }

  ~XGBoostOracle() { release(); }
  XGBoostOracle(const XGBoostOracle&) = delete;
  XGBoostOracle& operator=(const XGBoostOracle&) = delete;
  XGBoostOracle(XGBoostOracle&&) = delete;
  XGBoostOracle& operator=(XGBoostOracle&&) = delete;

  // The result is a borrowed CUDA FP32 [rows, classes] array, valid until the
  // next prediction or destruction. Consumers must finish before either occurs.
  // type=0 is native public softprob, never a local softmax or raw-score argmax.
  // margin=true exists solely for independent ordered-FP32 importer checks.
  const float* predict_values(const float* inputs, Size rows, bool margin = false) {
    if (!inputs || rows == 0 || rows > std::numeric_limits<Size>::max() / features_ ||
        rows > std::numeric_limits<Size>::max() / classes_) {
      throw std::invalid_argument("invalid oracle input shape");
    }
    int current_device = -1;
    cuda_check(cudaGetDevice(&current_device), "read oracle CUDA device");
    if (current_device != device_) throw std::runtime_error("oracle CUDA device changed");
    require_device_pointer(inputs, "oracle input");
    // The C API does not return its producer stream. Synchronizing here also
    // protects borrowed results consumed on another stream before buffer reuse.
    cuda_check(cudaDeviceSynchronize(), "synchronize native oracle inputs");
    const std::string array = "{\"data\":[" +
        std::to_string(reinterpret_cast<std::uintptr_t>(inputs)) +
        ",true],\"shape\":[" + std::to_string(rows) + "," +
        std::to_string(features_) +
        "],\"strides\":null,\"typestr\":\"<f4\",\"version\":3,\"stream\":1}";
    // XGBoost's JSON dialect accepts NaN, as used by its official Python API.
    const std::string configuration = std::string("{\"type\":") +
        (margin ? "1" : "0") +
        ",\"training\":false,\"iteration_begin\":0,\"iteration_end\":0,"
        "\"missing\":NaN,\"strict_shape\":true,\"cache_id\":0}";
    const Size* shape = nullptr;
    Size dimensions = 0;
    const float* prediction = nullptr;
    check(predict_(booster_, array.c_str(), configuration.c_str(), nullptr,
                   &shape, &dimensions, &prediction), "XGBoosterPredictFromCudaArray");
    if (!shape || dimensions != 2 || shape[0] != rows || shape[1] != classes_ ||
        !prediction) {
      throw std::runtime_error("native oracle prediction shape mismatch");
    }
    require_device_pointer(prediction, "oracle output");
    cuda_check(cudaDeviceSynchronize(), "synchronize native oracle probabilities");
    return prediction;
  }

  const std::array<int, 3>& version() const noexcept { return version_; }
  const std::string& build_info() const noexcept { return build_info_; }
  const std::string& configuration_json() const noexcept { return configuration_; }
  Size features() const noexcept { return features_; }
  Size classes() const noexcept { return classes_; }
  int device() const noexcept { return device_; }

  // Reduction happens in CUDA. The vector only transports the selected labels.
  std::vector<std::int32_t> predict_classes(const float* inputs, Size rows) {
    if (rows > static_cast<Size>(std::numeric_limits<int>::max()) ||
        classes_ > static_cast<Size>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("oracle class reduction exceeds CUDA index range");
    }
    const float* probabilities = predict_values(inputs, rows);
    return native_cuda::first_argmax_host(probabilities, static_cast<int>(rows),
                                          static_cast<int>(classes_));
  }

 private:
  using LastError = const char* (*)();
  using Version = void (*)(int*, int*, int*);
  using Info = int (*)(const char**);
  using Create = int (*)(const Handle[], Size, Handle*);
  using Free = int (*)(Handle);
  using Load = int (*)(Handle, const char*);
  using Parameter = int (*)(Handle, const char*, const char*);
  using NumFeature = int (*)(Handle, Size*);
  using SaveConfig = int (*)(Handle, Size*, const char**);
  using Predict = int (*)(Handle, const char*, const char*, Handle,
                         const Size**, Size*, const float**);

  template <typename Function>
  Function symbol(const char* name) {
    dlerror();
    void* address = dlsym(library_, name);
    const char* error = dlerror();
    if (error || !address) {
      throw std::runtime_error(std::string("missing XGBoost C API symbol ") + name);
    }
    return reinterpret_cast<Function>(address);
  }

  void check(int status, const char* operation) const {
    if (status != 0) {
      const char* error = last_error_ ? last_error_() : nullptr;
      throw std::runtime_error(std::string(operation) + ": " +
                               (error ? error : "unknown XGBoost error"));
    }
  }

  static void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
  }

  void require_device_pointer(const void* pointer, const char* role) const {
    cudaPointerAttributes attributes{};
    cuda_check(cudaPointerGetAttributes(&attributes, pointer), role);
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device_) {
      throw std::runtime_error(std::string(role) + " must be allocated on oracle CUDA device");
    }
  }

  static std::string compact_json(const std::string& text) {
    std::string compact;
    compact.reserve(text.size());
    bool quoted = false, escaped = false;
    for (char character : text) {
      if (quoted || (character != ' ' && character != '\t' && character != '\n' &&
                     character != '\r')) compact.push_back(character);
      if (!escaped && character == '"') quoted = !quoted;
      if (quoted && character == '\\') escaped = !escaped;
      else escaped = false;
    }
    return compact;
  }

  void release() noexcept {
    if (booster_ && free_) free_(std::exchange(booster_, nullptr));
    if (library_) dlclose(std::exchange(library_, nullptr));
  }

  void* library_ = nullptr;
  Handle booster_ = nullptr;
  LastError last_error_ = nullptr;
  Create create_ = nullptr;
  Free free_ = nullptr;
  Load load_ = nullptr;
  Parameter parameter_ = nullptr;
  Predict predict_ = nullptr;
  std::array<int, 3> version_{};
  std::string build_info_, configuration_;
  Size features_, classes_;
  int device_;
};

}  // namespace dpnative
