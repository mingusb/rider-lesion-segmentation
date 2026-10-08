#include "class_study_holdout.hpp"
#include "class_study_convert.hpp"
#include "class_model_dataset_contract.hpp"
#include "class_io.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace class_study::holdout_detail {
using J = nlohmann::json;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
void need(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void ck(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void done() {
  ck(cudaGetLastError(), "HOLDOUT CUDA launch");
  ck(cudaDeviceSynchronize(), "HOLDOUT CUDA completion");
}
unsigned blocks(u64 count) {
  return unsigned(std::min<u64>(count / 256 + (count % 256 != 0), 65535));
}
template<class T> struct Buffer {
  T* p = nullptr;
  u64 n = 0;
  Buffer() = default;
  explicit Buffer(u64 count) : n(count) {
    need(n > 0 && n <= SIZE_MAX / sizeof(T), "HOLDOUT allocation extent overflow");
    ck(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "HOLDOUT allocation");
  }
  ~Buffer() { release(); }
  void release() noexcept {
    if (!p) return;
    int previous = -1;
    const bool restore = cudaGetDevice(&previous) == cudaSuccess && previous != 0;
    if (restore) cudaSetDevice(0);
    cudaFree(p);
    if (restore) cudaSetDevice(previous);
    p = nullptr;
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept
      : p(std::exchange(other.p, nullptr)), n(std::exchange(other.n, 0)) {}
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      release(); p = std::exchange(other.p, nullptr); n = std::exchange(other.n, 0);
    }
    return *this;
  }
  void zero() { ck(cudaMemset(p, 0, n * sizeof(T)), "HOLDOUT counter initialization"); }
};
struct Owner { Buffer<float> values; Buffer<u32> labels; };
struct Counts {
  unsigned long long candidate_errors = 0, baseline_errors = 0;
  unsigned long long invalid_predictions = 0, invalid_labels = 0, accepted = 0;
};
void device_span(const void* pointer, u64 elements) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  need(pointer && address % 4 == 0 && elements > 0 && elements <= SIZE_MAX / 4 &&
       elements * 4 <= UINTPTR_MAX - address, "HOLDOUT borrowed pointer/extent");
  cudaPointerAttributes attributes{};
  ck(cudaPointerGetAttributes(&attributes, pointer), "HOLDOUT pointer attributes");
  need(attributes.type == cudaMemoryTypeDevice && attributes.device == 0,
       "HOLDOUT requires CUDA0 device storage");
  CUdeviceptr base = 0; std::size_t bytes = 0;
  need(cuMemGetAddressRange(&base, &bytes, static_cast<CUdeviceptr>(address)) == CUDA_SUCCESS &&
       address >= base && elements * 4 <= bytes && address - base <= bytes - elements * 4,
       "HOLDOUT borrowed span exceeds allocation");
}
void active_device() {
  int device = -1;
  ck(cudaGetDevice(&device), "HOLDOUT active device");
  need(device == 0, "HOLDOUT gate requires active CUDA0");
}
std::string read_exact(const std::filesystem::path& path, u64 bytes) {
  need(bytes <= SIZE_MAX && bytes <= u64(std::numeric_limits<std::streamsize>::max()),
       "HOLDOUT file extent exceeds supported I/O range");
  need(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path) == bytes,
       "HOLDOUT dense file byte extent differs");
  std::ifstream input(path, std::ios::binary);
  need(bool(input), "HOLDOUT dense file open failed");
  std::string result(std::size_t(bytes), '\0');
  input.read(result.data(), std::streamsize(bytes));
  need(bool(input) && input.peek() == std::char_traits<char>::eof() && !input.bad(),
       "HOLDOUT dense file read/extent differs");
  return result;
}
__global__ void validate_labels(const u32* labels, u64 rows, u32 classes, u32* bad) {
  for (u64 r = u64(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += u64(blockDim.x) * gridDim.x)
    if (labels[r] >= classes) atomicOr(bad, 1u);
}
__global__ void pack_interval(const float* source, float* packed, u64 rows,
                              u64 stride, u32 features) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * u64(features);
       i += u64(blockDim.x) * gridDim.x)
    packed[i] = source[(i / features) * stride + i % features];
}
__global__ void score_classes(const float* predictions, const u32* labels, u64 rows,
                              u32 classes, bool candidate, Counts* counts) {
  for (u64 r = u64(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += u64(blockDim.x) * gridDim.x) {
    const float prediction = predictions[r];
    const bool invalid = !isfinite(prediction) || prediction < 0 ||
        double(prediction) >= double(classes) || prediction != floorf(prediction);
    if (invalid) atomicAdd(&counts->invalid_predictions, 1ull);
    if (labels[r] >= classes) atomicAdd(&counts->invalid_labels, 1ull);
    if (!invalid && labels[r] < classes && u32(prediction) != labels[r])
      atomicAdd(candidate ? &counts->candidate_errors : &counts->baseline_errors, 1ull);
  }
}
__global__ void decide(Counts* counts) {
  if (!blockIdx.x && !threadIdx.x)
    counts->accepted = !counts->invalid_predictions && !counts->invalid_labels &&
        counts->candidate_errors < counts->baseline_errors;
}
} // namespace class_study::holdout_detail

namespace class_study {
nlohmann::json validate_holdout_dataset(const nlohmann::json& descriptor) {
  namespace c = class_model_contract;
  c::require(descriptor.is_object(), "HOLDOUT descriptor must be an object");
  c::require(descriptor.at("format").is_string() &&
             descriptor.at("format") == "dense-fp32-u32-holdout-labels-1",
             "unsupported HOLDOUT-only dense format");
  c::require(descriptor.at("role").is_string() && descriptor.at("role") == "HOLDOUT" &&
             descriptor.at("TEST_read").is_boolean() && !descriptor.at("TEST_read").get<bool>() &&
             descriptor.at("training_allowed").is_boolean() && !descriptor.at("training_allowed").get<bool>() &&
             descriptor.at("selection_allowed").is_boolean() && descriptor.at("selection_allowed").get<bool>(),
             "HOLDOUT must exclude TEST/training and explicitly allow confirmation selection");
  const auto integer = [&](const char* field) {
    return c::unsigned_integer(descriptor.at(field), field);
  };
  const auto features = integer("features"), classes = integer("classes");
  const auto rows = integer("rows"), stride = integer("row_stride");
  c::require(features > 0 && features <= INT32_MAX && classes >= 2 && classes <= 16777217 &&
             rows > 0 && stride >= features && rows <= UINT64_MAX / stride &&
             rows * stride <= SIZE_MAX / sizeof(float) && rows <= SIZE_MAX / sizeof(std::uint32_t),
             "HOLDOUT dynamic shape/storage extent");
  for (const char* field : {"FIT_rows", "VALID_rows"})
    if (descriptor.contains(field))
      c::require(integer(field) == 0, "HOLDOUT has no FIT/VALID roles");
  c::require(descriptor.at("preprocessing").is_string() &&
             !descriptor.at("preprocessing").get_ref<const std::string&>().empty(),
             "HOLDOUT preprocessing declaration");
  for (const char* field : {"values_path", "labels_path"}) {
    c::require(descriptor.at(field).is_string() &&
               !descriptor.at(field).get_ref<const std::string&>().empty(), "HOLDOUT file path");
    const auto& path = descriptor.at(field).get_ref<const std::string&>();
    c::require(path.find('\0') == std::string::npos && std::filesystem::path(path).is_absolute(),
               "HOLDOUT paths must be absolute without NUL");
  }
  c::require(c::sha256(descriptor.at("values_sha256")) && c::sha256(descriptor.at("labels_sha256")),
             "HOLDOUT file pins must be lowercase SHA256");
  return {{"format", "dense-fp32-u32-holdout-labels-1"}, {"role", "HOLDOUT"},
          {"features", features}, {"classes", classes}, {"rows", rows}, {"row_stride", stride},
          {"FIT_rows", 0}, {"VALID_rows", 0}, {"TEST_read", false},
          {"training_allowed", false}, {"selection_allowed", true}, {"metadata_only", true}};
}

ResidentHoldoutDataView stage_holdout_dataset(const nlohmann::json& descriptor) {
  namespace d = holdout_detail;
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4 && sizeof(std::uint32_t) == 4);
  const auto shape = validate_holdout_dataset(descriptor);
  const auto rows = shape.at("rows").get<d::u64>(), stride = shape.at("row_stride").get<d::u64>();
  const auto features = shape.at("features").get<d::u32>(), classes = shape.at("classes").get<d::u32>();
  const auto values = d::read_exact(descriptor.at("values_path").get<std::string>(), rows * stride * 4);
  const auto labels = d::read_exact(descriptor.at("labels_path").get<std::string>(), rows * 4);
  d::need(dpnative::sha256(values) == descriptor.at("values_sha256").get<std::string>() &&
          dpnative::sha256(labels) == descriptor.at("labels_sha256").get<std::string>(),
          "HOLDOUT dense file byte pins differ");
  d::ck(cudaSetDevice(0), "HOLDOUT staging device");
  auto owner = std::make_shared<d::Owner>();
  owner->values = d::Buffer<float>(rows * stride);
  owner->labels = d::Buffer<d::u32>(rows);
  d::Buffer<d::u32> bad(1); bad.zero();
  d::ck(cudaMemcpy(owner->values.p, values.data(), values.size(), cudaMemcpyHostToDevice), "HOLDOUT values upload");
  d::ck(cudaMemcpy(owner->labels.p, labels.data(), labels.size(), cudaMemcpyHostToDevice), "HOLDOUT labels upload");
  d::validate_labels<<<d::blocks(rows), 256>>>(owner->labels.p, rows, classes, bad.p);
  d::done();
  d::u32 invalid = 0;
  d::ck(cudaMemcpy(&invalid, bad.p, sizeof(invalid), cudaMemcpyDeviceToHost), "HOLDOUT label-validation result");
  d::need(!invalid, "HOLDOUT labels outside declared classes");
  auto binding = descriptor;
  binding["FIT_rows"] = 0; binding["VALID_rows"] = 0;
  binding["dataset_loads"] = 1; binding["input_content_reads"] = 2;
  binding["derived_dataset_files_written"] = 0; binding["CUDA_staged"] = true;
  binding["all_labels_validated_on_CUDA"] = true;
  binding["input_dtype"] = "little-endian FP32";
  binding["label_dtype"] = "little-endian uint32";
  return {owner->values.p, owner->labels.p, rows, stride, features, classes,
          std::move(binding), std::static_pointer_cast<void>(owner)};
}

nlohmann::json evaluate_holdout_gate(const ResidentHoldoutDataView& data,
                                    std::uint64_t offset, std::uint64_t count,
                                    const NativeOracle& candidate,
                                    const NativeOracle& baseline) {
  namespace d = holdout_detail;
  const auto shape = validate_holdout_dataset(data.binding);
  d::need(bool(data.owner) && shape.at("rows") == data.rows && shape.at("row_stride") == data.row_stride &&
          shape.at("features") == data.features && shape.at("classes") == data.classes,
          "HOLDOUT resident shape/ownership differs from binding");
  d::need(count > 0 && offset < data.rows && count <= data.rows - offset,
          "HOLDOUT gate interval must be nonempty and within declared rows");
  for (const auto* oracle : {&candidate, &baseline})
    d::need(bool(oracle->predict) && oracle->features == data.features && oracle->classes == data.classes &&
            oracle->objective == "multi:softmax" && class_model_contract::sha256(oracle->source_sha256) &&
            class_model_contract::sha256(oracle->library_sha256),
            "HOLDOUT oracle requires matching native class-ID shape and immutable identity pins");
  d::active_device();
  d::device_span(data.values, (data.rows - 1) * data.row_stride + data.features);
  d::device_span(data.labels, data.rows);
  d::ck(cudaDeviceSynchronize(), "HOLDOUT source producer");
  const float* raw = data.values + offset * data.row_stride;
  const auto* labels = data.labels + offset;
  d::Buffer<float> packed;
  if (data.row_stride != data.features) {
    packed = d::Buffer<float>(count * data.features);
    d::pack_interval<<<d::blocks(count * data.features), 256>>>(raw, packed.p, count, data.row_stride, data.features);
    d::done(); raw = packed.p;
  }
  d::Buffer<d::Counts> counts(1); counts.zero();
  const auto* candidate_classes = candidate.predict(raw, count, false);
  d::active_device(); d::device_span(candidate_classes, count);
  d::score_classes<<<d::blocks(count), 256>>>(candidate_classes, labels, count, data.classes, true, counts.p);
  // Consume the borrowed result before baseline.predict, even when both
  // callbacks share a model owner or use the same native output allocation.
  d::done();
  const auto* baseline_classes = baseline.predict(raw, count, false);
  d::active_device(); d::device_span(baseline_classes, count);
  d::score_classes<<<d::blocks(count), 256>>>(baseline_classes, labels, count, data.classes, false, counts.p);
  d::decide<<<1, 1>>>(counts.p); d::done();
  d::Counts result;
  d::ck(cudaMemcpy(&result, counts.p, sizeof(result), cudaMemcpyDeviceToHost), "HOLDOUT gate result");
  d::need(!result.invalid_predictions && !result.invalid_labels, "HOLDOUT gate invalid native class ID or label");
  return {{"rows", count}, {"candidate_errors", result.candidate_errors},
          {"baseline_errors", result.baseline_errors}, {"accepted", result.accepted != 0},
          {"offset", offset}, {"role", "HOLDOUT"}, {"TEST_read", false},
          {"training_allowed", false}, {"selection_allowed", true},
          {"acceptance_rule", "candidate_errors < baseline_errors"},
          {"candidate_source_sha256", candidate.source_sha256}, {"baseline_source_sha256", baseline.source_sha256},
          {"candidate_native_library_sha256", candidate.library_sha256},
          {"baseline_native_library_sha256", baseline.library_sha256},
          {"CUDA_computed", true}, {"filesystem_reads", 0}, {"filesystem_writes", 0}};
}
} // namespace class_study
