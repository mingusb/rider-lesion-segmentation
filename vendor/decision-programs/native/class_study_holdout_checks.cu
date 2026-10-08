#include "class_study_holdout.hpp"
#include "class_study_convert.hpp"
#include "class_study_data.hpp"
#include "class_io.hpp"
#include <cuda_runtime.h>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using J = nlohmann::json;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using class_study::NativeOracle;
using class_study::ResidentHoldoutDataView;
static_assert(!std::is_convertible_v<ResidentHoldoutDataView, class_study::ResidentDataView>);
static_assert(!std::is_convertible_v<ResidentHoldoutDataView, class_study::ResidentEvaluationDataView>);
static_assert(!std::is_convertible_v<class_study::ResidentEvaluationDataView, ResidentHoldoutDataView>);
void need(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void ck(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class F> void refused(F action, const char* message) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error(message);
}
template<class T> struct Buffer {
  T* p = nullptr;
  explicit Buffer(u64 count) { ck(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T))); }
  ~Buffer() { if (p) cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
J declaration() {
  return {{"format", "dense-fp32-u32-holdout-labels-1"}, {"role", "HOLDOUT"},
          {"rows", 8}, {"row_stride", 4}, {"features", 2}, {"classes", 3},
          {"TEST_read", false}, {"training_allowed", false}, {"selection_allowed", true},
          {"values_path", "/var/caller-fixtures/holdout.fp32"}, {"labels_path", "/var/caller-fixtures/holdout.u32"},
          {"values_sha256", std::string(64, 'a')}, {"labels_sha256", std::string(64, 'b')},
          {"preprocessing", "synthetic raw FP32 words; padded stride"}};
}
void metadata_checks() {
  const auto valid = declaration();
  const auto shape = class_study::validate_holdout_dataset(valid);
  need(shape.at("role") == "HOLDOUT" && shape.at("rows") == 8 && shape.at("row_stride") == 4 &&
       shape.at("training_allowed") == false && shape.at("selection_allowed") == true &&
       shape.at("TEST_read") == false && shape.at("metadata_only") == true,
       "HOLDOUT preflight lost explicit roles/geometry");
  const std::vector<J> changes = {
    {{"format", "dense-fp32-u32-class-labels-1"}}, {{"format", "dense-fp32-u32-evaluation-labels-1"}},
    {{"role", "TEST"}}, {{"role", "VALID"}}, {{"TEST_read", true}}, {{"TEST_read", 0}},
    {{"training_allowed", true}}, {{"training_allowed", 0}}, {{"selection_allowed", false}}, {{"selection_allowed", 1}},
    {{"FIT_rows", 1}}, {{"VALID_rows", 1}}, {{"FIT_rows", -1}}, {{"VALID_rows", 0.0}},
    {{"rows", 0}}, {{"rows", -1}}, {{"rows", 2.5}}, {{"rows", std::numeric_limits<u64>::max()}},
    {{"row_stride", 1}}, {{"row_stride", std::numeric_limits<u64>::max()}},
    {{"features", 0}}, {{"features", u64(INT32_MAX) + 1}}, {{"classes", 1}}, {{"classes", 16777218}},
    {{"values_path", "relative.fp32"}}, {{"labels_path", ""}}, {{"labels_path", std::string("/tmp/label\0data", 15)}},
    {{"values_sha256", std::string(64, 'A')}}, {{"labels_sha256", std::string(63, 'b')}}, {{"labels_sha256", false}},
    {{"preprocessing", ""}}, {{"preprocessing", false}}
  };
  for (const auto& update : changes) {
    auto bad = valid; bad.update(update);
    refused([&] { (void)class_study::validate_holdout_dataset(bad); }, "HOLDOUT accepted malformed descriptor");
  }
  for (const char* field : {"rows", "row_stride", "features", "classes", "role", "TEST_read", "values_sha256"}) {
    auto bad = valid; bad.erase(field);
    refused([&] { (void)class_study::validate_holdout_dataset(bad); }, "HOLDOUT accepted incomplete descriptor");
  }
  auto explicit_zero = valid; explicit_zero["FIT_rows"] = 0; explicit_zero["VALID_rows"] = 0;
  (void)class_study::validate_holdout_dataset(explicit_zero);
}
template<class T> std::string bytes(const std::vector<T>& values) {
  return std::string(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}
void write(const std::filesystem::path& path, const std::string& content) {
  std::ofstream file(path, std::ios::binary); file.exceptions(std::ios::badbit | std::ios::failbit);
  file.write(content.data(), std::streamsize(content.size())); file.close();
}
__global__ void prediction_fixture(const float* x, u64 rows, u32 features, u32 classes,
                                    const u32* expected, float* output, int mode, u32* bad) {
  for (u64 r = u64(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += u64(blockDim.x) * gridDim.x) {
    for (u32 f = 0; f < features; ++f)
      if (__float_as_uint(x[r * features + f]) != expected[r * features + f]) atomicOr(bad, 1u);
    const float correct = x[r * features];
    switch (mode) {
      case 0: output[r] = correct; break;
      case 1: output[r] = float((u32(correct) + 1) % classes); break;
      case 2: output[r] = r == 0 ? float((u32(correct) + 1) % classes) : correct; break;
      case 3: output[r] = __uint_as_float(0x7fc00001); break;
      case 4: output[r] = -1.0f; break;
      case 5: output[r] = 0.5f; break;
      default: output[r] = float(classes); break;
    }
  }
}
struct OracleOwner {
  Buffer<float> output;
  Buffer<u32> expected, bad{1};
  const float* last_input = nullptr;
  u64 last_rows = 0, calls = 0;
  OracleOwner(const std::vector<u32>& words, u64 rows) : output(rows), expected(words.size()) {
    ck(cudaMemcpy(expected.p, words.data(), words.size() * sizeof(u32), cudaMemcpyHostToDevice));
  }
};
NativeOracle oracle(const std::shared_ptr<OracleOwner>& owner, u64 offset, int mode) {
  NativeOracle result;
  result.features = 2; result.classes = 3; result.objective = "multi:softmax";
  result.source_sha256 = std::string(64, mode == 0 ? 'a' : 'b'); result.library_sha256 = std::string(64, 'c');
  result.predict = [owner, offset, mode](const float* x, u64 rows, bool margin) {
    need(!margin, "HOLDOUT fixture asked for margin predictions");
    owner->last_input = x; owner->last_rows = rows; ++owner->calls;
    ck(cudaMemset(owner->bad.p, 0, sizeof(u32)));
    prediction_fixture<<<1, 64>>>(x, rows, 2, 3, owner->expected.p + offset * 2,
                                  owner->output.p, mode, owner->bad.p);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    u32 invalid = 0; ck(cudaMemcpy(&invalid, owner->bad.p, sizeof(invalid), cudaMemcpyDeviceToHost));
    need(!invalid, "HOLDOUT gate changed input raw words/order/stride");
    return owner->output.p;
  };
  return result;
}
void gpu_checks(const std::filesystem::path& root) {
  need(root.is_absolute() && !std::filesystem::exists(root), "fresh absolute HOLDOUT fixture directory required");
  std::filesystem::create_directories(root);
  const std::vector<u32> y{0, 1, 2, 0, 2, 1, 0, 2};
  const std::vector<u32> second_words{0x80000000, 0x00000001, 0x7fc01234, 0xff800000,
                                     0x00800000, 0x3f800001, 0x00000000, 0x7f800000};
  std::vector<float> raw(8 * 4, std::bit_cast<float>(u32(0x7fc0ffff)));
  std::vector<u32> active_words;
  for (u64 r = 0; r < y.size(); ++r) {
    raw[r * 4] = float(y[r]); raw[r * 4 + 1] = std::bit_cast<float>(second_words[r]);
    active_words.push_back(std::bit_cast<u32>(raw[r * 4])); active_words.push_back(second_words[r]);
  }
  const auto values_bytes = bytes(raw), label_bytes = bytes(y);
  auto descriptor = declaration();
  descriptor["values_path"] = (root / "values.fp32").string();
  descriptor["labels_path"] = (root / "labels.u32").string();
  descriptor["values_sha256"] = dpnative::sha256(values_bytes);
  descriptor["labels_sha256"] = dpnative::sha256(label_bytes);
  write(root / "values.fp32", values_bytes); write(root / "labels.u32", label_bytes);
  auto data = class_study::stage_holdout_dataset(descriptor);
  need(data.rows == 8 && data.row_stride == 4 && data.binding.at("all_labels_validated_on_CUDA") == true &&
       data.binding.at("dataset_loads") == 1 && data.binding.at("input_content_reads") == 2,
       "HOLDOUT staging receipt/geometry differs");
  auto shared = std::make_shared<OracleOwner>(active_words, 8);
  const auto good = oracle(shared, 2, 0), wrong = oracle(shared, 2, 1), one_wrong = oracle(shared, 2, 2);
  const auto win = class_study::evaluate_holdout_gate(data, 2, 3, good, wrong);
  need(win.at("rows") == 3 && win.at("candidate_errors") == 0 && win.at("baseline_errors") == 3 &&
       win.at("accepted") == true && win.at("CUDA_computed") == true && win.at("filesystem_reads") == 0,
       "HOLDOUT strict win/shared borrowed output accounting failed");
  const auto loss = class_study::evaluate_holdout_gate(data, 2, 3, wrong, good);
  need(loss.at("candidate_errors") == 3 && loss.at("baseline_errors") == 0 && loss.at("accepted") == false,
       "HOLDOUT accepted strictly worse candidate");
  const auto tie = class_study::evaluate_holdout_gate(data, 2, 3, one_wrong, one_wrong);
  need(tie.at("candidate_errors") == 1 && tie.at("baseline_errors") == 1 && tie.at("accepted") == false,
       "HOLDOUT accepted tie");
  need(shared->calls == 6 && shared->last_rows == 3, "HOLDOUT gate evaluated extra rows/models");
  const auto before_refusals = shared->calls;
  for (const auto& interval : std::vector<std::pair<u64, u64>>{{0, 0}, {8, 1}, {7, 2}, {UINT64_MAX, 1}, {1, UINT64_MAX}})
    refused([&] { (void)class_study::evaluate_holdout_gate(data, interval.first, interval.second, good, wrong); },
            "HOLDOUT gate accepted invalid interval");
  auto bad_view = data; bad_view.rows = 9;
  refused([&] { (void)class_study::evaluate_holdout_gate(bad_view, 2, 3, good, wrong); }, "HOLDOUT accepted forged extent");
  bad_view = data; bad_view.owner.reset();
  refused([&] { (void)class_study::evaluate_holdout_gate(bad_view, 2, 3, good, wrong); }, "HOLDOUT accepted ownerless view");
  auto bad_oracle = good; bad_oracle.objective = "multi:softprob";
  refused([&] { (void)class_study::evaluate_holdout_gate(data, 2, 3, bad_oracle, wrong); }, "HOLDOUT accepted probability output");
  bad_oracle = good; bad_oracle.features = 3;
  refused([&] { (void)class_study::evaluate_holdout_gate(data, 2, 3, bad_oracle, wrong); }, "HOLDOUT accepted mismatched oracle");
  bad_oracle = good; bad_oracle.source_sha256 = "unbound";
  refused([&] { (void)class_study::evaluate_holdout_gate(data, 2, 3, bad_oracle, wrong); }, "HOLDOUT accepted unpinned oracle");
  need(shared->calls == before_refusals, "HOLDOUT interval/schema refusal invoked predictions");
  for (int mode : {3, 4, 5, 6}) {
    auto invalid = oracle(shared, 2, mode);
    refused([&] { (void)class_study::evaluate_holdout_gate(data, 2, 3, invalid, wrong); }, "HOLDOUT accepted invalid float class ID");
    refused([&] { (void)class_study::evaluate_holdout_gate(data, 2, 3, good, invalid); }, "HOLDOUT accepted invalid baseline class ID");
  }
  std::vector<float> host_output{0, 0, 0};
  bad_oracle = good; bad_oracle.predict = [&](const float*, u64, bool) { return host_output.data(); };
  refused([&] { (void)class_study::evaluate_holdout_gate(data, 2, 3, bad_oracle, wrong); }, "HOLDOUT accepted CPU prediction buffer");
  Buffer<float> short_output(1);
  bad_oracle.predict = [&](const float*, u64, bool) { return short_output.p; };
  refused([&] { (void)class_study::evaluate_holdout_gate(data, 2, 3, bad_oracle, wrong); }, "HOLDOUT consumed undersized prediction allocation");
  Buffer<u32> invalid_labels(8); auto bad_labels = y; bad_labels[2] = 3;
  ck(cudaMemcpy(invalid_labels.p, bad_labels.data(), 8 * sizeof(u32), cudaMemcpyHostToDevice));
  bad_view = data; bad_view.labels = invalid_labels.p;
  refused([&] { (void)class_study::evaluate_holdout_gate(bad_view, 2, 3, good, wrong); }, "HOLDOUT accepted invalid borrowed label");
  auto invalid_stage = descriptor; bad_labels = y; bad_labels.back() = 3;
  const auto invalid_label_bytes = bytes(bad_labels);
  write(root / "invalid-labels.u32", invalid_label_bytes);
  invalid_stage["labels_path"] = (root / "invalid-labels.u32").string();
  invalid_stage["labels_sha256"] = dpnative::sha256(invalid_label_bytes);
  refused([&] { (void)class_study::stage_holdout_dataset(invalid_stage); }, "HOLDOUT staging failed to validate all labels");
  auto pin_mismatch = descriptor; pin_mismatch["values_sha256"] = std::string(64, '0');
  refused([&] { (void)class_study::stage_holdout_dataset(pin_mismatch); }, "HOLDOUT accepted wrong source byte pin");
  write(root / "oversized.fp32", values_bytes + std::string(4, '\0'));
  auto oversized = descriptor; oversized["values_path"] = (root / "oversized.fp32").string();
  oversized["values_sha256"] = dpnative::sha256(values_bytes + std::string(4, '\0'));
  refused([&] { (void)class_study::stage_holdout_dataset(oversized); }, "HOLDOUT accepted excess source bytes");
  // Contiguous transport reaches the callback directly and also supports the
  // final one-row interval; padded fixtures above verify GPU packing bitwise.
  std::vector<float> contiguous;
  for (u32 word : active_words) contiguous.push_back(std::bit_cast<float>(word));
  auto compact_descriptor = descriptor; compact_descriptor["row_stride"] = 2;
  const auto compact_bytes = bytes(contiguous);
  write(root / "contiguous.fp32", compact_bytes);
  compact_descriptor["values_path"] = (root / "contiguous.fp32").string();
  compact_descriptor["values_sha256"] = dpnative::sha256(compact_bytes);
  auto compact_data = class_study::stage_holdout_dataset(compact_descriptor);
  auto last_good = oracle(shared, 7, 0), last_wrong = oracle(shared, 7, 1);
  auto last = class_study::evaluate_holdout_gate(compact_data, 7, 1, last_good, last_wrong);
  need(last.at("candidate_errors") == 0 && last.at("baseline_errors") == 1 && last.at("accepted") == true &&
       shared->last_input == compact_data.values + 7 * 2,
       "HOLDOUT contiguous final interval changed its extent/input");
}
} // namespace
int main(int argc, char** argv) {
  try {
    metadata_checks();
    if (argc == 2 && std::string(argv[1]) == "--metadata-only") {
      std::cout << "HOLDOUT metadata checks passed without CUDA initialization\n"; return 0;
    }
    need(argc == 2, "usage: class_study_holdout_checks --metadata-only|/absolute/fresh-fixture-directory");
    ck(cudaSetDevice(0)); gpu_checks(argv[1]);
    std::cout << "HOLDOUT CUDA checks passed\n"; return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
