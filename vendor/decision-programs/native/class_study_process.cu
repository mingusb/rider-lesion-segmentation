#include "class_io.hpp"
#include "class_model_dataset_contract.hpp"
#include "class_study_process.hpp"
#include "class_study_data.hpp"
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <limits>
#include <iostream>
#include <numeric>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <utility>
namespace class_study_process_detail {
#include "class_dag_storage.cuh"
void require(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
}
// Existing evaluator setup guard; no path operations occur per trial.
void Cpath(const fs::path &p) {
  require(p.is_absolute()&&p.string().find('\0')==std::string::npos,
          "absolute dataset or output path required");
  (void)fs::weakly_canonical(p);
}
#include "class_study_simplify_algorithms.cuh"
struct Stats {
  unsigned long long bad, mismatch, fit_errors, valid_errors;
};
__global__ void pack_rows(const float *src, float *dst, u64 n, u64 stride, u32 F) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < n * u64(F);
       i += u64(blockDim.x) * gridDim.x)
    dst[i] = src[(i / F) * stride + i % F];
}
__global__ void compare_score(const float *native, const u32 *pred, const u32 *y, u64 n,
                              u64 fit, u32 K, bool direct, Stats *s) {
  for (u64 r = u64(blockIdx.x) * blockDim.x + threadIdx.x; r < n;
       r += u64(blockDim.x) * gridDim.x) {
    u32 expected = 0;
    bool bad = false;
    if (direct) {
      float q = native[r];
      bad = !isfinite(q) || q < 0 || double(q) >= K || q != floorf(q);
      if (!bad)
        expected = u32(q);
    } else {
      float best = native[r * K];
      for (u32 k = 0; k < K; ++k) {
        float q = native[r * K + k];
        bad |= !isfinite(q) || q < 0 || q > 1;
        if (q > best) {
          best = q;
          expected = k;
        }
      }
    }
    bad |= y[r] >= K || pred[r] >= K;
    if (bad) {
      atomicAdd(&s->bad, 1ull);
      continue;
    }
    atomicAdd(&s->mismatch, (unsigned long long)(expected != pred[r]));
    if (pred[r] != y[r])
      atomicAdd(r < fit ? &s->fit_errors : &s->valid_errors, 1ull);
  }
}
// Checks the declared borrowed span before any kernel consumes it. This is
// metadata/ownership validation, not host numerical evaluation.
void borrowed_device_span(const void *pointer, u64 elements, int device) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  require(pointer && address % 4 == 0 && elements > 0 &&
              elements <= SIZE_MAX / 4 && elements * 4 <= UINTPTR_MAX - address,
          "resident evaluation borrowed pointer/extent");
  cudaPointerAttributes attributes{};
  ck(cudaPointerGetAttributes(&attributes, pointer), "evaluation pointer attributes");
  require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
          "resident evaluation requires same-device CUDA storage");
  CUdeviceptr base = 0;
  std::size_t bytes = 0;
  auto status = cuMemGetAddressRange(&base, &bytes, static_cast<CUdeviceptr>(address));
  require(status == CUDA_SUCCESS && address >= base && elements * 4 <= bytes &&
              address - base <= bytes - elements * 4,
          "resident evaluation borrowed span exceeds allocation");
}
__global__ void native_score(const float *native, const u32 *y, u64 n,
                             u64 fit, u32 K, bool direct, Stats *s) {
  for (u64 r = u64(blockIdx.x) * blockDim.x + threadIdx.x; r < n;
       r += u64(blockDim.x) * gridDim.x) {
    u32 predicted = 0;
    bool bad = false;
    if (direct) {
      const float q = native[r];
      bad = !isfinite(q) || q < 0 || double(q) >= K || q != floorf(q);
      if (!bad) predicted = u32(q);
    } else {
      float best = native[r * K];
      for (u32 k = 0; k < K; ++k) {
        const float q = native[r * K + k];
        bad |= !isfinite(q) || q < 0 || q > 1;
        if (q > best) { best = q; predicted = k; }
      }
    }
    bad |= y[r] >= K;
    if (bad) { atomicAdd(&s->bad, 1ull); continue; }
    if (predicted != y[r])
      atomicAdd(r < fit ? &s->fit_errors : &s->valid_errors, 1ull);
  }
}
__global__ void prefer_native_candidate(const Stats *stats, u64 candidate_bytes,
                                        u64 incumbent_errors, u64 incumbent_bytes,
                                        u32 *choice) {
  if (!blockIdx.x && !threadIdx.x) {
    const auto errors = stats->valid_errors;
    *choice = errors < incumbent_errors ||
              (errors == incumbent_errors && candidate_bytes < incumbent_bytes);
  }
}
__global__ void finish_accuracy(const Stats *s, double *out, u64 fit, u64 valid) {
  if (!blockIdx.x && !threadIdx.x) {
    out[0] = fit ? double(fit - s->fit_errors) / fit : 0.0;
    out[1] = valid ? double(valid - s->valid_errors) / valid : 0.0;
  }
}
} // namespace class_study_process_detail
namespace class_study {
namespace d = class_study_process_detail;
ConvertedModel simplify_model(ConvertedModel input, std::uint32_t passes) {
  d::require(bool(input.runtime), "simplification requires a model runtime");
  auto m = input.runtime->metadata();
  d::require(input.canonical_bytes.size() >= 64,
             "simplification canonical header extent");
  std::uint32_t F = 0, K = 0;
  std::memcpy(&F, input.canonical_bytes.data() + 12, 4);
  std::memcpy(&K, input.canonical_bytes.data() + 16, 4);
  d::require(F == m.features && K == m.classes,
             "simplification runtime/canonical shape differs");
  auto original =
      d::decode(input.canonical_bytes, m.features, m.classes, m.source_sha256);
  auto reduced = d::simplify(original, m.classes, passes);
  ConvertedModel out;
  out.canonical_bytes = d::encode(reduced.dag, m.features, m.classes, m.source_sha256);
  out.runtime = class_runtime::Runtime::load(
      out.canonical_bytes, dpnative::sha256(out.canonical_bytes), m.source_sha256);
  if (out.runtime->metadata().compact_resident)
    out.compact_bytes = out.runtime->compact_bytes();
  out.metrics = {{"input_nodes", m.nodes},
                 {"nodes", out.runtime->metadata().nodes},
                 {"canonical_bytes", out.canonical_bytes.size()},
                 {"compact_bytes", out.compact_bytes.size()},
                 {"passes", reduced.passes},
                 {"fixed_point", reduced.fixed},
                 {"filesystem_reads", 0},
                 {"filesystem_writes", 0}};
  return out;
}
struct ResidentEvaluation::Impl {
  std::shared_ptr<void> source_owner;
  nlohmann::json binding;
  int device = -1;
  bool native_ready = false;
  bool evaluation_only = false;
  d::Dev<float> values;
  d::Dev<std::uint32_t> labels, pred;
  d::Dev<d::Stats> stats{1};
  d::Dev<double> accuracy{2};
  d::Dev<std::uint32_t> native_choice{1};
  std::uint64_t rows, fit, valid;
  std::uint32_t F, K;
};
ResidentEvaluation::ResidentEvaluation(const nlohmann::json &b) {
  auto shape = class_model_contract::dense(b, false);
  d::require(shape.fit_rows > 0 && shape.valid_rows > 0,
             "resident evaluation requires nonempty FIT and VALID");
  auto xp = b.at("values_path").get<std::string>(),
       yp = b.at("labels_path").get<std::string>();
  d::Cpath(xp);
  d::Cpath(yp);
  auto xb = dpnative::read_text(xp), yb = dpnative::read_text(yp);
  d::require(dpnative::sha256(xb) == b.at("values_sha256").get<std::string>() &&
                 dpnative::sha256(yb) == b.at("labels_sha256").get<std::string>(),
             "initial evaluation data differs");
  d::require(xb.size() == shape.rows * shape.stride * 4 && yb.size() == shape.rows * 4,
             "evaluation data extent");
  p_ = std::make_unique<Impl>();
  auto &p = *p_;
  d::ck(cudaGetDevice(&p.device), "evaluation device");
  p.binding = b;
  p.rows = shape.rows;
  p.fit = shape.fit_rows;
  p.valid = shape.valid_rows;
  p.F = shape.features;
  p.K = shape.classes;
  d::Dev<float> raw(p.rows * shape.stride);
  d::ck(cudaMemcpy(raw.p, xb.data(), xb.size(), cudaMemcpyHostToDevice),
        "evaluation upload");
  p.values = d::Dev<float>(p.rows * p.F);
  p.labels = d::Dev<std::uint32_t>(p.rows);
  p.pred = d::Dev<std::uint32_t>(p.rows);
  d::ck(cudaMemcpy(p.labels.p, yb.data(), yb.size(), cudaMemcpyHostToDevice),
        "labels upload");
  d::pack_rows<<<d::blocks(p.rows * p.F), 256>>>(raw.p, p.values.p, p.rows,
                                                 shape.stride, p.F);
  d::done();
}
ResidentEvaluation::ResidentEvaluation(const ResidentDataView &b) {
  d::require(b.features > 0 && b.features <= std::uint32_t(INT32_MAX) &&
                 b.classes >= 2 && b.rows > 0 && b.row_stride >= b.features &&
                 b.fit_rows > 0 && b.fit_rows < b.rows &&
                 b.valid_rows == b.rows - b.fit_rows &&
                 b.rows <= UINT64_MAX / b.row_stride &&
                 b.rows * b.row_stride <= SIZE_MAX / sizeof(float) &&
                 b.rows <= SIZE_MAX / sizeof(std::uint32_t),
             "resident evaluation data shape/FIT-VALID roles");
  // Keep caller ownership until the synchronous copies finish (and through
  // the session); kernels read active features only, never padded columns.
  p_ = std::make_unique<Impl>();
  auto &p = *p_;
  p.source_owner = b.owner;
  p.binding = b.binding;
  d::ck(cudaGetDevice(&p.device), "evaluation device");
  const auto active_span = (b.rows - 1) * b.row_stride + b.features;
  d::borrowed_device_span(b.values, active_span, p.device);
  d::borrowed_device_span(b.labels, b.rows, p.device);
  d::ck(cudaDeviceSynchronize(), "resident evaluation source producer");
  p.rows = b.rows; p.fit = b.fit_rows; p.valid = b.valid_rows;
  p.F = b.features; p.K = b.classes;
  p.values = d::Dev<float>(p.rows * p.F);
  p.labels = d::Dev<std::uint32_t>(p.rows);
  p.pred = d::Dev<std::uint32_t>(p.rows);
  d::ck(cudaMemcpy(p.labels.p, b.labels, p.rows * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToDevice), "resident evaluation labels copy");
  d::pack_rows<<<d::blocks(p.rows * p.F), 256>>>(
      b.values, p.values.p, p.rows, b.row_stride, p.F);
  d::done();
}
ResidentEvaluation::ResidentEvaluation(const ResidentEvaluationDataView &b) {
  d::require(b.features > 0 && b.features <= std::uint32_t(INT32_MAX) &&
                 b.classes >= 2 && b.classes <= 16777217 && b.rows > 0 && b.row_stride >= b.features &&
                 b.rows <= UINT64_MAX / b.row_stride && b.rows * b.row_stride <= SIZE_MAX / sizeof(float) &&
                 b.rows <= SIZE_MAX / sizeof(std::uint32_t) && bool(b.owner) && b.binding.is_object(),
             "final resident evaluation shape/owner");
  d::require(b.binding.at("role") == "TEST" && b.binding.at("TEST_read").is_boolean() &&
                 b.binding.at("TEST_read").get<bool>() && b.binding.at("training_allowed") == false &&
                 b.binding.at("training_allowed").is_boolean() && b.binding.at("selection_allowed").is_boolean() &&
                 b.binding.at("selection_allowed") == false && class_model_contract::unsigned_integer(b.binding.at("rows"),"final bound rows") == b.rows &&
                 class_model_contract::unsigned_integer(b.binding.at("features"),"final bound features") == b.features && class_model_contract::unsigned_integer(b.binding.at("classes"),"final bound classes") == b.classes &&
                 class_model_contract::unsigned_integer(b.binding.at("row_stride"),"final bound stride") == b.row_stride,
             "final resident evaluation explicit holdout binding");
  p_ = std::make_unique<Impl>(); auto &p = *p_;
  p.evaluation_only = true; p.source_owner = b.owner; p.binding = b.binding;
  d::ck(cudaGetDevice(&p.device), "final evaluation device");
  d::borrowed_device_span(b.values, (b.rows - 1) * b.row_stride + b.features, p.device);
  d::borrowed_device_span(b.labels, b.rows, p.device);
  d::ck(cudaDeviceSynchronize(), "final evaluation source producer");
  p.rows = b.rows; p.fit = 0; p.valid = b.rows; p.F = b.features; p.K = b.classes;
  p.values = d::Dev<float>(p.rows * p.F); p.labels = d::Dev<std::uint32_t>(p.rows);
  p.pred = d::Dev<std::uint32_t>(p.rows);
  d::ck(cudaMemcpy(p.labels.p, b.labels, p.rows * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice), "final labels copy");
  d::pack_rows<<<d::blocks(p.rows * p.F), 256>>>(b.values, p.values.p, p.rows, b.row_stride, p.F);
  d::done();
}
ResidentEvaluation::~ResidentEvaluation() = default;
nlohmann::json ResidentEvaluation::evaluate(const ConvertedModel &m,
                                            const NativeOracle &oracle) {
  auto &p = *p_;
  p.native_ready = false;
  d::require(!p.evaluation_only, "final holdout only supports native inference; no compiled candidate comparisons");
  d::require(bool(m.runtime), "evaluation requires a model runtime");
  const auto &metadata = m.runtime->metadata();
  d::require(metadata.features == p.F && metadata.classes == p.K &&
                 oracle.features == p.F && oracle.classes == p.K &&
                 metadata.source_sha256 == oracle.source_sha256,
             "evaluation model identity/shape");
  d::require(oracle.objective == "multi:softmax" ||
                 oracle.objective == "multi:softprob",
             "evaluation objective");
  auto native = oracle.predict(p.values.p, p.rows, false);
  auto layout = m.runtime->metadata().compact_resident
                    ? class_runtime::Layout::compact8
                    : class_runtime::Layout::canonical16;
  m.runtime->predict({p.values.p, p.rows * p.F, 0, p.rows, p.F}, {p.pred.p, p.rows},
                     layout, class_runtime::Traversal::validated);
  p.stats.zero();
  d::compare_score<<<d::blocks(p.rows), 256>>>(
      native, p.pred.p, p.labels.p, p.rows, p.fit, p.K,
      oracle.objective == "multi:softmax", p.stats.p);
  d::finish_accuracy<<<1, 1>>>(p.stats.p, p.accuracy.p, p.fit, p.valid);
  d::done();
  auto s = p.stats.at(0);
  auto a = p.accuracy.get();
  d::require(!s.bad && !s.mismatch, "resident native/runtime disagreement");
  return {{"rows", p.rows},
          {"class_mismatches", s.mismatch},
          {"FIT_errors", s.fit_errors},
          {"VALID_errors", s.valid_errors},
          {"FIT_accuracy", a[0]},
          {"VALID_accuracy", a[1]},
          {"CUDA_computed", true},
          {"filesystem_reads", 0},
          {"filesystem_writes", 0}};
}
nlohmann::json ResidentEvaluation::evaluate_native(const NativeOracle &oracle) {
  auto &p = *p_;
  p.native_ready = false;
  d::require(bool(oracle.predict) && oracle.features == p.F &&
                 oracle.classes == p.K,
             "native evaluation oracle shape/callback");
  d::require(oracle.objective == "multi:softmax" ||
                 oracle.objective == "multi:softprob",
             "native evaluation objective");
  int current = -1;
  d::ck(cudaGetDevice(&current), "native evaluation active device");
  d::require(current == p.device, "native evaluation device changed");
  const bool direct = oracle.objective == "multi:softmax";
  d::require(direct || p.rows <= UINT64_MAX / p.K,
             "native evaluation output extent overflow");
  const auto *native = oracle.predict(p.values.p, p.rows, false);
  d::borrowed_device_span(native, direct ? p.rows : p.rows * p.K, p.device);
  p.stats.zero();
  d::native_score<<<d::blocks(p.rows), 256>>>(
      native, p.labels.p, p.rows, p.fit, p.K, direct, p.stats.p);
  d::finish_accuracy<<<1, 1>>>(p.stats.p, p.accuracy.p, p.fit, p.valid);
  // Finish consuming the borrowed native result before another callback/train.
  d::done();
  const auto s = p.stats.at(0);
  const auto a = p.accuracy.get();
  d::require(!s.bad, "native evaluation invalid class/probability or label");
  if (p.evaluation_only) {
    return {{"evaluation_role", "TEST"}, {"evaluation_rows", p.rows}, {"evaluation_errors", s.valid_errors},
            {"evaluation_accuracy", a[1]}, {"evaluation_engine", "native CUDA public prediction"},
            {"objective", oracle.objective}, {"source_sha256", oracle.source_sha256},
            {"native_library_sha256", oracle.library_sha256}, {"CUDA_computed", true},
            {"compiled_model_evaluated", false}, {"selection_allowed", false},
            {"filesystem_reads", 0}, {"filesystem_writes", 0}};
  }
  nlohmann::json result = {{"rows", p.rows}, {"FIT_rows", p.fit}, {"VALID_rows", p.valid},
          {"FIT_errors", s.fit_errors}, {"VALID_errors", s.valid_errors},
          {"FIT_accuracy", a[0]}, {"VALID_accuracy", a[1]},
          {"evaluation_engine", "native CUDA public prediction"},
          {"objective", oracle.objective},
          {"source_sha256", oracle.source_sha256},
          {"native_library_sha256", oracle.library_sha256},
          {"CUDA_computed", true}, {"compiled_model_evaluated", false},
          {"filesystem_reads", 0}, {"filesystem_writes", 0}};
  p.native_ready = true;
  return result;
}
bool ResidentEvaluation::prefer_last_native_candidate(
    std::uint64_t candidate_bytes, std::uint64_t incumbent_VALID_errors,
    std::uint64_t incumbent_bytes) {
  auto &p = *p_;
  d::require(!p.evaluation_only && p.native_ready, "native candidate selection excludes final holdout and requires a successful fresh score");
  int current = -1;
  d::ck(cudaGetDevice(&current), "native selection active device");
  d::require(current == p.device, "native selection device changed");
  d::prefer_native_candidate<<<1, 1>>>(
      p.stats.p, candidate_bytes, incumbent_VALID_errors, incumbent_bytes,
      p.native_choice.p);
  d::done();
  return p.native_choice.at(0) != 0;
}
} // namespace class_study
