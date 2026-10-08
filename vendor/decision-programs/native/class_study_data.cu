#include "class_study_data.hpp"
#include <cuda_runtime.h>
#include <openssl/evp.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace class_study::data_detail {
#include "class_model_dataset.cuh"
struct Owner { ModelDataset data; };
__global__ void evaluation_pack_idx(const std::uint8_t* raw, const std::uint8_t* raw_labels,
                                   float* x, u32* y, u64 rows, u32 F, u32 K, u32* bad) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * u64(F);
       i += u64(blockDim.x) * gridDim.x) x[i] = float(raw[i]);
  for (u64 r = u64(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += u64(blockDim.x) * gridDim.x) {
    y[r] = raw_labels[r]; if (y[r] >= K) atomicExch(bad, 1u);
  }
}

// Pure metadata preflight; no file reads or numerical work.
class_model_contract::Dense fit_shape(const nlohmann::json& descriptor) {
  need(descriptor.at("TEST_read").is_boolean() &&
           !descriptor.at("TEST_read").get<bool>(),
       "FIT staging must exclude TEST");
  const auto format = descriptor.at("format").get<std::string>();
  if (format == "dense-fp32-u32-class-labels-1")
    return class_model_contract::dense(descriptor, true);
  need(format == "mnist-idx-permutation-1", "unsupported FIT dataset format");
  const auto field = [&](const char* name) {
    return class_model_contract::unsigned_integer(descriptor.at(name), name);
  };
  need(field("features") == 784 && field("classes") == 10 &&
           field("FIT_rows") == 60000 && field("VALID_rows") == 0,
       "all-training IDX staging requires the official 60000 training rows");
  if (descriptor.contains("rows"))
    need(field("rows") == 60000, "all-training IDX declared row extent");
  if (descriptor.contains("row_stride"))
    need(field("row_stride") == 784, "all-training IDX declared row stride");
  for (const char* name : {"images_path", "labels_path"})
    need(descriptor.at(name).is_string() &&
             !descriptor.at(name).get_ref<const std::string&>().empty(),
         "all-training IDX path type");
  for (const char* name : {"images_sha256", "labels_sha256", "row_ids_sha256"})
    need(class_model_contract::sha256(descriptor.at(name)),
         "all-training IDX byte pins");
  return {784, 10, 60000, 784, 60000, 0};
}

ModelDataset load_fit_idx(const nlohmann::json& descriptor) {
  const auto shape = fit_shape(descriptor);
  fs::path images = descriptor.at("images_path").get<std::string>();
  fs::path labels = descriptor.at("labels_path").get<std::string>();
  Cpath(images);
  Cpath(labels);
  auto image_bytes = mn_read(images), label_bytes = mn_read(labels);
  const auto image_pin = sha256(image_bytes), label_pin = sha256(label_bytes);
  need(image_pin == descriptor.at("images_sha256").get<std::string>() &&
           label_pin == descriptor.at("labels_sha256").get<std::string>(),
       "all-training IDX byte pins differ");
  need(image_bytes.size() == 47040016 && label_bytes.size() == 60008 &&
           be32(image_bytes.data()) == 2051 && be32(image_bytes.data() + 4) == shape.rows &&
           be32(image_bytes.data() + 8) == 28 && be32(image_bytes.data() + 12) == 28 &&
           be32(label_bytes.data()) == 2049 && be32(label_bytes.data() + 4) == shape.rows,
       "official training IDX header/extent differs");
  ModelDataset result;
  result.F = shape.features;
  result.K = shape.classes;
  result.rows = shape.rows;
  result.stride = shape.stride;
  result.fit_rows = shape.fit_rows;
  result.valid_rows = 0;
  Dev<std::uint8_t> raw(result.rows * result.F), raw_labels(result.rows);
  cu(cudaMemcpy(raw.p, image_bytes.data() + 16, raw.n, cudaMemcpyHostToDevice));
  cu(cudaMemcpy(raw_labels.p, label_bytes.data() + 8, raw_labels.n, cudaMemcpyHostToDevice));
  result.x = Dev<float>(result.rows * result.stride);
  result.labels = Dev<u32>(result.rows);
  Dev<float> float_labels(result.rows);
  Dev<u32> ids(result.rows), seen(result.rows), bad(1);
  Dev<PackStats> statistics(1);
  seen.zero();
  bad.zero();
  statistics.zero();
  // Reuse the exact historical uint8->FP32 packing and bit transport audit.
  // All rows belong to FIT; the permutation remains identical to study staging.
  pack<<<blocks(result.rows), 256>>>(raw.p, raw_labels.p, u32(result.rows),
      u32(result.fit_rows), result.x.p, float_labels.p, ids.p, seen.p, statistics.p);
  audit_pack<<<blocks(result.rows), 256>>>(raw.p, raw_labels.p, u32(result.rows),
      result.x.p, float_labels.p, ids.p, seen.p, statistics.p);
  class_labels_from_exact_float<<<blocks(result.rows), 256>>>(
      float_labels.p, result.labels.p, result.rows, bad.p);
  done();
  need(!statistics.at(0).bad && !bad.at(0), "all-training IDX CUDA packing differs");
  auto row_ids = ids.get();
  std::string row_bytes(reinterpret_cast<const char*>(row_ids.data()), row_ids.size() * 4);
  const auto row_pin = sha256(row_bytes);
  need(row_pin == descriptor.at("row_ids_sha256").get<std::string>(),
       "all-training IDX row identities differ");
  result.binding = {{"format", "mnist-idx-permutation-1"}, {"features", result.F},
      {"classes", result.K}, {"rows", result.rows}, {"row_stride", result.stride},
      {"FIT_rows", result.fit_rows}, {"VALID_rows", 0}, {"images_path", images.string()},
      {"images_sha256", image_pin}, {"labels_path", labels.string()},
      {"labels_sha256", label_pin}, {"row_ids_sha256", row_pin},
      {"preprocessing", "GPU exact uint8-to-FP32 raw pixel values, no scaling"},
      {"FIT_only", true}, {"roles", "all official training rows FIT; no VALID or TEST input"},
      {"TEST_read", false}};
  return result;
}
}
namespace class_study {
ResidentDataView stage_dataset(const nlohmann::json& descriptor) {
  namespace d = data_detail;
  d::need(descriptor.at("TEST_read")==false,"study input must exclude TEST");
  auto format=descriptor.at("format").get<std::string>();
  d::need(format=="dense-fp32-u32-class-labels-1" || format=="mnist-idx-permutation-1",
          "unsupported resident study dataset format");
  auto owner=std::make_shared<d::Owner>();
  auto request=format=="dense-fp32-u32-class-labels-1"
      ? nlohmann::json{{"dataset",descriptor}} : descriptor;
  d::ck(cudaSetDevice(0),"study dataset device");
  owner->data=d::load_model_dataset(request,{},false);
  auto& x=owner->data;
  d::need(x.fit_rows>0 && x.valid_rows>0 && x.rows==x.fit_rows+x.valid_rows,
          "accuracy study requires disjoint nonempty FIT and VALID roles");
  auto binding=x.binding;
  binding["dataset_loads"]=1;
  binding["derived_dataset_files_written"]=0;
  binding["FIT_and_VALID_staged_on_GPU"]=true;
  return {x.x.p,x.labels.p,x.rows,x.stride,x.fit_rows,x.valid_rows,x.F,x.K,
          std::move(binding),std::static_pointer_cast<void>(owner)};
}
ResidentDataView stage_fit_dataset(const nlohmann::json& descriptor) {
  namespace d = data_detail;
  const auto shape = d::fit_shape(descriptor);
  auto owner = std::make_shared<d::Owner>();
  d::ck(cudaSetDevice(0), "FIT dataset device");
  if (descriptor.at("format") == "dense-fp32-u32-class-labels-1")
    owner->data = d::load_model_dataset(nlohmann::json{{"dataset", descriptor}}, {}, true);
  else
    owner->data = d::load_fit_idx(descriptor);
  auto& data = owner->data;
  d::need(data.rows == shape.rows && data.fit_rows == data.rows && data.valid_rows == 0 &&
              data.F == shape.features && data.K == shape.classes && data.stride == shape.stride,
          "FIT-only staged extents differ");
  auto binding = data.binding;
  binding["dataset_loads"] = 1;
  binding["derived_dataset_files_written"] = 0;
  binding["FIT_only"] = true;
  binding["FIT_staged_on_GPU"] = true;
  binding["VALID_read"] = false;
  binding["TEST_read"] = false;
  return {data.x.p, data.labels.p, data.rows, data.stride, data.rows, 0,
          data.F, data.K, std::move(binding), std::static_pointer_cast<void>(owner)};
}

nlohmann::json validate_evaluation_dataset(const nlohmann::json& b) {
  namespace c = data_detail::class_model_contract;
  const auto integer = [&](const char* field) { return c::unsigned_integer(b.at(field), field); };
  c::require(b.at("role") == "TEST" && b.at("TEST_read").is_boolean() && b.at("TEST_read").get<bool>(),
             "evaluation staging requires explicit TEST role");
  c::require(b.at("training_allowed").is_boolean() && !b.at("training_allowed").get<bool>() &&
             b.at("selection_allowed").is_boolean() && !b.at("selection_allowed").get<bool>(),
             "evaluation staging forbids training/selection");
  const auto F = integer("features"), K = integer("classes"), rows = integer("rows"), stride = integer("row_stride");
  c::require(F > 0 && F <= INT32_MAX && K >= 2 && K <= 16777217 && rows > 0 && stride >= F &&
             rows <= UINT64_MAX / stride && rows * stride <= SIZE_MAX / 4 && rows <= SIZE_MAX / 4,
             "evaluation staging dynamic shape/extent");
  for (const auto* key : {"FIT_rows", "VALID_rows"})
    if (b.contains(key)) c::require(integer(key) == 0, "final evaluation has no FIT/VALID roles");
  c::require(b.at("preprocessing").is_string() && !b.at("preprocessing").get_ref<const std::string&>().empty(),
             "evaluation input preprocessing declaration");
  const auto format = b.at("format").get<std::string>();
  c::require(format == "dense-fp32-u32-evaluation-labels-1" || format == "mnist-official-test-idx-1",
             "unsupported evaluation-only dataset format");
  const bool idx = format == "mnist-official-test-idx-1";
  if (idx) c::require(F == 784 && K == 10 && rows == 10000 && stride == 784 &&
                      b.at("pixel_transform") == "uint8_to_fp32_exact_no_scaling" && b.at("row_order") == "original_IDX_order",
                      "official test IDX geometry/raw-pixel contract");
  for (const auto* key : {idx ? "images_path" : "values_path", "labels_path"}) {
    c::require(b.at(key).is_string() && !b.at(key).get_ref<const std::string&>().empty(), "evaluation input path");
    const auto& path = b.at(key).get_ref<const std::string&>();
    c::require(path.find('\0') == std::string::npos && std::filesystem::path(path).is_absolute(), "evaluation path must be absolute without NUL");
  }
  c::require(c::sha256(b.at(idx ? "images_sha256" : "values_sha256")) && c::sha256(b.at("labels_sha256")),
             "evaluation input byte pins");
  return {{"format",format}, {"role","TEST"}, {"features",F}, {"classes",K}, {"rows",rows}, {"row_stride",stride},
          {"training_allowed",false}, {"selection_allowed",false}, {"TEST_read",true}, {"metadata_only",true}};
}
ResidentEvaluationDataView stage_evaluation_dataset(const nlohmann::json& b) {
  namespace d = data_detail;
  const auto shape = validate_evaluation_dataset(b);
  const auto rows = shape.at("rows").get<d::u64>(), stride = shape.at("row_stride").get<d::u64>();
  const auto F = shape.at("features").get<d::u32>(), K = shape.at("classes").get<d::u32>();
  const bool idx = b.at("format") == "mnist-official-test-idx-1";
  auto values = d::read_text(b.at(idx ? "images_path" : "values_path").get<std::string>());
  auto labels = d::read_text(b.at("labels_path").get<std::string>());
  const auto values_pin = d::sha256(values), labels_pin = d::sha256(labels);
  d::need(values_pin == b.at(idx ? "images_sha256" : "values_sha256").get<std::string>() &&
          labels_pin == b.at("labels_sha256").get<std::string>(), "final evaluation input bytes differ");
  if (idx) {
    d::need(values.size() == 16 + rows * F && labels.size() == 8 + rows &&
            d::be32(values.data()) == 2051 && d::be32(values.data()+4) == rows &&
            d::be32(values.data()+8) == 28 && d::be32(values.data()+12) == 28 &&
            d::be32(labels.data()) == 2049 && d::be32(labels.data()+4) == rows,
            "official test IDX header/extent differs");
  } else d::need(values.size() == rows * stride * 4 && labels.size() == rows * 4, "evaluation dense byte extent");
  auto owner = std::make_shared<d::Owner>(); auto& data = owner->data;
  data.F=F; data.K=K; data.rows=rows; data.stride=stride; data.fit_rows=0; data.valid_rows=0;
  d::ck(cudaSetDevice(0), "final evaluation dataset device");
  data.x=d::Dev<float>(rows*stride); data.labels=d::Dev<d::u32>(rows);
  if (idx) {
    d::Dev<std::uint8_t> raw(rows*F), raw_labels(rows); d::Dev<d::u32> bad(1); bad.zero();
    d::cu(cudaMemcpy(raw.p,values.data()+16,raw.n,cudaMemcpyHostToDevice));
    d::cu(cudaMemcpy(raw_labels.p,labels.data()+8,raw_labels.n,cudaMemcpyHostToDevice));
    d::evaluation_pack_idx<<<d::blocks(rows*F),256>>>(raw.p,raw_labels.p,data.x.p,data.labels.p,rows,F,K,bad.p);
    d::done(); d::need(!bad.at(0), "final IDX labels outside declared classes");
  } else {
    d::cu(cudaMemcpy(data.x.p,values.data(),values.size(),cudaMemcpyHostToDevice));
    d::cu(cudaMemcpy(data.labels.p,labels.data(),labels.size(),cudaMemcpyHostToDevice));
    d::done();
  }
  auto binding=b; binding["dataset_loads"]=1; binding["derived_dataset_files_written"]=0;
  binding["input_content_reads"]=2; binding["CUDA_staged"]=true; binding["training_allowed"]=false;
  binding["selection_allowed"]=false; binding["FIT_rows"]=0; binding["VALID_rows"]=0;
  return {data.x.p,data.labels.p,rows,stride,F,K,std::move(binding),std::static_pointer_cast<void>(owner)};
}
ResidentDataView fit_prefix(const ResidentDataView& all) {
  data_detail::need(bool(all.owner)&&all.values&&all.labels&&all.fit_rows>0&&
      all.fit_rows<=all.rows&&all.valid_rows==all.rows-all.fit_rows,
      "FIT view requires a complete owned dataset and valid role extents");
  auto fit=all;
  fit.rows=fit.fit_rows;
  fit.valid_rows=0;
  fit.binding["source_rows"]=all.rows;
  fit.binding["rows"]=fit.rows;
  fit.binding["FIT_rows"]=fit.rows;
  fit.binding["VALID_rows"]=0;
  fit.binding["FIT_only"]=true;
  fit.binding["role_scope"]="only the declared FIT prefix is exposed to native optimization";
  return fit;
}
} // namespace class_study
