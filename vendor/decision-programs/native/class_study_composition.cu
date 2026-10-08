#include "class_study_composition.hpp"
#include "class_study_train.hpp"
#include "class_model_dataset_contract.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <climits>
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace class_study::composition_detail {
using J = nlohmann::json;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
void need(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void ck(cudaError_t code, const char* operation) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
u64 integer(const J& value, const char* message) {
  need(value.is_number_integer(), message);
  if (value.is_number_unsigned()) return value.get<u64>();
  auto n = value.get<std::int64_t>(); need(n >= 0, message); return u64(n);
}
u64 product(u64 a, u64 b, const char* message) {
  need(!b || a <= UINT64_MAX / b, message); return a * b;
}
u64 byte_extent(u64 rows, u64 columns, const char* message) {
  auto n = product(product(rows, columns, message), sizeof(float), message);
  need(n <= SIZE_MAX, message); return n;
}
std::string hash(const void* bytes, std::size_t length) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{}; unsigned count = 0;
  need(EVP_Digest(bytes, length, digest.data(), &count, EVP_sha256(), nullptr) == 1,
       "composition SHA256 failed");
  std::ostringstream out; out << std::hex << std::setfill('0');
  for (unsigned i = 0; i < count; ++i) out << std::setw(2) << unsigned(digest[i]);
  return out.str();
}
std::string model_bytes(const J& model) {
  need(model.at("model").is_binary(), "composition model payload is not opaque binary");
  const auto& bytes = model.at("model").get_binary();
  need(!bytes.empty(), "composition model payload is empty");
  return std::string(bytes.begin(), bytes.end());
}
J model_identity(const J& model) {
  need(model.is_object() && model.at("sha256").is_string(), "composition model identity type");
  const auto& bytes = model.at("model");
  need(bytes.is_binary() && !bytes.get_binary().empty(), "composition opaque model payload");
  auto pin = model.at("sha256").get<std::string>();
  const auto& binary = bytes.get_binary();
  need(hash(binary.data(), binary.size()) == pin, "composition model SHA256 differs");
  return {{"sha256", pin}, {"bytes", binary.size()}};
}
void allocation(const void* pointer, u64 bytes, const char* operation) {
  need(pointer && reinterpret_cast<std::uintptr_t>(pointer) % alignof(float) == 0 &&
       bytes && bytes <= UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(pointer), operation);
  cudaPointerAttributes attributes{}; ck(cudaPointerGetAttributes(&attributes, pointer), operation);
  need(attributes.type == cudaMemoryTypeDevice && attributes.device == 0, operation);
  CUdeviceptr base = 0; std::size_t size = 0;
  auto address = reinterpret_cast<CUdeviceptr>(pointer);
  need(cuMemGetAddressRange(&base, &size, address) == CUDA_SUCCESS && address >= base &&
       bytes <= size && address - base <= size - bytes, operation);
}
} // namespace class_study::composition_detail

namespace class_study {
namespace cd = composition_detail;
using J = nlohmann::json;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
J ResidentComposition::validate_bundle(const J& bundle, const CompositionOptions& options) {
  cd::need(bundle.is_object() && bundle.at("format") == "native-nonlinear-deployment-bundle-1",
           "unsupported composition bundle format");
  auto kind = bundle.at("selected_kind").get<std::string>();
  cd::need(kind == "native_teacher" || kind == "nonlinear_combination", "unsupported composition selected kind");
  u64 F = cd::integer(bundle.at("raw_features"), "composition raw feature type");
  u64 K = cd::integer(bundle.at("classes"), "composition class type");
  cd::need(F > 0 && F <= INT32_MAX && K >= 2 && K <= 16777217,
           "composition native feature/class capacity");
  cd::need(options.response_gpu_byte_budget, "composition response byte budget must be positive");
  cd::need(bundle.at("native_library_path").is_string() &&
           bundle.at("native_library_path").get<std::string>().find('\0') == std::string::npos &&
           std::filesystem::path(bundle.at("native_library_path").get<std::string>()).is_absolute() &&
           class_model_contract::sha256(bundle.at("native_library_sha256")),
           "composition native library identity differs");
  const auto& contract = bundle.at("raw_dataset_contract");
  cd::need(contract.is_object() && cd::integer(contract.at("features"), "composition dataset feature type") == F &&
           cd::integer(contract.at("classes"), "composition dataset class type") == K &&
           contract.at("TEST_read").is_boolean() && !contract.at("TEST_read").get<bool>(),
           "composition raw dataset provenance shape/TEST role differs");
  cd::need(bundle.at("final_class_contract") == "selected native multi:softmax public class" &&
           bundle.at("compiled_single_tree").is_boolean() && !bundle.at("compiled_single_tree").get<bool>(),
           "composition class/representation contract differs");
  const auto& models = bundle.at("teacher_models"); const auto& order = bundle.at("teacher_order");
  cd::need(models.is_object() && order.is_array(), "composition teacher map/order type");
  J declarations = J::object(), normalized_order = J::array();
  u64 columns = 0, probability_columns = 0, bytes = 0, teacher_bytes = 0;
  J raw_mapping;
  std::vector<u32> raw_indices;
  if(bundle.contains("meta_input_mapping")){
    const auto& mapping=bundle.at("meta_input_mapping");
    cd::need(kind=="nonlinear_combination"&&mapping.is_object()&&mapping.size()==2&&
      mapping.contains("version")&&cd::integer(mapping.at("version"),"composition mapping version type")==1&&
      mapping.contains("raw_feature_indices")&&mapping.at("raw_feature_indices").is_array()&&
      !mapping.at("raw_feature_indices").empty(),"composition raw mapping schema/version differs");
    std::set<u32> used;
    for(const auto& value:mapping.at("raw_feature_indices")){
      const auto feature=cd::integer(value,"composition raw feature index type");
      cd::need(feature<F&&used.insert(u32(feature)).second,"composition raw mapping must be unique and in range");
      raw_indices.push_back(u32(feature));
    }
    raw_mapping={{"version",1},{"raw_feature_indices",raw_indices}};
  }
  J selected;
  if (kind == "native_teacher") {
    cd::need(models.empty() && order.empty() && !bundle.contains("meta_model") && bundle.contains("native_model"),
             "single native teacher bundle contains composition models");
    selected = cd::model_identity(bundle.at("native_model")); bytes = selected.at("bytes").get<u64>();
  } else {
    cd::need(!order.empty() && order.size() <= UINT32_MAX && !bundle.contains("native_model") &&
             bundle.at("teacher_probability_contract") ==
               "native full-round gbtree clone; derived multi:softprob; FP32 teacher-major/class-minor; no CPU transform",
             "composition teacher response contract differs");
    probability_columns = cd::product(order.size(), K, "composition response feature overflow");
    cd::need(probability_columns <= INT32_MAX && raw_indices.size() <= u64(INT32_MAX)-probability_columns, "composition meta feature capacity");
    columns = probability_columns + raw_indices.size();
    std::set<std::string> referenced;
    for (const auto& entry : order) {
      cd::need(entry.is_string(), "composition teacher reference type");
      auto pin = entry.get<std::string>(); cd::need(models.contains(pin), "composition teacher reference missing");
      normalized_order.push_back(pin); referenced.insert(pin);
    }
    cd::need(referenced.size() == models.size(), "composition model map contains unreferenced models");
    for (const auto& [pin, model] : models.items()) {
      auto item = cd::model_identity(model);
      cd::need(item.at("sha256") == pin && referenced.contains(pin), "composition teacher model-map key differs");
      auto n = item.at("bytes").get<u64>(); cd::need(bytes <= UINT64_MAX - n, "composition model byte overflow");
      bytes += n; declarations[pin] = std::move(item);
    }
    teacher_bytes = bytes;
    if (options.teacher_owner_window)
      cd::need(options.teacher_model_host_byte_budget && teacher_bytes <= options.teacher_model_host_byte_budget,
               "composition retained teacher-model host byte budget refusal");
    selected = cd::model_identity(bundle.at("meta_model"));
    auto n = selected.at("bytes").get<u64>(); cd::need(bytes <= UINT64_MAX - n, "composition total model bytes overflow"); bytes += n;
    cd::need(cd::byte_extent(1, columns, "composition minimum response buffer") <= options.response_gpu_byte_budget,
             "composition budget cannot hold one response row");
  }
  J semantic = {{"format", bundle.at("format")}, {"selected_kind", kind}, {"raw_features", F}, {"classes", K},
                {"native_library_path", bundle.at("native_library_path")}, {"native_library_sha256", bundle.at("native_library_sha256")},
                {"raw_dataset_contract", contract}, {"teacher_order", normalized_order},
                {"teacher_models", declarations}, {"selected_model", selected}};
  if(!raw_indices.empty()) semantic["meta_input_mapping"]=raw_mapping;
  auto encoded = semantic.dump();
  semantic["composition_identity_sha256"] = cd::hash(encoded.data(), encoded.size());
  semantic["identity_scope"] = "normalized declared bundle semantics and opaque model pins; not original CBOR byte encoding";
  if(!raw_indices.empty()){
    semantic["native_probability_columns"]=probability_columns;
    semantic["raw_passthrough_columns"]=raw_indices.size();
  }
  semantic["response_features"] = columns; semantic["response_gpu_byte_budget"] = options.response_gpu_byte_budget;
  semantic["unique_native_model_bytes"] = bytes;
  // Keep the legacy eager validation declaration byte-for-byte compatible with
  // retained study checkpoints. Runtime metadata() still reports this policy.
  if(options.teacher_owner_window){
    semantic["unique_teacher_model_bytes"] = teacher_bytes;
    semantic["teacher_owner_window"] = options.teacher_owner_window;
    semantic["teacher_model_host_byte_budget"] = options.teacher_model_host_byte_budget;
    semantic["teacher_owner_policy"] = "bounded sequential native owners with retained RAM model buffers";
  }
  semantic["native_model_shape_checked_by_CAPI_at_load"] = false;
  return semantic;
}
struct ResidentComposition::Impl {
  J declaration;
  CompositionOptions options;
  u32 F = 0, K = 0, columns = 0, probability_columns = 0;
  std::vector<u32> raw_indices;
  bool combined = false;
  std::vector<std::unique_ptr<ResidentPredictor>> teachers;
  std::vector<std::size_t> order;
  // Used only by optional bounded ownership; indices retain unique teacher
  // identity independently of the reusable native owner's current model.
  std::vector<std::string> teacher_buffers, teacher_pins;
  std::vector<std::size_t> slot_sources;
  std::vector<bool> checked_teachers;
  std::size_t next_slot = 0;
  u64 teacher_model_loads = 0, retained_teacher_host_bytes = 0;
  std::unique_ptr<ResidentPredictor> selected;
  float* response = nullptr;
  u64 capacity_bytes = 0, prediction_calls = 0, probability_calls = 0, packing_calls = 0;
  u64 response_allocations = 0, maximum_response_bytes = 0, raw_packing_calls = 0;
  explicit Impl(const J& bundle, const CompositionOptions& opts) : options(opts) {
    declaration = ResidentComposition::validate_bundle(bundle, options);
    F = u32(declaration.at("raw_features").get<u64>()); K = u32(declaration.at("classes").get<u64>());
    columns = u32(declaration.at("response_features").get<u64>());
    combined = declaration.at("selected_kind") == "nonlinear_combination";
    if(declaration.contains("meta_input_mapping"))raw_indices=declaration.at("meta_input_mapping").at("raw_feature_indices").get<std::vector<u32>>();
    probability_columns=columns-u32(raw_indices.size());
    J plan = {{"native_library_path", bundle.at("native_library_path")},
              {"native_library_sha256", bundle.at("native_library_sha256")}};
    if (!combined) {
      selected = std::make_unique<ResidentPredictor>(plan, F, K);
      const auto& model = bundle.at("native_model");
      selected->restore_model(cd::model_bytes(model), model.at("sha256").get<std::string>());
    } else {
      std::map<std::string, std::size_t> owners;
      for (const auto& ref : bundle.at("teacher_order")) {
        auto pin = ref.get<std::string>(); auto found = owners.find(pin);
        if (found == owners.end()) {
          const auto& model = bundle.at("teacher_models").at(pin);
          const auto index = owners.size(); owners.emplace(pin, index); order.push_back(index);
          if (options.teacher_owner_window) {
            teacher_buffers.push_back(cd::model_bytes(model)); teacher_pins.push_back(pin);
            retained_teacher_host_bytes += teacher_buffers.back().size();
          } else {
            auto owner = teachers.empty() ? std::make_unique<ResidentPredictor>(plan, F, K)
                : std::make_unique<ResidentPredictor>(plan, F, K, *teachers.front());
            owner->restore_model(cd::model_bytes(model), pin); ++teacher_model_loads;
            teachers.push_back(std::move(owner));
          }
        } else order.push_back(found->second);
      }
      if (options.teacher_owner_window) {
        checked_teachers.resize(teacher_buffers.size(), false);
        const auto count = std::min<std::size_t>(options.teacher_owner_window, teacher_buffers.size());
        for (std::size_t source = 0; source < count; ++source) {
          auto owner = teachers.empty() ? std::make_unique<ResidentPredictor>(plan, F, K)
              : std::make_unique<ResidentPredictor>(plan, F, K, *teachers.front());
          owner->restore_model(teacher_buffers[source], teacher_pins[source]); ++teacher_model_loads;
          teachers.push_back(std::move(owner)); slot_sources.push_back(source); checked_teachers[source] = true;
        }
      }
      selected = std::make_unique<ResidentPredictor>(plan, columns, K, *teachers.front());
      const auto& model = bundle.at("meta_model");
      selected->restore_model(cd::model_bytes(model), model.at("sha256").get<std::string>());
    }
  }
  ~Impl() { if (response) cudaFree(response); }
  ResidentPredictor& teacher_owner(std::size_t source) {
    if (!options.teacher_owner_window) return *teachers.at(source);
    for (std::size_t slot = 0; slot < slot_sources.size(); ++slot)
      if (slot_sources[slot] == source) return *teachers.at(slot);
    cd::need(!teachers.empty() && source < teacher_buffers.size(), "composition bounded teacher owner/index missing");
    const auto slot = next_slot;
    // Loading is transactional in ResidentPredictor. A refused replacement
    // leaves the old slot binding/current model usable. Copy consumers have
    // synchronized before a following column can reach this replacement.
    teachers[slot]->restore_model(teacher_buffers[source], teacher_pins[source]);
    slot_sources[slot] = source; checked_teachers[source] = true; ++teacher_model_loads;
    next_slot = (slot + 1) % teachers.size(); return *teachers[slot];
  }
  void ensure_response(u64 bytes) {
    cd::need(bytes && bytes <= options.response_gpu_byte_budget, "composition response buffer budget refusal");
    if (bytes <= capacity_bytes) return;
    // Beginning a new predict invalidates the previous borrowed result. Complete
    // any default-stream reads and free the old buffer before growth; old+new
    // response allocations never silently exceed the declared buffer budget.
    cd::ck(cudaStreamSynchronize(nullptr), "composition response growth completion");
    if (response) { cd::ck(cudaFree(response), "composition old response free"); response = nullptr; capacity_bytes = 0; }
    cd::ck(cudaMalloc(reinterpret_cast<void**>(&response), std::size_t(bytes)), "composition response allocate");
    cd::allocation(response, bytes, "composition private response allocation/extent");
    capacity_bytes = bytes; ++response_allocations; maximum_response_bytes = std::max(maximum_response_bytes, bytes);
  }
};
ResidentComposition::ResidentComposition(const J& bundle, const CompositionOptions& options)
    : p_(std::make_unique<Impl>(bundle, options)) {}
ResidentComposition::~ResidentComposition() = default;
ResidentComposition::ResidentComposition(ResidentComposition&&) noexcept = default;
ResidentComposition& ResidentComposition::operator=(ResidentComposition&&) noexcept = default;
const float* ResidentComposition::predict(const float* input, u64 rows, bool margin) {
  cd::need(bool(p_), "moved-from composition owner"); auto& i = *p_;
  cd::need(rows > 0, "composition prediction requires nonempty rows");
  auto input_bytes = cd::byte_extent(rows, i.F, "composition input geometry overflow");
  cd::byte_extent(rows, margin ? i.K : 1, "composition output geometry overflow");
  cd::ck(cudaSetDevice(0), "composition prediction device");
  cd::allocation(input, input_bytes, "composition input allocation/extent/device");
  if (!i.combined) {
    auto* out = i.selected->predict(input, rows, margin); ++i.prediction_calls; return out;
  }
  auto response_bytes = cd::byte_extent(rows, i.columns, "composition response geometry overflow");
  if (i.response) {
    auto a = reinterpret_cast<std::uintptr_t>(input), b = reinterpret_cast<std::uintptr_t>(i.response);
    cd::need(a >= b + i.capacity_bytes || b >= a + input_bytes, "composition input aliases private response buffer");
  }
  i.ensure_response(response_bytes);
  auto width = std::size_t(u64(i.K) * sizeof(float)), pitch = std::size_t(u64(i.columns) * sizeof(float));
  cd::need(width <= pitch, "composition column copy geometry");
  for (std::size_t teacher = 0; teacher < i.order.size(); ++teacher) {
    auto* source = i.teacher_owner(i.order[teacher]).predict_probabilities(input, rows);
    cd::allocation(source, cd::byte_extent(rows, i.K, "composition borrowed response geometry"),
                   "composition borrowed probability extent/device");
    cd::ck(cudaMemcpy2D(i.response + teacher * u64(i.K), pitch, source, width,
                       width, std::size_t(rows), cudaMemcpyDeviceToDevice), "composition teacher column pack");
    // Finish reading the borrowed native view before the next native API call.
    cd::ck(cudaStreamSynchronize(nullptr), "composition borrowed response copy completion");
    ++i.probability_calls; ++i.packing_calls;
  }
  for(std::size_t column=0;column<i.raw_indices.size();++column){
    cd::ck(cudaMemcpy2D(i.response+i.probability_columns+column,pitch,input+i.raw_indices[column],
      std::size_t(u64(i.F)*sizeof(float)),sizeof(float),std::size_t(rows),cudaMemcpyDeviceToDevice),
      "composition raw column pack");
    ++i.raw_packing_calls;
  }
  if(!i.raw_indices.empty())cd::ck(cudaStreamSynchronize(nullptr),"composition raw copy completion");
  auto* out = i.selected->predict(i.response, rows, margin); ++i.prediction_calls; return out;
}
u32 ResidentComposition::features() const { cd::need(bool(p_), "moved-from composition owner"); return p_->F; }
u32 ResidentComposition::classes() const { cd::need(bool(p_), "moved-from composition owner"); return p_->K; }
u32 ResidentComposition::response_features() const { cd::need(bool(p_), "moved-from composition owner"); return p_->columns; }
J ResidentComposition::metadata() const {
  cd::need(bool(p_), "moved-from composition owner"); const auto& i = *p_;
  auto result = i.declaration;
  result["unique_teacher_model_bytes"] = i.combined
      ? i.declaration.at("unique_native_model_bytes").get<u64>()-i.declaration.at("selected_model").at("bytes").get<u64>() : 0;
  result["teacher_owner_window"] = i.options.teacher_owner_window;
  result["teacher_model_host_byte_budget"] = i.options.teacher_model_host_byte_budget;
  result["teacher_owner_policy"] = i.options.teacher_owner_window ? "bounded sequential native owners with retained RAM model buffers" : "eager unique native teacher owners";
  const auto checked = i.checked_teachers.empty() ? i.teachers.size()
      : std::count(i.checked_teachers.begin(), i.checked_teachers.end(), true);
  result["native_model_shape_checked_by_CAPI_at_load"] = !i.combined || !i.options.teacher_owner_window || checked == i.teacher_buffers.size();
  result["teacher_shape_validation_scope"] = "eager owners check every model at setup; bounded owners check each model before its first response, never bypassing native shape/objective checks";
  result["inference_only"] = true; result["FIT_data_retained"] = false;
  result["DMatrix_creations"] = 0; result["native_training_calls"] = 0;
  result["component_file_reads"] = 0; result["component_file_writes"] = 0;
  result["native_library_content_checks_at_session_setup"] = 1;
  result["CPU_model_predictions"] = false;
  result["prediction_calls"] = i.prediction_calls; result["teacher_probability_calls"] = i.probability_calls;
  result["D2D_column_copy_calls"] = i.packing_calls;
  result["response_allocations"] = i.response_allocations; result["current_response_CUDA_bytes"] = i.capacity_bytes;
  result["maximum_response_CUDA_bytes"] = i.maximum_response_bytes;
  result["native_internal_allocation_peak_bounded"] = false;
  result["memory_scope"] = "response buffer explicitly bounded; native original/derived models and prediction buffers opaque";
  result["unique_teacher_owners"] = i.teachers.size();
  result["unique_teacher_models"] = i.declaration.at("teacher_models").size();
  result["retained_teacher_host_bytes"] = i.retained_teacher_host_bytes;
  result["teacher_model_buffer_loads"] = i.teacher_model_loads;
  result["teacher_models_shape_checked_unique"] = checked;
  result["bounded_owner_host_budget_scope"] = "retained opaque teacher buffers only; caller bundle/checkpoint copies and native host objects are additional";
  result["packing"] = "checked cudaMemcpy2D D2D; exact FP32 words; teacher-major/class-minor";
  if(!i.raw_indices.empty()){
    result["raw_D2D_column_copy_calls"]=i.raw_packing_calls;
    result["packing"]="checked cudaMemcpy2D D2D; exact FP32 words; teacher-major/class-minor followed by declared raw feature order";
  }
  result["source_hard_class_probability_argmax_equivalence_claim"] = false;
  result["selected_predictor"] = i.selected->metadata();
  J native_teachers = J::array(); for (const auto& t : i.teachers) native_teachers.push_back(t->metadata());
  result["teacher_predictors"] = std::move(native_teachers);
  return result;
}
} // namespace class_study

