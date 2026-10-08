#include "class_study_oof.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <climits>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <utility>

namespace class_study::oof_detail {
using J = nlohmann::json;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
void need(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
void ck(cudaError_t code, const char* operation) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
void done() {
  ck(cudaGetLastError(), "OOF CUDA launch");
  ck(cudaDeviceSynchronize(), "OOF CUDA completion");
}
u64 mul(u64 a, u64 b, const char* message) {
  need(!b || a <= UINT64_MAX / b, message); return a * b;
}
u64 add(u64 a, u64 b, const char* message) {
  need(a <= UINT64_MAX - b, message); return a + b;
}
u64 extent(u64 n, u64 word, const char* message) {
  auto bytes = mul(n, word, message); need(bytes <= SIZE_MAX, message); return bytes;
}
std::string hash(const void* data, std::size_t bytes) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{}; unsigned size = 0;
  need(EVP_Digest(data, bytes, digest.data(), &size, EVP_sha256(), nullptr) == 1,
       "OOF opaque buffer SHA256 failed");
  std::ostringstream out; out << std::hex << std::setfill('0');
  for (unsigned i = 0; i < size; ++i) out << std::setw(2) << unsigned(digest[i]);
  return out.str();
}
u64 integer(const J& value, const char* message) {
  need(value.is_number_integer(), message);
  if (value.is_number_unsigned()) return value.get<u64>();
  auto result = value.get<std::int64_t>(); need(result >= 0, message); return u64(result);
}
unsigned blocks(u64 n) { return unsigned(std::min<u64>((n + 255) / 256, 65535)); }
void allocation(const void* p, u64 bytes, const char* operation) {
  need(p && bytes && bytes <= UINTPTR_MAX - reinterpret_cast<std::uintptr_t>(p), operation);
  cudaPointerAttributes a{}; ck(cudaPointerGetAttributes(&a, p), operation);
  need(a.type == cudaMemoryTypeDevice && a.device == 0, operation);
  CUdeviceptr base = 0; std::size_t size = 0;
  auto address = reinterpret_cast<CUdeviceptr>(p);
  need(cuMemGetAddressRange(&base, &size, address) == CUDA_SUCCESS && address >= base &&
       bytes <= size && address - base <= size - bytes, operation);
}
template<class T> struct Buffer {
  T* p = nullptr; u64 n = 0;
  Buffer() = default;
  explicit Buffer(u64 count) : n(count) {
    if (n) ck(cudaMalloc(reinterpret_cast<void**>(&p), extent(n, sizeof(T), "OOF allocation overflow")), "OOF allocate");
  }
  ~Buffer() { if (p) cudaFree(p); }
  Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : p(std::exchange(other.p, nullptr)), n(std::exchange(other.n, 0)) {}
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      if (p) cudaFree(p);
      p = std::exchange(other.p, nullptr); n = std::exchange(other.n, 0);
    }
    return *this;
  }
  void zero() { if (n) ck(cudaMemset(p, 0, n * sizeof(T)), "OOF initialize"); }
  u64 bytes() const { return n * sizeof(T); }
};
__device__ u32 assigned_fold(u64 row, u32 folds, u32 rotation) {
  return u32((row % folds + u64(rotation)) % folds);
}
__device__ u64 fold_row(u64 ordinal, u32 folds, u32 first, bool complement) {
  if (!complement) return u64(first) + ordinal * folds;
  u64 group = ordinal / (folds - 1), within = ordinal % (folds - 1);
  return group * folds + within + u64(within >= first);
}
__global__ void copy_labels(const u32* input, u32* output, u64 rows, u32 K, u32* bad) {
  for (u64 r = u64(blockIdx.x) * blockDim.x + threadIdx.x; r < rows; r += u64(blockDim.x) * gridDim.x) {
    u32 y = input[r]; if (y >= K) atomicOr(bad, 1u); output[r] = y;
  }
}
__global__ void gather_fold(const float* input, const u32* labels, u64 FIT_rows,
    u64 stride, u32 F, u32 K, u32 folds, u32 rotation, u32 fold, u32 first,
    bool complement, u64 rows, float* output, u32* out_labels, u64* row_ids, u32* bad) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * u64(F); i += u64(blockDim.x) * gridDim.x) {
    u64 local = i / F, r = fold_row(local, folds, first, complement); u32 c = u32(i % F);
    if (r >= FIT_rows || (assigned_fold(r, folds, rotation) != fold) != complement) {
      atomicOr(bad, 2u); continue;
    }
    float value = input[r * stride + c]; if (isinf(value)) atomicOr(bad, 4u); output[i] = value;
    if (!c) {
      row_ids[local] = r;
      if (complement) {
        u32 label = labels[r]; if (label >= K) atomicOr(bad, 8u); out_labels[local] = label;
      }
    }
  }
}
__global__ void audit_gather(const float* input, const u32* labels, u64 FIT_rows,
    u64 stride, u32 F, u32 folds, u32 rotation, u32 fold, bool complement,
    u64 rows, const float* output, const u32* out_labels, const u64* row_ids, u32* bad) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * u64(F); i += u64(blockDim.x) * gridDim.x) {
    u64 local = i / F, r = row_ids[local]; u32 c = u32(i % F);
    if (r >= FIT_rows || (assigned_fold(r, folds, rotation) != fold) != complement) {
      atomicOr(bad, 16u); continue;
    }
    if (__float_as_uint(output[i]) != __float_as_uint(input[r * stride + c]) ||
        (!c && local && row_ids[local - 1] >= r) ||
        (!c && complement && out_labels[local] != labels[r])) atomicOr(bad, 32u);
  }
}
__global__ void gather_valid(const float* input, u64 start, u64 rows, u64 stride, u32 F, float* output, u32* bad) {
  for (u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * u64(F); i += u64(blockDim.x) * gridDim.x) {
    float value = input[(start + i / F) * stride + i % F]; if (isinf(value)) atomicOr(bad, 64u); output[i] = value;
  }
}
__global__ void scatter(const float* probabilities, u64 rows, u32 teacher, u32 M,
    u32 K, const u64* row_ids, u64 base, u64 total_rows, float* response, u32* coverage, u32* bad) {
  for (u64 local = u64(blockIdx.x) * blockDim.x + threadIdx.x; local < rows; local += u64(blockDim.x) * gridDim.x) {
    u64 r = row_ids ? row_ids[local] : base + local;
    if (r >= total_rows) { atomicOr(bad, 128u); continue; }
    double sum = 0; bool invalid = false;
    for (u32 k = 0; k < K; ++k) {
      float p = probabilities[local * K + k]; invalid |= !isfinite(p) || p < 0 || p > 1; sum += double(p);
    }
    if (invalid || fabs(sum - 1.0) > 0.001) { atomicOr(bad, 256u); continue; }
    if (atomicCAS(coverage + r * M + teacher, 0u, 1u) != 0) { atomicOr(bad, 512u); continue; }
    for (u32 k = 0; k < K; ++k) response[(r * M + teacher) * K + k] = probabilities[local * K + k];
  }
}
__global__ void audit_cells(const float* response, const u32* coverage, u64 rows,
    u64 FIT_rows, u32 M, u32 K, u32 folds, u32 rotation, u64 oof_cursor, u64 valid_cursor, u32* bad,
    const u32* imported = nullptr) {
  for (u64 cell = u64(blockIdx.x) * blockDim.x + threadIdx.x; cell < rows * u64(M); cell += u64(blockDim.x) * gridDim.x) {
    u64 r = cell / M; u32 teacher = u32(cell % M);
    bool expected = (imported && imported[teacher]) || (r < FIT_rows
        ? u64(assigned_fold(r, folds, rotation)) * M + teacher < oof_cursor : teacher < valid_cursor);
    if (coverage[cell] != u32(expected)) atomicOr(bad, 1024u);
    double sum = 0;
    for (u32 k = 0; k < K; ++k) {
      float p = response[cell * K + k];
      if (expected) { if (!isfinite(p) || p < 0 || p > 1) atomicOr(bad, 2048u); sum += double(p); }
      else if (__float_as_uint(p) != 0u) atomicOr(bad, 4096u);
    }
    if (expected && fabs(sum - 1.0) > 0.001) atomicOr(bad, 8192u);
  }
}
struct FoldOwner {
  Buffer<float> train_values, holdout_values; Buffer<u32> train_labels;
  Buffer<u64> train_ids, holdout_ids; u64 training_rows, prediction_rows;
  FoldOwner(u64 train, u64 held, u32 F) : train_values(train * F), holdout_values(held * F),
      train_labels(train), train_ids(train), holdout_ids(held), training_rows(train), prediction_rows(held) {}
};

J round_groups(const J& declarations) {
  std::map<std::string,u32> ids; J groups = J::array(), by_teacher = J::array();
  for (u32 t = 0; t < declarations.size(); ++t) {
    auto hp = declarations.at(t).at("hyperparameters");
    const auto rounds = integer(hp.at("rounds"), "OOF group round extent"); hp.erase("rounds");
    const auto key = hp.dump(); auto found = ids.find(key);
    if (found == ids.end()) {
      const auto g = u32(groups.size()); ids.emplace(key, g);
      groups.push_back({{"hyperparameters_except_rounds",hp},{"teachers",J::array()},{"maximum_rounds",rounds}});
      found = ids.find(key);
    }
    auto& group = groups.at(found->second); group.at("teachers").push_back(t);
    group["maximum_rounds"] = std::max(integer(group.at("maximum_rounds"), "OOF group maximum rounds"), rounds);
    by_teacher.push_back(found->second);
  }
  return {{"format","resident-native-oof-round-groups-1"},{"groups",groups},{"group_by_teacher",by_teacher},
          {"contract","exact normalized hyperparameters except rounds; original target order; maximum unmatched rounds per fold"}};
}
bool grouped(const J& layout) { return layout.contains("round_prefix_groups"); }
J remaining_import_work(const J& layout,const J& map) {
  const u64 M=integer(layout.at("teachers"),"OOF remaining teacher extent");
  const u64 folds=integer(layout.at("folds"),"OOF remaining fold extent");
  need(M&&map.is_array()&&map.size()==M&&folds>=2,"OOF remaining import map/shape");
  u64 unmatched=0;for(const auto& from:map)unmatched+=from.is_null();
  const auto units=mul(unmatched,folds,"OOF remaining response units");
  J result={{"remaining_teacher_fold_response_units",units}};
  if(!grouped(layout)){
    result["remaining_teacher_fold_FIT_calls"]=units;
    result["remaining_native_FIT_calls"]=units;
    result["remaining_native_FIT_calls_basis"]="one independent complement FIT per unmatched teacher-fold response unit";
  }else{
    const auto& groups=layout.at("round_prefix_groups").at("groups");need(groups.is_array(),"OOF remaining round groups type");
    u64 active=0;for(const auto& group:groups){bool needed=false;
      need(group.at("teachers").is_array()&&!group.at("teachers").empty(),"OOF remaining round group extent");
      for(const auto& t:group.at("teachers")){const auto id=integer(t,"OOF remaining group teacher index");need(id<M,"OOF remaining group teacher outside bank");needed|=map.at(std::size_t(id)).is_null();}
      active+=needed;
    }
    result["remaining_teacher_fold_FIT_calls"]=nullptr;
    result["remaining_teacher_fold_FIT_calls_scope"]="deprecated for grouped prefixes; null because response units are not FIT calls";
    result["round_trajectory_groups"]=groups.size();result["unmatched_round_trajectory_groups"]=active;
    result["remaining_native_FIT_calls"]=mul(active,folds,"OOF remaining maximum-round FIT calls");
    result["remaining_native_FIT_calls_basis"]="scheduled maximum-round complement FITs: unmatched normalized trajectories times folds; conditional on completing the declared grouped route";
  }
  return result;
}
u32 teacher_group(const J& layout, u32 teacher) {
  return u32(integer(layout.at("round_prefix_groups").at("group_by_teacher").at(teacher), "OOF teacher group"));
}
u32 required_rounds(const J& layout, const J& imports, u32 group) {
  u64 maximum = 0;
  for (const auto& t : layout.at("round_prefix_groups").at("groups").at(group).at("teachers")) {
    const auto teacher = u32(integer(t, "OOF group teacher"));
    if (imports.is_null() || imports.at("teacher_map").at(teacher).is_null())
      maximum = std::max(maximum, integer(layout.at("teacher_declarations").at(teacher).at("hyperparameters").at("rounds"), "OOF required group rounds"));
  }
  need(maximum && maximum <= INT_MAX, "OOF unmatched group maximum round extent"); return u32(maximum);
}

// Metadata/opaque-byte checks only. The numerical donor cells are audited on
// CUDA before their first copy, and target cells are audited on every restore.
void saved_receipts(const J& s, const J& layout, bool rigorous = false) {
  const u64 M = integer(layout.at("teachers"), "OOF saved teacher extent");
  const u64 fit = integer(layout.at("FIT_rows"), "OOF saved FIT extent");
  const u64 valid_rows = integer(layout.at("VALID_rows"), "OOF saved VALID extent");
  const u64 folds = integer(layout.at("folds"), "OOF saved fold extent");
  const u64 rotation = integer(layout.at("rotation"), "OOF saved rotation");
  need(M && M <= UINT32_MAX && folds >= 2 && folds <= fit && rotation < folds,
       "OOF saved fold/teacher capacity");
  const u64 oof = integer(s.at("oof_cursor"), "OOF saved fold cursor type");
  const u64 valid = integer(s.at("valid_cursor"), "OOF saved VALID cursor type");
  const u64 units = mul(M, folds, "OOF saved unit extent");
  need(oof <= units && valid <= (valid_rows ? M : 0) && (!valid || oof == units),
       "OOF saved cursor ordering/extent");
  const char* phase = oof < units ? "oof_training" : valid_rows && valid < M ? "valid_deployment" : "complete";
  need(s.at("phase") == phase && s.at("fold_results").is_array() && s.at("fold_results").size() == oof &&
       s.at("VALID_results").is_array() && s.at("VALID_results").size() == valid,
       "OOF saved phase/result extents");
  const auto& declared = layout.at("teacher_declarations");
  need(declared.is_array() && declared.size() == M, "OOF saved declaration extent");
  auto probability = [&](const J& view, const J& pin, const J& hp) {
    need(view.at("source_model_sha256") == pin && view.at("source_objective") == "multi:softmax" &&
         view.at("derived_objective") == "multi:softprob" && view.at("operation") == "native_full_round_probability_view" &&
         view.at("native_clone_method") == "XGBoosterSlice" && view.at("FIT_performed") == false &&
         view.at("source_objective_unchanged") == true && view.at("native_library_sha256") == layout.at("native_library_sha256") &&
         view.at("features") == layout.at("features") && view.at("classes") == layout.at("classes") &&
         view.at("rounds") == hp.at("rounds"), "OOF imported probability contract differs");
  };
  for (u64 unit = 0; unit < oof; ++unit) {
    const u64 fold = unit / M, teacher = unit % M;
    const auto& r = s.at("fold_results").at(std::size_t(unit));
    const u64 first = (fold + folds - rotation) % folds;
    const u64 held = 1 + (fit - 1 - first) / folds;
    const auto& training = r.at("training");
    const bool range = training.value("operation", std::string()) == "native_existing_round_prefix_response";
    need(integer(r.at("fold"), "OOF saved fold type") == fold &&
         integer(r.at("teacher"), "OOF saved teacher type") == teacher &&
         integer(r.at("training_rows"), "OOF saved complement rows") == fit - held &&
         integer(r.at("prediction_rows"), "OOF saved held rows") == held &&
         training.at("hyperparameters") == declared.at(std::size_t(teacher)).at("hyperparameters") &&
         (range ? !r.contains("model_sha256") && !training.contains("model_sha256") :
                  r.at("model_sha256") == training.at("model_sha256")) &&
         r.at("FIT_exclusion_verified_on_CUDA") == true && r.at("VALID_read_for_training") == false &&
         r.at("TEST_read") == false && training.at("VALID_read") == false && training.at("TEST_read") == false,
         "OOF saved fold receipt differs");
    const bool prefix = range || training.value("operation", std::string()) == "native_existing_round_prefix";
    if (prefix) {
      const auto& parent = r.at("round_prefix_parent");
      const auto& fit_receipt = parent.at("training"); auto hp = fit_receipt.at("hyperparameters"); hp.erase("rounds");
      auto wanted_hp = declared.at(std::size_t(teacher)).at("hyperparameters"); wanted_hp.erase("rounds");
      need(integer(parent.at("fold"), "OOF prefix parent fold") == fold &&
           integer(parent.at("group"), "OOF prefix parent group") <= UINT32_MAX && hp == wanted_hp &&
           parent.at("model_sha256") == fit_receipt.at("model_sha256") &&
           parent.at("model_bytes") == fit_receipt.at("model_bytes") &&
           training.at("source_parent_sha256") == parent.at("model_sha256") &&
           training.at("original_rounds") == fit_receipt.at("hyperparameters").at("rounds") &&
           training.at("selected_rounds") == training.at("hyperparameters").at("rounds") &&
           integer(training.at("selected_rounds"), "OOF selected prefix rounds") <=
             integer(training.at("original_rounds"), "OOF original prefix rounds") &&
           training.at("FIT_performed") == false && training.at("CUDA_training") == false &&
           training.at("native_library_sha256") == layout.at("native_library_sha256") &&
           integer(fit_receipt.at("FIT_rows"), "OOF parent FIT extent") == fit - held &&
           fit_receipt.at("features") == layout.at("features") && fit_receipt.at("classes") == layout.at("classes") &&
           fit_receipt.at("native_objective") == "multi:softmax" && fit_receipt.at("CUDA_training") == true &&
           fit_receipt.at("VALID_read") == false && fit_receipt.at("TEST_read") == false,
           "OOF saved native prefix/parent FIT contract differs");
    }
    if (rigorous || prefix) {
      need(training.at("features") == layout.at("features") && training.at("classes") == layout.at("classes") &&
           integer(training.at("FIT_rows"), "OOF imported training FIT extent") == fit - held &&
           training.at("native_objective") == "multi:softmax" && (prefix || training.at("CUDA_training") == true),
           "OOF imported fold training contract differs");
      if(range){
        const auto& parent=r.at("round_prefix_parent");const auto& source=r.at("response_source");
        const auto& view=r.at("derived_probability_view");
        need(source==J{{"format","native-round-prefix-response-1"},{"parent_model_sha256",parent.at("model_sha256")},
          {"iteration_begin",0},{"iteration_end",training.at("selected_rounds")},{"source_objective","multi:softmax"},
          {"derived_objective","multi:softprob"},{"native_library_sha256",layout.at("native_library_sha256")}},
          "OOF prefix response provenance differs");
        need(view.at("response_source")==source&&view.at("source_model_sha256")==parent.at("model_sha256")&&
          view.at("source_objective")=="multi:softmax"&&view.at("derived_objective")=="multi:softprob"&&
          view.at("operation")=="native_iteration_range_probability_view"&&view.at("native_clone_method")=="XGBoosterSlice"&&
          view.at("FIT_performed")==false&&view.at("source_objective_unchanged")==true&&
          view.at("native_library_sha256")==layout.at("native_library_sha256")&&
          view.at("features")==layout.at("features")&&view.at("classes")==layout.at("classes")&&
          view.at("iteration_begin")==0&&view.at("iteration_end")==training.at("selected_rounds")&&
          view.at("rounds")==training.at("selected_rounds")&&view.at("source_parent_rounds")==training.at("original_rounds"),
          "OOF native range probability contract differs");
      }else probability(r.at("derived_probability_view"), r.at("model_sha256"), training.at("hyperparameters"));
    }
  }
  for (u64 teacher = 0; teacher < valid; ++teacher) {
    const auto& r = s.at("VALID_results").at(std::size_t(teacher));
    need(integer(r.at("teacher"), "OOF saved VALID teacher") == teacher &&
         r.at("full_fit_model_sha256") == declared.at(std::size_t(teacher)).at("full_fit_model_sha256") &&
         integer(r.at("rows"), "OOF saved VALID rows") == valid_rows &&
         integer(r.at("native_training_calls"), "OOF saved VALID training count") == 0,
         "OOF saved deployment receipt differs");
    if (rigorous) probability(r.at("derived_probability_view"), r.at("full_fit_model_sha256"),
                            declared.at(std::size_t(teacher)).at("hyperparameters"));
  }
}
void common_import_contract(const J& donor, const J& target) {
  need(donor.at("format") == "resident-native-oof-identity-1" &&
       donor.at("fold_assignment") == target.at("fold_assignment") &&
       donor.at("column_order") == target.at("column_order") &&
       donor.at("response_objective") == target.at("response_objective"), "OOF imported response/fold grammar differs");
  const auto& a = donor.at("layout"); const auto& b = target.at("layout");
  for (const char* key : {"features","classes","rows","FIT_rows","VALID_rows","folds","seed","rotation",
                         "source_dataset_binding","native_library_path","native_library_sha256"})
    need(a.at(key) == b.at(key), std::string("OOF import common contract differs: ") + key);
}
J canonical_complete_donor(J s) {
  need((s.at("format") == "resident-native-oof-state-1" || s.at("format") == "resident-native-oof-state-2" ||
        s.at("format") == "resident-native-oof-state-3" || s.at("format") == "resident-native-oof-state-4") &&
       s.at("phase") == "complete", "OOF column import requires a complete supported donor");
  // A complete donor has canonical evidence for every unit. Flatten older
  // import history instead of retaining recursively nested donor snapshots.
  s["format"] = "resident-native-oof-state-1"; s.erase("imports");
  s.erase("round_prefix_parents"); s.erase("round_prefix_actual_FIT_calls"); s.erase("round_prefix_slice_units");
  s.erase("round_prefix_response_units");
  s.at("identity").erase("completed_column_import");
  saved_receipts(s, s.at("identity").at("layout"), true); return s;
}
J match_columns(const J& donor_identity, const J& target_identity) {
  common_import_contract(donor_identity, target_identity);
  const auto& a = donor_identity.at("layout").at("teacher_declarations");
  const auto& b = target_identity.at("layout").at("teacher_declarations");
  need(a.is_array() && !a.empty() && b.is_array() && !b.empty(), "OOF import teacher declarations missing");
  std::map<std::string,u64> lookup; std::map<std::string,bool> target_seen;
  for (std::size_t i = 0; i < a.size(); ++i)
    need(lookup.emplace(a.at(i).dump(), i).second, "OOF donor teacher match is ambiguous");
  J map = J::array(); u64 matches = 0;
  for (const auto& declaration : b) {
    const auto key = declaration.dump();
    need(target_seen.emplace(key, true).second, "OOF target teacher match is ambiguous");
    const auto found = lookup.find(key);
    if (found == lookup.end()) map.push_back(nullptr);
    else { map.push_back(found->second); ++matches; }
  }
  need(matches, "OOF completed donor has no exact matching target teachers"); return map;
}
void payload(const OofCheckpoint& s, const J& layout) {
  const u64 rows = integer(layout.at("rows"), "OOF payload rows");
  const u64 M = integer(layout.at("teachers"), "OOF payload teachers");
  const u64 K = integer(layout.at("classes"), "OOF payload classes");
  const u64 cells = mul(rows, M, "OOF payload cells");
  const u64 rb = extent(mul(cells, K, "OOF payload probabilities"), 4, "OOF payload response bytes");
  const u64 cb = extent(cells, 4, "OOF payload coverage bytes");
  need(s.response_bytes.size() == rb && s.coverage_bytes.size() == cb &&
       layout.at("response_bytes") == rb && layout.at("coverage_bytes") == cb &&
       s.state.at("response_sha256") == hash(s.response_bytes.data(), s.response_bytes.size()) &&
       s.state.at("coverage_sha256") == hash(s.coverage_bytes.data(), s.coverage_bytes.size()),
       "OOF checkpoint payload extent/SHA differs");
}
J import_binding(const J& donor, const J& map, const J& provenance) {
  need(provenance.is_object() && provenance.at("checkpoint").is_string() &&
       provenance.at("checkpoint").get<std::string>().starts_with("/") &&
       provenance.at("generation").is_string(), "OOF import committed donor provenance missing");
  const auto generation = provenance.at("generation").get<std::string>();
  need(generation.starts_with("generation-") && generation.find_first_of("/\\") == std::string::npos,
       "OOF import donor generation invalid");
  return {{"format", "resident-native-oof-column-import-1"}, {"provenance", provenance},
          {"donor_identity", donor.at("identity")}, {"donor_response_sha256", donor.at("response_sha256")},
          {"donor_coverage_sha256", donor.at("coverage_sha256")}, {"teacher_map", map}};
}
J checked_imports(const J& imports, const J& base_identity) {
  need(imports.is_object(), "OOF saved imports missing");
  auto donor = canonical_complete_donor(imports.at("donor_state"));
  need(donor == imports.at("donor_state"), "OOF saved donor evidence is not canonical");
  auto map = match_columns(donor.at("identity"), base_identity);
  need(map == imports.at("teacher_map"), "OOF saved imported teacher mapping differs");
  auto binding = import_binding(donor, map, imports.at("provenance"));
  need(binding == imports.at("binding"), "OOF saved import identity differs"); return binding;
}
struct PrefixRestore {
  std::vector<TrainingResult> parents;
  u32 fold = UINT32_MAX;
  u64 fits = 0, slices = 0;
};
PrefixRestore checked_prefix_parents(const OofCheckpoint& saved, const J& layout, const J& imports) {
  const auto& s = saved.state; const u64 M = integer(layout.at("teachers"), "OOF prefix teacher count");
  const u64 cursor = integer(s.at("oof_cursor"), "OOF prefix cursor");
  const u64 units = mul(M, integer(layout.at("folds"), "OOF prefix folds"), "OOF prefix units");
  PrefixRestore out; out.parents.resize(layout.at("round_prefix_groups").at("groups").size());
  std::set<std::pair<u64,u32>> fits; u64 slices = 0; J expected = J::array();
  std::map<u32,J> current;
  for (u64 unit = 0; unit < cursor; ++unit) {
    const auto teacher = u32(unit % M), group = teacher_group(layout, teacher);
    if (!imports.is_null() && !imports.at("teacher_map").at(teacher).is_null()) continue;
    const auto& r = s.at("fold_results").at(std::size_t(unit)); const auto& parent = r.at("round_prefix_parent");
    need(r.at("training").at("operation") == (layout.value("round_prefix_iteration_range",false)?
           "native_existing_round_prefix_response":"native_existing_round_prefix") &&
         integer(parent.at("group"), "OOF saved group identity") == group &&
         integer(parent.at("training").at("hyperparameters").at("rounds"), "OOF saved group rounds") ==
           required_rounds(layout, imports, group), "OOF saved target prefix trajectory differs");
    fits.emplace(unit / M, group); ++slices;
    if (cursor < units && unit / M == cursor / M) {
      auto [at, inserted] = current.emplace(group, parent);
      need(inserted || at->second == parent, "OOF saved current-fold parent receipt changed");
    }
  }
  for (const auto& [group,parent] : current) expected.push_back(parent);
  need(s.at("round_prefix_parents") == expected && saved.round_prefix_parent_bytes.size() == expected.size(),
       "OOF saved current-fold parent extent/metadata differs");
  out.fits = integer(s.at("round_prefix_actual_FIT_calls"), "OOF saved actual prefix FIT count");
  out.slices = integer(s.at(layout.value("round_prefix_iteration_range",false)?
      "round_prefix_response_units":"round_prefix_slice_units"), "OOF saved prefix response count");
  need(out.fits == fits.size() && out.slices == slices, "OOF actual FIT/prefix response counters differ");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto& parent = expected.at(index); const auto group = u32(integer(parent.at("group"), "OOF restored group"));
    const auto& bytes = saved.round_prefix_parent_bytes[index];
    need(!bytes.empty() && parent.at("model_bytes") == bytes.size() &&
         parent.at("model_sha256") == hash(bytes.data(), bytes.size()), "OOF retained prefix parent extent/SHA differs");
    auto& dst = out.parents.at(group); dst.model_json.assign(bytes.begin(), bytes.end());
    dst.model_sha256 = parent.at("model_sha256").get<std::string>(); dst.metrics = parent.at("training");
    out.fold = u32(cursor / M);
  }
  return out;
}
} // namespace class_study::oof_detail

namespace class_study {
namespace d = oof_detail;
using J = nlohmann::json; using u32 = std::uint32_t; using u64 = std::uint64_t;
OofFailure::OofFailure(const std::string& message, J statistics)
    : std::runtime_error(message), partial_statistics(std::move(statistics)) {}
J ResidentOof::validate_metadata(const J& plan, const ResidentDataView& data,
    const std::vector<OofTeacher>& teachers, const OofOptions& options) {
  d::need(data.owner && data.values && data.labels && data.binding.is_object(), "OOF requires owned resident data and binding");
  d::need(data.features && data.features <= u32(INT32_MAX) && data.classes >= 2, "OOF source feature/class capacity");
  d::need(data.fit_rows && data.fit_rows <= data.rows && data.valid_rows == data.rows - data.fit_rows &&
          data.row_stride >= data.features, "OOF role/stride extents");
  d::need(data.binding.at("TEST_read").is_boolean() && !data.binding.at("TEST_read").get<bool>(), "OOF input cannot contain TEST");
  d::extent(d::mul(data.rows, data.row_stride, "OOF source row extent"), 4, "OOF source byte extent");
  d::extent(data.rows, 4, "OOF label extent");
  auto fit = fit_prefix(data); auto training = ResidentTrainer::validate_metadata(plan, fit);
  d::need(options.folds >= 2 && options.folds <= data.fit_rows && options.gpu_byte_budget,
          "OOF folds require nonempty complement/holdout and positive byte budget");
  d::need(!options.round_prefix_iteration_range||options.group_round_prefixes,"OOF iteration-range responses require round grouping");
  d::need(!teachers.empty() && teachers.size() <= UINT32_MAX, "OOF nonempty teacher capacity");
  u64 M = teachers.size(), columns = d::mul(M, data.classes, "OOF teacher/class capacity");
  d::need(columns <= u64(INT32_MAX), "OOF derived feature capacity");
  u64 cells = d::mul(data.rows, M, "OOF coverage extent");
  u64 response_bytes = d::extent(d::mul(cells, data.classes, "OOF response extent"), 4, "OOF response bytes");
  u64 coverage_bytes = d::extent(cells, 4, "OOF coverage bytes");
  u64 label_bytes = d::extent(data.rows, 4, "OOF copied label bytes");
  u64 persistent = d::add(d::add(response_bytes, coverage_bytes, "OOF persistent extent"),
      d::add(label_bytes, 4, "OOF persistent label/flag extent"), "OOF persistent total");
  u64 anchor = d::extent(d::mul(data.fit_rows, u64(data.features) + 1, "OOF full-FIT trainer copies"), 4, "OOF full-FIT trainer bytes");
  u64 max_train = data.fit_rows - data.fit_rows / options.folds;
  u64 gather = d::add(d::extent(d::mul(data.fit_rows, data.features, "OOF fold gathered values"), 4, "OOF fold values bytes"),
      d::add(d::extent(max_train, 4, "OOF fold labels bytes"), d::extent(data.fit_rows, 8, "OOF fold IDs bytes"),
             "OOF fold labels/IDs"), "OOF fold gather total");
  u64 trainer = d::extent(d::mul(max_train, u64(data.features) + 1, "OOF complement trainer copies"), 4, "OOF complement trainer bytes");
  u64 valid = d::extent(d::mul(data.valid_rows, data.features, "OOF packed VALID extent"), 4, "OOF packed VALID bytes");
  u64 peak = d::add(d::add(persistent, anchor, "OOF persistent/trainer peak"),
      std::max(d::add(gather, trainer, "OOF fold peak"), valid), "OOF known peak");
  d::need(peak <= options.gpu_byte_budget, "OOF explicit/known-copy GPU byte budget refusal");
  J declared = J::array();
  for (const auto& teacher : teachers) {
    auto hp = ResidentTrainer::validate_hyperparameters(teacher.hyperparameters, data.classes);
    const auto& model = teacher.full_fit_model;
    d::need(teacher.full_fit_data_binding == fit.binding, "OOF deployment full-FIT binding differs");
    d::need(!model.model_json.empty() && d::hash(model.model_json.data(), model.model_json.size()) == model.model_sha256,
            "OOF deployment model buffer identity differs");
    const auto& m = model.metrics;
    d::need(m.is_object() && m.at("hyperparameters") == hp &&
            d::integer(m.at("features"), "OOF teacher feature type") == data.features &&
            d::integer(m.at("classes"), "OOF teacher class type") == data.classes &&
            d::integer(m.at("FIT_rows"), "OOF teacher FIT row type") == data.fit_rows &&
            m.at("native_objective") == "multi:softmax" &&
            m.at("VALID_read").is_boolean() && !m.at("VALID_read").get<bool>() &&
            m.at("TEST_read").is_boolean() && !m.at("TEST_read").get<bool>(),
            "OOF deployment teacher shape/HP/FIT-only receipt differs");
    if (m.contains("native_library_sha256"))
      d::need(m.at("native_library_sha256") == training.at("native_library_sha256"), "OOF deployment teacher library differs");
    if (m.contains("model_sha256")) d::need(m.at("model_sha256") == model.model_sha256, "OOF teacher metric model pin differs");
    if (m.contains("model_bytes")) d::need(d::integer(m.at("model_bytes"), "OOF model byte type") == model.model_json.size(), "OOF teacher metric model extent differs");
    declared.push_back({{"hyperparameters", hp}, {"full_fit_model_sha256", model.model_sha256},
                        {"full_fit_model_bytes", model.model_json.size()}, {"full_fit_data_binding", teacher.full_fit_data_binding}});
  }
  J result = {{"features", data.features}, {"classes", data.classes}, {"teachers", M}, {"response_features", columns},
          {"rows", data.rows}, {"FIT_rows", data.fit_rows}, {"VALID_rows", data.valid_rows},
          {"folds", options.folds}, {"seed", options.seed}, {"rotation", options.seed % options.folds},
          {"response_bytes", response_bytes}, {"coverage_bytes", coverage_bytes}, {"copied_label_bytes", label_bytes},
          {"persistent_explicit_bytes", persistent}, {"known_explicit_peak_bytes", peak},
          {"gpu_byte_budget", options.gpu_byte_budget}, {"native_opaque_peak_bounded", false},
          {"source_dataset_binding", data.binding}, {"teacher_declarations", std::move(declared)},
          {"native_library_path", plan.at("native_library_path")}, {"native_library_sha256", training.at("native_library_sha256")}};
  if (options.group_round_prefixes) result["round_prefix_groups"] = d::round_groups(result.at("teacher_declarations"));
  if (options.round_prefix_iteration_range) result["round_prefix_iteration_range"] = true;
  return result;
}

struct ResidentOof::Impl {
  ResidentDataView source;
  J plan, layout, identity, fold_results = J::array(), valid_results = J::array();
  OofOptions options; std::vector<OofTeacher> teachers;
  d::Buffer<float> response; d::Buffer<u32> coverage, labels, bad, imported_device;
  J imports = nullptr;
  std::unique_ptr<ResidentTrainer> anchor;
  // The trainer dies before the gather owner, whose values back its native API.
  std::shared_ptr<d::FoldOwner> fold_owner;
  std::unique_ptr<ResidentTrainer> fold_trainer;
  d::Buffer<float> packed_valid;
  u64 oof_cursor = 0, valid_cursor = 0, fits_this_process = 0, valid_this_process = 0;
  u64 restored_oof = 0, restored_valid = 0;
  std::vector<TrainingResult> round_parents;
  u32 round_parent_fold = UINT32_MAX;
  u64 round_actual_fits = 0, round_slice_units = 0, slices_this_process = 0;
  bool busy = false, failed = false, finalized = false;
  std::string failure;
  std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
  u32 M = 0, K = 0, F = 0, rotation = 0;
  u64 initial_free_bytes = 0, maximum_observed_used_delta = 0;

  Impl(const J& p, const ResidentDataView& data, std::vector<OofTeacher> bank, const OofOptions& opts)
      : source(data), plan(p), options(opts), teachers(std::move(bank)) {
    static_assert(std::endian::native == std::endian::little);
    layout = ResidentOof::validate_metadata(plan, source, teachers, options);
    M = u32(teachers.size()); K = source.classes; F = source.features; rotation = u32(options.seed % options.folds);
    if (options.group_round_prefixes) round_parents.resize(layout.at("round_prefix_groups").at("groups").size());
    identity = {{"format", "resident-native-oof-identity-1"}, {"layout", layout},
                {"fold_assignment", "FIT ordinal modulo folds with seed rotation; not stratified"},
                {"column_order", "teacher-major/class-minor"}, {"response_objective", "derived multi:softprob"}};
    d::ck(cudaSetDevice(0), "OOF device");
    std::size_t free = 0, total = 0; d::ck(cudaMemGetInfo(&free, &total), "OOF free memory"); initial_free_bytes = free;
    d::need(layout.at("known_explicit_peak_bytes").get<u64>() <= free, "OOF known buffer peak exceeds device free memory");
    d::allocation(source.values, ((source.rows - 1) * source.row_stride + F) * 4, "OOF source values extent/device");
    d::allocation(source.labels, source.rows * 4, "OOF source labels extent/device");
    response = d::Buffer<float>(source.rows * u64(M) * K); coverage = d::Buffer<u32>(source.rows * u64(M));
    labels = d::Buffer<u32>(source.rows); bad = d::Buffer<u32>(1); response.zero(); coverage.zero(); bad.zero();
    d::copy_labels<<<d::blocks(source.rows), 256>>>(source.labels, labels.p, source.rows, K, bad.p);
    checked("OOF source class labels invalid");
    // Setup checks the library once. Fold constructors reuse its live verified
    // authorization and own their references, without a library file reread.
    anchor = std::make_unique<ResidentTrainer>(plan, fit_prefix(source)); observe_memory();
  }
  std::string phase() const {
    if (failed) return "failed";
    if (oof_cursor < u64(M) * options.folds) return "oof_training";
    if (source.valid_rows && valid_cursor < M) return "valid_deployment";
    return "complete";
  }
  void observe_memory() {
    std::size_t free = 0, total = 0; d::ck(cudaMemGetInfo(&free, &total), "OOF observed memory");
    u64 delta = free < initial_free_bytes ? initial_free_bytes - free : 0;
    maximum_observed_used_delta = std::max(maximum_observed_used_delta, delta);
  }
  void checked(const char* why) {
    d::done(); u32 result = 0;
    d::ck(cudaMemcpy(&result, bad.p, sizeof(result), cudaMemcpyDeviceToHost), "OOF audit flag"); d::need(!result, why);
  }
  void audit() {
    bad.zero();
    d::audit_cells<<<d::blocks(source.rows * u64(M)), 256>>>(response.p, coverage.p, source.rows,
        source.fit_rows, M, K, options.folds, rotation, oof_cursor, valid_cursor, bad.p, imported_device.p);
    checked("OOF response/coverage does not match completed boundary");
  }
  void clear_round_parents() {
    for (auto& parent : round_parents) parent = {};
    round_parent_fold = UINT32_MAX;
  }
  J parent_record(u32 group) const {
    const auto& parent = round_parents.at(group);
    return {{"fold",round_parent_fold},{"group",group},{"model_sha256",parent.model_sha256},
            {"model_bytes",parent.model_json.size()},{"training",parent.metrics}};
  }
  J parent_records() const {
    J result = J::array();
    for (u32 g = 0; g < round_parents.size(); ++g)
      if (!round_parents[g].model_json.empty()) result.push_back(parent_record(g));
    return result;
  }
  TrainingResult& bind_round_parent(u32 fold, u32 teacher) {
    const u32 group = d::teacher_group(layout, teacher);
    if (round_parent_fold == UINT32_MAX) round_parent_fold = fold;
    d::need(round_parent_fold == fold, "OOF retained round parent belongs to another fold");
    auto& parent = round_parents.at(group);
    if (parent.model_json.empty()) {
      auto hp = layout.at("round_prefix_groups").at("groups").at(group).at("hyperparameters_except_rounds");
      hp["rounds"] = d::required_rounds(layout, imports, group);
      parent = fold_trainer->train(hp); ++fits_this_process; ++round_actual_fits;
    }
    if (fold_trainer->retained_prefix_source_sha256() != parent.model_sha256 &&
        !fold_trainer->select_retained_prefix_source(parent.model_sha256))
      fold_trainer->retain_prefix_source(parent.model_json, parent.model_sha256);
    return parent;
  }
  TrainingResult round_prefix(u32 fold, u32 teacher) {
    auto&parent=bind_round_parent(fold,teacher);
    auto result = fold_trainer->slice_retained_prefix(
        u32(d::integer(layout.at("teacher_declarations").at(teacher).at("hyperparameters").at("rounds"), "OOF requested prefix rounds")));
    d::need(result.metrics.at("original_rounds") == parent.metrics.at("hyperparameters").at("rounds") &&
            result.metrics.at("source_parent_sha256") == parent.model_sha256,
            "OOF retained native parent round/source contract differs");
    result.metrics["hyperparameters"] = layout.at("teacher_declarations").at(teacher).at("hyperparameters");
    ++slices_this_process; ++round_slice_units; return result;
  }
  void prepare_fold(u32 fold) {
    if (fold_trainer) return;
    u32 first = u32((u64(fold) + options.folds - rotation) % options.folds);
    u64 held = 1 + (source.fit_rows - 1 - first) / options.folds, train = source.fit_rows - held;
    d::need(held && train, "OOF empty fold complement/holdout");
    fold_owner = std::make_shared<d::FoldOwner>(train, held, F); bad.zero(); auto& part = *fold_owner;
    d::gather_fold<<<d::blocks(train * F), 256>>>(source.values, source.labels, source.fit_rows,
        source.row_stride, F, K, options.folds, rotation, fold, first, true, train,
        part.train_values.p, part.train_labels.p, part.train_ids.p, bad.p);
    d::gather_fold<<<d::blocks(held * F), 256>>>(source.values, source.labels, source.fit_rows,
        source.row_stride, F, K, options.folds, rotation, fold, first, false, held,
        part.holdout_values.p, nullptr, part.holdout_ids.p, bad.p);
    d::done();
    d::audit_gather<<<d::blocks(train * F), 256>>>(source.values, source.labels, source.fit_rows,
        source.row_stride, F, options.folds, rotation, fold, true, train,
        part.train_values.p, part.train_labels.p, part.train_ids.p, bad.p);
    d::audit_gather<<<d::blocks(held * F), 256>>>(source.values, source.labels, source.fit_rows,
        source.row_stride, F, options.folds, rotation, fold, false, held,
        part.holdout_values.p, nullptr, part.holdout_ids.p, bad.p);
    checked("OOF CUDA fold gather/exclusion audit failed");
    J binding = {{"format", "resident-FIT-complement-1"}, {"features", F}, {"classes", K},
                 {"rows", train}, {"row_stride", F}, {"FIT_rows", train}, {"VALID_rows", 0},
                 {"FIT_only", true}, {"TEST_read", false}, {"source_dataset_binding", source.binding},
                 {"excluded_fold", fold}, {"folds", options.folds}, {"seed", options.seed},
                 {"row_order", "ascending source FIT ordinal; excluded fold absent"}};
    ResidentDataView view{part.train_values.p, part.train_labels.p, train, F, train, 0,
                          F, K, std::move(binding), std::static_pointer_cast<void>(fold_owner)};
    fold_trainer = std::make_unique<ResidentTrainer>(plan, view, *anchor); observe_memory();
    if(plan.contains("experiment_checkpoint_host_byte_budget")){
      const auto host_budget=d::integer(plan.at("experiment_checkpoint_host_byte_budget"),"OOF parent-cache host byte budget");
      if(host_budget)fold_trainer->set_retained_prefix_cache_byte_budget(std::min<u64>(1ULL<<30,host_budget));
    }
  }
  bool imported(u32 teacher) const {
    return !imports.is_null() && !imports.at("teacher_map").at(teacher).is_null();
  }
  void skip_imported() {
    if (imports.is_null()) return;
    const auto& donor = imports.at("donor_state");
    const u64 donor_M = d::integer(donor.at("identity").at("layout").at("teachers"), "OOF imported donor teacher count");
    while (oof_cursor < u64(M) * options.folds && imported(u32(oof_cursor % M))) {
      const u32 teacher = u32(oof_cursor % M); const u64 fold = oof_cursor / M;
      const u64 from = d::integer(imports.at("teacher_map").at(teacher), "OOF imported teacher index");
      auto receipt = donor.at("fold_results").at(std::size_t(fold * donor_M + from));
      receipt["teacher"] = teacher; fold_results.push_back(std::move(receipt)); ++oof_cursor;
      if (oof_cursor % M == 0) { fold_trainer.reset(); fold_owner.reset(); clear_round_parents(); }
    }
    if (oof_cursor == u64(M) * options.folds && source.valid_rows) {
      while (valid_cursor < M && imported(u32(valid_cursor))) {
        const u32 teacher = u32(valid_cursor);
        const u64 from = d::integer(imports.at("teacher_map").at(teacher), "OOF imported VALID teacher index");
        auto receipt = donor.at("VALID_results").at(std::size_t(from));
        receipt["teacher"] = teacher; valid_results.push_back(std::move(receipt)); ++valid_cursor;
      }
    }
  }
  u64 imported_count() const {
    if (imports.is_null()) return 0;
    u64 count = 0; for (const auto& value : imports.at("teacher_map")) count += !value.is_null(); return count;
  }
  void finalize() {
    if (finalized) return;
    audit(); fold_trainer.reset(); fold_owner.reset(); clear_round_parents(); anchor.reset(); packed_valid = {};
    finalized = true; observe_memory();
  }
  J statistics() const {
    auto result = layout; result["format"] = "resident-native-oof-result-1";
    result["complete"] = !failed && phase() == "complete" && finalized;
    result["phase"] = phase(); result["failed"] = failed;
    result["oof_cursor"] = oof_cursor; result["valid_cursor"] = valid_cursor;
    result["completed_teacher_folds"] = oof_cursor; result["completed_VALID_teachers"] = valid_cursor;
    result["native_training_calls_this_process"] = fits_this_process;
    result["native_training_calls_completed_total"] = oof_cursor;
    result["restored_completed_teacher_folds"] = restored_oof;
    result["restored_completed_VALID_teachers"] = restored_valid;
    if (!imports.is_null()) {
      result["completed_column_import"] = identity.at("completed_column_import");
      result["imported_teacher_columns"] = imported_count();
      result["imported_completed_teacher_folds"] = imported_count() * options.folds;
      result["imported_completed_VALID_teachers"] = source.valid_rows ? imported_count() : 0;
      result["new_native_training_calls_this_process"] = fits_this_process;
      result["new_VALID_predictions_this_process"] = valid_this_process;
      result["import_mask_CUDA_bytes"] = imported_device.bytes();
      // The scalar cursor is the materialized evidence prefix; imported cells
      // ahead of it are already covered, so report their total separately.
      u64 newly_covered = 0;
      for (u64 unit = 0; unit < oof_cursor; ++unit) newly_covered += !imported(u32(unit % M));
      result["covered_teacher_fold_units"] = imported_count() * options.folds + newly_covered;
      result["new_completed_teacher_folds_retained"] = newly_covered;
      result["native_training_calls_in_materialized_prefix"] = oof_cursor;
      result["native_training_calls_completed_total"] = imported_count() * options.folds + newly_covered;
      result["native_training_calls_completed_total_scope"] = "historical completed fold FIT calls represented by imported and retained response units; not calls performed by this process";
      // A grouped donor may have many response prefixes from one FIT. Do not
      // reinterpret those imported response units as independent FIT calls.
      u64 imported_prefixes = 0;
      const auto& donor = imports.at("donor_state");
      const u64 donor_M = d::integer(donor.at("identity").at("layout").at("teachers"), "OOF metric donor extent");
      for (u32 t = 0; t < M; ++t) if (imported(t)) {
        const auto from = d::integer(imports.at("teacher_map").at(t), "OOF metric donor teacher");
        for (u32 fold = 0; fold < options.folds; ++fold){
          const auto operation=donor.at("fold_results").at(std::size_t(u64(fold) * donor_M + from)).at("training")
            .value("operation", std::string());
          imported_prefixes += operation == "native_existing_round_prefix" || operation == "native_existing_round_prefix_response";
        }
      }
      if (imported_prefixes) {
        result["imported_native_prefix_response_units"] = imported_prefixes;
        if (!options.group_round_prefixes) {
          result["native_training_calls_completed_total"] = newly_covered;
          result["actual_native_FIT_calls_completed_total"] = newly_covered;
          result["native_training_calls_completed_total_scope"] = "actual independent target FIT calls retained across resume; imported response prefixes are separate and not recounted as FIT calls";
        }
      }
    }
    result["VALID_predictions_this_process"] = valid_this_process;
    if (options.group_round_prefixes) {
      result["group_round_prefixes"] = true;
      result["round_trajectory_groups"] = round_parents.size();
      result["response_teacher_fold_units_completed"] = oof_cursor;
      result["native_training_calls_completed_total"] = round_actual_fits;
      result["actual_native_FIT_calls_completed_total"] = round_actual_fits;
      result["native_training_calls_completed_total_scope"] = "actual maximum-round complement FIT calls in this target owner, including restored calls; donor-imported response units are separate";
      result["round_prefix_slice_units_completed_total"] = options.round_prefix_iteration_range?0:round_slice_units;
      result["round_prefix_slice_calls_this_process"] = options.round_prefix_iteration_range?0:slices_this_process;
      if(options.round_prefix_iteration_range){
        result["round_prefix_iteration_range"]=true;
        result["round_prefix_response_units_completed_total"]=round_slice_units;
        result["round_prefix_response_calls_this_process"]=slices_this_process;
      }
      result["round_prefix_parents"] = parent_records();
      u64 bytes = 0; for (const auto& parent : round_parents) bytes += parent.model_json.size();
      result["current_fold_parent_host_bytes"] = bytes;
      result["round_prefix_scope"] = options.round_prefix_iteration_range?
        "native retained maximum-round fold parents with explicit iteration-range responses; no sliced model identity claimed":
        "native retained maximum-round fold parents, requested prefix slices; not independent FIT calls per response column";
    }
    result["fold_results"] = fold_results; result["VALID_results"] = valid_results;
    result["TEST_read"] = false; result["VALID_used_for_teacher_training"] = false;
    result["full_FIT_models_used_for_OOF"] = false;
    result["CPU_model_predictions"] = false; result["component_file_reads"] = 0; result["component_file_writes"] = 0;
    result["native_library_content_checks_at_session_setup"] = 1; result["fold_library_file_rereads"] = 0;
    result["exactly_once_coverage_verified_on_CUDA"] = !failed && finalized;
    result["OOF_exclusion_verified_on_CUDA"] = !failed && oof_cursor > 0;
    result["response_semantics"] = "native derived softprob, no asserted source-hard-class/argmax equivalence";
    result["FIT_metric_scope"] = "meta-training on OOF responses; not full-FIT deployment composition FIT accuracy";
    result["checkpoint_scope"] = "completed teacher-fold/VALID-teacher boundaries; no mid-native-fit snapshot";
    result["gpu_budget_scope"] = "explicit OOF buffers and known trainer value/label copies; native internal peak is not prebounded";
    result["maximum_observed_device_used_delta_bytes"] = maximum_observed_used_delta;
    result["device_measurement_scope"] = "CUDA free-memory delta at completed setup/operations, not an allocation peak trace";
    result["process_wall_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (failed) result["failure"] = failure; return result;
  }
};
ResidentOof::ResidentOof(const J& plan, const ResidentDataView& data,
    std::vector<OofTeacher> teachers, const OofOptions& options)
    : p_(std::make_shared<Impl>(plan, data, std::move(teachers), options)) {}
ResidentOof::~ResidentOof() = default;
ResidentOof::ResidentOof(ResidentOof&&) noexcept = default;
ResidentOof& ResidentOof::operator=(ResidentOof&&) noexcept = default;
bool ResidentOof::advance() {
  d::need(bool(p_), "moved-from OOF session"); auto& i = *p_;
  d::need(!i.busy && !i.failed, "OOF session busy/failed"); if (i.phase() == "complete") return false;
  i.busy = true;
  try {
    const auto start = std::chrono::steady_clock::now();
    if (i.oof_cursor < u64(i.M) * i.options.folds) {
      u32 fold = u32(i.oof_cursor / i.M), teacher = u32(i.oof_cursor % i.M);
      i.prepare_fold(fold); auto& part = *i.fold_owner;
      TrainingResult trained;
      const float* probabilities=nullptr;J range_view=nullptr;
      if(i.options.round_prefix_iteration_range){
        auto&parent=i.bind_round_parent(fold,teacher);
        const auto rounds=u32(d::integer(i.layout.at("teacher_declarations").at(teacher).at("hyperparameters").at("rounds"),"OOF response prefix rounds"));
        probabilities=i.fold_trainer->predict_retained_prefix_probabilities(part.holdout_values.p,part.prediction_rows,rounds);
        range_view=i.fold_trainer->metadata().at("retained_prefix_probability_response");
        d::need(range_view.at("source_model_sha256")==parent.model_sha256&&
          range_view.at("source_parent_rounds")==parent.metrics.at("hyperparameters").at("rounds"),"OOF native response parent differs");
        trained.metrics={{"operation","native_existing_round_prefix_response"},{"FIT_performed",false},{"CUDA_training",false},
          {"features",i.F},{"classes",i.K},{"FIT_rows",part.training_rows},{"native_objective","multi:softmax"},
          {"native_library_sha256",i.layout.at("native_library_sha256")},{"source_parent_sha256",parent.model_sha256},
          {"original_rounds",range_view.at("source_parent_rounds")},{"selected_rounds",rounds},
          {"hyperparameters",i.layout.at("teacher_declarations").at(teacher).at("hyperparameters")},
          {"VALID_read",false},{"TEST_read",false},{"native_model_exports",0},{"native_prefix_slice_calls",0},
          {"native_parent_model_loads",range_view.at("native_parent_model_loads")}};
        ++i.slices_this_process;++i.round_slice_units;
      }
      else if (i.options.group_round_prefixes) trained = i.round_prefix(fold, teacher);
      else { trained = i.fold_trainer->train(i.teachers[teacher].hyperparameters); ++i.fits_this_process; }
      if(!probabilities)probabilities=i.fold_trainer->predict_probabilities(part.holdout_values.p, part.prediction_rows); i.bad.zero();
      d::scatter<<<d::blocks(part.prediction_rows), 256>>>(probabilities, part.prediction_rows,
          teacher, i.M, i.K, part.holdout_ids.p, 0, i.source.fit_rows, i.response.p, i.coverage.p, i.bad.p);
      i.checked("OOF native response scatter/coverage invalid");
      J receipt = {{"fold", fold}, {"teacher", teacher},
          {"training_rows", part.training_rows}, {"prediction_rows", part.prediction_rows},
          {"training", std::move(trained.metrics)},
          {"derived_probability_view", i.options.round_prefix_iteration_range?range_view:
            i.fold_trainer->metadata().value("probability_response_view", J::object())},
          {"FIT_exclusion_verified_on_CUDA", true}, {"VALID_read_for_training", false}, {"TEST_read", false},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}};
      if(i.options.round_prefix_iteration_range)receipt["response_source"]=range_view.at("response_source");
      else receipt["model_sha256"]=trained.model_sha256;
      if (i.options.group_round_prefixes) receipt["round_prefix_parent"] = i.parent_record(d::teacher_group(i.layout, teacher));
      i.fold_results.push_back(std::move(receipt));
      ++i.oof_cursor;
      if (i.oof_cursor % i.M == 0) { i.fold_trainer.reset(); i.fold_owner.reset(); i.clear_round_parents(); }
    } else {
      u32 teacher = u32(i.valid_cursor);
      if (!i.packed_valid.p) {
        i.packed_valid = d::Buffer<float>(i.source.valid_rows * i.F); i.bad.zero();
        d::gather_valid<<<d::blocks(i.packed_valid.n), 256>>>(i.source.values, i.source.fit_rows,
            i.source.valid_rows, i.source.row_stride, i.F, i.packed_valid.p, i.bad.p);
        i.checked("OOF VALID packing invalid");
      }
      const auto& source_model = i.teachers[teacher].full_fit_model;
      i.anchor->restore_model(source_model.model_json, source_model.model_sha256);
      auto* probabilities = i.anchor->predict_probabilities(i.packed_valid.p, i.source.valid_rows); i.bad.zero();
      d::scatter<<<d::blocks(i.source.valid_rows), 256>>>(probabilities, i.source.valid_rows,
          teacher, i.M, i.K, nullptr, i.source.fit_rows, i.source.rows, i.response.p, i.coverage.p, i.bad.p);
      i.checked("OOF deployment VALID response scatter/coverage invalid");
      d::need(i.anchor->current_model_sha256() == source_model.model_sha256, "OOF VALID derived view changed source identity");
      i.valid_results.push_back({{"teacher", teacher}, {"full_fit_model_sha256", source_model.model_sha256},
          {"rows", i.source.valid_rows}, {"native_training_calls", 0},
          {"derived_probability_view", i.anchor->metadata().value("probability_response_view", J::object())},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}});
      ++i.valid_cursor; ++i.valid_this_process;
    }
    i.skip_imported(); i.observe_memory(); if (i.phase() == "complete") i.finalize(); i.busy = false; return true;
  } catch (const std::exception& e) {
    i.busy = false; i.failed = true; i.failure = e.what(); throw OofFailure(i.failure, i.statistics());
  }
}
bool ResidentOof::complete() const {
  d::need(bool(p_), "moved-from OOF session"); return !p_->failed && p_->phase() == "complete" && p_->finalized;
}
J ResidentOof::progress() const {
  d::need(bool(p_)&&!p_->busy,"OOF progress requires an owned completed boundary");const auto& i=*p_;
  J out={{"phase",i.phase()},{"failed",i.failed},{"complete",!i.failed&&i.phase()=="complete"&&i.finalized},
      {"oof_cursor",i.oof_cursor},{"valid_cursor",i.valid_cursor},{"native_training_calls_this_process",i.fits_this_process}};
  if(i.options.group_round_prefixes){
    out["group_round_prefixes"]=true;out["round_trajectory_groups"]=i.round_parents.size();
    out["actual_native_FIT_calls_completed_total"]=i.round_actual_fits;
    out["actual_native_FIT_calls_this_process"]=i.fits_this_process;
    out["round_prefix_slice_units_completed_total"]=i.options.round_prefix_iteration_range?0:i.round_slice_units;
    if(i.options.round_prefix_iteration_range){out["round_prefix_iteration_range"]=true;out["round_prefix_response_units_completed_total"]=i.round_slice_units;}
    out["response_teacher_fold_units_in_materialized_prefix"]=i.oof_cursor;
  }
  return out;
}
J ResidentOof::state() const {
  d::need(bool(p_), "moved-from OOF session");
  J s = {{"format", p_->options.round_prefix_iteration_range?"resident-native-oof-state-4":p_->options.group_round_prefixes ? "resident-native-oof-state-3" :
           p_->imports.is_null() ? "resident-native-oof-state-1" : "resident-native-oof-state-2"}, {"identity", p_->identity}, {"phase", p_->phase()},
          {"oof_cursor", p_->oof_cursor}, {"valid_cursor", p_->valid_cursor},
          {"fold_results", p_->fold_results}, {"VALID_results", p_->valid_results}};
  if (!p_->imports.is_null()) s["imports"] = p_->imports;
  if (p_->options.group_round_prefixes) {
    s["round_prefix_parents"] = p_->parent_records();
    s["round_prefix_actual_FIT_calls"] = p_->round_actual_fits;
    s[p_->options.round_prefix_iteration_range?"round_prefix_response_units":"round_prefix_slice_units"] = p_->round_slice_units;
  }
  return s;
}
J ResidentOof::metrics() const { d::need(bool(p_), "moved-from OOF session"); return p_->statistics(); }
OofCheckpoint ResidentOof::checkpoint() const {
  d::need(bool(p_) && !p_->busy && !p_->failed, "OOF checkpoint requires a successful completed boundary");
  auto& i = *p_; i.audit(); OofCheckpoint saved; saved.state = state();
  saved.response_bytes.resize(std::size_t(i.response.bytes())); saved.coverage_bytes.resize(std::size_t(i.coverage.bytes()));
  d::ck(cudaMemcpy(saved.response_bytes.data(), i.response.p, saved.response_bytes.size(), cudaMemcpyDeviceToHost), "OOF checkpoint response bytes");
  d::ck(cudaMemcpy(saved.coverage_bytes.data(), i.coverage.p, saved.coverage_bytes.size(), cudaMemcpyDeviceToHost), "OOF checkpoint coverage bytes");
  saved.state["response_sha256"] = d::hash(saved.response_bytes.data(), saved.response_bytes.size());
  saved.state["coverage_sha256"] = d::hash(saved.coverage_bytes.data(), saved.coverage_bytes.size());
  if (i.options.group_round_prefixes) for (const auto& parent : i.round_parents)
    if (!parent.model_json.empty()) saved.round_prefix_parent_bytes.emplace_back(parent.model_json.begin(), parent.model_json.end());
  return saved;
}
void ResidentOof::restore(const OofCheckpoint& saved) {
  d::need(bool(p_) && !p_->busy && !p_->failed && !p_->oof_cursor && !p_->valid_cursor && p_->imports.is_null(),
          "OOF restore requires a fresh session");
  auto& i = *p_; const auto& s = saved.state;
  const bool version2 = s.at("format") == "resident-native-oof-state-2";
  const bool version3 = s.at("format") == "resident-native-oof-state-3";
  const bool version4 = s.at("format") == "resident-native-oof-state-4";
  const bool has_imports = s.contains("imports");
  d::need(version2 || version3 || version4 || s.at("format") == "resident-native-oof-state-1", "OOF saved state version unsupported");
  d::need(version4==i.options.round_prefix_iteration_range&&(version3||version4) == i.options.group_round_prefixes &&
          (version3 || version4 || saved.round_prefix_parent_bytes.empty()),
          "OOF checkpoint round-prefix policy/payload differs");
  d::need(!version2 || has_imports, "OOF version2 import evidence missing");
  auto base_identity = s.at("identity");
  if (has_imports) base_identity.erase("completed_column_import");
  d::need(base_identity == i.identity, "OOF checkpoint source/plan identity differs");
  d::saved_receipts(s, i.layout);
  const auto oof = d::integer(s.at("oof_cursor"), "OOF saved fold cursor type");
  const auto valid = d::integer(s.at("valid_cursor"), "OOF saved VALID cursor type");
  J next_imports = nullptr; d::Buffer<u32> next_mask; std::vector<u32> mask;
  if (has_imports) {
    next_imports = s.at("imports"); const auto binding = d::checked_imports(next_imports, i.identity);
    d::need(s.at("identity").at("completed_column_import") == binding, "OOF checkpoint import binding differs");
    const auto& donor = next_imports.at("donor_state");
    const auto donor_M = d::integer(donor.at("identity").at("layout").at("teachers"), "OOF saved donor extent");
    mask.resize(i.M, 0);
    for (u32 t = 0; t < i.M; ++t) mask[t] = !next_imports.at("teacher_map").at(t).is_null();
    for (u64 unit = 0; unit < oof; ++unit) {
      const u32 teacher = u32(unit % i.M);
      if (mask[teacher]) {
        const auto from = d::integer(next_imports.at("teacher_map").at(teacher), "OOF saved source teacher");
        auto r = donor.at("fold_results").at(std::size_t((unit / i.M) * donor_M + from)); r["teacher"] = teacher;
        d::need(r == s.at("fold_results").at(std::size_t(unit)), "OOF saved imported fold receipt changed");
      }
    }
    for (u32 teacher = 0; teacher < valid; ++teacher) if (mask[teacher]) {
      const auto from = d::integer(next_imports.at("teacher_map").at(teacher), "OOF saved source VALID teacher");
      auto r = donor.at("VALID_results").at(std::size_t(from)); r["teacher"] = teacher;
      d::need(r == s.at("VALID_results").at(teacher), "OOF saved imported VALID receipt changed");
    }
    d::need(oof == u64(i.M) * i.options.folds || !mask[u32(oof % i.M)], "OOF saved fold cursor did not skip import");
    d::need(!i.source.valid_rows || oof != u64(i.M) * i.options.folds || valid == i.M || !mask[u32(valid)],
            "OOF saved VALID cursor did not skip import");
    d::need(d::add(i.layout.at("known_explicit_peak_bytes").get<u64>(), u64(i.M) * 4, "OOF restored import-mask peak") <= i.options.gpu_byte_budget,
            "OOF restored import-mask byte budget refusal");
  } else d::need(!s.at("identity").contains("completed_column_import"), "OOF import identity has no retained evidence");
  d::need(!has_imports || version2 || version3 || version4, "OOF version1 cannot declare imports");
  d::payload(saved, i.layout);
  d::PrefixRestore prefix;
  if (version3||version4) prefix = d::checked_prefix_parents(saved, i.layout, next_imports);
  try {
    if (has_imports) {
      next_mask = d::Buffer<u32>(i.M);
      d::ck(cudaMemcpy(next_mask.p, mask.data(), u64(i.M) * 4, cudaMemcpyHostToDevice), "OOF restore import mask");
    }
    d::ck(cudaMemcpy(i.response.p, saved.response_bytes.data(), saved.response_bytes.size(), cudaMemcpyHostToDevice), "OOF restore response bytes");
    d::ck(cudaMemcpy(i.coverage.p, saved.coverage_bytes.data(), saved.coverage_bytes.size(), cudaMemcpyHostToDevice), "OOF restore coverage bytes");
    i.oof_cursor = oof; i.valid_cursor = valid; i.fold_results = s.at("fold_results"); i.valid_results = s.at("VALID_results");
    i.imports = std::move(next_imports); i.imported_device = std::move(next_mask); i.identity = s.at("identity");
    if (version3||version4) {
      i.round_parents = std::move(prefix.parents); i.round_parent_fold = prefix.fold;
      i.round_actual_fits = prefix.fits; i.round_slice_units = prefix.slices;
    }
    i.audit(); i.restored_oof = oof; i.restored_valid = valid;
    if (i.phase() == "complete") i.finalize();
  } catch (const std::exception& e) {
    i.failed = true; i.failure = e.what(); throw OofFailure(i.failure, i.statistics());
  }
}
OofImportReport ResidentOof::import_completed_columns(const OofCheckpoint& saved, const J& provenance) {
  d::need(bool(p_) && !p_->busy && !p_->failed && !p_->oof_cursor && !p_->valid_cursor &&
          p_->imports.is_null() && p_->fold_results.empty() && p_->valid_results.empty(),
          "OOF column import requires one donor and a fresh target");
  auto& i = *p_; auto donor = d::canonical_complete_donor(saved.state);
  d::need(saved.round_prefix_parent_bytes.empty(), "OOF complete donor must not retain unfinished fold-parent payloads");
  auto map = d::match_columns(donor.at("identity"), i.identity);
  const auto remaining_work=d::remaining_import_work(i.layout,map);
  auto binding = d::import_binding(donor, map, provenance);
  const auto& layout = donor.at("identity").at("layout");
  d::payload(saved, layout);
  const auto donor_M64 = d::integer(layout.at("teachers"), "OOF donor count");
  d::need(donor_M64 <= UINT32_MAX, "OOF donor count overflow"); const u32 donor_M = u32(donor_M64);
  const auto donor_bytes = d::add(saved.response_bytes.size(), saved.coverage_bytes.size(), "OOF donor upload extent");
  const u64 mask_bytes = u64(i.M) * 4;
  const auto persistent_anchor = d::add(i.layout.at("persistent_explicit_bytes").get<u64>(),
      d::extent(d::mul(i.source.fit_rows, u64(i.F) + 1, "OOF import anchor copies"), 4, "OOF import anchor bytes"),
      "OOF import current explicit bytes");
  const auto staging_peak = d::add(persistent_anchor, d::add(donor_bytes, mask_bytes, "OOF import scratch bytes"), "OOF import staging peak");
  const auto future_peak = d::add(i.layout.at("known_explicit_peak_bytes").get<u64>(), mask_bytes, "OOF import future peak");
  d::need(std::max(staging_peak, future_peak) <= i.options.gpu_byte_budget, "OOF completed-column import byte budget refusal");
  std::size_t free = 0, total = 0; d::ck(cudaMemGetInfo(&free, &total), "OOF import free memory");
  d::need(d::add(donor_bytes, mask_bytes, "OOF import allocations") <= free, "OOF donor upload exceeds current device free memory");
  std::vector<u32> mask(i.M, 0); u64 matched = 0;
  for (u32 t = 0; t < i.M; ++t) { mask[t] = !map.at(t).is_null(); matched += mask[t]; }
  J next_imports = {{"binding", binding}, {"provenance", provenance}, {"teacher_map", map}, {"donor_state", donor}};
  auto next_identity = i.identity; next_identity["completed_column_import"] = binding;
  bool target_written = false;
  try {
    d::Buffer<float> donor_response(saved.response_bytes.size() / 4);
    d::Buffer<u32> donor_coverage(saved.coverage_bytes.size() / 4), next_mask(i.M);
    d::ck(cudaMemcpy(donor_response.p, saved.response_bytes.data(), saved.response_bytes.size(), cudaMemcpyHostToDevice), "OOF upload donor responses");
    d::ck(cudaMemcpy(donor_coverage.p, saved.coverage_bytes.data(), saved.coverage_bytes.size(), cudaMemcpyHostToDevice), "OOF upload donor coverage");
    d::ck(cudaMemcpy(next_mask.p, mask.data(), mask_bytes, cudaMemcpyHostToDevice), "OOF import teacher mask");
    i.bad.zero();
    d::audit_cells<<<d::blocks(i.source.rows * u64(donor_M)), 256>>>(donor_response.p, donor_coverage.p,
        i.source.rows, i.source.fit_rows, donor_M, i.K, i.options.folds, i.rotation,
        u64(donor_M) * i.options.folds, i.source.valid_rows ? donor_M : 0, i.bad.p);
    i.checked("OOF complete donor probability/coverage audit failed");
    i.audit();
    for (u32 teacher = 0; teacher < i.M; ++teacher) if (mask[teacher]) {
      const auto from = d::integer(map.at(teacher), "OOF donor column index");
      target_written = true;
      d::ck(cudaMemcpy2D(i.response.p + u64(teacher) * i.K, u64(i.M) * i.K * 4,
          donor_response.p + from * i.K, u64(donor_M) * i.K * 4,
          u64(i.K) * 4, i.source.rows, cudaMemcpyDeviceToDevice), "OOF import probability columns");
      d::ck(cudaMemcpy2D(i.coverage.p + teacher, u64(i.M) * 4,
          donor_coverage.p + from, u64(donor_M) * 4, 4, i.source.rows, cudaMemcpyDeviceToDevice), "OOF import coverage column");
    }
    i.bad.zero();
    d::audit_cells<<<d::blocks(i.source.rows * u64(i.M)), 256>>>(i.response.p, i.coverage.p,
        i.source.rows, i.source.fit_rows, i.M, i.K, i.options.folds, i.rotation, 0, 0, i.bad.p, next_mask.p);
    i.checked("OOF imported target coverage/probability audit failed");
    i.imports.swap(next_imports); i.identity.swap(next_identity); i.imported_device = std::move(next_mask);
    i.skip_imported(); i.observe_memory(); if (i.phase() == "complete") i.finalize();
    J report={{"complete", true}, {"matched_teacher_columns", matched}, {"target_teacher_columns", i.M},
      {"donor_teacher_columns", donor_M}, {"imported_completed_teacher_folds", matched * i.options.folds},
      {"imported_completed_VALID_teachers", i.source.valid_rows ? matched : 0},
      {"new_native_FIT_calls", 0},
      {"remaining_VALID_response_calls", i.source.valid_rows ? i.M - matched : 0}, {"teacher_map", map},
      {"donor_provenance", provenance}, {"CUDA_donor_and_target_coverage_verified", true},
      {"import_staging_explicit_peak_bytes", staging_peak}, {"future_known_explicit_peak_bytes", future_peak},
      {"import_mask_bytes", mask_bytes}, {"native_opaque_peak_bounded", false}};
    report.update(remaining_work);return {std::move(report)};
  } catch (const std::exception& e) {
    if (target_written) { i.failed = true; i.failure = e.what(); throw OofFailure(i.failure, i.statistics()); }
    throw;
  }
}
ResidentDataView ResidentOof::completed_data() const {
  d::need(complete(), "OOF response data is not complete"); const auto& i = *p_;
  J binding = {{"format", "resident-native-oof-response-matrix-1"},
      {"features", u64(i.M) * i.K}, {"classes", i.K}, {"rows", i.source.rows},
      {"row_stride", u64(i.M) * i.K}, {"FIT_rows", i.source.fit_rows}, {"VALID_rows", i.source.valid_rows},
      {"input_dtype", "little-endian FP32"}, {"label_dtype", "little-endian uint32"},
      {"feature_semantics", "teacher-major/class-minor native derived multi:softprob responses"},
      {"preprocessing", "FIT uses held-out-complement OOF teachers; VALID uses supplied full-FIT deployment teachers"},
      {"source_dataset_binding", i.source.binding}, {"teacher_declarations", i.layout.at("teacher_declarations")},
      {"fold_assignment", i.identity.at("fold_assignment")}, {"folds", i.options.folds}, {"seed", i.options.seed},
      {"row_order", "unchanged source FIT prefix then source VALID suffix"}, {"labels_preserved_on_CUDA", true},
      {"FIT_response_role", "OOF meta-training features"}, {"VALID_response_role", "full-FIT teacher deployment validation features"},
      {"TEST_read", false}, {"VALID_used_for_teacher_training", false},
      {"native_library_sha256", i.layout.at("native_library_sha256")}};
  if (i.source.binding.contains("row_ids_sha256")) binding["row_ids_sha256"] = i.source.binding.at("row_ids_sha256");
  return {i.response.p, i.labels.p, i.source.rows, u64(i.M) * i.K, i.source.fit_rows,
          i.source.valid_rows, u32(u64(i.M) * i.K), i.K, std::move(binding), std::static_pointer_cast<void>(p_)};
}
} // namespace class_study
