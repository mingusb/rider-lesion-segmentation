#pragma once
// Setup-only bridge to the EXISTING qualified native softprob gap gate.
// Source identity remains restricted until generic native-margin correspondence
// is qualified. Kernel, library, device, configuration and fresh-process
// evidence remain pinned. Metadata eligibility alone creates no authority.
// No converter algorithm, local softmax, or per-batch journal.
#include "class_native_softprob_gap.hpp"
#include "class_constant_oracle.hpp"
#include <cstdlib>
#include <memory>
#include <string>

namespace class_study {
struct NativeGateSetup {
 std::shared_ptr<const native_softprob_gap::RuntimeGate> gate;
 nlohmann::json receipt={{"enabled",false}};
 std::string reason;
};
namespace native_gate_detail {
inline void require(bool v,const char*why){if(!v)throw std::runtime_error(why);}
inline void cuda_check(cudaError_t status,const char*operation){if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));}
inline void checked_setup_path(const std::filesystem::path&p){require(p.is_absolute()&&p.string().find('\0')==std::string::npos,"native gate setup paths must be absolute");}
struct WarmBuffer {
 float*pointer=nullptr;
 ~WarmBuffer(){if(pointer){int previous=-1;bool restore=cudaGetDevice(&previous)==cudaSuccess&&previous!=0;if(restore)cudaSetDevice(0);cudaFree(pointer);if(restore)cudaSetDevice(previous);}}
 void allocate(){cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer),7*sizeof(float)),"native gate warmup allocation");cuda_check(cudaMemset(pointer,0,7*sizeof(float)),"native gate warmup initialization");}
};
struct SetupOwner {
 // Reverse destruction keeps the warm CUDA row alive through proxy teardown.
 WarmBuffer warm;
 std::unique_ptr<dpnative::XGBoostConstantOracle> oracle;
 native_softprob_gap::RuntimeGate gate;
};
}
// Before process startup set CUDA_INJECTION64_PATH to the exact pinned helper
// and DP_CUPTI_MODULE_CAPTURE_DIR to a fresh existing capture directory. The
// snapshot must be a fresh filename in that directory. Unsupported setups
// return a null authorization with the exact refusal; ordinary conversion may
// continue without this optional shortcut. The factory never trains a model.
inline NativeGateSetup try_qualify_native_gap(
 const std::string&source_sha,const std::filesystem::path&library_path,
 const std::filesystem::path&helper_path,const std::filesystem::path&snapshot_path,
 const std::filesystem::path&qualification_path,
 std::uint32_t model_classes=native_softprob_gap::reviewed_classes,
 const std::string&model_objective="multi:softprob"){
 NativeGateSetup out;
 try{
  using namespace native_gate_detail;
  const auto eligibility=native_softprob_gap::source_eligibility(source_sha,model_classes,model_objective);
  require(eligibility.reviewed_shape_available,eligibility.reason.c_str());
  require(source_sha==native_softprob_gap::source_sha,"native gate source has no qualified native-margin correspondence contract");
  checked_setup_path(library_path);checked_setup_path(helper_path);checked_setup_path(snapshot_path);checked_setup_path(qualification_path);
  const char*injection=std::getenv("CUDA_INJECTION64_PATH");const char*capture=std::getenv("DP_CUPTI_MODULE_CAPTURE_DIR");
  require(injection&&*injection&&capture&&*capture,"native gate requires capture injection before process startup");
  checked_setup_path(injection);checked_setup_path(capture);
  require(std::filesystem::canonical(injection)==std::filesystem::canonical(helper_path),"native gate injection helper path differs");
  require(std::filesystem::canonical(capture)==std::filesystem::canonical(snapshot_path.parent_path()),"native gate snapshot and capture directories differ");
  const auto qualification_bytes=dpnative::read_text(qualification_path);
  require(dpnative::sha256(qualification_bytes)==native_softprob_gap::qualification_sha,"native gate zero-tree qualification receipt is not pinned");
  const auto qualification=nlohmann::json::parse(qualification_bytes);
  require(qualification.at("passed")==true&&qualification.at("trained_rounds")==0&&qualification.at("base_margin_fp32_bits_equal")==true&&qualification.at("native_probability_fp32_bits_equal")==true,"native gate qualification scope differs");
  require(dpnative::sha256(dpnative::read_text(library_path))==native_softprob_gap::library_sha,"native gate library bytes differ");
  require(dpnative::sha256(dpnative::read_text(helper_path))==native_softprob_gap::helper_sha,"native gate helper bytes differ");
  require(native_softprob_gap::detail::environment_ok(),"native gate floating-point environment differs");
  cuda_check(cudaSetDevice(0),"native gate setup device");
  auto owner=std::make_shared<SetupOwner>();
  owner->oracle=std::make_unique<dpnative::XGBoostConstantOracle>(library_path.string(),7,0);
  const auto configuration=nlohmann::json::parse(owner->oracle->configuration_json());
  const auto build=nlohmann::json::parse(owner->oracle->build_info());
  require(configuration==qualification.at("configuration"),"native gate current zero-tree configuration differs");
  require(dpnative::sha256(configuration.dump())==native_softprob_gap::configuration_sha&&dpnative::sha256(build.dump())==native_softprob_gap::build_sha,"native gate current configuration/build pin differs");
  cudaDeviceProp properties{};int driver=0,runtime=0;
  cuda_check(cudaGetDeviceProperties(&properties,0),"native gate device properties");cuda_check(cudaDriverGetVersion(&driver),"native gate driver version");cuda_check(cudaRuntimeGetVersion(&runtime),"native gate runtime version");
  nlohmann::json live={{"format","source-live-native-transform-1"},{"source_sha256",source_sha},{"model_classes",model_classes},{"model_objective",model_objective},{"library_sha256",native_softprob_gap::library_sha},{"qualified_zero_tree_receipt_sha256",dpnative::sha256(qualification_bytes)},
   {"xgboost_version",owner->oracle->version()},{"native_configuration",configuration},{"native_build_info",build},{"gpu",properties.name},{"compute_capability",nlohmann::json::array({properties.major,properties.minor})},{"cuda_driver_version",driver},{"cuda_runtime_version",runtime},{"training_performed",false},{"dataset_records_read",false}};
  // Validate the queried runtime facts before warming. No receipt field is
  // invented from a different booster or a previous process's capture.
  (void)native_softprob_gap::detail::runtime_identity(source_sha,live);
  owner->warm.allocate();
  (void)owner->oracle->predict_values(owner->warm.pointer,1,true);
  (void)owner->oracle->predict_values(owner->warm.pointer,1,false);
  owner->gate=native_softprob_gap::RuntimeGate::qualify(source_sha,live,helper_path,snapshot_path);
  out.receipt=owner->gate.evidence();
  require(owner->gate.enabled()&&owner->gate.matches_runtime(source_sha,live)&&owner->gate.matches_source(source_sha,model_classes,model_objective,native_softprob_gap::library_sha),out.receipt.value("reason",std::string("native gate same-process qualification failed")).c_str());
  // Aliasing ownership retains the genuine zero-tree handle, CUDA proxy/input,
  // loaded capture helper and private RuntimeGate through conversion.
  out.gate=std::shared_ptr<const native_softprob_gap::RuntimeGate>(owner,&owner->gate);
  out.receipt["setup_warmup_rows"]=1;out.receipt["setup_only"]=true;out.receipt["per_batch_file_operations"]=0;
  out.receipt["full_model_callback_unchanged"]=true;out.receipt["warmup_is_universal_numerical_proof"]=false;
  out.receipt["source_identity_is_eligibility_whitelist"]=true;
  out.receipt["generic_native_margin_correspondence_qualified"]=false;
  out.receipt["reviewed_transform_contract"]=native_softprob_gap::reviewed_contract;
  out.receipt["model_classes"]=model_classes;out.receipt["model_objective"]=model_objective;
  out.receipt["setup_library_content_reads"]=3;out.receipt["factory_library_content_checks"]=1;out.receipt["inherited_gate_library_readbacks"]=2;
  out.receipt["inherited_gate_pin_and_readback_protocol_unchanged"]=true;
 }catch(const std::exception&e){out.gate.reset();out.reason=e.what();out.receipt["enabled"]=false;out.receipt["reason"]=out.reason;out.receipt["exact_fallback_required"]=true;}
 return out;
}
} // namespace class_study
