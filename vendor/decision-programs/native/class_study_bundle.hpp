#pragma once
#include "class_study_composition.hpp"
#include "class_study_data.hpp"
#include "class_study_process.hpp"
#include "class_io.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace class_study::bundle_evaluation {
// Evaluate an exported deployment independently of every training/OOF owner.
// This is development FIT/VALID evaluation; it is not a final TEST protocol.
inline int run(const std::filesystem::path& bundle_path,
               const std::filesystem::path& descriptor_path,
               const std::filesystem::path& output) {
  using J = nlohmann::json;
  if (!output.is_absolute() || std::filesystem::exists(output))
    throw std::runtime_error("bundle evaluation output must be a fresh absolute directory");
  const auto started = std::chrono::steady_clock::now();
  const auto bytes = dpnative::read_text(bundle_path);
  const auto bundle = J::from_cbor(bytes);
  const auto descriptor = J::parse(dpnative::read_text(descriptor_path));
  if (!descriptor.contains("TEST_read") || descriptor.at("TEST_read") != false)
    throw std::runtime_error("bundle development evaluation requires TEST_read=false");
  ResidentComposition model(bundle);
  const auto metadata = model.metadata();
  NativeOracle oracle;
  oracle.features = model.features();
  oracle.classes = model.classes();
  oracle.source_sha256 = metadata.at("composition_identity_sha256").get<std::string>();
  oracle.library_sha256 = bundle.at("native_library_sha256").get<std::string>();
  oracle.predict = [&model](const float* values, std::uint64_t rows, bool margin) {
    return model.predict(values, rows, margin);
  };
  auto data = stage_dataset(descriptor);
  ResidentEvaluation evaluation(data);
  auto metrics = evaluation.evaluate_native(oracle);
  metrics["FIT_accuracy_scope"] = "deployed composition on declared evaluation FIT rows; not OOF meta-training accuracy";
  metrics["VALID_accuracy_scope"] = "deployed composition on declared development VALID rows";
  J report = {{"format", "native-bundle-evaluation-1"}, {"complete", true},
    {"bundle_path", bundle_path.string()}, {"bundle_sha256", dpnative::sha256(bytes)},
    {"evaluation_descriptor_path", descriptor_path.string()}, {"evaluation_binding", data.binding},
    {"evaluation", metrics}, {"composition", model.metadata()},
    {"FIT_calls", 0}, {"training_or_OOF_owners_created", false},
    {"TEST_read", false}, {"selection_performed", false},
    {"process_wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}};
  std::filesystem::create_directories(output);
  std::ofstream file(output/"result.json", std::ios::binary);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file << report.dump(2) << '\n'; file.flush();
  std::cout << J{{"event", "bundle_evaluation_complete"},
    {"VALID_errors", metrics.at("VALID_errors")}, {"output", output.string()}, {"FIT_calls", 0}}.dump() << '\n';
  return 0;
}
} // namespace class_study::bundle_evaluation
