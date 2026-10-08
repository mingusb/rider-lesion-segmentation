#pragma once
#include "class_io.hpp"
#include "streaming_file_hash.hpp"
#include "public_rl_source_manifest.hpp"
#include <filesystem>
#include <stdexcept>

// Build identity is evidence about this linked executable and its embedded source
// set. It confers no source-model authority; Session still constructs and audits
// every class graph and default runtime on CUDA in the current process.
namespace public_rl_build {
inline std::string source_sha256(const std::string& name) {
  for (const auto& entry : public_rl_source_manifest::files)
    if (name == entry.name) return entry.sha256;
  throw std::runtime_error("RL source not present in embedded build manifest");
}
inline dpnative::json binding() {
  dpnative::json sources = dpnative::json::object();
  for (const auto& entry : public_rl_source_manifest::files)
    sources[entry.name] = entry.sha256;
  const auto executable = std::filesystem::read_symlink("/proc/self/exe");
  return {{"format", "public-RL-embedded-source-build-1"},
          {"executable_sha256", dp_streaming::sha256_file(executable)},
          {"manifest_sha256", dpnative::sha256(sources.dump())},
          {"source_hashes", sources},
          {"external_receipts_authorize_model", false}};
}
}
