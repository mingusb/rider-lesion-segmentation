#pragma once
#include "rl_category_policy.hpp"
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>

namespace rl_qualified_session {
using ResidentWord=std::uint64_t;
class Session;
// Pure metadata plan for table copies plus one engine's explicit CUDA buffers.
// Does not include existing Session/Policy/native allocations or thread stacks.
ResidentWord resident_device_plan(ResidentWord source_words,ResidentWord nodes,
    ResidentWord entries,ResidentWord chunk_records,bool full_trajectories);
struct ResidentArtifact {
  ResidentWord entry=0,encoded_bytes=0,nodes=0;
  std::string runtime_sha256,certificate_sha256;
  std::filesystem::path runtime_path;
};
namespace testing {
struct ResidentCheckReport {bool passed=false,CUDA_executed=false;ResidentWord assertions=0,rejections=0,episode_word_checks=0;};
ResidentCheckReport resident_gpu_checks(int device=0);
}
// Same-process immutable authority; serialized summaries cannot create it.
class QualifiedCostTable {
  struct Data;
  std::shared_ptr<const Data> p_;
  explicit QualifiedCostTable(std::shared_ptr<const Data>);
  static QualifiedCostTable seal(std::shared_ptr<const void>,ResidentWord process_id,
      int device,ResidentWord generation,ResidentWord baseline_bytes,
      std::string source_binding,std::string semantics_binding,
      const std::vector<ResidentWord>&source_words,
      const std::vector<std::vector<ResidentWord>>&keys,
      std::vector<ResidentArtifact>artifacts,ResidentWord maximum_device_bytes);
  void require_owner(const std::shared_ptr<const void>&,ResidentWord,
      const std::string&)const;
  friend class Session;
  friend class ResidentTraining;
  friend testing::ResidentCheckReport testing::resident_gpu_checks(int);
public:
  QualifiedCostTable(const QualifiedCostTable&)=default;
  ResidentWord entries()const;
  ResidentWord generation()const;
  const std::string& source_binding()const;
  const std::string& semantics_binding()const;
  const std::vector<ResidentArtifact>& artifacts()const;
};

struct ResidentSchedule {
  ResidentWord seed0=0,episode0=0,episodes=0;
};
struct ResidentOptions {
  ResidentWord maximum_chunk_records=64;
  bool retain_full_trajectory_words=false; // Diagnostic comparison only.
};
// Bounded word transport. These records grant no source/cost authority.
struct ResidentEpisodeRecord {
  ResidentWord seed=0,episode=0,sampled_version=0,committed_version=0;
  ResidentWord updates=0,entry=UINT64_MAX,encoded_bytes=0,incumbent_entry=UINT64_MAX;
  ResidentWord baseline_bytes=0,learning_rate_bits=0,reward_bits=0;
  unsigned error=0,reserved=0;
  std::array<unsigned,44> order44{};
  std::array<ResidentWord,44> after_logit_words{};
};
static_assert(sizeof(ResidentEpisodeRecord)==624);
struct ResidentChunk {
  bool complete=false,CUDA_executed=false;
  ResidentWord requested=0,completed=0,failed_episode=UINT64_MAX;
  ResidentWord table_generation=0,policy_version=0,updates=0,incumbent_entry=UINT64_MAX;
  unsigned error=0;
  std::array<ResidentWord,44> final_logit_words{};
  std::vector<ResidentEpisodeRecord> records;
  std::vector<std::array<rl_category_policy::Trace,2>> diagnostic_trajectories;
};
class ResidentTraining {
  struct Impl;
  std::unique_ptr<Impl> p_;
  ResidentTraining(const QualifiedCostTable&,const std::array<ResidentWord,44>&,
      ResidentWord initial_version,ResidentWord learning_rate_bits,
      ResidentSchedule declared_schedule,ResidentOptions);
  void require_complete(const QualifiedCostTable&)const;
  friend class Session;
  friend testing::ResidentCheckReport testing::resident_gpu_checks(int);
public:
  ~ResidentTraining();
  ResidentTraining(const ResidentTraining&)=delete;
  ResidentTraining&operator=(const ResidentTraining&)=delete;
  // One bounded numerical launch. No host callbacks/downloads/JSON per episode.
  // A miss/error prevents that episode's credit. An incomplete chunk cannot
  // qualify publication even if earlier private updates have completed.
  ResidentChunk run_chunk(ResidentSchedule);
};
// Planned Session members, defined by the private implementation:
// QualifiedCostTable Session::seal_resident_table();
// std::unique_ptr<ResidentTraining> Session::make_resident(
//   const QualifiedCostTable&, const rl_category_policy::Policy& initial,
//   ResidentWord learning_rate_bits, ResidentSchedule, ResidentOptions = {});
} // namespace rl_qualified_session
