#pragma once
#include "rl_category_lowering.hpp"
#include "class_rank_gpu_binary_grid_validator.hpp"
#include "rl_category_policy.hpp"
#include "resident_session.hpp"
#include <filesystem>
namespace rl_qualified_session {
using U=std::uint64_t;using Stop=rank_gpu_class_apply::Stop;
struct Options {
  U maximum_device_bytes=128ull*1024*1024;
  U maximum_validation_cells=536870912,native_batch_rows=16384;
  U maximum_runtime_bytes=1024ull*1024*1024;
  U maximum_cache_entries=64; // Storage optimization bound; zero disables cache.
  bool per_trial_diagnostics=false;
};
struct Summary {
  bool qualified=false,CUDA_executed=false,publication_identity_current=false;
  U process_id=0,baseline_bytes=0,baseline_nodes=0,base_native_cells=0;
  U class_builds=0,base_native_audits=0,proposals=0,identity_retained_device_bytes=0;
  U cache_hits=0,cache_misses=0,distinct_runtime_artifacts=1,cache_retained_device_bytes=0;
  std::string binding,source_sha256,library_sha256,rank_sha256,class_word_sha256;
  std::string baseline_runtime_sha256,base_native_audit_sha256,build_binding_sha256;
  double setup_seconds=0,finalization_seconds=0;
};
struct ProposalTimings {double identity_before=0,lowering=0,canonical=0,codec_IO=0,identity_after=0,total=0;};
class Session;
// Opaque same-process token. JSON/provided booleans cannot construct this.
class QualifiedProposal {
  struct Data;std::shared_ptr<const Data> p_;
  explicit QualifiedProposal(std::shared_ptr<const Data>);friend class Session;
public:
  QualifiedProposal(const QualifiedProposal&)=default;
  U encoded_bytes()const;U nodes()const;U policy_version()const;
  const std::string& runtime_sha256()const;const std::string& certificate_sha256()const;
  const std::filesystem::path& directory()const;
  const std::filesystem::path& runtime_path()const;
  bool cache_hit()const;
  const std::string& certificate_record()const;
  ProposalTimings timings()const;
};
class Session {
  struct Impl;std::unique_ptr<Impl> p_;
  QualifiedProposal propose_impl(rl_category_lowering::DeviceOrder,const rl_category_policy::OrderView*,
                                 std::filesystem::path,const Stop&);
public:
  // Builds graph internally and fully validates one default collected/decoded
  // runtime. No imported class graph, archive acceptance or public flag path.
  Session(const std::string&library,const std::string&teacher,
          const std::string&expected_source_sha256,std::filesystem::path fresh_base,
          std::filesystem::path frozen_source,Options={},const Stop& = {},int device=0);
  ~Session();Session(const Session&)=delete;Session&operator=(const Session&)=delete;
  Summary summary()const;
  // Rechecks captured teacher/library/build and every published artifact once
  // outside the steady-state loop. Proposals never reload the native teacher.
  void finalize();
  // Freezes cache admission for this Session. Only internally qualified keys
  // and complete immutable artifact costs can enter a resident table.
  QualifiedCostTable seal_resident_table();
  std::unique_ptr<ResidentTraining> make_resident(const QualifiedCostTable&,
      const rl_category_policy::Policy& initial, U learning_rate_bits,ResidentSchedule,
      ResidentOptions = {});
  void finalize_resident(const QualifiedCostTable&,const ResidentTraining&);
  QualifiedProposal propose(rl_category_lowering::DeviceOrder,
                            std::filesystem::path fresh_output,const Stop& = {});
  QualifiedProposal propose(const rl_category_policy::OrderView&,
                            std::filesystem::path fresh_output,const Stop& = {});
  // Verifies sealed private token/session. Unique persisted artifacts are
  // rechecked at finalize; opt-in diagnostics also check them during credit.
  // Credit metadata transport only; reward/REINFORCE arithmetic stays CUDA.
  rl_category_policy::Credit credit(const QualifiedProposal&,
                                    const rl_category_policy::Policy&,
                                    U learning_rate_bits)const;
};
}
