#pragma once
#include "class_rank_gpu_class_apply.hpp"
#include "rl_category_policy.hpp"
#include "rl_category_lowering.hpp"
namespace rl_qualified_session::detail {
using U=rank_gpu_class_apply::U;
// CUDA full-word identity only. This does not manufacture source authority.
class ClassIdentity {
  struct Impl;std::unique_ptr<Impl> p_;
  friend class EffectiveCache;
public:
  ClassIdentity(const rank_gpu_class_apply::Result&,
                const std::array<std::vector<unsigned>,10>&,U device_cap,int device);
  ~ClassIdentity();ClassIdentity(const ClassIdentity&)=delete;
  void audit(const rank_gpu_class_apply::Result&,
             const std::array<std::vector<unsigned>,10>&);
  std::vector<U> words()const;
  U retained_device_bytes()const;U owned_device_peak_bytes()const;
};
U identity_word_count(U nodes,U arcs,U states,U edges,U rank_words);
// Full exact order identity between a private candidate and current sampled
// policy trace. Rejects mismatched versions/actions before reward credit.
void audit_policy_order(const std::array<unsigned,44>&,
                        const std::array<rl_category_policy::Trace,2>&,
                        U expected_version,U device_cap,int device);
// Integer artifact-cost comparison on CUDA; ties retain the incumbent.
bool smaller_encoded_cost(U candidate,U incumbent,int device=0);
struct EffectiveAction {std::array<unsigned,44>order{};std::string key_sha256;U hit=UINT64_MAX;};
// Exact per-node fallback/mask/selected-order key, owned by one private Session.
// References are admitted only after the Session finishes universal qualification.
class EffectiveCache {
  struct Impl;std::unique_ptr<Impl>p_;
public:
  EffectiveCache(const ClassIdentity&,U maximum_entries,U device_cap,int device);
  ~EffectiveCache();EffectiveCache(const EffectiveCache&)=delete;
  EffectiveAction normalize(const unsigned*borrowed_order);
  void audit_order(const unsigned*borrowed_order);
  void audit_traces(const std::vector<rl_category_lowering::Trace>&);
  U admit(); // Returns UINT64_MAX if storage optimization is full, never an accepted partial key.
  U retained_device_bytes()const;U entries()const;
  std::vector<std::vector<U>> snapshot_keys()const;
};
namespace testing {struct Report{bool passed=false,CUDA_executed=false;U assertions=0,rejections=0,word_checks=0;};Report gpu_checks(int device=0);Report cache_gpu_checks(int device=0);}
}
