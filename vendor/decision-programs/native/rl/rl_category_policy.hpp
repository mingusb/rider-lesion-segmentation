#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rl_category_policy {
using U=std::uint64_t;
inline constexpr unsigned categories=44,maximum_actions=40;
inline constexpr unsigned sampler_schema_version=1;
inline constexpr U wild_mask=15,soil_mask=((U(1)<<44)-1)^wild_mask;
struct Request {U seed=0,episode=0,allowed_mask=0,fallback_mask=0;unsigned group=0,reserved=0;};
struct Trace {
 U version=0,seed=0,episode=0,allowed_mask=0,fallback_mask=0;
 unsigned group=0,steps=0;
 unsigned actions[maximum_actions]{};
 U remaining[maximum_actions]{};
 double log_probability[maximum_actions]{};
 double gradient[categories]{};
 double total_log_probability=0;
};
static_assert(sizeof(Request)==40&&sizeof(Trace)==1208);
// These flags are a caller declaration for reward eligibility, not a private
// source/native certificate. Policy outputs never authorize classifier roots.
struct Credit {U sampled_version=0,sampled_seed=0,sampled_episode=0,encoded_bytes=0,baseline_bytes=0;U learning_rate_bits=0;unsigned complete=0,independently_validated=0,scope_qualified=0,reserved=0;};
struct OrderView {const unsigned*order44=nullptr;const Trace*traces=nullptr;U version=0,seed=0,episode=0;int device=0;U baseline_bytes=0;};
struct Update {bool applied=false;U old_version=0,new_version=0,updates=0;int error=0;std::string reason;};
class Policy {
 struct Impl;std::unique_ptr<Impl>p_;
public:
 explicit Policy(int device=0);
 ~Policy();Policy(Policy&&)noexcept;Policy&operator=(Policy&&)noexcept;
 Policy(const Policy&)=delete;Policy&operator=(const Policy&)=delete;
 // CUDA samples two full Plackett-Luce permutations. Lowering keeps its own
 // fixed fallback, skips those bits and emits this preference in reverse.
 // Borrowed pointers stay valid until next sample/update/upload/destruction;
 // caller must finish synchronized lowering before changing the policy.
 // Baseline cost must be independent of this action, e.g. the frozen baseline
 // constructor's complete bytes or an EMA from earlier episodes. It is bound
 // before sampling and cannot be changed by a post-action Credit.
 OrderView sample44(U seed,U episode,U baseline_bytes=0);
 std::array<Trace,2>trace()const;
 Update reinforce(const Credit&credit);
 std::array<U,categories>logit_words()const;
 // Exact word transport. CUDA rejects nonfinite words and nonpositive version.
 void upload(const std::array<U,categories>&words,U version);
 U version()const;U owned_device_bytes()const;
};
struct CheckResult {bool passed=false,CUDA_executed=false,native_class_authority=false;U assertions=0,rejections=0,gradient_cases=0,update_cases=0,first_failure=0;};
CheckResult gpu_checks(int device=0);
} // namespace rl_category_policy
