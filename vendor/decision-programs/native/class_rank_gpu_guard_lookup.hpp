#pragma once
#include "class_rank_gpu_bounds.hpp"
#include <functional>
#include <string>
namespace rank_gpu_guard_lookup {
using Stop=std::function<bool()>;using Box=rank_gpu_bounds::Box;
inline constexpr std::uint64_t none=UINT64_MAX;
// Six complete 256-bit registry identities, in this order: source bytes,
// rank-threshold words, importer semantics, ordered arithmetic, current native
// qualification, module-arena/predicate words. Compare ALL words+generations;
// an aggregate hash is never substituted for exact equality of this key.
struct AuthorityKey {std::array<std::uint64_t,24>words{};std::uint64_t authority_generation=0,module_generation=0;bool operator==(const AuthorityKey&)const=default;};
struct Entry {std::uint64_t entry_id=none,module_id=none,certificate_id=none;Box guard;};
struct Bank {AuthorityKey authority;Box domain;std::uint64_t module_count=0;std::vector<Entry>entries;std::string binding;};
// REQUIRED live registry boundary: true means the exact module+guard+authority
// conclusion named by this committed certificate ID was already checked.
// This module neither checks source equivalence nor mints a certificate.
using Admission=std::function<bool(const AuthorityKey&,const Entry&)>;
struct Options {std::uint64_t maximum_bank_bytes=64ull*1024*1024,maximum_query_bytes=16ull*1024*1024;std::uint32_t entries_per_window=4096,queries_per_window=256;};
struct Preparation {bool complete=false,CUDA_executed=false;std::uint64_t entries=0;std::string binding,reason;};
struct Batch {AuthorityKey authority;std::vector<Box>queries;std::string bank_binding,binding;};
enum class Outcome:std::int32_t {miss=0,hit=1,empty=2};
struct Hit {Outcome outcome=Outcome::miss;std::uint64_t entry_id=none,module_id=none,certificate_id=none;bool operator==(const Hit&)const=default;};
struct Result {bool complete=false,CUDA_executed=false;std::string bank_binding,binding,reason;std::vector<Hit>hits;std::uint64_t comparisons=0,windows=0;};
// Immutable snapshot, no concurrent/reentrant calls. Caller must invalidate the
// generation when live authority or module arena changes. Entry IDs must be
// strictly increasing (lowest ID wins); certificate IDs may be shared only as
// authorized by Admission. Entries are expanded full source-guard contracts;
// residual-projection entries require another checked premise and are NOT
// accepted by this API. Rank cells only: raw fractional FP32 inputs require the
// separately qualified source-rank mapper. No graph traversal occurs here.
class DeviceBank {struct Impl;std::unique_ptr<Impl>p_;public:
 DeviceBank(Bank bank,Admission admission,Options options={},int device=0);~DeviceBank();
 DeviceBank(const DeviceBank&)=delete;DeviceBank&operator=(const DeviceBank&)=delete;
 Preparation prepare(const Stop&stop={});const Preparation&preparation()const;
 // CUDA checks every authority word, query/guard/domain bounds and 10 interval
 // plus two category-group inclusion tests. Valid empty queries return EMPTY
 // with no IDs. Invalid words/nonempty out-of-domain queries throw. A hit only
 // identifies an already admitted certificate; it adds no class authority.
 // Cancellation/resource exhaustion returns incomplete with no published hits.
 Result lookup(Batch batch,const Stop&stop={});
};
// Proposal transport only; these boxes have NO applicability authority.
// Numeric axis0..9: side0 lowers lo, side1 raises hi, by amount then clips.
// Category axis10/11: side MUST0, amount is local bit index0..3/0..39;
// add that bit only when allowed in domain (otherwise an unchanged proposal).
struct Action {std::uint64_t guard_index=0;std::uint32_t axis=0,side=0,amount=1;};
struct ExpansionInput {Box domain;std::vector<Box>guards;std::vector<Action>actions;std::string binding;};
struct Expansion {Box box;std::int32_t changed=0;};
struct ExpansionResult {bool complete=false,CUDA_executed=false;std::string binding,reason;std::vector<Expansion>proposals;};
ExpansionResult propose_expansions(ExpansionInput,Options={},const Stop&stop={},int device=0);

}
