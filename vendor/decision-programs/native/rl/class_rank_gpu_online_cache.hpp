#pragma once
#include "class_rank_disk_ledger.hpp"
#include <array>
#include <functional>
#include <memory>
#include <optional>
namespace rank_gpu_online_cache {
namespace dl=rank_disk_ledger;
inline constexpr std::uint64_t none=UINT64_MAX;
using Stop=std::function<bool()>;
struct Prefix {
 std::array<std::uint64_t,4> genesis{};
 std::uint64_t generation=0,nodes=0,terms=0;
 std::string head_sha256;
 bool operator==(const Prefix&)const=default;
};
// Ref.kind: 0 absent, 1 committed module ID, 2 earlier request-row index.
// No bits are stolen from the 64-bit ID space. Leaf refs must be absent.
struct Ref {std::uint64_t id=none;std::uint32_t kind=0;bool operator==(const Ref&)const=default;};
struct Draft {
 Ref left,right;
 std::uint64_t first_term=0,threshold_bits=0;
 std::uint32_t term_count=0,cut_bits=0;
 std::int32_t kind=2,label=-1,feature=-1;
 bool operator==(const Draft&)const=default;
};
struct Module {
 std::uint64_t id=none,expanded_nodes=0,expanded_terms=0,height=0;
 // Flag1 means actual virtual count >=2^64; value is UINT64_MAX, a lower
 // sentinel only. Flag0 means exact, including an exact UINT64_MAX count.
 std::uint32_t nodes_saturated=0,terms_saturated=0;
 bool operator==(const Module&)const=default;
};
struct Options {
 std::uint64_t maximum_nodes=262144,maximum_terms=1048576;
 std::uint64_t maximum_batch_nodes=65536,maximum_batch_terms=1048576;
 std::uint64_t maximum_device_bytes=512ull*1024*1024;
 std::uint32_t rows_per_chunk=4096,hash_bits=64;
};
// authority_sha256 binds source/rank/importer/arithmetic/native policy. nonce
// must be unique for a NEW arena lineage. Same nonce+seed intentionally denotes
// the same genesis; independent diverging writers must never share a lineage.
// Seed is candidate-only: structural import grants no archived class authority.
struct Seed {
 std::vector<dl::Node> nodes;std::vector<dl::Term> terms;
 std::string authority_sha256,nonce;
};
struct Request {Prefix expected;std::vector<Draft> nodes;std::vector<dl::Term> terms;};
struct Extension {
 Prefix before,after;
 std::string request_sha256,delta_sha256,mapping_sha256,receipt_sha256;
 bool operator==(const Extension&)const=default;
};
struct Work {
 std::uint64_t rows=0,created_nodes=0,created_terms=0,duplicate_hits=0;
 std::uint64_t equal_child_eliminations=0,hash_probes=0,exact_key_comparisons=0;
 std::uint64_t audited_rows=0,audited_new_nodes=0;
 bool operator==(const Work&)const=default;
};
struct Preparation {bool complete=false,CUDA_executed=false;Prefix prefix;std::string reason;};
struct Result {
 bool complete=false,CUDA_executed=false,structural_verified=false,published=false;
 Prefix before,after;std::optional<Extension> extension;
 std::vector<dl::Node> nodes;std::vector<dl::Term> terms;
 std::vector<Module> mapping;Work work;std::string reason;
};
struct Snapshot {Prefix prefix;std::vector<dl::Node>nodes;std::vector<dl::Term>terms;std::vector<Module>modules;};
// Host-only byte identity/checked storage sizing, never structural selection.
void validate_options(const Options&);
std::uint64_t device_bytes(const Options&);
Prefix genesis(const Seed&);
std::string request_digest(const Request&);
std::string delta_digest(const std::vector<dl::Node>&,const std::vector<dl::Term>&);
std::string mapping_digest(const std::vector<Module>&);
Extension make_extension(const Prefix&,std::uint64_t new_nodes,std::uint64_t new_terms,
 const std::string&request_sha256,const std::string&delta_sha256,const std::string&mapping_sha256);
class Arena {
 struct Impl;std::unique_ptr<Impl>p_;
public:
 Arena(Seed,Options={},int device=0);~Arena();Arena(Arena&&)noexcept;Arena&operator=(Arena&&)noexcept;
 Arena(const Arena&)=delete;Arena&operator=(const Arena&)=delete;
 Preparation prepare(const Stop& = {});
 Result append(Request,const Stop& = {});
 Prefix prefix()const;
 bool current(const Prefix&)const;
 // Exact private latest receipt only, not caller-editable proof flags. Apply
 // durable journal + registry extension synchronously before the next append.
 bool confirms_extension(const Prefix&before,const Prefix&after,const std::string&receipt_sha256)const;
 bool confirms_extension(const Extension&)const;
 Snapshot snapshot()const;
 void poison();bool poisoned()const;
};
// GPU validates every descriptor/ref, preserves arbitrary ordered finite terms
// (features0..53, duplicates and signed zero included), performs exact full-key
// hash-consing and equal-child elimination, and audits all returned mappings.
// Children of every canonical branch are smaller than the parent. Module size
// is its unfolded canonical program, not original donor work or rank volume.
// Stored counts/IDs/height are checked exact. Virtual counts saturate explicitly;
// their overflow never blocks a valid structural append or masquerades as exact.
// Existing committed payload/IDs never change. Append stages an isolated delta;
// cancellation before publication rolls back hash slots and returns no delta or
// map. Runtime failure poisons the handle. Coordinator must poison on disk or
// registry publication failure. No stop callback occurs after linearization.
// No-op requests return mappings but leave prefix and latest receipt unchanged.
// Single coordinator, not thread-safe. Do not move/destroy Arena from callbacks.
// Bounded resident implementation, not a paged/4TB inference claim. No source,
// numerical class, guard applicability or certificate authority is created.
}
