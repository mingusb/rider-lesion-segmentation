#pragma once
#include "class_rank_gpu_online_cache.hpp"
#include "class_rank_gpu_guard_lookup.hpp"
#include <functional>
#include <memory>
namespace rank_gpu_task_resolution {
namespace arena=rank_gpu_online_cache;namespace guard=rank_gpu_guard_lookup;
using U=std::uint64_t;using Box=guard::Box;using Stop=std::function<bool()>;
inline constexpr U none=UINT64_MAX;
struct Question {std::int32_t feature=-1;std::uint32_t cut_bits=0;bool operator==(const Question&)const=default;};
enum class Status:std::int32_t {pending=0,split=1,resolved=2};
struct Task {
 U id=none,parent=none;std::int32_t side=-1;Box box;Status status=Status::pending;
 Question question;U left=none,right=none,partition_certificate=none;
 arena::Module module;U certificate=none;
};
enum class Kind:std::int32_t {split=0,guard_hit=1,source_leaf=2,parent_completion=3};
struct Event {
 Kind kind=Kind::guard_hit;U task=none;Box expected_box;
 // Split: exact strict complementary axis boxes, assigned fresh child IDs by CUDA.
 Question question;Box left_box,right_box;U partition_certificate=none;
 // Resolution: checked canonical module metadata and source contract ID.
 arena::Module module;U certificate=none;
 // Hit guard is supplied by the current qualified lookup/registry.
 guard::Entry hit;
 // Source leaf class is qualified by registry, never by this structural module.
 std::int32_t source_label=-1;
 // Completion must name BOTH current child contracts.
 U left_certificate=none,right_certificate=none;
};
struct ParentRequest {
 U task=none,partition_certificate=none;Box box;Question question;
 U left_task=none,right_task=none,left_certificate=none,right_certificate=none;
 arena::Module left,right;arena::Draft draft;
};
struct Application {Event event;Task before,after;Task left,right;};
struct Work {
 U splits=0,guard_hits=0,source_leaves=0,parents_completed=0;
 U guard_module_nodes_referenced=0,guard_module_terms_referenced=0;
 U task_resolutions=0;
 std::uint32_t guard_nodes_saturated=0,guard_terms_saturated=0;
 // These are operations/references, NOT observed saved source work.
 bool operator==(const Work&)const=default;
};
struct State {
 bool prepared=false;U revision=0,tasks=0,pending=0,split=0,resolved=0;
 arena::Prefix prefix;Work work;std::string state_sha256,last_commit_sha256;
};
struct Seed {guard::AuthorityKey authority;arena::Prefix prefix;Box domain;std::string binding;};
struct Batch {
 guard::AuthorityKey authority;U expected_revision=0;
 arena::Prefix expected_prefix,target_prefix;std::vector<Event>events;std::string binding;
};
struct Options {U maximum_tasks=262144,maximum_events=4096,maximum_device_bytes=512ull*1024*1024;};
struct Prepared {
 State before,after;std::string binding,delta_sha256;
 // state_sha256 is a genesis+exact-delta replay-chain identity, not a fresh
 // whole-buffer hash. after.last_commit_sha256 is filled only in Result.state.
 std::vector<Application>applications;std::vector<ParentRequest>ready_parents;
 // Every returned parent is a proposal for exact structural interning. Its
 // source contract requires partition + BOTH child contracts via registry.
};
// All callbacks are trusted live synchronous boundaries, not JSON import.
// context accepts a currently checked immutable arena prefix. extension proves
// old words/IDs unchanged. admit validates module mapping/proof provenance;
// split admission may mint only a structural partition contract after CUDA
// has checked exact strict nonempty child boxes.
struct Registry {
 std::function<bool(const guard::AuthorityKey&,const arena::Prefix&)> context;
 std::function<bool(const arena::Prefix&,const arena::Prefix&)> extension;
 std::function<bool(const guard::AuthorityKey&,const arena::Prefix&,const Application&)> admit;
};
// Must durably commit EXACT Prepared delta/proof/application records. Return
// its committed SHA256. It must not mutate arena/source/native identities.
// After it succeeds there is no stop callback; host pointer swap publishes the
// already checked CUDA draft. Failure poisons this handle; poison any arena
// whose append was part of the external transaction too.
using Publish=std::function<std::string(const Prepared&)>;
struct Result {bool complete=false,CUDA_executed=false,published=false;State state;Prepared transaction;std::string reason;};
// Structural import provenance only. Ledger digest is the old authenticated
// committed-prefix chain SHA, never a hash of an appendable unbounded file.
struct ImportDescriptor {
 std::string origin_head_sha256,origin_tasks_sha256,origin_ledger_sha256;
 std::string snapshot_sha256,gpu_validation_sha256;
 bool operator==(const ImportDescriptor&)const=default;
};
struct ImportTask {U node_id=none;Box box;};
struct ImportInput {
 std::vector<ImportTask>tasks;std::vector<rank_disk_ledger::Node>nodes;
 std::string origin_head_sha256,origin_tasks_sha256,origin_ledger_sha256;
};
struct ImportPreparation {bool complete=false,CUDA_executed=false;State state;ImportDescriptor descriptor;U archived_leaves=0;std::string reason;};
struct Snapshot {State state;std::vector<Task>tasks;};
enum class PendingOrder:std::int32_t {oldest_first=0,newest_first=1};
struct PendingSelection {
 bool complete=false,CUDA_executed=false;State state;PendingOrder order=PendingOrder::oldest_first;
 U examined_tasks=0,total_pending=0;std::vector<Task>rows;std::string reason;
};
// Canonical metadata only: little-endian words, length-prefixed strings, no padding.
// AFTER state/commit digests are excluded to avoid circular hashes.
std::string delta_words(const Prepared&);
std::string delta_digest(const Prepared&);
std::string genesis_digest(const Seed&);
std::string state_extension_digest(const State&,const std::string& delta);
U device_bytes(const Options&);
class DeviceTasks {
 struct Impl;std::unique_ptr<Impl>p_;
public:
 DeviceTasks(Seed,Registry,Options={},int device=0);~DeviceTasks();
 DeviceTasks(const DeviceTasks&)=delete;DeviceTasks&operator=(const DeviceTasks&)=delete;
 Result prepare(const Stop& = {});
 // NEW structural genesis only: preserves every original task ID/box and
 // stored strict axis question, but imports all old leaves/frontier PENDING.
 // No old labels, proof IDs, modules or guards become current authority.
 // Caller must separately authenticate origin HEAD/ledger/task bindings.
 ImportPreparation prepare_import(ImportInput,const Stop& = {});
 bool confirms_import(const Seed&,const State&,const ImportDescriptor&)const;
 // Available only while the exact imported genesis is current (revision0).
 // Used by the durable imported checkpoint writer, bounded to4096 rows/call.
 std::vector<Task> read_imported_page(U first,U count)const;
 Result apply(Batch,Publish,const Stop& = {});
 // Read-only bounded CUDA selection. Scans resident statuses in parallel;
 // host receives only min(maximum_rows,pending) exact Task records. Ordering
 // is deterministic by stable ID. Invalid order/zero request is rejected;
 // cancellation/resource limits return no rows and never mutate task state.
 PendingSelection select_pending(U maximum_rows,PendingOrder,const Stop& = {});
 Snapshot snapshot()const;State state()const;void poison();bool poisoned()const;
};
// V1 splits use strict integer rank-axis / one-hot indicator predicates;
// weighted cached MODULES are allowed via qualified guard contracts, but no
// weighted pending-task geometry is approximated by an outer box. No source
// or class computation, source-query-avoidance claim, archive authority, paged
// state, resume importer or whole-source acceptance is provided here.
}
