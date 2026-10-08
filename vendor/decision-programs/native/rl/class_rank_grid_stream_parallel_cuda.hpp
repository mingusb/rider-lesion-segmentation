#pragma once
#include "class_rank_grid_whole_cuda.hpp"
namespace rank_grid_stream_parallel {
using U=std::uint64_t;using Profile=rank_grid_whole::Profile;
namespace arena=rank_gpu_online_cache;namespace dl=rank_disk_ledger;
struct Progress {U total_cells=0,constructed_cells=0,batches=0,unique_nodes=0;};
struct Options {
 U maximum_cells=1048576,native_batch_rows=4096;
 U maximum_device_bytes=512ull*1024*1024;
 U maximum_source_nodes=100000,maximum_source_trees=4096;
 U maximum_arena_nodes=262144,maximum_arena_device_bytes=512ull*1024*1024;
 // Caller supplies a unique fresh lineage name (e.g. fresh output directory).
 std::string arena_nonce;
 // Qualification can force collisions; equality always compares full keys.
 std::uint32_t structural_hash_bits=64;
 // Complete aligned suffix subtrees only;1 retains the serial reference path.
 std::uint32_t parallel_chunk_limit=64;
 // Called after checked batch commit; no completion/source-acceptance claim.
 // A throwing callback fails and poisons this private construction.
 std::function<void(const Progress&)> progress;
};
struct ProfileReport {
 Profile profile;
 // Little-endian exact256-bit product;12 radices<=2^24+1 fit this format.
 std::array<U,4> cell_count_words{};
 bool fits_uint64=false,within_cell_budget=false;
};
struct Result {
 Profile profile;U cells=0,source_rows=0,leaf_drafts=0,branch_drafts=0;
 U batches=0,maximum_live_roots=0,maximum_batch_drafts=0;
 U logical_fold_branches=0,equal_child_eliminations=0,numeric_absorptions=0;
 U local_duplicate_hits=0,committed_duplicate_hits=0,exact_key_probes=0;
 U mirror_and_index_device_bytes=0;
 U parallel_staging_device_bytes=0,parallel_chunks=0,parallel_cells=0;
 U leaf_cache_hits=0,chunk_cells=1,chunk_first_dimension=12;
 arena::Snapshot graph;arena::Module root;
 std::string source_sha256;
 bool native_margin_bits_equal=false;
};
// All numerical work is CUDA. Cells are consumed M-1,...,0, batching native
// evaluation and exact right-associated dimension folds. No full-cell labels,
// ranks/frontier or2M-1 draft allocation. State has at most12 committed roots.
// Exact GPU key interning and same-feature numeric absorption precede append.
// leaf_drafts/branch_drafts count actually submitted unique drafts; the separate
// logical_fold_branches remains cells-1. The read-only committed mirror/index
// is charged to owned_device_peak_bytes. No hash-only equivalence is used.
// Arena nodes remain resident and capacity-limited: no spill/GC/resume claim.
// Fresh final audit regenerates every cell in bounded batches and compares
// native public classes plus original ordered source-margin bits.
class Compiler {
 struct Impl;std::unique_ptr<Impl>p_;
public:
 Compiler(const std::string&library,const std::string&model,Options);
 ~Compiler();Compiler(const Compiler&)=delete;Compiler&operator=(const Compiler&)=delete;
 ProfileReport profile_only();
 Result construct(const arena::Stop& = {});
 U audit(const std::vector<dl::Node>&,U root,const arena::Stop& = {});
};
}