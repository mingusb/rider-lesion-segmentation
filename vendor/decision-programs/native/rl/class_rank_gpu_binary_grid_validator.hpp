#pragma once
#include "class_rank_grid_stream_parallel_cuda.hpp"
#include "class_rank_regional_model_export.hpp"
namespace rank_gpu_binary_grid_validator {
using U=std::uint64_t;using Profile=rank_grid_whole::Profile;using ProfileReport=rank_grid_stream_parallel::ProfileReport;
namespace codec=rank_regional_model_export;namespace dl=rank_disk_ledger;using Stop=rank_gpu_online_cache::Stop;
struct Options {U maximum_cells=67108864,native_batch_rows=4096,maximum_device_bytes=256ull*1024*1024,maximum_source_nodes=100000,maximum_source_trees=4096,maximum_candidate_nodes=1048576;std::function<void(U,U)>progress;};
struct Result {bool complete=false,CUDA_executed=false,candidate_source_cell_constancy=false;U cells=0,native_rows=0,native_margin_words=0,batches=0,checked_nodes=0,numeric_questions=0,category_entries=0,category_values=0,category_equivalence_comparisons=0,owned_device_peak_bytes=0;std::string reason,native_configuration,native_contract;Profile profile;};
// Independent persisted-binary validator. No imported Apply receipts confer
// authority. Every categorical dimension-block entry is checked on all4/40
// values; source-equivalent categories must reach exactly the same exit ID.
// Numeric rank predicates must be source boundaries; dimensions never decrease.
// This sufficient certificate can reject extensionally equal distinct exits.
// Complete finite-FP32-domain class equality additionally requires every source
// quotient cell's native public first-argmax and ordered margin-word replay.
struct BatchAudit {bool complete=false,CUDA_executed=false;U cells=0,probability_words=0,margin_words=0,native_calls=0,native_rows=0,owned_device_peak_bytes=0;std::string reason;};
// Metadata sizing only. No numerical model work.
U planned_batch_device_bytes(U rows);void validate_options(const Options&);
class Validator {struct Impl;std::unique_ptr<Impl>p_;public:Validator(const std::string&library,const std::string&source,Options={});~Validator();Validator(const Validator&)=delete;Validator&operator=(const Validator&)=delete;ProfileReport profile_only();Result audit(const codec::Model&,const Stop& = {});BatchAudit audit_batch_words(U first_rows,U second_rows,const Stop& = {});};
}
