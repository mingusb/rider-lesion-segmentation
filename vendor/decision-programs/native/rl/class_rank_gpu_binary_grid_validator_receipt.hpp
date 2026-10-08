#pragma once
#include "class_rank_gpu_binary_grid_validator.hpp"
#include "class_rank_gpu_leaf_partitions_receipt.hpp"
namespace binary_grid_receipt {using namespace leaf_receipt;namespace vg=rank_gpu_binary_grid_validator;namespace codec=rank_regional_model_export;
inline J describe(const vg::Result&r){return J{{"complete",r.complete},{"CUDA_executed",r.CUDA_executed},{"candidate_source_cell_constancy",r.candidate_source_cell_constancy},{"cells",r.cells},{"native_rows",r.native_rows},{"native_margin_words",r.native_margin_words},{"batches",r.batches},{"checked_nodes",r.checked_nodes},{"numeric_questions",r.numeric_questions},{"category_entries",r.category_entries},{"category_values",r.category_values},{"category_equivalence_comparisons",r.category_equivalence_comparisons},{"owned_device_peak_bytes",r.owned_device_peak_bytes},{"reason",r.reason},{"native_configuration",r.native_configuration.empty()?J():J::parse(r.native_configuration)},{"native_contract",r.native_contract.empty()?J():J::parse(r.native_contract)},{"radices",r.profile.radices},{"source_rank_cut_bits",r.profile.cuts},{"minimum_rank",r.profile.minimum_rank}};}
inline J build(const fs::path&source){return bind(source);}
inline void same(const codec::Model&a,const codec::Model&b){need(a.source_sha256==b.source_sha256&&a.rank_sha256==b.rank_sha256&&a.rank_cut_bits==b.rank_cut_bits&&a.scope.lo==b.scope.lo&&a.scope.hi==b.scope.hi&&a.scope.allowed==b.scope.allowed&&a.nodes==b.nodes&&a.terms==b.terms&&a.root==b.root,"validator codec metadata readback differs");}
}
