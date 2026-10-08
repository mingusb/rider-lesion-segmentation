#include "streaming_file_hash.hpp"
#pragma once
#include "class_rank_gpu_leaf_partitions.hpp"
#include "class_io.hpp"
#include "portable_build_identity.hpp"
#include <bit>
#include <unistd.h>
namespace leaf_receipt {
using J=dpnative::json;namespace fs=std::filesystem;namespace lp=rank_gpu_leaf_partitions;namespace sf=rank_gpu_score_factors;
inline void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
inline J box(const rank_gpu_bounds::Box& b){return J{{"lo",b.lo},{"hi",b.hi},{"allowed",b.allowed}};}
inline J describe(const lp::Result& r){return J{{"complete",r.complete},{"CUDA_executed",r.CUDA_executed},{"reason",r.reason},{"source_sha256",r.source_sha256},{"library_sha256",r.library_sha256},{"rank_sha256",r.rank_sha256},{"source_binding",r.source.binding},{"domain",box(r.domain)},{"original_leaves",r.original_leaves},{"feasible_leaves",r.feasible_leaves},{"empty_leaves",r.empty_leaves},{"disjoint_pairs_checked",r.disjoint_pairs_checked},{"complete_tree_volumes_checked",r.complete_tree_volumes_checked},{"native_witness_rows",r.native_witness_rows},{"native_margin_words",r.native_margin_words},{"owned_device_peak_bytes",r.owned_device_peak_bytes},{"owned_device_resident_bytes",r.owned_device_resident_bytes}};}
inline J describe(const lp::Audit& a){return J{{"complete",a.complete},{"CUDA_executed",a.CUDA_executed},{"rows",a.rows},{"native_margin_words",a.native_margin_words},{"factor_margin_words",a.factor_margin_words},{"owned_device_peak_bytes",a.owned_device_peak_bytes},{"reason",a.reason}};}
inline J describe(const sf::Result& a){return J{{"complete",a.complete},{"CUDA_executed",a.CUDA_executed},{"source_binding",a.source_binding},{"reason",a.reason},{"combinations",a.combinations},{"compatible",a.compatible},{"cartesian_offsets",a.cartesian_offsets},{"factor_offsets",a.factor_offsets},{"owned_device_peak_bytes",a.owned_device_peak_bytes}};}
inline J source_payload(const lp::Result& r){J leaves=J::array(),bias=J::array();for(const auto& a:r.source.value.leaves)leaves.push_back(J{{"box",box(a.box)},{"ordinal",a.ordinal},{"value_bits",std::bit_cast<std::uint32_t>(a.value)}});for(float x:r.source.value.bias)bias.push_back(std::bit_cast<std::uint32_t>(x));return J{{"format","GPU-complete-source-leaf-partitions-1"},{"binding",r.source.binding},{"source_sha256",r.source_sha256},{"library_sha256",r.library_sha256},{"rank_sha256",r.rank_sha256},{"domain",box(r.domain)},{"rank_cut_bits",r.rank_cut_bits},{"tree_offsets",r.source.value.offsets},{"channels",r.source.value.channels},{"bias_bits",bias},{"leaves",leaves},{"import_is_not_source_authority",true}};}
inline J factor_payload(const sf::Result& r){J rows=J::array();for(const auto& a:r.factors)rows.push_back(J{{"box",box(a.box)},{"score_bits",a.score_bits},{"channel",a.channel},{"combination",a.combination}});return J{{"format","GPU-complete-source-score-factors-1"},{"binding",r.source_binding},{"factor_offsets",r.factor_offsets},{"factors",rows},{"class_authority",false}};}
inline J hashes(const fs::path& dir){J h=J::object();for(const auto& e:fs::directory_iterator(dir))if(e.is_regular_file())h[e.path().filename().string()]=dp_streaming::sha256_file(e.path());return h;}
inline J bind(const fs::path&){return public_rl_build::binding();}
}
