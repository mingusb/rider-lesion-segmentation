#pragma once
#include "class_rank_gpu_dag_collect.hpp"
#include "class_rank_gpu_online_cache.hpp"
#include "class_rank_gpu_task_resolution.hpp"
#include <filesystem>
namespace rank_regional_model_export {
namespace arena=rank_gpu_online_cache;namespace tasks=rank_gpu_task_resolution;namespace collect=rank_gpu_dag_collect;
using U=std::uint64_t;using RankWords=std::array<std::vector<std::uint32_t>,10>;using Stop=collect::Stop;
struct Request {
 arena::Snapshot graph;tasks::Task resolved_root;tasks::Box scope;
 std::string source_sha256,rank_sha256,graph_checkpoint_sha256,current_policy_sha256,root_proof_sha256;
 RankWords rank_cut_bits;std::string binding;
};
// This is CURRENT resolved module/source scope authority, not an archive flag.
// Must check exact root, source/rank/native policy, graph prefix and guard proof.
using Admission=std::function<bool(const Request&)>;
struct Options {collect::Options collector;U maximum_runtime_bytes=1024ull*1024*1024;};
struct Usage {U files=0,logical_bytes=0,allocated_bytes=0;};
struct Result {
 bool complete=false,CUDA_executed=false;std::string reason,model_sha256,audit_sha256;
 U runtime_model_bytes=0,construction_audit_bytes=0,nodes=0,terms=0,root=arena::none;
 U header_bytes=0,rank_bytes=0,node_bytes=0,term_bytes=0,checksum_bytes=0;
};
struct Model {
 std::string source_sha256,rank_sha256;tasks::Box scope;RankWords rank_cut_bits;
 std::vector<collect::dl::Node>nodes;std::vector<collect::dl::Term>terms;U root=arena::none;
};
// Canonical exact storage codec only. It does NOT collect, route, select or
// authorize a graph. The single inference file contains no maps/proof journals.
std::string encode(const Model&);Model decode(const std::string&);
Model read_model(const std::filesystem::path&,const std::string& expected_sha256,U maximum_bytes=1024ull*1024*1024);
std::string rank_digest(const RankWords&);
// Calls CUDA collector with ONLY resolved_root.module.id, then exact serialization
// and independent decoding. Fresh directories; model.bin is the sole runtime
// artifact. Collection/input maps and source proof references remain audit-only.
// Initial schema is explicitly REGION scoped; never claims whole-model coverage.
Result export_model(std::filesystem::path runtime_directory,std::filesystem::path audit_directory,
 Request,Admission,Options={},const Stop& = {},int device=0);
// Snapshot logical and filesystem allocated bytes; skips the designated runtime
// subtree when accounting construction. Counts files once by pathname, not dedup.
Usage directory_usage(const std::filesystem::path&,const std::filesystem::path& excluded_subtree={});
}
