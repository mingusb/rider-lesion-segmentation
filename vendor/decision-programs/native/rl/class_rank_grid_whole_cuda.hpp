#pragma once
#include "class_io.hpp"
#include "class_rank_gpu_online_cache.hpp"
#include "class_rank_gpu_dag_inference.hpp"
#include <array>
#include <memory>
namespace rank_grid_whole {
using U=std::uint64_t;
namespace arena=rank_gpu_online_cache; namespace dl=rank_disk_ledger;
struct Options {
 U maximum_cells=262144;
 // Bounds owned CUDA buffers, excluding the native library's internal workspace.
 U maximum_device_bytes=512ull*1024*1024;
 U native_batch_rows=4096;
 U maximum_source_nodes=100000,maximum_source_trees=4096;
};
struct CategoryProfile {
 // Tested categories ascending, followed by lowest untested OTHER representative.
 std::vector<std::uint32_t> representatives;
 std::vector<std::uint32_t> tested_features,cut_bits;
};
struct Profile {
 std::array<std::vector<std::uint32_t>,10> cuts;
 std::array<std::uint32_t,10> minimum_rank{};
 std::array<CategoryProfile,2> categories;
 std::array<U,12> radices{},strides{};
 U distinct_count=0,cells=0,owned_device_peak_bytes=0;
};
struct Result {
 Profile profile;
 U cells=0;
 std::vector<arena::Draft> drafts;
 std::string source_sha256;
 bool native_margin_bits_equal=false;
};
// Enumerates every feasible product cell of the complete source-question
// profile over finite FP32 numeric inputs and exactly-one 4/40 categories.
// Numeric rank0 is omitted only if the minimum source cut is -FLT_MAX.
// Cells use numeric0..9,wilderness,soil order, last dimension fastest.
// All classification, representative construction, coverage checks, native
// reduction and binary draft construction run on CUDA. Result drafts are a
// unique-parent full binary tree before the caller's qualified hash-consing.
// construct/audit throw on unsupported resources or failed checks: no partial
// result is a source certificate. Capacity does not imply practical scaling.
class Compiler {
 struct Impl; std::unique_ptr<Impl>p_;
public:
 Compiler(const std::string&library,const std::string&model,Options={});
 ~Compiler(); Compiler(const Compiler&)=delete; Compiler&operator=(const Compiler&)=delete;
 Result construct();
 U audit(const std::vector<dl::Node>&,U root);
};
}