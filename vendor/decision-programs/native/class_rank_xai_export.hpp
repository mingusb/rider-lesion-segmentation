#pragma once
#include "class_rank_regional_model_export.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace rank_xai_export {
namespace codec=rank_regional_model_export;
using U=std::uint64_t;
// Structural lowering only: no input routing, score calculation or predictions.
enum class GateKind : std::uint32_t {
 leaf=0, constant_false=1, constant_true=2, numeric_less=3, category_less=4
};
struct Node {
 U id=codec::arena::none,left=codec::arena::none,right=codec::arena::none;
 std::int32_t kind=2,feature=-1,label=-1;
 std::uint32_t stored_cut_bits=0,raw_threshold_bits=0;
 GateKind gate=GateKind::leaf;
 bool operator==(const Node&)const=default;
};
struct Model {
 std::string runtime_sha256,source_sha256,rank_sha256;
 codec::tasks::Box scope;
 codec::RankWords rank_cut_bits;
 std::vector<Node>nodes;
 U root=codec::arena::none,runtime_bytes=0;
};
// Supported: axis questions on 10 source-rank numeric and 44 one-hot features,
// and seven class leaves. Ordered arithmetic planes reject explicitly.
// Numeric r<c lowers to false if c<=0, true if c>L, otherwise raw x<cuts[ceil(c)-1].
// Threshold words, including signed zero, are authoritative; displayed decimal
// values are commentary. Full source/native equivalence is not re-proved here.
Model lower(const codec::Model&);
struct Options {bool smooth_numeric=false; U maximum_runtime_bytes=1024ull*1024*1024;};
struct Result {
 std::string runtime_sha256,equations_sha256,text_sha256,inspect_sha256,result_sha256;
 U nodes=0,runtime_bytes=0,export_bytes=0; bool structural_roundtrip=false;
};
// Fresh output only. Reconstructs codec words from JSON and demands exact runtime
// bytes before publishing result.json. All CPU work is schema/metadata/I/O only.
Result write(const std::filesystem::path&runtime,const std::string&expected_sha256,
             const std::filesystem::path&fresh_output,Options={});
void verify(const std::filesystem::path&export_directory);
} // namespace rank_xai_export

