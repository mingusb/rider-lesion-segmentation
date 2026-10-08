#pragma once
// Standalone disabled prototype: source bounds only, never a class certificate.
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
namespace rank_gpu_bounds {
inline constexpr uint64_t wilderness=15,soil=((uint64_t(1)<<44)-1)^wilderness;
struct Box {std::array<int32_t,10> lo{},hi{};uint64_t allowed=wilderness|soil;};
struct Term {int32_t feature=0;float weight=0;};
struct Condition {int32_t first_term=0,term_count=0,side=0,pad=0;double threshold=0;};
struct Query {Box box;int32_t first_condition=0,condition_count=0;};
struct Leaf {Box box;int32_t ordinal=-1;float value=0;};
struct Source {
  std::vector<Leaf> leaves;
  std::vector<int32_t> offsets,channels;
  std::array<float,7> bias{};
};
struct Batch {std::vector<Query> queries;std::vector<Condition> conditions;std::vector<Term> terms;};
struct TreeBound {
  float low=0,high=0;int32_t first=-1,last=-1,low_ordinal=-1,high_ordinal=-1;
  uint32_t minimum_bits=UINT32_MAX,maximum_bits=0;
  uint64_t leaf_tests=0,axis_feasible=0,plane_tests=0,plane_rejected=0;
};
struct RegionBound {
  std::array<float,7> lower{},upper{};
  uint64_t leaf_tests=0,axis_feasible=0,plane_tests=0,plane_rejected=0;
  int32_t first_empty_tree=-1,constant_numeric=1,constant_bits=1,bounds_valid=0;
};
struct Timing {double host_ms=0,upload_ms=0,kernel_ms=0,download_ms=0;};
struct Result {std::vector<TreeBound> trees;std::vector<RegionBound> regions;Timing timing;};
void validate(const Source&,const Batch&);
void validate_batch(const Batch&);
Result reference(const Source&,const Batch&,bool include_trees=true);
class DeviceSource {
  struct Impl;std::unique_ptr<Impl> impl_;
public:
  explicit DeviceSource(const Source&,int device=0);
  ~DeviceSource();
  DeviceSource(const DeviceSource&)=delete;DeviceSource& operator=(const DeviceSource&)=delete;
  Result evaluate(const Batch&,bool include_trees=true);
  double source_upload_ms()const;
};
}
