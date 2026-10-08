#pragma once
// Host metadata/state only. Prediction, error counts and gate acceptance use CUDA.
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace class_study::nested_holdout {
using J = nlohmann::json;
inline void need(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
inline std::uint64_t number(const J& value) {
  need(value.is_number_integer(), "nested holdout expects a nonnegative integer");
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  const auto n = value.get<std::int64_t>();
  need(n >= 0, "nested holdout expects a nonnegative integer");
  return static_cast<std::uint64_t>(n);
}
inline J partitions(std::uint64_t rows, std::uint64_t depth) {
  need(depth > 0 && depth <= rows, "holdout depth must be positive and leave every gate nonempty");
  // Avoid an unbounded metadata allocation for malformed plans.
  need(depth <= UINT32_MAX, "holdout depth exceeds supported gate count");
  J out = J::array();
  const auto base = rows / depth, extra = rows % depth;
  std::uint64_t offset = 0;
  for (std::uint64_t i = 0; i < depth; ++i) {
    const auto count = base + (i < extra);
    out.push_back({{"offset", offset}, {"rows", count}});
    offset += count;
  }
  return out;
}
inline void distinct_inputs(const J& holdout, const J& development) {
  for (const char* key : {"values_path", "source_path", "source_csv"}) {
    if (holdout.contains(key) && development.contains(key) &&
        holdout.at(key).is_string() && development.at(key).is_string()) {
      const std::filesystem::path a = holdout.at(key).get<std::string>();
      const std::filesystem::path b = development.at(key).get<std::string>();
      need(std::filesystem::weakly_canonical(a) != std::filesystem::weakly_canonical(b),
           "holdout must be separate from development data");
    }
  }
  // A relabelled copy of the same examples is not fresh holdout data.
  for (const char* key : {"values_sha256", "source_sha256"})
    if (holdout.contains(key) && development.contains(key))
      need(holdout.at(key) != development.at(key), "holdout duplicates development data contents");
}
inline J configuration(const J& option, const J& development) {
  need(option.is_object(), "nested_holdout must be an object");
  for (auto it = option.begin(); it != option.end(); ++it)
    need(it.key() == "depth" || it.key() == "dataset" || it.key() == "acceptance_rule" || it.key() == "partition_rows",
         "unknown nested_holdout option");
  const auto depth = number(option.at("depth"));
  const auto& data = option.at("dataset");
  need(data.at("features") == development.at("features") && data.at("classes") == development.at("classes"),
       "holdout feature/class dimensions differ from development data");
  need(option.value("acceptance_rule", std::string("strictly_fewer_errors_at_every_level")) ==
       "strictly_fewer_errors_at_every_level", "unsupported nested holdout acceptance rule");
  distinct_inputs(data, development);
  const auto rows=number(data.at("rows"));
  J parts;
  if(option.contains("partition_rows")){
    const auto& counts=option.at("partition_rows");
    need(depth>0&&depth<=UINT32_MAX&&depth<=rows&&counts.is_array()&&counts.size()==depth,
         "holdout partition_rows must contain exactly depth nonempty intervals");
    parts=J::array();std::uint64_t offset=0;
    for(const auto& value:counts){const auto count=number(value);
      need(count>0&&count<=rows-offset,"holdout partition_rows exceed row extent or contain an empty interval");
      parts.push_back({{"offset",offset},{"rows",count}});offset+=count;
    }
    need(offset==rows,"holdout partition_rows must cover every holdout row exactly once");
  }else parts=partitions(rows,depth);
  return {{"depth", depth}, {"dataset", data}, {"partitions", parts},
          {"acceptance_rule", "strictly_fewer_errors_at_every_level"}};
}
inline J frozen(const J& config, std::size_t winner_index, const J& winner,
                std::size_t baseline_index, const J& baseline) {
  const bool same = winner_index == baseline_index;
  return {{"format", "nested-holdout-confirmation-1"}, {"configuration", config},
          {"inner_selected_index", winner_index}, {"baseline_index", baseline_index},
          {"inner_selected", winner}, {"baseline", baseline}, {"gates", J::array()},
          {"next_gate", 0}, {"status", same ? "baseline_already_selected" : "pending"},
          {"final_selected_index", same ? J(baseline_index) : J(nullptr)},
          {"TEST_read", false}, {"gate_data_used_for_selection", false},
          {"candidate_frozen_before_outer_scoring", true}};
}
inline void record(J& state, const J& score) {
  need(state.at("status") == "pending", "nested holdout decision is already terminal");
  const auto cursor = number(state.at("next_gate"));
  const auto& parts = state.at("configuration").at("partitions");
  need(cursor < parts.size(), "nested holdout gate cursor out of range");
  const auto rows = number(parts.at(cursor).at("rows"));
  const auto candidate = number(score.at("candidate_errors"));
  const auto baseline = number(score.at("baseline_errors"));
  need(score.at("offset") == parts.at(cursor).at("offset") &&
       score.at("candidate_source_sha256") == state.at("inner_selected").at("model_sha256") &&
       score.at("baseline_source_sha256") == state.at("baseline").at("model_sha256") &&
       score.at("candidate_native_library_sha256") == state.at("inner_selected").at("evaluation").at("native_library_sha256") &&
       score.at("baseline_native_library_sha256") == state.at("baseline").at("evaluation").at("native_library_sha256"),
       "nested holdout score belongs to a different frozen pair or interval");
  need(score.at("rows") == rows && candidate <= rows && baseline <= rows &&
       score.at("accepted").is_boolean() && score.at("CUDA_computed") == true,
       "invalid nested holdout CUDA gate result");
  // Consistency validation of a device decision, not a host candidate search.
  need(score.at("accepted") == (candidate < baseline), "nested holdout gate decision disagrees with errors");
  J gate = score; gate["level"] = cursor + 1; gate["offset"] = parts.at(cursor).at("offset");
  state["gates"].push_back(gate); state["next_gate"] = cursor + 1;
  state["gate_data_used_for_selection"] = true;
  if (!score.at("accepted").get<bool>()) {
    state["status"] = "baseline_retained";
    state["final_selected_index"] = state.at("baseline_index");
  } else if (cursor + 1 == parts.size()) {
    state["status"] = "candidate_accepted";
    state["final_selected_index"] = state.at("inner_selected_index");
  }
}
inline void validate_state(const J& state, const J& config, std::size_t winner_index,
                           const J& winner, std::size_t baseline_index, const J& baseline) {
  auto expected = frozen(config, winner_index, winner, baseline_index, baseline);
  need(state.at("gates").is_array(), "saved holdout gates must be an array");
  for (const auto& gate : state.at("gates")) record(expected, gate);
  need(state == expected, "saved nested holdout decision or frozen identities differ");
}
// OOF reuse may import development work, never a spent outer confirmation set.
inline void reject_spent_donor(const J& plan, const J& donor) {
  if (!plan.contains("nested_holdout") || !donor.contains("nested_holdout") || donor.at("nested_holdout").is_null()) return;
  const auto& old = donor.at("nested_holdout");
  if (old.at("gates").empty()) return;
  distinct_inputs(plan.at("nested_holdout").at("dataset"), old.at("configuration").at("dataset"));
}
} // namespace class_study::nested_holdout
