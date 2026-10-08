#pragma once

// Exact symbolic class-tree optimization. No floating-point arithmetic, CUDA,
// Torch, Python, model fitting, or observation-based equivalence is used here.
// Predicate bit patterns are opaque Boolean variable identities. Missing-value
// defaults are part of that identity, so every rewrite remains pointwise valid.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace decision_programs::symbolic {

using Id = std::uint64_t;
inline constexpr Id kLeaf = std::numeric_limits<Id>::max();
using Deadline = std::function<void()>;

struct Predicate {
  std::int32_t feature = -1;
  std::uint32_t cut_bits = 0;
  std::uint8_t missing_left = 0;
  bool operator==(const Predicate&) const = default;
};

struct Node {
  Predicate predicate{};
  Id left = kLeaf;
  Id right = kLeaf;
  std::int64_t label = -1;
  [[nodiscard]] bool terminal() const noexcept { return label >= 0; }
};

struct PredicateHash {
  std::size_t operator()(const Predicate& p) const noexcept {
    auto x = (std::uint64_t{static_cast<std::uint32_t>(p.feature)} << 32) | p.cut_bits;
    x ^= std::uint64_t{p.missing_left} * 0x9e3779b97f4a7c15ULL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return static_cast<std::size_t>(x ^ (x >> 31));
  }
};

struct BranchKey {
  Predicate predicate;
  Id left;
  Id right;
  bool operator==(const BranchKey&) const = default;
};

struct BranchHash {
  std::size_t operator()(const BranchKey& k) const noexcept {
    auto x = static_cast<std::uint64_t>(PredicateHash{}(k.predicate));
    x ^= k.left + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    x ^= k.right + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return static_cast<std::size_t>(x);
  }
};

struct PassStats {
  std::uint64_t nodes_before = 0;
  std::uint64_t nodes_after = 0;
  std::uint64_t accepted_rotations = 0;
  std::uint64_t accepted_cofactors = 0;
  std::uint64_t cofactor_states = 0;
  bool cofactor_budget_exhausted = false;
};

struct OptimizationStats {
  std::uint64_t nodes_before = 0;
  std::uint64_t nodes_after = 0;
  bool converged = false;
  bool shared_nodes_at_deployment = false;
  std::string stop_reason = "pass limit";
  std::vector<PassStats> passes;
};

struct DeploymentStats {
  std::uint64_t physical_nodes = 0;
  std::uint64_t physical_leaves = 0;
  std::uint64_t max_depth = 0;
  std::uint64_t reachable_canonical_records = 0;
};

class SymbolicStore {
 public:
  // Child-before-parent canonical records are directly serializable. Call
  // rebuild_index() after loading or otherwise replacing these records.
  std::vector<Node> records;
  OptimizationStats optimization;
  // On a cooperative deadline exception, this remains a valid completed
  // incumbent in the current records, including after an earlier collection.
  Id checkpoint_root = kLeaf;

  explicit SymbolicStore(std::uint64_t outputs) : outputs_(outputs) {
    if (outputs == 0 || outputs > static_cast<Id>(std::numeric_limits<std::int64_t>::max()))
      throw std::invalid_argument("invalid symbolic class count");
    for (Id label = 0; label < outputs; ++label) leaf(static_cast<std::int64_t>(label));
  }

  [[nodiscard]] std::uint64_t outputs() const noexcept { return outputs_; }

  Id leaf(std::int64_t label) {
    if (label < 0) throw std::invalid_argument("symbolic leaf labels must be nonnegative");
    if (auto it = leaves_.find(label); it != leaves_.end()) return it->second;
    const Id result = records.size();
    records.push_back(Node{Predicate{}, kLeaf, kLeaf, label});
    sizes_.push_back(1);
    leaves_.emplace(label, result);
    return result;
  }

  Id make(Predicate predicate, Id left, Id right) {
    validate_predicate(predicate);
    require_id(left); require_id(right);
    while (!records[left].terminal() && records[left].predicate == predicate) left = records[left].left;
    while (!records[right].terminal() && records[right].predicate == predicate) right = records[right].right;
    if (left == right) return left;
    const BranchKey key{predicate, left, right};
    if (auto it = branches_.find(key); it != branches_.end()) return it->second;
    const auto size = checked_size(sizes_[left], sizes_[right]);
    const Id result = records.size();
    records.push_back(Node{predicate, left, right, -1});
    sizes_.push_back(size);
    branches_.emplace(key, result);
    return result;
  }

  [[nodiscard]] std::uint64_t physical_size(Id root) const {
    require_id(root);
    return sizes_[root];
  }

  // Analysis may contain opaque unfinished functions encoded as labels beyond
  // outputs(). They are never valid deployment classes. Check the reachable
  // program before creating a prediction artifact; unused analysis records do
  // not affect the completed root. Counts charge every physical occurrence.
  [[nodiscard]] DeploymentStats deployment_stats(Id root, std::uint64_t features,
                                                  const Deadline& deadline = {}) const {
    require_id(root);
    if (!features) throw std::invalid_argument("deployment requires positive features");
    if (deadline) deadline();
    DeploymentStats result;
    result.physical_nodes = sizes_[root];
    result.physical_leaves = sizes_[root] / 2 + 1;
    std::vector<std::uint8_t> reached(static_cast<std::size_t>(root) + 1, 0);
    std::vector<std::uint64_t> depths(static_cast<std::size_t>(root) + 1, 0);
    reached[root] = 1;
    for (Id cursor = root + 1; cursor > 0;) {
      const Id node = --cursor;
      poll(deadline, node);
      if (!reached[node]) continue;
      const Node& record = records[node];
      ++result.reachable_canonical_records;
      if (record.terminal()) {
        if (static_cast<std::uint64_t>(record.label) >= outputs_)
          throw std::invalid_argument("cannot deploy an opaque unfinished symbolic leaf");
        if (record.left != kLeaf || record.right != kLeaf)
          throw std::invalid_argument("deployment terminal has children");
      } else {
        validate_predicate(record.predicate);
        if (static_cast<std::uint64_t>(record.predicate.feature) >= features)
          throw std::invalid_argument("deployment predicate exceeds feature dimension");
        if (record.label != -1 || record.left >= node || record.right >= node)
          throw std::invalid_argument("deployment requires child-before-parent canonical records");
        reached[record.left] = reached[record.right] = 1;
      }
    }
    for (Id node = 0; node <= root; ++node) {
      poll(deadline, node);
      if (reached[node] && !records[node].terminal())
        depths[node] = 1 + std::max(depths[records[node].left], depths[records[node].right]);
    }
    result.max_depth = depths[root];
    if (deadline) deadline();
    return result;
  }

  // Rebuild cached sizes and exact-key indexes without renumbering loaded
  // records. Hash collisions are resolved by complete field equality.
  void rebuild_index(const Deadline& deadline = {}) {
    std::vector<std::uint64_t> sizes;
    std::unordered_map<std::int64_t, Id> leaves;
    std::unordered_map<BranchKey, Id, BranchHash> branches;
    sizes.reserve(records.size());
    for (Id node = 0; node < records.size(); ++node) {
      poll(deadline, node);
      const auto& record = records[node];
      if (record.terminal()) {
        if (record.left != kLeaf || record.right != kLeaf)
          throw std::invalid_argument("symbolic leaf has children");
        sizes.push_back(1);
        leaves.try_emplace(record.label, node);
      } else {
        if (record.label != -1 || record.left >= node || record.right >= node)
          throw std::invalid_argument("symbolic records require child-before-parent IDs");
        validate_predicate(record.predicate);
        sizes.push_back(checked_size(sizes[record.left], sizes[record.right]));
        branches.try_emplace(BranchKey{record.predicate, record.left, record.right}, node);
      }
    }
    sizes_.swap(sizes); leaves_.swap(leaves); branches_.swap(branches);
  }

  // Import one physical hierarchy. Opaque pending regions must have distinct
  // caller-assigned labels >= outputs(); no unknown is treated as a class.
  Id canonicalize(const std::vector<Node>& input, Id root = 0, const Deadline& deadline = {}) {
    if (root >= input.size()) throw std::invalid_argument("invalid physical root");
    std::vector<std::uint8_t> seen(input.size(), 0);
    std::vector<Id> mapped(input.size(), kLeaf);
    std::vector<std::pair<Id, bool>> pending{{root, false}};
    std::uint64_t visited = 0;
    while (!pending.empty()) {
      poll(deadline, visited);
      const auto [node, finish] = pending.back(); pending.pop_back();
      if (node >= input.size()) throw std::invalid_argument("physical child out of range");
      const Node record = input[node];
      if (finish) {
        mapped[node] = make(record.predicate, mapped[record.left], mapped[record.right]);
        ++visited;
        continue;
      }
      if (seen[node]++) throw std::invalid_argument("physical input has shared parents or a cycle");
      if (record.terminal()) {
        if (record.left != kLeaf || record.right != kLeaf)
          throw std::invalid_argument("physical terminal has children");
        mapped[node] = leaf(record.label);
        ++visited;
      } else {
        if (record.label != -1 || record.left >= input.size() || record.right >= input.size())
          throw std::invalid_argument("invalid physical fork");
        pending.emplace_back(node, true);
        pending.emplace_back(record.right, false);
        pending.emplace_back(record.left, false);
      }
    }
    if (visited != input.size()) throw std::invalid_argument("unreachable physical input nodes");
    checkpoint_root = mapped[root];
    return checkpoint_root;
  }

  // Each accepted rewrite is an exact Boolean identity and must decrease the
  // fully unfolded physical size. Construction sharing is never its objective.
  Id optimize(Id root, std::uint32_t passes = 8, std::uint32_t cofactor_depth = 2,
              std::uint64_t max_states = 100000, const Deadline& deadline = {}) {
    require_id(root);
    if (cofactor_depth > 2 || max_states == 0)
      throw std::invalid_argument("invalid symbolic cofactor limits");
    checkpoint_root = root;
    optimization = OptimizationStats{};
    optimization.nodes_before = optimization.nodes_after = physical_size(root);
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
      if (deadline) deadline();
      const Id boundary = records.size();
      PassStats stats;
      stats.nodes_before = physical_size(root);
      std::vector<Id> rewritten(boundary, kLeaf);
      using CofactorMap = std::unordered_map<Id, std::pair<Id, Id>>;
      std::unordered_map<Predicate, CofactorMap, PredicateHash> cache;

      auto cofactors = [&](Id source, Predicate predicate) -> std::pair<Id, Id> {
        if (stats.cofactor_budget_exhausted) return {kLeaf, kLeaf};
        auto& memo = cache[predicate];
        std::vector<std::pair<Id, bool>> pending{{source, false}};
        while (!pending.empty()) {
          const auto [node, finish] = pending.back(); pending.pop_back();
          if (memo.contains(node)) continue;
          if (stats.cofactor_states >= max_states) {
            stats.cofactor_budget_exhausted = true;
            return {kLeaf, kLeaf};
          }
          poll(deadline, stats.cofactor_states);
          const Node record = records[node];
          if (record.terminal()) {
            memo.emplace(node, std::pair{node, node});
            ++stats.cofactor_states;
          } else if (finish) {
            const auto a = memo.at(record.left), b = memo.at(record.right);
            const auto value = record.predicate == predicate
                ? std::pair{a.first, b.second}
                : std::pair{make(record.predicate, a.first, b.first),
                            make(record.predicate, a.second, b.second)};
            memo.emplace(node, value);
            ++stats.cofactor_states;
          } else {
            pending.emplace_back(node, true);
            pending.emplace_back(record.right, false);
            pending.emplace_back(record.left, false);
          }
        }
        return memo.at(source);
      };

      for (Id node = 0; node < boundary; ++node) {
        poll(deadline, node);
        const Node old = records[node];
        if (old.terminal()) { rewritten[node] = node; continue; }
        const Id current = make(old.predicate, rewritten[old.left], rewritten[old.right]);
        const Node record = records[current];
        if (record.terminal()) { rewritten[node] = current; continue; }
        std::vector<Predicate> alternatives;
        std::vector<Id> frontier{record.left, record.right};
        for (std::uint32_t level = 0; level < std::max(1u, cofactor_depth); ++level) {
          std::vector<Id> following;
          for (Id child : frontier) {
            const Node candidate = records[child];
            if (candidate.terminal()) continue;
            if (alternatives.size() < 6 && std::ranges::find(alternatives, candidate.predicate) == alternatives.end())
              alternatives.push_back(candidate.predicate);
            following.push_back(candidate.left); following.push_back(candidate.right);
          }
          frontier.swap(following);
        }
        Id best = current;
        bool recursive_best = false;
        for (Predicate lifted : alternatives) {
          const Node a = records[record.left], b = records[record.right];
          const Id at = !a.terminal() && a.predicate == lifted ? a.left : record.left;
          const Id af = !a.terminal() && a.predicate == lifted ? a.right : record.left;
          const Id bt = !b.terminal() && b.predicate == lifted ? b.left : record.right;
          const Id bf = !b.terminal() && b.predicate == lifted ? b.right : record.right;
          const Id left = make(record.predicate, at, bt);
          const Id right = make(record.predicate, af, bf);
          const Id rotated = make(lifted, left, right);
          if (physical_size(rotated) < physical_size(best)) { best = rotated; recursive_best = false; }
          if (cofactor_depth) {
            const auto pair = cofactors(current, lifted);
            if (pair.first != kLeaf) {
              const Id expanded = make(lifted, pair.first, pair.second);
              if (physical_size(expanded) < physical_size(best)) { best = expanded; recursive_best = true; }
            }
          }
        }
        stats.accepted_rotations += best != current;
        stats.accepted_cofactors += recursive_best;
        rewritten[node] = best;
      }
      root = rewritten[root];
      checkpoint_root = root;
      stats.nodes_after = physical_size(root);
      if (stats.nodes_after > stats.nodes_before) throw std::logic_error("symbolic optimization increased physical size");
      optimization.passes.push_back(stats);
      optimization.nodes_after = stats.nodes_after;
      root = collect(root, deadline);
      if (stats.nodes_after == stats.nodes_before) {
        optimization.converged = !stats.cofactor_budget_exhausted;
        optimization.stop_reason = stats.cofactor_budget_exhausted
            ? "cofactor state budget reached; completed exact rewrites retained"
            : "no strict physical size decrease";
        break;
      }
    }
    if (deadline) deadline();
    return checkpoint_root = root;
  }

  // Expand every occurrence: the result has one root and unique parents even
  // when records share canonical subprograms during construction.
  [[nodiscard]] std::vector<Node> emit(Id root, std::uint64_t max_nodes = kLeaf,
                                       const Deadline& deadline = {}) const {
    const auto size = physical_size(root);
    if (size > max_nodes || size > std::vector<Node>{}.max_size())
      throw std::length_error("physical class tree exceeds emission capacity");
    std::vector<Node> output;
    output.reserve(static_cast<std::size_t>(size));
    struct Visit { Id source; Id parent; bool right; };
    std::vector<Visit> pending{{root, kLeaf, false}};
    while (!pending.empty()) {
      poll(deadline, output.size());
      const Visit item = pending.back(); pending.pop_back();
      Node record = records[item.source];
      const Id node = output.size();
      if (item.parent != kLeaf) {
        if (item.right) output[item.parent].right = node;
        else output[item.parent].left = node;
      }
      if (!record.terminal()) {
        pending.push_back({record.right, node, true});
        pending.push_back({record.left, node, false});
        record.left = record.right = kLeaf;
      }
      output.push_back(record);
    }
    if (output.size() != size) throw std::logic_error("physical emission count mismatch");
    return output;
  }

 private:
  std::uint64_t outputs_;
  std::vector<std::uint64_t> sizes_;
  std::unordered_map<std::int64_t, Id> leaves_;
  std::unordered_map<BranchKey, Id, BranchHash> branches_;

  static void poll(const Deadline& deadline, std::uint64_t iteration) {
    if (deadline && (iteration & 4095u) == 0) deadline();
  }
  static std::uint64_t checked_size(std::uint64_t left, std::uint64_t right) {
    if (left >= kLeaf - right) throw std::overflow_error("unfolded physical class-tree size overflows uint64");
    return 1 + left + right;
  }
  static void validate_predicate(Predicate predicate) {
    const bool nan_bits = (predicate.cut_bits & 0x7f800000u) == 0x7f800000u
                       && (predicate.cut_bits & 0x007fffffu) != 0;
    if (predicate.feature < 0 || predicate.missing_left > 1 || nan_bits)
      throw std::invalid_argument("invalid serialized class predicate");
  }
  void require_id(Id id) const {
    if (sizes_.size() != records.size()) throw std::logic_error("rebuild symbolic indexes after replacing records");
    if (id >= records.size()) throw std::out_of_range("invalid symbolic record ID");
  }
  Id collect(Id root, const Deadline& deadline) {
    std::vector<std::uint8_t> reached(records.size(), 0);
    std::vector<Id> pending{root};
    std::uint64_t visited = 0;
    while (!pending.empty()) {
      const Id node = pending.back(); pending.pop_back();
      if (reached[node]) continue;
      poll(deadline, visited++);
      reached[node] = 1;
      if (!records[node].terminal()) {
        pending.push_back(records[node].right); pending.push_back(records[node].left);
      }
    }
    SymbolicStore compact(outputs_);
    std::vector<Id> mapping(records.size(), kLeaf);
    for (Id node = 0; node < records.size(); ++node) {
      poll(deadline, node);
      if (!reached[node]) continue;
      const Node record = records[node];
      mapping[node] = record.terminal() ? compact.leaf(record.label)
          : compact.make(record.predicate, mapping[record.left], mapping[record.right]);
    }
    const Id result = mapping[root];
    // Commit collection atomically with respect to cooperative callbacks.
    records.swap(compact.records); sizes_.swap(compact.sizes_);
    leaves_.swap(compact.leaves_); branches_.swap(compact.branches_);
    checkpoint_root = result;
    return result;
  }
};

}  // namespace decision_programs::symbolic
