#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace class_tree_adapter {

// Storage/structure transport only. No predictions, floating-point arithmetic,
// pruning, interning or source-model calls run on the CPU.
struct Limits {
  std::uint32_t maximum_nodes = 1000000;
};
struct Result {
  std::uint32_t features, classes, nodes, root;
  std::uint64_t maximum_depth, leaves, branches;
  std::string original_sha256, canonical_sha256, canonical_bytes;
  // Bijection, not a deployment dependency. Original bytes retain inactive leaf
  // fields that the canonical runtime intentionally omits.
  std::vector<std::uint32_t> original_to_canonical;
};

// Completed, unique-parent CLSTREE1; finite FP32 axis cuts and per-split NaN
// defaults. Dynamic F/K use representable signed feature/U32 class fields;
// allocation is bounded independently by the caller's explicit node capacity.
// The source identity in CLSGDAG1 is the exact original CLSTREE1 SHA256.
Result adapt(const std::string& original_bytes,
             const std::string& expected_original_sha256, Limits = {});

// Checks the full graph isomorphism, then reconstructs the exact original wire
// bytes, including inactive leaf metadata. Throws on altered bytes or mapping.
std::string inverse(const Result&, const std::string& original_bytes);

}  // namespace class_tree_adapter
