#include "class_symbolic.hpp"

#include <cassert>
#include <iostream>
#include <set>

using namespace decision_programs::symbolic;

static std::int64_t evaluate(const std::vector<Node>& tree, std::uint32_t assignment) {
  Id node = 0;
  while (!tree[node].terminal()) {
    // This is a Boolean symbolic truth table, never FP32 observations or
    // numerical tree predictions. Each test feature names one Boolean atom.
    const bool truth = (assignment & (1u << tree[node].predicate.feature)) != 0;
    node = truth ? tree[node].left : tree[node].right;
  }
  return tree[node].label;
}

static void equal_truth(const std::vector<Node>& a, const std::vector<Node>& b, unsigned variables) {
  for (std::uint32_t value = 0; value < (1u << variables); ++value)
    assert(evaluate(a, value) == evaluate(b, value));
}

static void unique_parents(const std::vector<Node>& nodes) {
  std::vector<unsigned> incoming(nodes.size(), 0);
  for (const auto& node : nodes) {
    if (node.terminal()) {
      assert(node.left == kLeaf && node.right == kLeaf);
    } else {
      assert(node.left < nodes.size() && node.right < nodes.size());
      ++incoming[node.left]; ++incoming[node.right];
    }
  }
  assert(incoming[0] == 0);
  for (Id node = 1; node < incoming.size(); ++node) assert(incoming[node] == 1);
}

static Id hidden_conjunction(SymbolicStore& store) {
  const Predicate a{0, 0, 0}, b{1, 0, 1}, g{2, 0, 1};
  const Id zero = store.leaf(0), one = store.leaf(1);
  const Id ab = store.make(a, store.make(b, one, zero), zero);
  const Id ba = store.make(b, store.make(a, one, zero), zero);
  return store.make(g, ab, ba);
}

int main() {
  unsigned checks = 0;
  {
    SymbolicStore store(3);
    const Predicate p{0, 0, 1}, q{1, 0, 0};
    Id root = store.make(p, store.make(q, store.leaf(0), store.leaf(1)),
                           store.make(q, store.leaf(2), store.leaf(1)));
    const auto before = store.emit(root);
    assert(before.size() == 7);
    root = store.optimize(root, 8, 0, 10000);
    const auto after = store.emit(root);
    assert(after.size() == 5);
    equal_truth(before, after, 2);
    unique_parents(after);
    ++checks;
  }
  {
    SymbolicStore immediate(2);
    Id root = hidden_conjunction(immediate);
    const auto before = immediate.emit(root);
    assert(before.size() == 11);
    root = immediate.optimize(root, 8, 0, 10000);
    assert(immediate.physical_size(root) == 11);
    SymbolicStore deeper(2);
    root = deeper.canonicalize(before);
    root = deeper.optimize(root, 8, 2, 10000);
    const auto after = deeper.emit(root);
    assert(after.size() == 5);
    equal_truth(before, after, 3);
    unique_parents(after);
    ++checks;
  }
  {
    SymbolicStore store(2);
    const Predicate p{0, 0, 1}, q{1, 0, 0};
    const Id u = store.leaf(100), v = store.leaf(101);
    assert(u != v);
    Id root = store.make(p, store.make(q, u, store.leaf(0)),
                           store.make(q, v, store.leaf(0)));
    const auto before = store.emit(root);
    root = store.optimize(root, 8, 2, 10000);
    const auto after = store.emit(root);
    equal_truth(before, after, 2);
    std::set<std::int64_t> opaque;
    for (const auto& node : after) if (node.label >= 2) opaque.insert(node.label);
    assert((opaque == std::set<std::int64_t>{100, 101}));
    unique_parents(after);
    ++checks;
  }
  {
    SymbolicStore store(2);
    const Predicate p{0, 0, 1};
    const Id a = store.make(p, store.leaf(0), store.leaf(1));
    assert(store.make(p, a, store.leaf(1)) == a);
    assert(store.make(p, store.leaf(0), a) == a);
    assert(store.make(p, a, a) == a);
    assert(store.make(Predicate{0, 0, 0}, store.leaf(0), store.leaf(1)) != a);
    assert(store.make(Predicate{0, 0x80000000u, 1}, store.leaf(0), store.leaf(1)) != a);
    SymbolicStore restored(2);
    restored.records = store.records;
    restored.rebuild_index();
    assert(restored.physical_size(a) == store.physical_size(a));
    equal_truth(store.emit(a), restored.emit(a), 1);
    ++checks;
  }
  {
    SymbolicStore store(2);
    Id root = hidden_conjunction(store);
    const auto before = store.emit(root);
    root = store.optimize(root, 8, 2, 1);
    equal_truth(before, store.emit(root), 3);
    assert(!store.optimization.converged);
    assert(store.optimization.passes.back().cofactor_budget_exhausted);
    ++checks;
  }
  {
    SymbolicStore store(2);
    Id root = hidden_conjunction(store);
    const auto before = store.emit(root);
    unsigned calls = 0;
    bool stopped = false;
    try {
      store.optimize(root, 8, 2, 10000, [&] {
        if (++calls == 4) throw std::runtime_error("test deadline");
      });
    } catch (const std::runtime_error&) { stopped = true; }
    assert(stopped);
    equal_truth(before, store.emit(store.checkpoint_root), 3);
    unique_parents(store.emit(store.checkpoint_root));
    ++checks;
  }
  {
    SymbolicStore store(2);
    Id root = store.leaf(1);
    for (std::int32_t feature = 0; feature < 10000; ++feature)
      root = store.make(Predicate{feature, 0, 0}, root, store.leaf(0));
    auto physical = store.emit(root);
    assert(physical.size() == 20001);
    unique_parents(physical);
    SymbolicStore imported(2);
    const Id restored = imported.canonicalize(physical);
    assert(imported.physical_size(restored) == physical.size());
    ++checks;
  }
  {
    SymbolicStore store(2);
    const Predicate p{0, 0, 1};
    std::vector<Node> shared{{p, 1, 1, -1}, {Predicate{}, kLeaf, kLeaf, 0}};
    bool rejected = false;
    try { store.canonicalize(shared); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    const Id root = store.make(p, store.leaf(0), store.leaf(1));
    rejected = false;
    try { (void)store.emit(root, 2); } catch (const std::length_error&) { rejected = true; }
    assert(rejected);
    ++checks;
  }
  {
    SymbolicStore store(2);
    const Predicate p{0, 0, 1}, q{1, 0, 0};
    const Id shared = store.make(q, store.leaf(0), store.leaf(1));
    const Id root = store.make(p, shared, store.make(Predicate{2, 0, 1}, shared, store.leaf(0)));
    const auto statistics = store.deployment_stats(root, 3);
    assert(statistics.physical_nodes == store.emit(root).size());
    assert(statistics.physical_nodes == 9 && statistics.physical_leaves == 5);
    assert(statistics.max_depth == 3 && statistics.reachable_canonical_records == 5);
    // Unreachable opaque analysis tokens do not invalidate a completed root.
    const Id unfinished = store.leaf(100);
    assert(store.deployment_stats(root, 3).physical_nodes == 9);
    bool rejected = false;
    try { (void)store.deployment_stats(unfinished, 3); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { (void)store.deployment_stats(root, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    ++checks;
  }
  {
    // Exercise every cooperative interruption point, including collection
    // after a completed improving pass has renumbered canonical records.
    SymbolicStore reference(2);
    const Id root = hidden_conjunction(reference);
    const auto before = reference.emit(root);
    unsigned total_calls = 0;
    (void)reference.optimize(root, 8, 2, 10000, [&] { ++total_calls; });
    for (unsigned stop = 1; stop <= total_calls; ++stop) {
      SymbolicStore store(2);
      const Id original = hidden_conjunction(store);
      unsigned calls = 0;
      bool stopped = false;
      try {
        (void)store.optimize(original, 8, 2, 10000, [&] {
          if (++calls == stop) throw std::runtime_error("test every deadline boundary");
        });
      } catch (const std::runtime_error&) { stopped = true; }
      assert(stopped);
      const auto incumbent = store.emit(store.checkpoint_root);
      equal_truth(before, incumbent, 3);
      unique_parents(incumbent);
      SymbolicStore restored(2);
      restored.records = store.records;
      restored.rebuild_index();
      equal_truth(before, restored.emit(store.checkpoint_root), 3);
      assert(restored.deployment_stats(store.checkpoint_root, 3).physical_nodes <= before.size());
    }
    assert(total_calls > 8);
    ++checks;
  }
  std::cout << "{\"passed\":true,\"symbolic_check_groups\":" << checks
            << ",\"floating_point_evaluations\":0,\"cuda_executions\":0}\n";
}
