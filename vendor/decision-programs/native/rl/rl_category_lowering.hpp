#pragma once
#include "class_rank_gpu_class_apply.hpp"

namespace rl_category_lowering {
namespace ca = rank_gpu_class_apply;
using U = ca::U;

// Borrowed CUDA device storage. First4 entries permute bits0..3; the next40
// permute bits4..43. This is root-to-fallback preference within each group.
// Lifetime/read-only storage must cover the entire synchronized lower call.
// A null pointer delegates the qualified default lowering without changes.
struct DeviceOrder {
  const unsigned* bits = nullptr;
  U policy_version = 0;
};

struct Trace {
  U source_node = ca::none, fallback_arc = ca::none;
  U selected_mask = 0, policy_version = 0;
  unsigned dimension = 12, count = 0;
  unsigned test_order[40]{};  // Actual root-to-fallback tests; unused entries=UINT_MAX.
};
static_assert(sizeof(Trace) == 200);

struct Result {
  ca::Lowered lowered;
  bool order_injected = false, trace_audited = false;
  U policy_version = 0;
  std::array<unsigned,44> selected_order{};
  std::vector<Trace> traces;  // One initialized record per source node.
};

// Pure size/overflow metadata. Includes lowering arrays, snapshot order and trace.
U planned_device_bytes(U nodes, U arcs, U drafts);
Result lower_binary(const ca::Result&, DeviceOrder = {},
                    U maximum_drafts = 1048576,
                    U maximum_device_bytes = 128ull*1024*1024,
                    const ca::Stop& = {}, int device = 0);
namespace testing {
struct Report {
  bool passed=false,CUDA_executed=false,native_class_authority=false;
  U assertions=0,rejections=0,stop_boundaries=0,class_checks=0,ordered_variants=0;
  U default_word_checks=0,default_byte_checks=0;
};
Report gpu_checks(int device=0);
}
}
