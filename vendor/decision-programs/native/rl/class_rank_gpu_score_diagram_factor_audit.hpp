#pragma once
#include "class_rank_gpu_score_diagram.hpp"
namespace rank_gpu_score_diagram {
struct FactorAuditOptions {
    U maximum_factor_node_pairs=16777216;
    U maximum_device_bytes=128ull*1024*1024;
};
struct FactorAudit {
    bool complete=false, CUDA_executed=false;
    U factors=0, nodes=0, possible_factor_node_pairs=0, visited_factor_node_pairs=0;
    U intersecting_arcs=0, terminal_score_word_checks=0, checked_nodes=0;
    U owned_device_peak_bytes=0;
    std::string reason;
};
// Independent universal congruence check against the SAME qualified complete
// factor partition passed to construct. Binding strings do not authenticate an
// imported archive. CUDA independently checks node arc coverage and strictly
// increasing dimensions, then propagates factor-box-compatible reachability.
// Because a product factor's suffix is independent of earlier coordinates,
// (factor ID, node ID) is a sufficient state after a path reaches that node.
// Every reachable terminal must match the factor's exact score word. No joint
// quotient enumeration or sampling; no native class/probability authority.
FactorAudit audit_factors(const Input&,const Result&,FactorAuditOptions={},
                         const Stop& = {},int device=0);
}
