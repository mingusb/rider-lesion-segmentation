#pragma once
#include "class_rank_disk_ledger.hpp"
#include <functional>
namespace rank_gpu_dag_collect {
namespace dl=rank_disk_ledger;
using U=std::uint64_t;using Stop=std::function<bool()>;
// Roots are an explicit caller-owned retention set: model/task roots plus any
// desired live proof, guard, reader, or checkpoint pins. An empty set collects
// everything. This routine cannot discover missing external references.
struct Input {std::vector<dl::Node>nodes;std::vector<dl::Term>terms;std::vector<U>roots;std::string binding;};
struct Options {U maximum_nodes=1048576,maximum_terms=4194304,maximum_roots=1048576,maximum_device_bytes=1024ull*1024*1024;};
struct Work {U stored_nodes_before=0,stored_terms_before=0,retained_nodes=0,retained_terms=0,frontier_rounds=0,visited_edges=0;};
struct Result {
 bool complete=false,CUDA_executed=false,structural_verified=false;
 std::string binding,input_sha256,output_sha256,reason;
 std::vector<dl::Node>nodes;std::vector<dl::Term>terms;std::vector<U>roots;
 // Dense old->new maps; UINT64_MAX means unreachable/removed. Old input words
 // and IDs are never modified. Returned IDs belong to a NEW namespace.
 std::vector<U>node_map,term_map;Work work;
};
// CUDA validates the complete input DAG, marks BOTH branches transitively,
// copies exactly the reachable nodes/terms, and audits closure, minimality,
// exact predicate words, term order and all remapped references. Arithmetic is
// never reevaluated. No source-equivalence or complete-root authority is minted.
// Cancellation/resource limits publish no compact payload or partial maps.
// Bounded resident implementation; not paging or live in-place reclamation.
Result collect(Input,Options={},const Stop& = {},int device=0);
}
