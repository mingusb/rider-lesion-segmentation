#pragma once
#include "class_rank_disk_ledger.hpp"
#include <functional>
namespace rank_gpu_dag_inference {
namespace dl=rank_disk_ledger;using Stop=std::function<bool()>;
struct Program {std::vector<dl::Node>nodes;std::vector<dl::Term>terms;std::string binding;};
// Already source-rank encoded cells, not original raw Forest feature values.
// No labels enter inference. Numeric ranks0..2^24, exactly one category bit in
// each group0..3 and4..43. Upstream source-table mapping is a separate operation.
struct Row {std::array<std::int32_t,10>rank{};std::uint64_t categories=17;};
struct Batch {std::vector<Row>rows;std::vector<std::uint64_t>roots;std::string program_binding,binding;};
struct Options {std::uint64_t maximum_model_bytes=512ull*1024*1024,maximum_batch_bytes=256ull*1024*1024;std::uint32_t steps_per_round=128;};
struct Preparation {bool complete=false,CUDA_executed=false;std::string binding,reason;std::uint64_t nodes=0,terms=0;};
struct Result {bool complete=false,CUDA_executed=false;std::string program_binding,binding,reason;std::vector<std::int32_t>labels;std::vector<std::uint64_t>terminals,questions;std::uint64_t rounds=0;double host_ms=0;};
// Resident bounded shared-node program. All numerical/predicate/input/reference
// validation and routing run on CUDA. Every internal child ID must be strictly
// smaller than its parent; shared children and arbitrary per-row roots are legal.
// Axis: FP32 rank/one-hot value < exact FP32 cut. Plane: original ordered separate
// RN FP64 products and additions, starting+0, then < exact FP64 threshold.
// Terms allow every feature0..53, finite signed-zero weights, duplicates and
// arbitrary original order; no sparse-family canonicalization is performed.
// This evaluates persisted words only: no source ensemble, neural model, fitting
// or native class authority is consulted. Loading is not a fresh source proof.
// Inputs/options are value snapshots. No concurrent/reentrant object use.
class DeviceGraph {struct Impl;std::unique_ptr<Impl>p_;public:
 DeviceGraph(Program program,Options options={},int device=0);~DeviceGraph();
 DeviceGraph(const DeviceGraph&)=delete;DeviceGraph&operator=(const DeviceGraph&)=delete;
 Preparation prepare(const Stop&stop={});const Preparation&preparation()const;
 Result evaluate(Batch batch,const Stop&stop={});
};
}
