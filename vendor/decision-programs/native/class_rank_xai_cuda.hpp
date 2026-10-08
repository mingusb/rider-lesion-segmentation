#pragma once
#include "class_rank_xai_export.hpp"
namespace rank_xai_cuda {
using U=std::uint64_t;
struct Options { U maximum_rows=256,maximum_nodes=100000,maximum_device_bytes=512ull<<20; };
struct TraceStep { U node=0; int feature=-1; std::uint32_t threshold_bits=0,stored_cut_bits=0; rank_xai_export::GateKind gate=rank_xai_export::GateKind::leaf; bool left=false; };
struct Answer {
 int hard_class=-1,smooth_class=-1;
 std::array<double,7> smooth_scores{};
 // Class-major derivatives for the ten continuous input features.
 std::array<double,70> smooth_derivatives{};
 std::vector<TraceStep> path;
};
struct Result { bool CUDA_executed=false,raw_and_rank_routes_equal=false;U owned_device_peak_bytes=0;std::vector<Answer> rows; };
// The hard path is independently checked against the stored rank predicates.
// Smooth membership scores and gradients are for the explicit soft equation
// system; they are not native XGBoost probabilities or accuracy guarantees.
Result explain(const rank_xai_export::Model&,const std::vector<std::array<float,54>>&,
               const std::array<float,10>&positive_temperatures,Options={});
}
