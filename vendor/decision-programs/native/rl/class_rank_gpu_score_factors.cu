#include "class_rank_gpu_score_factors.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace rank_gpu_score_factors { namespace {
void need(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
void check(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
U multiply(U a, U b) {
    need(!b || a <= UINT64_MAX / b, "score-factor byte arithmetic overflow");
    return a * b;
}
struct Budget {
    U maximum, used = 0, peak = 0;
    void add(U size) {
        need(size <= maximum - used, "score-factor device budget exceeded");
        used += size; peak = std::max(peak, used);
    }
};
template<class T> struct Buffer {
    T* data = nullptr;
    U count, bytes;
    Budget& budget;
    Buffer(Budget& owner, U n) : count(n), bytes(multiply(n, sizeof(T))), budget(owner) {
        budget.add(bytes);
        if (bytes) {
            auto status = cudaMalloc(reinterpret_cast<void**>(&data), bytes);
            if (status != cudaSuccess) { budget.used -= bytes; check(status); }
            // Includes padding: every byte subsequently copied is initialized.
            auto cleared = cudaMemset(data, 0, bytes);
            if (cleared != cudaSuccess) {
                cudaFree(data); data = nullptr; budget.used -= bytes; check(cleared);
            }
        }
    }
    ~Buffer() { if (data) cudaFree(data); budget.used -= bytes; }
    Buffer(const Buffer&) = delete;
    void put(const T* input) {
        if (bytes) check(cudaMemcpy(data, input, bytes, cudaMemcpyHostToDevice));
    }
    std::vector<T> get(U n) const {
        need(n <= count, "score-factor download range");
        std::vector<T> output(n);
        if (n) check(cudaMemcpy(output.data(), data, n * sizeof(T), cudaMemcpyDeviceToHost));
        return output;
    }
};
struct Metadata {
    U cartesian[8], offsets[8], combinations, compatible;
    int trees[7][8], counts[7], bad, capped;
};
__device__ bool valid_box(const b::Box& box) {
    if ((box.allowed & ~(b::wilderness | b::soil)) ||
        !(box.allowed & b::wilderness) || !(box.allowed & b::soil)) return false;
    for (int f = 0; f < 10; ++f)
        if (box.lo[f] < 0 || box.hi[f] > 16777216 || box.lo[f] > box.hi[f]) return false;
    return true;
}
__device__ bool nonempty(const b::Box& box) {
    if (!(box.allowed & b::wilderness) || !(box.allowed & b::soil)) return false;
    for (int f = 0; f < 10; ++f) if (box.lo[f] > box.hi[f]) return false;
    return true;
}
__global__ void validate(const b::Leaf* leaves, int nl, const int* offsets,
                         const int* channels, int nt, const float* bias, Metadata* meta) {
    U i = U(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < 7 && !isfinite(bias[i])) atomicExch(&meta->bad, 1);
    if (i <= U(nt) && (offsets[i] < 0 || offsets[i] > nl ||
        (!i && offsets[i] != 0) || (i == U(nt) && offsets[i] != nl))) atomicExch(&meta->bad, 1);
    if (i < U(nt) && (channels[i] < 0 || channels[i] >= 7 || offsets[i] >= offsets[i+1]))
        atomicExch(&meta->bad, 1);
    if (i < U(nl) && (!valid_box(leaves[i].box) || !isfinite(leaves[i].value)))
        atomicExch(&meta->bad, 1);
}
__global__ void layout(const int* offsets, const int* channels, int nt,
                       unsigned max_trees, U max_combinations, Metadata* meta) {
    if (blockIdx.x || threadIdx.x) return;
    for (int t = 0; t < nt; ++t) {
        int c = channels[t], count = meta->counts[c];
        if (count >= int(max_trees)) { meta->capped = 1; return; }
        meta->trees[c][count] = t; ++meta->counts[c];
    }
    U total = 0;
    for (int c = 0; c < 7; ++c) {
        // Deliberately restricted to complete positive-round 7-class teachers.
        if (!meta->counts[c]) { meta->bad = 1; return; }
        meta->cartesian[c] = total;
        U combinations = 1;
        for (int j = 0; j < meta->counts[c]; ++j) {
            int t = meta->trees[c][j]; U size = U(offsets[t+1] - offsets[t]);
            if (combinations > max_combinations / size) { meta->capped = 2; return; }
            combinations *= size;
        }
        if (combinations > max_combinations - total) { meta->capped = 2; return; }
        total += combinations;
    }
    meta->cartesian[7] = total; meta->combinations = total;
}
__global__ void candidates(const b::Leaf* leaves, const int* offsets, const float* bias,
                           Metadata* meta, Factor* output, unsigned char* keep) {
    U i = U(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= meta->combinations) return;
    int c = 0; while (i >= meta->cartesian[c+1]) ++c;
    U combination = i - meta->cartesian[c], remainder = combination;
    int ids[8];
    for (int j = meta->counts[c] - 1; j >= 0; --j) {
        int t = meta->trees[c][j]; U size = U(offsets[t+1] - offsets[t]);
        ids[j] = offsets[t] + int(remainder % size); remainder /= size;
    }
    b::Box box = leaves[ids[0]].box;
    for (int j = 1; j < meta->counts[c]; ++j) {
        const auto& next = leaves[ids[j]].box; box.allowed &= next.allowed;
        for (int f = 0; f < 10; ++f) {
            box.lo[f] = max(box.lo[f], next.lo[f]);
            box.hi[f] = min(box.hi[f], next.hi[f]);
        }
    }
    if (!nonempty(box)) return;
    float score = bias[c];
    for (int j = 0; j < meta->counts[c]; ++j) {
        score = __fadd_rn(score, leaves[ids[j]].value);
        if (!isfinite(score)) { atomicExch(&meta->bad, 2); return; }
    }
    output[i].box = box; output[i].score_bits = __float_as_uint(score);
    output[i].channel = unsigned(c); output[i].combination = combination; keep[i] = 1;
}
__global__ void compact_layout(const unsigned char* keep, U* destinations, Metadata* meta) {
    if (blockIdx.x || threadIdx.x) return;
    U count = 0;
    for (int c = 0; c < 7; ++c) {
        meta->offsets[c] = count;
        for (U i = meta->cartesian[c]; i < meta->cartesian[c+1]; ++i) {
            destinations[i] = count; count += keep[i];
        }
        if (count == meta->offsets[c]) { meta->bad = 3; return; }
    }
    meta->offsets[7] = count; meta->compatible = count;
}
__global__ void compact(const Factor* input, const unsigned char* keep,
                        const U* destinations, Factor* output, U n) {
    U i = U(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && keep[i]) {
        const auto& source = input[i]; auto& target = output[destinations[i]];
        target.box = source.box; target.score_bits = source.score_bits;
        target.channel = source.channel; target.combination = source.combination;
    }
}
unsigned blocks(U n) {
    need(n && n <= U(INT_MAX) * 128, "score-factor launch extent exceeded");
    return unsigned((n+127)/128);
}
}

Result prepare(Source input, Options options, const Stop& stop, int device) {
    Result result; result.source_binding = input.binding;
    auto halt = [&](const char* why) {
        if (stop && stop()) { result.reason = why; return true; } return false;
    };
    need(!input.binding.empty(), "score-factor source binding missing");
    need(options.maximum_trees_per_class >= 1 && options.maximum_trees_per_class <= 8 &&
         options.maximum_combinations, "score-factor options invalid");
    const auto& source = input.value; U nt = source.channels.size(), nl = source.leaves.size();
    need(nt && nt < INT_MAX && nl <= INT_MAX && source.offsets.size() == nt+1,
         "score-factor source shape invalid");
    if (halt("cancelled_before_preparation")) return result;
    // No arithmetic/source validation takes place on the host.
    U base = multiply(nl, sizeof(b::Leaf)) + multiply(nt*2+1, sizeof(int)) +
             sizeof(float)*7 + sizeof(Metadata);
    if (base > options.maximum_device_bytes) { result.reason = "source_device_budget"; return result; }
    check(cudaSetDevice(device)); Budget budget{options.maximum_device_bytes};
    Buffer<b::Leaf> leaves(budget, nl); Buffer<int> offsets(budget, nt+1), channels(budget, nt);
    Buffer<float> bias(budget, 7); Buffer<Metadata> metadata(budget, 1);
    // Source buffers exist on every later incomplete return, including caps.
    result.owned_device_peak_bytes = budget.peak;
    leaves.put(source.leaves.data()); offsets.put(source.offsets.data());
    channels.put(source.channels.data()); bias.put(source.bias.data());
    validate<<<blocks(std::max<U>({nl, nt+1, 7})),128>>>(leaves.data, int(nl), offsets.data,
        channels.data, int(nt), bias.data, metadata.data);
    check(cudaGetLastError()); result.CUDA_executed = true;
    need(!metadata.get(1)[0].bad, "CUDA score-factor source layout/rank/value invalid");
    if (halt("cancelled_after_source_validation")) return result;
    layout<<<1,1>>>(offsets.data, channels.data, int(nt), options.maximum_trees_per_class,
                   options.maximum_combinations, metadata.data);
    check(cudaGetLastError()); auto meta = metadata.get(1)[0];
    need(!meta.bad, "CUDA score-factor class missing");
    if (meta.capped) { result.reason = meta.capped == 1 ? "trees_per_class_cap" : "combination_cap"; return result; }
    result.combinations = meta.combinations;
    for (int c = 0; c < 8; ++c) result.cartesian_offsets[c] = meta.cartesian[c];
    // Both full candidate and worst-case compact output are counted before work.
    U extra = multiply(meta.combinations, sizeof(Factor)*2 + sizeof(U) + 1);
    if (extra > options.maximum_device_bytes-base) { result.reason = "factor_device_budget"; return result; }
    if (halt("cancelled_before_factor_construction")) return result;
    Buffer<Factor> staged(budget, meta.combinations), collected(budget, meta.combinations);
    Buffer<unsigned char> keep(budget, meta.combinations); Buffer<U> destinations(budget, meta.combinations);
    result.owned_device_peak_bytes = budget.peak;
    candidates<<<blocks(meta.combinations),128>>>(leaves.data, offsets.data, bias.data,
                                                 metadata.data, staged.data, keep.data);
    check(cudaGetLastError()); need(!metadata.get(1)[0].bad, "CUDA score-factor nonfinite ordered score");
    compact_layout<<<1,1>>>(keep.data, destinations.data, metadata.data);
    check(cudaGetLastError()); meta = metadata.get(1)[0];
    need(!meta.bad, "CUDA score-factor empty class intersection partition");
    if (halt("cancelled_before_factor_compaction")) return result;
    compact<<<blocks(meta.combinations),128>>>(staged.data, keep.data, destinations.data,
                                             collected.data, meta.combinations);
    check(cudaGetLastError()); auto factors = collected.get(meta.compatible);
    if (halt("cancelled_before_factor_publication")) return result;
    result.compatible = meta.compatible;
    for (int c = 0; c < 8; ++c) result.factor_offsets[c] = meta.offsets[c];
    result.factors = std::move(factors); result.complete = true;
    result.reason = "CUDA_complete_class_score_factors_no_class_authority";
    return result;
}
}
