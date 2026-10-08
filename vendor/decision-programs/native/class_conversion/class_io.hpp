#pragma once
#include "../class_native_model_json.hpp"
#include <bit>
#include <span>
#include <tuple>
#include <map>

#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace class_conversion_native {
using json = nlohmann::json;
namespace fs = std::filesystem;

inline std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    auto count=stream.tellg();
    if(count<0 || static_cast<uint64_t>(count)>4ULL*1024*1024*1024)
        throw std::runtime_error("read extent exceeds4GiB");
    std::string bytes(static_cast<size_t>(count),'\0');
    stream.seekg(0);
    if(count)stream.read(bytes.data(),count);
    if(!stream || stream.peek()!=EOF || stream.bad())throw std::runtime_error("incomplete or changed read " + path.string());
    return bytes;
}

inline std::string sha256(const std::string& bytes) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned length = 0;
    if (!EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length, EVP_sha256(), nullptr))
        throw std::runtime_error("SHA256 failed");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned i = 0; i < length; ++i) out << std::setw(2) << unsigned(digest[i]);
    return out.str();
}

inline void atomic_text(const fs::path& path, const std::string& bytes) {
    fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream.write(bytes.data(), bytes.size());
        stream.flush();
    }
    fs::rename(temporary, path);
}

inline void atomic_json(const fs::path& path, const json& value) {
    atomic_text(path, value.dump(2) + "\n");
}

template<class T> void write_scalar(std::ostream& out, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::endian::native == std::endian::little);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<class T> T read_scalar(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated native checkpoint");
    return value;
}

template<class T> void write_vector(std::ostream& out, const std::vector<T>& values) {
    write_scalar<uint64_t>(out, values.size());
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
}

template<class T> std::vector<T> read_vector(std::istream& in, uint64_t maximum) {
    uint64_t count = read_scalar<uint64_t>(in);
    if (count > maximum || count > std::numeric_limits<size_t>::max() / sizeof(T))
        throw std::runtime_error("native checkpoint vector exceeds its declared bounds");
    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!in) throw std::runtime_error("truncated native checkpoint vector");
    return values;
}

enum class NativeObjective { softprob, softmax };
inline const char* native_objective_name(NativeObjective objective){
    switch(objective){case NativeObjective::softprob:return "multi:softprob";case NativeObjective::softmax:return "multi:softmax";}
    throw std::runtime_error("invalid native objective enum");
}
struct SourceData {
    // Zero means every channel is a native margin. Mixed composition appends
    // structural authority channels, which are never native model outputs.
    int32_t native_margin_outputs = 0;
    uint32_t structural_guard_trees = 0;
    NativeObjective objective = NativeObjective::softprob;
    int32_t features = 0, outputs = 0, depth = 0;
    std::array<int, 3> version{};
    std::string bytes, identity;
    std::vector<int32_t> feature, left, right, roots, channels;
    std::vector<float> cut, value, bias;
    std::vector<uint8_t> missing_left;
    std::vector<int32_t> leaf_offsets{0}, leaf_nodes, ancestors;
    std::vector<uint8_t> ancestor_right;
};

inline SourceData read_source_bytes(std::string bytes) {
    auto decimal = [](const json& input, const char* field) -> uint64_t {
        if (!input.is_string()) throw std::runtime_error(std::string(field) + " must be a decimal string");
        const auto text = input.get<std::string>();
        uint64_t value = 0;
        auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
            throw std::runtime_error(std::string("invalid decimal field: ") + field);
        return value;
    };
    auto integer = [](const json& input, const char* field) -> int32_t {
        if (!input.is_number_integer() ||
            (input.is_number_unsigned() && input.get<uint64_t>() > uint64_t(std::numeric_limits<int32_t>::max())))
            throw std::runtime_error(std::string("invalid integer field: ") + field);
        const auto value = input.get<int64_t>();
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
            throw std::runtime_error(std::string("integer field exceeds int32: ") + field);
        return int32_t(value);
    };
    auto integers = [&](const json& input, const char* field) {
        if (!input.is_array()) throw std::runtime_error(std::string(field) + " must be an integer array");
        std::vector<int32_t> result;
        result.reserve(input.size());
        for (const auto& item : input) result.push_back(integer(item, field));
        return result;
    };
    SourceData out;
    out.bytes = std::move(bytes);
    out.identity = sha256(out.bytes);
    auto document = dp_native_json::parse(out.bytes);
    auto& learner = document.at("learner");
    auto& booster = learner.at("gradient_booster");
    auto& parameters = learner.at("learner_model_param");
    const auto objective = learner.at("objective").at("name").get<std::string>();
    if (booster.at("name") != "gbtree" || (objective != "multi:softprob" && objective != "multi:softmax"))
        throw std::runtime_error("native conversion requires scalar numeric gbtree multi:softprob or multi:softmax");
    out.objective = objective == "multi:softprob" ? NativeObjective::softprob : NativeObjective::softmax;
    const auto features = decimal(parameters.at("num_feature"), "num_feature");
    const auto outputs = decimal(parameters.at("num_class"), "num_class");
    if (features < 1 || features > uint64_t(std::numeric_limits<int32_t>::max()) || outputs < 2 || outputs > 1024)
        throw std::runtime_error("native class contract requires positive features and 2..1024 classes");
    out.features = int32_t(features); out.outputs = int32_t(outputs);
    if (parameters.contains("num_target") && decimal(parameters.at("num_target"), "num_target") != 1)
        throw std::runtime_error("multi-target source is unsupported");
    auto version = integers(document.at("version"), "version");
    if (version != std::vector<int32_t>{3, 4, 1}) throw std::runtime_error("native source must be XGBoost 3.4.1");
    std::copy(version.begin(), version.end(), out.version.begin());
    auto initial = parameters.at("base_score");
    if (initial.is_string()) initial = json::parse(initial.get<std::string>());
    if (!initial.is_array()) initial = json::array({initial});
    out.bias = initial.get<std::vector<float>>();
    if (out.bias.size() == 1) out.bias.resize(out.outputs, out.bias.front());
    if (out.bias.size() != size_t(out.outputs)) throw std::runtime_error("source bias dimension differs");
    for (float x : out.bias) if (!std::isfinite(x)) throw std::runtime_error("nonfinite source bias");
    auto& model = booster.at("model");
    auto& trees = model.at("trees");
    out.channels = integers(model.at("tree_info"), "tree_info");
    if (!trees.is_array() || trees.size() != out.channels.size() || trees.empty()) throw std::runtime_error("invalid source tree count");
    if (decimal(model.at("gbtree_model_param").at("num_trees"), "num_trees") != trees.size())
        throw std::runtime_error("source num_trees differs from tree storage");
    const auto iteration = integers(model.at("iteration_indptr"), "iteration_indptr");
    if (iteration.size() < 2 || iteration.front() != 0 || iteration.back() < 0 ||
        size_t(iteration.back()) != trees.size() || !std::is_sorted(iteration.begin(), iteration.end()))
        throw std::runtime_error("source iteration offsets do not cover the stored trees in order");
    std::vector<std::vector<int32_t>> paths;
    std::vector<std::vector<uint8_t>> sides;
    for (size_t tree_index = 0; tree_index < trees.size(); ++tree_index) {
        const auto& tree = trees[tree_index];
        if (integer(tree.at("id"), "tree id") < 0 || size_t(integer(tree.at("id"), "tree id")) != tree_index)
            throw std::runtime_error("source tree id differs from its stored order");
        auto leaf_vector = decimal(tree.at("tree_param").at("size_leaf_vector"), "size_leaf_vector");
        if (leaf_vector != 0 && leaf_vector != 1) throw std::runtime_error("vector leaves are unsupported");
        if (!tree.at("categories").empty()) throw std::runtime_error("categorical source is unsupported");
        const auto split_types = integers(tree.at("split_type"), "split_type");
        for (int split_type : split_types)
            if (split_type != 0) throw std::runtime_error("categorical source split is unsupported");
        auto left = integers(tree.at("left_children"), "left_children");
        auto right = integers(tree.at("right_children"), "right_children");
        auto feature = integers(tree.at("split_indices"), "split_indices");
        auto cut = tree.at("split_conditions").get<std::vector<float>>();
        const auto& missing = tree.at("default_left");
        const size_t n = left.size();
        if (!n || right.size() != n || feature.size() != n || cut.size() != n || !missing.is_array() ||
            missing.size() != n || split_types.size() != n)
            throw std::runtime_error("source node arrays differ in size");
        if (decimal(tree.at("tree_param").at("num_nodes"), "num_nodes") != n)
            throw std::runtime_error("source num_nodes differs from node storage");
        if (out.feature.size() + n > size_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("source topology exceeds int32 node indexing");
        if (out.channels[tree_index] < 0 || out.channels[tree_index] >= out.outputs)
            throw std::runtime_error("invalid source class channel");
        int32_t offset = int32_t(out.feature.size());
        out.roots.push_back(offset);
        for (size_t i = 0; i < n; ++i) {
            bool terminal = left[i] == -1 && right[i] == -1;
            if (!terminal && (left[i] < 0 || right[i] < 0 || size_t(left[i]) >= n || size_t(right[i]) >= n))
                throw std::runtime_error("source fork requires two valid children");
            if (!terminal && (feature[i] < 0 || feature[i] >= out.features || std::isnan(cut[i])))
                throw std::runtime_error("invalid source predicate");
            if (terminal && !std::isfinite(cut[i])) throw std::runtime_error("nonfinite source leaf response");
            out.feature.push_back(terminal ? -1 : feature[i]);
            out.cut.push_back(terminal ? 0.f : cut[i]);
            out.value.push_back(terminal ? cut[i] : 0.f);
            out.left.push_back(terminal ? -1 : offset + left[i]);
            out.right.push_back(terminal ? -1 : offset + right[i]);
            const int missing_value = missing[i].is_boolean() ? int(missing[i].get<bool>()) : integer(missing[i], "default_left");
            if (missing_value != 0 && missing_value != 1) throw std::runtime_error("default_left must be Boolean or 0/1");
            out.missing_left.push_back(uint8_t(missing_value));
        }
        struct Visit { int32_t node; std::vector<int32_t> path; std::vector<uint8_t> right; };
        std::vector<Visit> pending{{0, {}, {}}};
        std::vector<uint8_t> visited(n, 0);
        size_t reached = 0;
        while (!pending.empty()) {
            Visit current = std::move(pending.back()); pending.pop_back();
            if (current.node < 0 || size_t(current.node) >= n || visited[current.node]++)
                throw std::runtime_error("source topology is cyclic or has shared children");
            ++reached;
            out.depth = std::max(out.depth, int32_t(current.path.size()));
            if (left[current.node] < 0) {
                out.leaf_nodes.push_back(offset + current.node);
                paths.push_back(std::move(current.path)); sides.push_back(std::move(current.right));
            } else {
                current.path.push_back(offset + current.node);
                auto right_path = current.right;
                current.right.push_back(0); right_path.push_back(1);
                pending.push_back({right[current.node], current.path, std::move(right_path)});
                pending.push_back({left[current.node], std::move(current.path), std::move(current.right)});
            }
        }
        if (reached != n) throw std::runtime_error("unreachable source nodes");
        out.leaf_offsets.push_back(int32_t(out.leaf_nodes.size()));
    }
    if (out.depth && out.leaf_nodes.size() > std::numeric_limits<size_t>::max() / size_t(out.depth))
        throw std::runtime_error("source ancestor storage exceeds host address range");
    out.ancestors.assign(out.leaf_nodes.size() * out.depth, -1);
    out.ancestor_right.assign(out.ancestors.size(), 0);
    for (size_t i = 0; i < paths.size(); ++i) {
        std::copy(paths[i].begin(), paths[i].end(), out.ancestors.begin() + i * out.depth);
        std::copy(sides[i].begin(), sides[i].end(), out.ancestor_right.begin() + i * out.depth);
    }
    return out;
}
inline SourceData read_source(const fs::path& path) { return read_source_bytes(read_text(path)); }

// Raw fields retain declared order after the native probability prefix. The
// key preserves cut words (including signed zero) and native missing direction.
struct MappedRawPredicate {
    uint32_t feature;
    float cut;
    bool missing_left;
};
inline std::vector<MappedRawPredicate> mapped_meta_raw_predicates(
        const SourceData& meta, uint32_t probability_columns,
        std::span<const uint32_t> raw_indices, uint32_t raw_features) {
    auto require = [](bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
    };
    require(probability_columns <= uint32_t(INT32_MAX) &&
            raw_indices.size() <= size_t(INT32_MAX) - probability_columns &&
            size_t(meta.features) == probability_columns + raw_indices.size(),
            "composed meta inputs must equal native probability prefix plus mapped raw columns");
    std::vector<uint32_t> ordered(raw_indices.begin(), raw_indices.end());
    std::sort(ordered.begin(), ordered.end());
    require(std::adjacent_find(ordered.begin(), ordered.end()) == ordered.end() &&
            (ordered.empty() || ordered.back() < raw_features),
            "composed raw feature mapping must be unique and in range");
    std::map<std::tuple<uint32_t,uint32_t,bool>, MappedRawPredicate> predicates;
    for (size_t n = 0; n < meta.feature.size(); ++n) {
        const auto f = meta.feature[n];
        if (f < 0 || uint32_t(f) < probability_columns) continue;
        require(size_t(f) - probability_columns < raw_indices.size(), "mapped raw predicate index exceeds meta inputs");
        const auto raw = raw_indices[size_t(f) - probability_columns];
        const bool missing = meta.missing_left[n] != 0;
        predicates.emplace(std::tuple{raw, std::bit_cast<uint32_t>(meta.cut[n]), missing},
                           MappedRawPredicate{raw, meta.cut[n], missing});
    }
    std::vector<MappedRawPredicate> result;
    result.reserve(predicates.size());
    for (const auto& [key, predicate] : predicates) result.push_back(predicate);
    return result;
}

// Same adaptive engine, with independent native base margins. For mixed meta
// inputs, each unresolved raw predicate additionally has a structural guard.
// A guard contributes 0 (left) or 2^bit (right) in a dedicated zero-bias channel.
// Packing 24 distinct bits per channel bounds every partial sum by 2^24-1, so
// all RN32 additions are exact in any order. Terminal authority words therefore
// distinguish every raw branch signature even when all native margins match.
// These channels are never classes, probabilities, or native meta inputs.
inline constexpr uint32_t structural_guard_bits = std::numeric_limits<float>::digits;
static_assert(structural_guard_bits == 24, "structural guards require IEEE binary32 precision");
inline SourceData compose_source_forests(const std::vector<SourceData>& bases,
                                        const SourceData& meta,
                                        std::span<const uint32_t> raw_indices = {}) {
    auto require = [](bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
    };
    require(!bases.empty(), "composed conversion requires base models");
    require(meta.objective == NativeObjective::softmax,
            "composed meta source must use native multi:softmax");
    SourceData out;
    out.features = bases.front().features;
    out.objective = NativeObjective::softmax;
    out.version = meta.version;
    json identities = json::array();
    size_t leaves = 0;
    for (const auto& base : bases) {
        require(base.features == out.features && base.outputs >= 2 &&
                    base.objective == NativeObjective::softmax && !base.structural_guard_trees,
                "composed base feature/objective shape differs");
        require(base.version == meta.version, "composed native versions differ");
        require(out.outputs <= 1024 - base.outputs,
                "composed native score channels exceed supported capacity");
        require(base.feature.size() <= size_t(INT32_MAX) - out.feature.size() &&
                    base.roots.size() <= size_t(INT32_MAX) - out.roots.size() &&
                    base.leaf_nodes.size() <= size_t(INT32_MAX) - leaves,
                "composed source index capacity exceeded");
        const int32_t node_offset = int32_t(out.feature.size());
        const int32_t channel_offset = out.outputs;
        for (auto root : base.roots) out.roots.push_back(node_offset + root);
        for (auto channel : base.channels) out.channels.push_back(channel_offset + channel);
        out.feature.insert(out.feature.end(), base.feature.begin(), base.feature.end());
        out.cut.insert(out.cut.end(), base.cut.begin(), base.cut.end());
        out.value.insert(out.value.end(), base.value.begin(), base.value.end());
        out.missing_left.insert(out.missing_left.end(), base.missing_left.begin(), base.missing_left.end());
        out.bias.insert(out.bias.end(), base.bias.begin(), base.bias.end());
        for (auto node : base.left) out.left.push_back(node < 0 ? -1 : node_offset + node);
        for (auto node : base.right) out.right.push_back(node < 0 ? -1 : node_offset + node);
        for (auto node : base.leaf_nodes) out.leaf_nodes.push_back(node_offset + node);
        for (size_t i = 1; i < base.leaf_offsets.size(); ++i)
            out.leaf_offsets.push_back(int32_t(leaves) + base.leaf_offsets[i]);
        leaves += base.leaf_nodes.size();
        out.outputs += base.outputs;
        out.depth = std::max(out.depth, base.depth);
        identities.push_back({{"source_sha256", base.identity}, {"score_channels", base.outputs}});
    }
    const auto native_columns = uint32_t(out.outputs);
    const auto native_trees = out.roots.size(), native_nodes = out.feature.size();
    const auto predicates = mapped_meta_raw_predicates(meta, native_columns, raw_indices, out.features);
    const size_t guard_channels = predicates.size()/structural_guard_bits + (predicates.size()%structural_guard_bits != 0);
    require(guard_channels <= size_t(1024 - out.outputs),
            "composed total authority channels exceed supported capacity");
    require(predicates.size() <= (size_t(INT32_MAX) - out.feature.size())/3 &&
            predicates.size() <= size_t(INT32_MAX) - out.roots.size() &&
            predicates.size() <= (size_t(INT32_MAX) - leaves)/2,
            "composed structural guard index capacity exceeded");
    const size_t native_leaves = leaves;
    json guard_predicates = json::array();
    for (size_t i = 0; i < predicates.size(); ++i) {
        const auto& predicate = predicates[i];
        const auto root = int32_t(out.feature.size());
        out.roots.push_back(root);
        out.channels.push_back(int32_t(native_columns + i/structural_guard_bits));
        out.feature.insert(out.feature.end(), {int32_t(predicate.feature), -1, -1});
        out.cut.insert(out.cut.end(), {predicate.cut, 0.f, 0.f});
        out.value.insert(out.value.end(), {0.f, 0.f, std::ldexp(1.f, int(i%structural_guard_bits))});
        out.left.insert(out.left.end(), {root+1, -1, -1});
        out.right.insert(out.right.end(), {root+2, -1, -1});
        out.missing_left.insert(out.missing_left.end(), {uint8_t(predicate.missing_left), 0, 0});
        out.leaf_nodes.insert(out.leaf_nodes.end(), {root+1, root+2});
        leaves += 2; out.leaf_offsets.push_back(int32_t(leaves));
        guard_predicates.push_back({{"feature", predicate.feature},
            {"threshold_FP32_bits", std::bit_cast<uint32_t>(predicate.cut)},
            {"missing_left", predicate.missing_left}, {"authority_channel", native_columns+i/structural_guard_bits},
            {"bit", i%structural_guard_bits}});
    }
    out.outputs += int32_t(guard_channels);
    out.bias.resize(out.outputs, 0.f);
    out.structural_guard_trees = uint32_t(predicates.size());
    if (!raw_indices.empty()) out.native_margin_outputs = int32_t(native_columns);
    if (!predicates.empty()) out.depth = std::max(out.depth, int32_t(1));
    require(!out.depth || leaves <= std::numeric_limits<size_t>::max() / size_t(out.depth),
            "composed source ancestor extent overflow");
    out.ancestors.assign(leaves * out.depth, -1);
    out.ancestor_right.assign(out.ancestors.size(), 0);
    size_t leaf_offset = 0;
    int32_t node_offset = 0;
    for (const auto& base : bases) {
        for (size_t leaf = 0; leaf < base.leaf_nodes.size(); ++leaf)
            for (int32_t depth = 0; depth < base.depth; ++depth) {
                const size_t from = leaf * base.depth + depth;
                const size_t to = (leaf_offset + leaf) * out.depth + depth;
                if (base.ancestors[from] >= 0) out.ancestors[to] = node_offset + base.ancestors[from];
                out.ancestor_right[to] = base.ancestor_right[from];
            }
        leaf_offset += base.leaf_nodes.size();
        node_offset += int32_t(base.feature.size());
    }
    for (size_t i = 0; i < predicates.size(); ++i) {
        const size_t left = (native_leaves + 2*i) * out.depth, right = left + out.depth;
        out.ancestors[left] = out.ancestors[right] = int32_t(native_nodes + 3*i);
        out.ancestor_right[right] = 1;
    }
    json descriptor = {{"format", "native-probability-stack-conversion-source-1"},
        {"bases", identities}, {"meta_sha256", meta.identity}, {"features", out.features},
        {"score_channels", out.outputs}, {"classes", meta.outputs},
        {"response_order", "teacher-major native full-round softprob FP32 columns"}};
    // Preserve the exact legacy descriptor identity when no raw fields exist.
    if (!raw_indices.empty()) {
        descriptor["meta_input_mapping"] = {{"version", 1},
            {"raw_feature_indices", std::vector<uint32_t>(raw_indices.begin(), raw_indices.end())}};
        descriptor["score_channels"] = native_columns;
        descriptor["native_probability_columns"] = native_columns;
        descriptor["native_meta_input_columns"] = meta.features;
        descriptor["native_margin_audit_columns"] = native_columns;
        descriptor["authority_channels"] = out.outputs;
        descriptor["native_source_trees"] = native_trees;
        descriptor["native_source_nodes"] = native_nodes;
        descriptor["structural_guards"] = {{"policy", "packed_raw_predicate_bits24_v1"},
            {"trees", predicates.size()}, {"nodes", 3*predicates.size()}, {"channels", guard_channels},
            {"predicates", guard_predicates}, {"native_model_inputs", false},
            {"proof", "Distinct right-branch bits have exact nonnegative RN32 sums at most 2^24-1 from zero bias; terminal cache words preserve each mapped raw predicate outcome"}};
    }
    out.bytes = descriptor.dump();
    out.identity = sha256(out.bytes);
    return out;
}
}  // namespace class_conversion_native
