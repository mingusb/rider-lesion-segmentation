#pragma once
#include "../class_native_model_json.hpp"

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

namespace dpnative {
using json = nlohmann::json;
namespace fs = std::filesystem;

inline std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
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

struct SourceData {
    int32_t features = 0, outputs = 0, depth = 0;
    std::array<int, 3> version{};
    std::string bytes, identity;
    std::vector<int32_t> feature, left, right, roots, channels;
    std::vector<float> cut, value, bias;
    std::vector<uint8_t> missing_left;
    std::vector<int32_t> leaf_offsets{0}, leaf_nodes, ancestors;
    std::vector<uint8_t> ancestor_right;
};

inline SourceData read_source(const fs::path& path) {
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
    out.bytes = read_text(path);
    out.identity = sha256(out.bytes);
    auto document = dp_native_json::parse(out.bytes);
    auto& learner = document.at("learner");
    auto& booster = learner.at("gradient_booster");
    auto& parameters = learner.at("learner_model_param");
    if (booster.at("name") != "gbtree" || learner.at("objective").at("name") != "multi:softprob")
        throw std::runtime_error("native conversion requires scalar numeric gbtree multi:softprob");
    const auto features = decimal(parameters.at("num_feature"), "num_feature");
    const auto outputs = decimal(parameters.at("num_class"), "num_class");
    if (features < 1 || features > uint64_t(std::numeric_limits<int32_t>::max()) || outputs < 2 || outputs > 7)
        throw std::runtime_error("native class contract requires positive features and 2..7 classes");
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
}  // namespace dpnative
