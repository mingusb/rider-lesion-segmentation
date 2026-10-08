// CPU-only JSON/parser and serialization checks. No numerical inference or fit.
#include "class_io.hpp"

#include <functional>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: class_io_checks SOURCE.json NEW_DIRECTORY");
        const dpnative::fs::path directory = argv[2];
        if (!dpnative::fs::create_directories(directory)) throw std::runtime_error("check directory already exists");
        const auto original = dpnative::read_source(argv[1]);
        auto fixture = dpnative::json::parse(original.bytes);
        auto& model = fixture["learner"]["gradient_booster"]["model"];
        model["trees"] = dpnative::json::array({model["trees"].at(0)});
        model["tree_info"] = dpnative::json::array({model["tree_info"].at(0)});
        model["iteration_indptr"] = dpnative::json::array({0, 1});
        model["gbtree_model_param"]["num_trees"] = "1";
        dpnative::atomic_json(directory / "valid-prefix.json", fixture);
        const auto valid = dpnative::read_source(directory / "valid-prefix.json");
        if (valid.roots.size() != 1 || valid.features != original.features || valid.outputs != original.outputs)
            throw std::runtime_error("valid prefix metadata changed");
        using J = dpnative::json;
        auto tree = [](J& value) -> J& { return value["learner"]["gradient_booster"]["model"]["trees"][0]; };
        auto ensemble = [](J& value) -> J& { return value["learner"]["gradient_booster"]["model"]; };
        const std::vector<std::pair<std::string, std::function<void(J&)>>> cases{
            {"wrong-version", [](J& j) { j["version"] = J::array({3, 4, 0}); }},
            {"fractional-version", [](J& j) { j["version"][0] = 3.0; }},
            {"decimal-suffix", [](J& j) { j["learner"]["learner_model_param"]["num_feature"] = "54junk"; }},
            {"fractional-channel", [&](J& j) { ensemble(j)["tree_info"][0] = 0.5; }},
            {"wrong-tree-id", [&](J& j) { tree(j)["id"] = 2; }},
            {"wrong-node-count", [&](J& j) { tree(j)["tree_param"]["num_nodes"] = "1"; }},
            {"wrong-tree-count", [&](J& j) { ensemble(j)["gbtree_model_param"]["num_trees"] = "2"; }},
            {"wrong-iteration-start", [&](J& j) { ensemble(j)["iteration_indptr"] = J::array({1, 1}); }},
            {"wrong-iteration-end", [&](J& j) { ensemble(j)["iteration_indptr"] = J::array({0, 0}); }},
            {"unordered-iterations", [&](J& j) { ensemble(j)["iteration_indptr"] = J::array({0, 2, 1}); }},
            {"invalid-default", [&](J& j) { tree(j)["default_left"][0] = 2; }},
            {"fractional-default", [&](J& j) { tree(j)["default_left"][0] = 1.0; }},
            {"fractional-feature", [&](J& j) { tree(j)["split_indices"][0] = 0.5; }},
            {"wrong-split-array", [&](J& j) { tree(j)["split_type"] = J::array(); }},
            {"shared-successor", [&](J& j) { tree(j)["right_children"][0] = tree(j)["left_children"][0]; }},
            {"cycle", [&](J& j) { tree(j)["left_children"][0] = 0; }}
        };
        std::vector<std::string> rejected;
        for (const auto& [name, modify] : cases) {
            auto candidate = fixture;
            modify(candidate);
            const auto path = directory / (name + ".json");
            dpnative::atomic_json(path, candidate);
            bool failed = false;
            try { (void)dpnative::read_source(path); }
            catch (const std::exception&) { failed = true; }
            if (!failed) throw std::runtime_error("reader accepted malformed source: " + name);
            rejected.push_back(name);
        }
        J result{{"complete", true}, {"cpu_scope", "JSON parsing and serialization only"},
                 {"python_runtime", false}, {"source_sha256", original.identity},
                 {"source_trees", original.roots.size()}, {"valid_prefix_trees", valid.roots.size()},
                 {"malformed_cases_rejected", rejected}, {"count", rejected.size() + 2}};
        dpnative::atomic_json(directory / "result.json", result);
        std::cout << result.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native reader checks failed: " << error.what() << '\n';
        return 1;
    }
}
