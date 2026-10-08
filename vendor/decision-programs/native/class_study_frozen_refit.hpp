#pragma once
// Fixed-recipe host orchestration. Numerical work uses the existing resident
// trainer and OOF implementation; no evaluator, search or TEST source exists here.
#include "class_study_checkpoint.hpp"
#include "class_study_composition.hpp"
#include "class_study_oof.hpp"
#include "class_model_dataset_contract.hpp"
#include "class_io.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>

namespace class_study::frozen_refit {
using J = nlohmann::json;
using u64 = std::uint64_t;
namespace cp = class_study::checkpoint;
inline constexpr const char* state_format = "native-frozen-refit-checkpoint-1";
inline void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
inline u64 number(const J& value, const char* message) { return class_model_contract::unsigned_integer(value, message); }
inline void no_role(const J& object, const char* field) {
  need(object.at(field).is_boolean() && !object.at(field).get<bool>(), "frozen refit excludes VALID/TEST");
}
struct Recipe {
  std::uint32_t F = 0, K = 0;
  u64 rows = 0;
  std::vector<J> teachers;
  J meta, identity, teacher_groups = nullptr;
  OofOptions oof;
  std::optional<std::size_t> standalone;
  bool keep_development_meta = false;
};
// Metadata grouping shared with the OOF contract: exact normalized keys except
// rounds, stable first occurrence, and the maximum declared rounds per group.
inline J round_groups(const std::vector<J>& teachers) {
  std::map<std::string, std::size_t> ids;
  J groups = J::array(), by_teacher = J::array();
  for (std::size_t i = 0; i < teachers.size(); ++i) {
    auto hp = teachers[i]; const auto rounds = number(hp.at("rounds"), "refit group rounds"); hp.erase("rounds");
    const auto key = hp.dump(); auto it = ids.find(key);
    if (it == ids.end()) {
      const auto group = groups.size(); ids.emplace(key, group);
      groups.push_back({{"hyperparameters_except_rounds", hp}, {"maximum_rounds", rounds}, {"teachers", J::array()}});
      it = ids.find(key);
    }
    auto& group = groups.at(it->second); group.at("teachers").push_back(i);
    group["maximum_rounds"] = std::max(number(group.at("maximum_rounds"), "refit group maximum"), rounds);
    by_teacher.push_back(it->second);
  }
  return {{"format", "native-frozen-refit-round-groups-1"}, {"groups", groups}, {"group_by_teacher", by_teacher},
          {"contract", "exact normalized hyperparameters except rounds; maximum FIT once; original teacher output order"}};
}
// Metadata only: does not open the dataset, prior receipt or native library.
inline Recipe validate_recipe(const J& plan, const J& descriptor) {
  need(plan.at("format") == "native-frozen-refit-recipe-1" &&
       plan.at("workflow") == "native-frozen-refit-1", "frozen refit recipe format/workflow");
  no_role(plan, "TEST_read"); no_role(plan, "VALID_read"); no_role(descriptor, "TEST_read");
  const auto F = number(descriptor.at("features"), "refit features"),
             K = number(descriptor.at("classes"), "refit classes"),
             rows = number(descriptor.at("FIT_rows"), "refit FIT rows");
  need(F > 0 && F <= INT32_MAX && K >= 2 && K <= 16777217 && rows > 0 &&
       number(descriptor.at("VALID_rows"), "refit VALID rows") == 0, "refit requires dynamic FIT-only shape");
  if (descriptor.contains("rows")) need(number(descriptor.at("rows"), "refit rows") == rows, "refit row role extent");
  if (plan.contains("dataset_source")) need(plan.at("dataset_source") == descriptor, "refit declared dataset differs");
  need(plan.at("native_library_path").is_string() && class_model_contract::sha256(plan.at("native_library_sha256")),
       "refit native library declaration");
  Recipe out; out.F = std::uint32_t(F); out.K = std::uint32_t(K); out.rows = rows;
  if (plan.contains("refit_policy")) {
    need(plan.at("refit_policy").is_string(), "refit policy type");
    const auto policy = plan.at("refit_policy").get<std::string>();
    need(policy == "full_fit_oof_meta" || policy == "teachers_only_frozen_meta", "unsupported refit policy");
    out.keep_development_meta = policy == "teachers_only_frozen_meta";
  }
  const auto& teachers = plan.at("teacher_hyperparameters");
  need(teachers.is_array() && !teachers.empty() && teachers.size() <= u64(INT32_MAX) / K, "refit ordered teacher list/response width");
  for (const auto& hp : teachers) out.teachers.push_back(ResidentTrainer::validate_hyperparameters(hp, out.K));
  const auto& meta = plan.at("meta_hyperparameters");
  need(meta.is_array() && meta.size() == 1, "refit requires exactly one frozen meta configuration");
  out.meta = ResidentTrainer::validate_hyperparameters(meta.at(0), out.K);
  const auto& o = plan.at("oof");
  const auto folds = number(o.at("folds"), "refit folds");
  need(folds >= 2 && folds <= rows && folds <= UINT32_MAX, "refit fold capacity");
  out.oof.folds = std::uint32_t(folds); out.oof.seed = number(o.at("seed"), "refit fold seed");
  out.oof.gpu_byte_budget = number(o.at("gpu_byte_budget"), "refit OOF budget");
  need(out.oof.gpu_byte_budget > 0, "refit OOF byte budget must be positive");
  if (o.contains("group_round_prefixes")) {
    need(o.at("group_round_prefixes").is_boolean(), "refit OOF grouping must be boolean");
    out.oof.group_round_prefixes = o.at("group_round_prefixes").get<bool>();
  }
  if (out.oof.group_round_prefixes) out.teacher_groups = round_groups(out.teachers);
  const auto& prior = plan.at("selected_prior_study");
  need(prior.is_object() && prior.at("result_path").is_string() &&
       !prior.at("result_path").get_ref<const std::string&>().empty() &&
       class_model_contract::sha256(prior.at("result_sha256")) &&
       class_model_contract::sha256(prior.at("selected_model_sha256")) &&
       prior.at("generation").is_string() && !prior.at("generation").get_ref<const std::string&>().empty(),
       "refit explicit prior receipt identity");
  if (out.keep_development_meta) {
    need(prior.contains("bundle_path") && prior.at("bundle_path").is_string() &&
         !prior.at("bundle_path").get_ref<const std::string&>().empty() &&
         std::filesystem::path(prior.at("bundle_path").get<std::string>()).is_absolute() &&
         prior.at("bundle_path").get_ref<const std::string&>().find('\0') == std::string::npos &&
         class_model_contract::sha256(prior.at("bundle_sha256")), "frozen development meta requires pinned absolute selected bundle");
  }
  if (plan.contains("standalone_baseline_teacher_index") && !plan.at("standalone_baseline_teacher_index").is_null()) {
    const auto index = number(plan.at("standalone_baseline_teacher_index"), "refit standalone teacher index");
    need(index < out.teachers.size(), "refit standalone teacher index outside frozen order");
    out.standalone = std::size_t(index);
  }
  auto semantic = plan;
  for (const char* key : {"experiment_checkpoint_path", "experiment_resume_from", "experiment_checkpoint_host_byte_budget", "protocol"}) semantic.erase(key);
  // The prior content pin, generation and selected model are semantic; its
  // location is provenance and may move between compatible restart processes.
  semantic["selected_prior_study"].erase("result_path");
  if (out.keep_development_meta) semantic["selected_prior_study"].erase("bundle_path");
  out.identity = {{"format", "native-frozen-refit-identity-1"}, {"recipe", semantic},
      {"dataset", descriptor}, {"teacher_hyperparameters", out.teachers}, {"meta_hyperparameters", out.meta},
      {"training_rule", "refit all frozen teachers on all FIT; honest complement OOF; one fixed meta FIT; no selection"}};
  if (out.keep_development_meta) out.identity["training_rule"] = "refit frozen teachers on all FIT; preserve exact development OOF meta bytes; no OOF/meta FIT/selection in refit";
  if (out.oof.group_round_prefixes) out.identity["full_teacher_round_groups"] = out.teacher_groups;
  return out;
}
inline J validate_prior(const Recipe& recipe, const J& plan, const J& source) {
  const auto& offered = plan.at("selected_prior_study");
  need(source.at("format") == "native-nonlinear-combination-result-1" && source.at("complete") == true &&
       source.at("TEST_read") == false && source.at("selected").at("kind") == "nonlinear_combination",
       "refit prior receipt must be a completed selected nonlinear study");
  const auto& checkpoint = source.at("experiment_checkpoint");
  need(checkpoint.at("committed") == true && checkpoint.at("failed") == false &&
       checkpoint.at("generation") == offered.at("generation") &&
       source.at("selected").at("model_sha256") == offered.at("selected_model_sha256"), "refit prior committed selection differs");
  const auto& identity = source.at("identity"); const auto& old_plan = identity.at("plan");
  no_role(old_plan, "TEST_read"); no_role(old_plan, "VALID_read");
  const auto& prior_teachers = identity.at("teacher_hyperparameters");
  need(prior_teachers.is_array() && !prior_teachers.empty(), "refit prior teacher bank extent");
  const auto selected_count = source.at("selected").contains("meta_teacher_count")
      ? number(source.at("selected").at("meta_teacher_count"), "prior selected teacher prefix") : u64(prior_teachers.size());
  need(selected_count > 0 && selected_count <= prior_teachers.size() && selected_count == recipe.teachers.size(),
       "refit recipe is not the selected prior teacher prefix");
  for (std::size_t i = 0; i < recipe.teachers.size(); ++i)
    need(prior_teachers.at(i) == recipe.teachers.at(i), "refit selected ordered teacher prefix differs");
  need(
       old_plan.at("native_library_path") == plan.at("native_library_path") &&
       old_plan.at("native_library_sha256") == plan.at("native_library_sha256") &&
       number(identity.at("dataset").at("features"), "prior features") == recipe.F &&
       number(identity.at("dataset").at("classes"), "prior classes") == recipe.K,
       "refit prior ordered teachers/library/input shape differs");
  const auto& old_data = identity.at("dataset");
  const auto old_fit = number(old_data.at("FIT_rows"), "prior FIT rows"),
             old_valid = number(old_data.at("VALID_rows"), "prior VALID rows");
  need(old_fit <= UINT64_MAX - old_valid && old_fit + old_valid == recipe.rows,
       "refit does not cover exactly the prior training-data rows");
  auto old_contract = old_data, current_contract = recipe.identity.at("dataset");
  // The frozen development partition becomes all-FIT; bytes, order, input
  // semantics and source pins remain identical. A different same-shaped data
  // source is not authorized by the selected prior study.
  for (const char* key : {"FIT_rows", "VALID_rows", "rows", "split"}) {
    old_contract.erase(key); current_contract.erase(key);
  }
  need(old_contract == current_contract, "refit underlying training-data contract differs from prior study");
  const auto index = number(source.at("selected").at("index"), "prior selected meta index");
  const auto& old_meta = identity.at("meta_hyperparameters");
  need(old_meta.is_array() && index < old_meta.size() && old_meta.at(std::size_t(index)) == recipe.meta,
       "refit fixed meta parameters differ from selected prior model");
  const bool prefix_declared = old_plan.contains("meta_teacher_counts");
  need(prefix_declared == source.at("selected").contains("meta_teacher_count"),
       "refit prior selected teacher prefix declaration differs");
  if (prefix_declared) {
    const auto& counts = old_plan.at("meta_teacher_counts");
    need(counts.is_array() && counts.size() == old_meta.size() &&
         number(counts.at(std::size_t(index)), "prior declared teacher prefix") == selected_count &&
         identity.at("meta_teacher_counts") == counts &&
         number(source.at("selected").at("response_features"), "prior selected response features") == selected_count * recipe.K,
         "refit selected teacher prefix differs from its declared meta candidate");
  }
  need(number(old_plan.at("oof").at("folds"), "prior folds") == recipe.oof.folds &&
       number(old_plan.at("oof").at("seed"), "prior fold seed") == recipe.oof.seed &&
       old_plan.at("oof").value("group_round_prefixes", false) == recipe.oof.group_round_prefixes,
       "refit fold policy differs from frozen selected recipe");
  // This is provenance from a pinned result receipt, not a new VALID score or
  // replay of the prior checkpoint's numerical response payload.
  return {{"result_path", offered.at("result_path")}, {"result_sha256", offered.at("result_sha256")},
      {"generation", offered.at("generation")}, {"selected_model_sha256", offered.at("selected_model_sha256")},
      {"prior_identity_sha256", dpnative::sha256(identity.dump())}, {"teacher_hyperparameters", recipe.teachers},
      {"meta_hyperparameters", recipe.meta}, {"folds", recipe.oof.folds}, {"seed", recipe.oof.seed},
      {"group_round_prefixes", recipe.oof.group_round_prefixes},
      {"native_library_sha256", plan.at("native_library_sha256")},
      {"scope", "completed development-study metadata fixes recipe; no prior numerical response/model payload is loaded"}};
}
inline J pack(const TrainingResult& model) {
  return {{"model", cp::binary(model.model_json)}, {"sha256", model.model_sha256}, {"metrics", model.metrics}};
}
inline TrainingResult unpack(const J& saved) {
  TrainingResult model{cp::bytes(saved.at("model")), saved.at("sha256").get<std::string>(), saved.at("metrics")};
  need(!model.model_json.empty() && dpnative::sha256(model.model_json) == model.model_sha256, "refit saved native buffer differs");
  return model;
}
inline void check_model(const TrainingResult& model, const J& hp, u64 F, u64 K, u64 rows) {
  const auto& m = model.metrics;
  need(m.at("hyperparameters") == hp && number(m.at("features"), "saved model F") == F &&
       number(m.at("classes"), "saved model K") == K && number(m.at("FIT_rows"), "saved model FIT") == rows &&
       m.at("native_objective") == "multi:softmax" && m.at("VALID_read") == false && m.at("TEST_read") == false &&
       m.at("model_sha256") == model.model_sha256 && number(m.at("model_bytes"), "saved model bytes") == model.model_json.size(),
       "refit saved native shape/HP/roles/extent differs");
}
// A captured development meta is opaque native bytes, not a new FIT result.
// Its original row role/OOF provenance remains distinct from the refit dataset.
inline void check_preserved_meta(const Recipe& recipe, const J& plan,
                                 const TrainingResult& model, const J& provenance) {
  need(recipe.keep_development_meta && provenance.at("format") == "frozen-development-meta-provenance-1",
       "frozen meta policy/provenance format");
  const auto& offered = plan.at("selected_prior_study");
  for (const char* key : {"result_sha256", "generation", "selected_model_sha256", "bundle_sha256"})
    need(provenance.at(key) == offered.at(key), "frozen meta captured source identity differs");
  const auto& development = provenance.at("development_dataset");
  const auto fit_rows = number(development.at("FIT_rows"), "frozen meta development FIT rows");
  const auto valid_rows = number(development.at("VALID_rows"), "frozen meta development VALID rows");
  need(fit_rows > 0 && valid_rows > 0 && fit_rows <= UINT64_MAX - valid_rows && fit_rows + valid_rows == recipe.rows &&
       number(development.at("features"), "frozen meta raw F") == recipe.F &&
       number(development.at("classes"), "frozen meta K") == recipe.K && development.at("TEST_read") == false,
       "frozen meta development shape/roles differ");
  auto old_contract = development, new_contract = recipe.identity.at("dataset");
  for (const char* key : {"FIT_rows", "VALID_rows", "rows", "split"}) { old_contract.erase(key); new_contract.erase(key); }
  need(old_contract == new_contract && provenance.at("teacher_hyperparameters") == J(recipe.teachers) &&
       provenance.at("meta_hyperparameters") == recipe.meta &&
       provenance.at("original_teacher_order").is_array() && provenance.at("original_teacher_order").size() == recipe.teachers.size() &&
       provenance.at("OOF_exclusion_verified_on_CUDA") == true && provenance.at("exactly_once_coverage_verified_on_CUDA") == true &&
       provenance.at("full_FIT_models_used_for_OOF") == false &&
       provenance.at("folds") == recipe.oof.folds && provenance.at("seed") == recipe.oof.seed,
       "frozen meta recipe/OOF source contract differs");
  check_model(model, recipe.meta, u64(recipe.teachers.size()) * recipe.K, recipe.K, fit_rows);
  need(model.model_sha256 == offered.at("selected_model_sha256").get<std::string>() &&
       model.metrics.at("training_performed_this_workflow") == false &&
       model.metrics.at("native_library_sha256") == plan.at("native_library_sha256"),
       "frozen meta bytes/training origin differs");
}
inline TrainingResult capture_development_meta(const Recipe& recipe, const J& plan, const J& source,
                                               const J& bundle, u64 bundle_bytes, J& provenance) {
  const auto& offered = plan.at("selected_prior_study");
  const auto declaration = ResidentComposition::validate_bundle(bundle);
  const auto& development = source.at("identity").at("dataset");
  need(recipe.keep_development_meta && bundle.at("selected_kind") == "nonlinear_combination" &&
       bundle.at("raw_dataset_contract") == development && bundle.at("raw_features") == recipe.F && bundle.at("classes") == recipe.K &&
       bundle.at("native_library_path") == plan.at("native_library_path") &&
       bundle.at("native_library_sha256") == plan.at("native_library_sha256") &&
       bundle.at("meta_model").at("sha256") == offered.at("selected_model_sha256") &&
       bundle.at("teacher_order").size() == recipe.teachers.size() && source.at("selected_bundle_bytes") == bundle_bytes,
       "frozen meta original selected bundle differs");
  const auto& evidence = source.at("OOF");
  need(evidence.at("complete") == true && evidence.at("OOF_exclusion_verified_on_CUDA") == true &&
       evidence.at("exactly_once_coverage_verified_on_CUDA") == true && evidence.at("full_FIT_models_used_for_OOF") == false &&
       evidence.at("FIT_rows") == development.at("FIT_rows") && evidence.at("VALID_rows") == development.at("VALID_rows") &&
       evidence.at("folds") == recipe.oof.folds && evidence.at("seed") == recipe.oof.seed,
       "frozen meta original honest OOF evidence differs");
  const auto& responses = source.at("response_binding");
  need(responses.at("teacher_declarations").is_array() && responses.at("teacher_declarations").size() >= recipe.teachers.size(),
       "frozen meta original teacher response extent");
  for (std::size_t i = 0; i < recipe.teachers.size(); ++i) {
    const auto& pin = bundle.at("teacher_order").at(i); const auto& teacher = responses.at("teacher_declarations").at(i);
    need(teacher.at("hyperparameters") == recipe.teachers.at(i) && teacher.at("full_fit_model_sha256") == pin &&
         teacher.at("full_fit_model_bytes") == cp::bytes(bundle.at("teacher_models").at(pin.get<std::string>()).at("model")).size(),
         "frozen meta source teacher order/response pin differs");
    bool found = false;
    for (const auto& candidate : source.at("candidates")) if (candidate.at("kind") == "native_teacher" && candidate.at("index") == i) {
      need(!found && candidate.at("model_sha256") == pin, "frozen meta source candidate teacher pin differs"); found = true;
    }
    need(found, "frozen meta source teacher result absent");
  }
  const auto fit_rows = number(development.at("FIT_rows"), "frozen meta FIT rows");
  provenance = {{"format", "frozen-development-meta-provenance-1"}, {"bundle_path", offered.at("bundle_path")},
      {"bundle_sha256", offered.at("bundle_sha256")}, {"result_sha256", offered.at("result_sha256")},
      {"generation", offered.at("generation")}, {"selected_model_sha256", offered.at("selected_model_sha256")},
      {"development_dataset", development}, {"teacher_hyperparameters", recipe.teachers}, {"meta_hyperparameters", recipe.meta},
      {"original_teacher_order", bundle.at("teacher_order")}, {"folds", recipe.oof.folds}, {"seed", recipe.oof.seed},
      {"development_fold_training_rows_min", fit_rows - (fit_rows / recipe.oof.folds + (fit_rows % recipe.oof.folds != 0))},
      {"development_fold_training_rows_max", fit_rows - fit_rows / recipe.oof.folds},
      {"OOF_exclusion_verified_on_CUDA", true}, {"exactly_once_coverage_verified_on_CUDA", true}, {"full_FIT_models_used_for_OOF", false},
      {"scope", "captured completed development OOF provenance; not recomputed or used for refit selection"},
      {"distribution_shift", "meta trained on development OOF fold-teacher probabilities; deployment uses newly refit all-FIT teachers; prior VALID accuracy is not preserved"}};
  TrainingResult model{cp::bytes(bundle.at("meta_model").at("model")), offered.at("selected_model_sha256").get<std::string>(),
      J{{"hyperparameters", recipe.meta}, {"features", u64(recipe.teachers.size()) * recipe.K}, {"classes", recipe.K},
        {"FIT_rows", fit_rows}, {"native_objective", "multi:softmax"}, {"VALID_read", false}, {"TEST_read", false},
        {"native_library_sha256", plan.at("native_library_sha256")}, {"model_sha256", offered.at("selected_model_sha256")},
        {"model_bytes", declaration.at("selected_model").at("bytes")}, {"training_performed_this_workflow", false},
        {"metadata_origin", "source receipt and validated selected bundle; original development training metrics are not recreated"}}};
  need(dpnative::sha256(model.model_json) == model.model_sha256, "frozen meta opaque bytes differ");
  check_preserved_meta(recipe, plan, model, provenance); return model;
}
inline void check_group_state(const Recipe& recipe, const std::vector<OofTeacher>& teachers,
                              const std::vector<std::optional<TrainingResult>>& parents,
                              const std::vector<std::string>& pins, u64 actual_fits) {
  if (!recipe.oof.group_round_prefixes) {
    need(parents.empty() && pins.empty() && actual_fits == teachers.size(), "refit independent FIT count differs"); return;
  }
  const auto& groups = recipe.teacher_groups.at("groups");
  need(parents.size() == groups.size() && pins.size() == groups.size(), "refit group parent extent");
  u64 started = 0;
  for (std::size_t g = 0; g < groups.size(); ++g) {
    const auto& group = groups.at(g); const auto& members = group.at("teachers");
    const auto first = number(members.front(), "refit group first"), last = number(members.back(), "refit group last");
    const bool fitted = first < teachers.size(), active = fitted && last >= teachers.size();
    started += fitted;
    need(bool(parents[g]) == active && (fitted ? class_model_contract::sha256(J(pins[g])) : pins[g].empty()),
         "refit required group parent/cursor pin differs");
    if (parents[g]) {
      auto hp = group.at("hyperparameters_except_rounds"); hp["rounds"] = group.at("maximum_rounds");
      check_model(*parents[g], hp, recipe.F, recipe.K, recipe.rows);
      need(parents[g]->model_sha256 == pins[g], "refit retained group parent bytes differ");
    }
  }
  need(started == actual_fits, "refit group FIT count differs from produced teachers");
  for (std::size_t i = 0; i < teachers.size(); ++i) {
    const auto g = std::size_t(number(recipe.teacher_groups.at("group_by_teacher").at(i), "refit teacher group"));
    const auto& m = teachers[i].full_fit_model.metrics;
    need(m.at("source_parent_sha256") == pins[g] &&
         m.at("original_rounds") == groups.at(g).at("maximum_rounds") &&
         m.at("selected_rounds") == recipe.teachers[i].at("rounds") && m.at("FIT_performed") == false,
         "refit produced teacher prefix binding differs");
  }
}
inline J pack_oof(const OofCheckpoint& saved) {
  J out = {{"state", saved.state}, {"responses", J::binary(saved.response_bytes)}, {"coverage", J::binary(saved.coverage_bytes)}};
  if (!saved.round_prefix_parent_bytes.empty()) {
    out["round_prefix_parents"] = J::array();
    for (const auto& bytes : saved.round_prefix_parent_bytes) out["round_prefix_parents"].push_back(J::binary(bytes));
  }
  return out;
}
inline OofCheckpoint unpack_oof(const J& saved) {
  need(saved.at("responses").is_binary() && saved.at("coverage").is_binary(), "refit OOF byte payload types");
  OofCheckpoint out{saved.at("state"), saved.at("responses").get_binary(), saved.at("coverage").get_binary(), {}};
  if (saved.contains("round_prefix_parents")) {
    need(saved.at("round_prefix_parents").is_array(), "refit OOF parent payload array");
    for (const auto& bytes : saved.at("round_prefix_parents")) {
      need(bytes.is_binary(), "refit OOF parent payload type"); out.round_prefix_parent_bytes.push_back(bytes.get_binary());
    }
  }
  return out;
}
inline void write(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream file(path, std::ios::binary); file.exceptions(std::ios::badbit | std::ios::failbit);
  file.write(bytes.data(), std::streamsize(bytes.size())); file.flush();
}
inline void write_cbor(const std::filesystem::path& path, const J& bundle) {
  std::ofstream file(path, std::ios::binary); file.exceptions(std::ios::badbit | std::ios::failbit);
  J::to_cbor(bundle, nlohmann::detail::output_adapter<char>(file)); file.flush();
}
inline int run(const J& plan, const J& descriptor, const std::filesystem::path& output,
               const std::string& requested_resume, volatile std::sig_atomic_t& stopping,
               volatile std::sig_atomic_t& checkpoint_requested) {
  const auto started = std::chrono::steady_clock::now(); const auto recipe = validate_recipe(plan, descriptor);
  need(output.is_absolute() && !std::filesystem::exists(output), "refit final output must be fresh absolute directory");
  const auto resume = requested_resume.empty() ? plan.value("experiment_resume_from", std::string{}) : requested_resume;
  const auto checkpoint_root = plan.value("experiment_checkpoint_path", resume);
  need(!checkpoint_root.empty(), "frozen refit requires whole-experiment checkpoint destination");
  const auto budget = number(plan.value("experiment_checkpoint_host_byte_budget", J(0)), "refit host checkpoint budget");
  cp::Coordinator coordinator(checkpoint_root, budget, state_format);
  std::string phase = "full_fit_teachers"; bool resumed = false;
  std::vector<OofTeacher> teachers; std::optional<TrainingResult> meta, frozen_meta;
  J frozen_meta_provenance = nullptr;
  const auto group_count = recipe.oof.group_round_prefixes ? recipe.teacher_groups.at("groups").size() : 0;
  std::vector<std::optional<TrainingResult>> group_parents(group_count);
  std::vector<std::string> group_parent_pins(group_count);
  u64 full_fits_total = 0;
  J prior, saved_oof = nullptr, oof_metrics = nullptr, response_binding = nullptr;
  u64 full_fits_this_process = 0, oof_fits_this_process = 0, meta_fits_this_process = 0;
  if (!resume.empty()) {
    auto loaded = cp::Coordinator::load(resume, budget, state_format); const auto& saved = loaded.state;
    need(loaded.engine_snapshot_path.empty() && saved.at("identity") == recipe.identity, "refit checkpoint identity differs");
    phase = saved.at("phase").get<std::string>();
    need(phase == "full_fit_teachers" || phase == "oof" || phase == "meta_fit" || phase == "complete", "refit saved phase");
    if (recipe.keep_development_meta) need(phase == "full_fit_teachers" || phase == "complete", "frozen meta saved phase contains new OOF/meta training");
    prior = saved.at("prior_study");
    for (const char* key : {"result_sha256", "generation", "selected_model_sha256"})
      need(prior.at(key) == plan.at("selected_prior_study").at(key), "refit saved prior-source identity differs");
    need(prior.at("teacher_hyperparameters") == J(recipe.teachers) &&
         prior.at("meta_hyperparameters") == recipe.meta &&
         number(prior.at("folds"), "saved prior folds") == recipe.oof.folds &&
         number(prior.at("seed"), "saved prior seed") == recipe.oof.seed &&
         prior.value("group_round_prefixes", false) == recipe.oof.group_round_prefixes &&
         prior.at("native_library_sha256") == plan.at("native_library_sha256"),
         "refit saved prior recipe binding differs");
    no_role(saved, "TEST_read"); no_role(saved, "VALID_read");
    need(saved.at("selection_performed") == false && saved.at("teachers").is_array(), "refit saved stage contract");
    for (const auto& teacher : saved.at("teachers"))
      teachers.push_back({teacher.at("hyperparameters"), unpack(teacher.at("native")), teacher.at("FIT_binding")});
    if (!saved.at("meta_model").is_null()) meta = unpack(saved.at("meta_model"));
    saved_oof = saved.at("oof_checkpoint"); oof_metrics = saved.at("OOF"); response_binding = saved.at("response_binding");
    if (recipe.keep_development_meta) {
      frozen_meta = unpack(saved.at("frozen_development_meta")); frozen_meta_provenance = saved.at("frozen_meta_provenance");
      check_preserved_meta(recipe, plan, *frozen_meta, frozen_meta_provenance);
      need(saved_oof.is_null() && oof_metrics.is_null() && response_binding.is_null(), "frozen meta checkpoint contains refit OOF state");
      if (meta) need(meta->model_json == frozen_meta->model_json && meta->model_sha256 == frozen_meta->model_sha256 &&
                     meta->metrics == frozen_meta->metrics, "frozen meta final bytes differ from captured source");
    }
    need(teachers.size() <= recipe.teachers.size() && number(saved.at("produced_teacher_models"), "refit produced teacher count") == teachers.size(),
         "refit produced teacher extent");
    full_fits_total = number(saved.at("full_FIT_calls"), "refit saved full FIT count");
    if (recipe.oof.group_round_prefixes) {
      need(saved.at("full_teacher_round_groups") == recipe.teacher_groups && saved.at("full_teacher_group_parents").size() == group_count &&
           saved.at("full_teacher_group_parent_sha256").size() == group_count, "refit saved round group contract");
      for (std::size_t g = 0; g < group_count; ++g) {
        if (!saved.at("full_teacher_group_parents").at(g).is_null()) group_parents[g] = unpack(saved.at("full_teacher_group_parents").at(g));
        group_parent_pins[g] = saved.at("full_teacher_group_parent_sha256").at(g).get<std::string>();
      }
    }
    check_group_state(recipe, teachers, group_parents, group_parent_pins, full_fits_total);
    need(phase == "full_fit_teachers" ? teachers.size() < recipe.teachers.size() : teachers.size() == recipe.teachers.size(), "refit teacher phase extent");
    need((phase == "complete") == bool(meta) && number(saved.at("meta_FIT_calls"), "refit saved meta FIT count") == (recipe.keep_development_meta ? 0 : u64(bool(meta))), "refit meta phase/model extent");
    need(phase == "full_fit_teachers" ? saved_oof.is_null() : true, "refit pre-OOF state contains response payload");
    need(!recipe.keep_development_meta && (phase == "meta_fit" || phase == "complete") ? !saved_oof.is_null() && saved_oof.at("state").at("phase") == "complete" : true,
         "refit meta stage lacks complete OOF evidence");
    for (std::size_t i = 0; i < teachers.size(); ++i) {
      need(teachers[i].hyperparameters == recipe.teachers[i], "refit restored teacher order differs");
      check_model(teachers[i].full_fit_model, recipe.teachers[i], recipe.F, recipe.K, recipe.rows);
    }
    if (meta && !recipe.keep_development_meta) check_model(*meta, recipe.meta, u64(recipe.teachers.size()) * recipe.K, recipe.K, recipe.rows);
    resumed = true;
  } else {
    const auto bytes = dpnative::read_text(plan.at("selected_prior_study").at("result_path").get<std::string>());
    need(dpnative::sha256(bytes) == plan.at("selected_prior_study").at("result_sha256").get<std::string>(), "refit prior result bytes differ");
    const auto source = J::parse(bytes); prior = validate_prior(recipe, plan, source);
    if (recipe.keep_development_meta) {
      const auto bundle_bytes = dpnative::read_text(plan.at("selected_prior_study").at("bundle_path").get<std::string>());
      need(dpnative::sha256(bundle_bytes) == plan.at("selected_prior_study").at("bundle_sha256").get<std::string>(),
           "frozen meta selected bundle bytes differ");
      frozen_meta = capture_development_meta(recipe, plan, source, J::from_cbor(bundle_bytes), bundle_bytes.size(), frozen_meta_provenance);
      prior["scope"] = "pinned completed development receipt and selected meta bytes captured; no prior response matrix is loaded";
    }
  }
  std::unique_ptr<ResidentOof> oof;
  auto snapshot = [&] {
    J bank = J::array(); for (const auto& teacher : teachers)
      bank.push_back({{"hyperparameters", teacher.hyperparameters}, {"native", pack(teacher.full_fit_model)}, {"FIT_binding", teacher.full_fit_data_binding}});
    J parents = J::array(); for (const auto& parent : group_parents) parents.push_back(parent ? pack(*parent) : J(nullptr));
    J saved = {{"format", state_format}, {"identity", recipe.identity}, {"prior_study", prior}, {"phase", phase},
        {"teachers", std::move(bank)}, {"meta_model", meta ? pack(*meta) : J(nullptr)},
        {"full_FIT_calls", full_fits_total}, {"produced_teacher_models", teachers.size()}, {"meta_FIT_calls", recipe.keep_development_meta ? 0 : (meta ? 1 : 0)},
        {"full_teacher_round_groups", recipe.teacher_groups}, {"full_teacher_group_parents", std::move(parents)},
        {"full_teacher_group_parent_sha256", group_parent_pins},
        {"oof_checkpoint", oof ? pack_oof(oof->checkpoint()) : saved_oof}, {"OOF", oof ? oof->metrics() : oof_metrics},
        {"response_binding", response_binding}, {"selection_performed", false}, {"VALID_read", false}, {"TEST_read", false},
        {"periodic_checkpoint_writes", false}, {"mid_native_FIT_resume", false}};
    if (recipe.keep_development_meta) { saved["frozen_development_meta"] = pack(*frozen_meta); saved["frozen_meta_provenance"] = frozen_meta_provenance; }
    return saved;
  };
  auto boundary = [&](bool final = false) {
    if (final || stopping || checkpoint_requested) {
      coordinator.finish(); need(!coordinator.poll().failed, "refit prior checkpoint writer failed"); checkpoint_requested = 0;
      need(coordinator.save_host(snapshot()), "refit checkpoint writer busy");
      if (final || stopping) { coordinator.finish(); need(!coordinator.poll().failed, "refit checkpoint publication failed"); }
    }
    J progress = oof ? oof->progress() : J(nullptr);
    std::cout << J{{"event", "frozen_refit_boundary"}, {"phase", phase}, {"full_FIT_calls", full_fits_total}, {"produced_teacher_models", teachers.size()},
        {"meta_FIT_calls", recipe.keep_development_meta ? 0 : (meta ? 1 : 0)}, {"oof_cursor", progress.is_null() ? J(nullptr) : progress.at("oof_cursor")},
        {"stop_requested", bool(stopping)}}.dump() << '\n' << std::flush;
    return bool(stopping);
  };
  if (phase != "complete") {
    auto data = stage_fit_dataset(descriptor); const auto fit = fit_prefix(data);
    need(data.features == recipe.F && data.classes == recipe.K && data.rows == recipe.rows &&
         data.fit_rows == data.rows && data.valid_rows == 0, "refit staged FIT-only shape differs");
    for (const auto& teacher : teachers) need(teacher.full_fit_data_binding == fit.binding, "refit restored full-FIT binding differs");
    if (phase == "full_fit_teachers") {
      ResidentTrainer trainer(plan, fit); trainer.restore_completed_trial_count(full_fits_total);
      while (teachers.size() < recipe.teachers.size()) {
        if (boundary()) return 2;
        const auto index = teachers.size(); TrainingResult trained;
        if (recipe.oof.group_round_prefixes) {
          const auto g = std::size_t(number(recipe.teacher_groups.at("group_by_teacher").at(index), "refit teacher group"));
          const auto& group = recipe.teacher_groups.at("groups").at(g);
          if (!group_parents[g]) {
            auto hp = group.at("hyperparameters_except_rounds"); hp["rounds"] = group.at("maximum_rounds");
            group_parents[g] = trainer.train(hp); group_parent_pins[g] = group_parents[g]->model_sha256;
            ++full_fits_this_process; ++full_fits_total;
          }
          const auto& parent = *group_parents[g];
          if (trainer.retained_prefix_source_sha256() != parent.model_sha256)
            trainer.retain_prefix_source(parent.model_json, parent.model_sha256);
          trained = trainer.slice_retained_prefix(std::uint32_t(number(recipe.teachers[index].at("rounds"), "refit requested rounds")));
          need(trained.metrics.at("source_parent_sha256") == parent.model_sha256 &&
               trained.metrics.at("original_rounds") == group.at("maximum_rounds"), "refit native prefix source/round extent differs");
          trained.metrics["hyperparameters"] = recipe.teachers[index];
          trained.metrics["full_FIT_round_group"] = g;
          trained.metrics["maximum_round_parent_hyperparameters"] = parent.metrics.at("hyperparameters");
          if (number(group.at("teachers").back(), "refit group last") == index) group_parents[g].reset();
        } else {
          trained = trainer.train(recipe.teachers[index]); ++full_fits_this_process; ++full_fits_total;
        }
        teachers.push_back({recipe.teachers[index], std::move(trained), fit.binding});
      }
      phase = recipe.keep_development_meta ? "complete" : "oof";
      if (recipe.keep_development_meta) meta = *frozen_meta;
    }
    if (!recipe.keep_development_meta) {
      oof = std::make_unique<ResidentOof>(plan, data, teachers, recipe.oof);
    if (!saved_oof.is_null()) oof->restore(unpack_oof(saved_oof));
    if (phase == "oof") {
      while (!oof->complete()) { if (boundary()) return 2; oof->advance(); }
      phase = "meta_fit";
    }
    need(oof->complete(), "refit fixed meta stage requires complete honest OOF responses");
    auto responses = oof->completed_data(); response_binding = responses.binding;
    need(responses.fit_rows == responses.rows && responses.valid_rows == 0, "refit responses contain VALID rows");
    oof_fits_this_process = number(oof->metrics().at("native_training_calls_this_process"), "refit OOF process FIT count");
    if (boundary()) return 2;
    auto meta_plan = plan; meta_plan.erase("dataset_source");
    ResidentTrainer trainer(meta_plan, fit_prefix(responses));
    meta = trainer.train(recipe.meta); ++meta_fits_this_process;
    oof_metrics = oof->metrics(); phase = "complete";
    }
  }
  if (boundary(true)) return 2;
  need(meta.has_value() && teachers.size() == recipe.teachers.size(), "refit completed deployment is absent");
  J deployment = {{"format", "native-nonlinear-deployment-bundle-1"}, {"selected_kind", "nonlinear_combination"},
      {"raw_features", recipe.F}, {"classes", recipe.K}, {"native_library_path", plan.at("native_library_path")},
      {"native_library_sha256", plan.at("native_library_sha256")}, {"raw_dataset_contract", descriptor},
      {"teacher_probability_contract", "native full-round gbtree clone; derived multi:softprob; FP32 teacher-major/class-minor; no CPU transform"},
      {"final_class_contract", "selected native multi:softmax public class"}, {"teacher_models", J::object()}, {"teacher_order", J::array()},
      {"compiled_single_tree", false}, {"deployment_replay_qualified", false}};
  for (const auto& teacher : teachers) {
    const auto& model = teacher.full_fit_model; deployment["teacher_order"].push_back(model.model_sha256);
    if (!deployment["teacher_models"].contains(model.model_sha256))
      deployment["teacher_models"][model.model_sha256] = {{"model", cp::binary(model.model_json)}, {"sha256", model.model_sha256}};
  }
  deployment["meta_model"] = {{"model", cp::binary(meta->model_json)}, {"sha256", meta->model_sha256}};
  if (recipe.keep_development_meta) { deployment["refit_policy"] = "teachers_only_frozen_meta"; deployment["meta_training_provenance"] = frozen_meta_provenance; }
  const auto bundle_metadata = ResidentComposition::validate_bundle(deployment);
  std::filesystem::create_directories(output); write_cbor(output / "selected-bundle.cbor", deployment);
  J standalone_artifact = nullptr;
  if (recipe.standalone) {
    const auto index = *recipe.standalone; const auto& model = teachers.at(index).full_fit_model;
    auto standalone = deployment; standalone["selected_kind"] = "native_teacher";
    standalone["teacher_models"] = J::object(); standalone["teacher_order"] = J::array(); standalone.erase("meta_model");
    standalone["native_model"] = {{"model", cp::binary(model.model_json)}, {"sha256", model.model_sha256}};
    ResidentComposition::validate_bundle(standalone);
    write_cbor(output / "standalone-bundle.cbor", standalone); write(output / "standalone-model.json", model.model_json);
    standalone_artifact = {{"teacher_index", index}, {"model_sha256", model.model_sha256},
        {"bundle_path", (output / "standalone-bundle.cbor").string()}, {"model_path", (output / "standalone-model.json").string()},
        {"scored", false}};
  }
  const auto checkpoint = coordinator.poll();
  J report = {{"format", "native-frozen-refit-result-1"}, {"complete", true}, {"identity", recipe.identity}, {"prior_study", prior},
      {"selected_kind", "nonlinear_combination"}, {"meta_model_sha256", meta->model_sha256}, {"meta_training", meta->metrics},
      {"teacher_models", J::array()}, {"response_binding", response_binding}, {"OOF", oof_metrics},
      {"OOF_metrics_scope", "retained construction evidence; separate top-level this-process FIT counters describe this invocation"},
      {"full_FIT_calls_total", full_fits_total}, {"full_teacher_models_total", teachers.size()},
      {"full_FIT_round_groups", recipe.teacher_groups}, {"full_teacher_prefix_slices_total", recipe.oof.group_round_prefixes ? teachers.size() : 0}, {"full_FIT_calls_this_process", full_fits_this_process},
      {"OOF_FIT_calls_total", recipe.keep_development_meta ? 0 : number(oof_metrics.at("native_training_calls_completed_total"), "refit total OOF FIT calls")}, {"OOF_FIT_calls_this_process", oof_fits_this_process},
      {"OOF_response_teacher_fold_units", recipe.keep_development_meta ? 0 : u64(teachers.size()) * recipe.oof.folds},
      {"meta_FIT_calls_total", recipe.keep_development_meta ? 0 : 1}, {"meta_FIT_calls_this_process", meta_fits_this_process}, {"resumed_experiment", resumed},
      {"selection_performed", false}, {"HPO_performed", false}, {"accuracy_evaluated", false},
      {"VALID_read", false}, {"TEST_read", false}, {"per_trial_file_writes", 0}, {"periodic_checkpoint_writes", false},
      {"mid_native_FIT_resume", false}, {"checkpoint_scope", "completed full-teacher/FIT-fold/fixed-meta boundaries; interrupted native calls finish before graceful save"},
      {"selected_bundle_path", (output / "selected-bundle.cbor").string()},
      {"selected_bundle_bytes", std::filesystem::file_size(output / "selected-bundle.cbor")}, {"bundle_metadata", bundle_metadata},
      {"standalone_baseline_artifact", standalone_artifact},
      {"experiment_checkpoint", {{"committed", checkpoint.committed}, {"failed", checkpoint.failed}, {"generation", checkpoint.generation}, {"host_bytes", checkpoint.host_bytes}}},
      {"process_wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}};
  if (recipe.keep_development_meta) {
    report["refit_policy"] = "teachers_only_frozen_meta";
    report["frozen_meta_provenance"] = frozen_meta_provenance;
    report["meta_training_scope"] = "exact development OOF meta bytes preserved; no meta FIT in this refit; deployed teachers refit on all FIT";
    report["OOF_metrics_scope"] = "no OOF response matrix or fold FIT constructed in this refit; original development provenance is reported separately";
    report["checkpoint_scope"] = "completed all-FIT teacher boundaries plus captured immutable development meta; no mid-native FIT restart";
    report["prior_VALID_accuracy_preserved"] = false;
  }
  for (const auto& teacher : teachers) report["teacher_models"].push_back({{"hyperparameters", teacher.hyperparameters},
      {"model_sha256", teacher.full_fit_model.model_sha256}, {"model_bytes", teacher.full_fit_model.model_json.size()}, {"training", teacher.full_fit_model.metrics}});
  write(output / "result.json", report.dump(2) + "\n");
  std::cout << J{{"event", "frozen_refit_complete"}, {"output", output.string()}, {"meta_model_sha256", meta->model_sha256},
      {"accuracy_evaluated", false}, {"selection_performed", false}, {"TEST_read", false}}.dump() << '\n' << std::flush;
  return 0;
}
} // namespace class_study::frozen_refit