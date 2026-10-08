#include "class_study_convert.hpp"
#include "class_native_softprob_gap.hpp"
#include "class_conversion/class_io.hpp"
#include "class_conversion/adaptive_engine.cuh"
#include "class_conversion/adaptive_parallel_schedule.cuh"
#include "class_conversion/grid_count_report.hpp"
#include "class_conversion/adaptive_work_summary.cuh"
#include "class_conversion/adaptive_eta_report.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

namespace class_study {
namespace {
namespace a = class_conversion_adaptive;
namespace d = class_conversion_adaptive_domain;
using class_conversion_native::SourceData;
using class_conversion_native::sha256;
using json = nlohmann::json;
bool active_conversion = false;
json calibrated_decision_rule(const NativeOracle& native) {
  if (!native.positive_probability_threshold) return nullptr;
  const float threshold = *native.positive_probability_threshold;
  a::require(native.classes == 2 && std::isfinite(threshold) &&
                 threshold >= 0.f && threshold <= 1.f,
             "calibrated conversion requires two classes and a finite FP32 probability threshold in [0,1]");
  return {{"format", "native-positive-probability-threshold-1"},
          {"positive_class", 1}, {"comparison", ">="},
          {"threshold_fp32", threshold},
          {"threshold_fp32_bits", std::bit_cast<std::uint32_t>(threshold)},
          {"probability_transform", "native multi:softprob"},
          {"iteration_range", "all source rounds"},
          {"public_callback_output", "direct_class_index"}};
}
struct ConversionGuard {
  ConversionGuard() {
    a::require(!active_conversion, "resident conversion is non-reentrant");
    active_conversion = true;
  }
  ~ConversionGuard() { active_conversion = false; }
};
json counters(const a::Status& h) {
  return {{"states", h.state_creations}, {"state_slots_high_water", h.states},
          {"state_slots_occupied", h.states - h.free_count}, {"state_slots_free", h.free_count},
          {"state_slots_reused", h.state_reuses}, {"completed_state_cache_evictions", h.state_evictions},
          {"nodes", h.nodes}, {"expansions", h.expansions},
          {"normalization_steps", h.normalization_steps}, {"prefix_additions", h.prefix_additions},
          {"state_cache_hits", h.state_hits}, {"node_cache_hits", h.node_hits},
          {"terminal_vector_cache_hits", h.terminal_hits}, {"native_terminal_vectors", h.native_terminals},
          {"class_pruned_states", h.class_pruned_states}, {"key_collisions", h.key_collisions},
          {"terminal_gap_pruned_states", h.terminal_gap_pruned_states},
          {"active_depth", h.depth}, {"CUDA_error_code", h.error}};
}
std::string encode(const std::vector<a::Node>& nodes, a::u32 root,
                   a::u32 F, a::u32 K, const std::string& source) {
  a::require(source.size() == 64 && !nodes.empty() && root < nodes.size(), "canonical graph identity");
  std::ostringstream out(std::ios::binary);
  out.write("CLSGDAG1", 8);
  for (a::u32 word : {1u, F, K, root, a::u32(nodes.size()), 1u})
    out.write(reinterpret_cast<const char*>(&word), sizeof(word));
  for (size_t i = 0; i < source.size(); i += 2)
    out.put(char(std::stoul(source.substr(i, 2), nullptr, 16)));
  out.write(reinterpret_cast<const char*>(nodes.data()), nodes.size() * sizeof(a::Node));
  return out.str();
}
struct SourceStorage {
  a::Buffer<std::int32_t> feature, left, right, roots, channels;
  a::Buffer<float> cut, value, bias;
  a::Buffer<std::uint8_t> missing;
  explicit SourceStorage(a::Budget& b, const SourceData& s)
      : feature(b, s.feature.size()), left(b, s.left.size()), right(b, s.right.size()),
        roots(b, s.roots.size()), channels(b, s.channels.size()), cut(b, s.cut.size()),
        value(b, s.value.size()), bias(b, s.bias.size()), missing(b, s.missing_left.size()) {
    feature.upload(s.feature); left.upload(s.left); right.upload(s.right);
    roots.upload(s.roots); channels.upload(s.channels); cut.upload(s.cut);
    value.upload(s.value); bias.upload(s.bias); missing.upload(s.missing_left);
  }
  a::SourceView view(const SourceData& s) const {
    return {feature.data, left.data, right.data, roots.data, channels.data,
            cut.data, value.data, bias.data, missing.data,
            a::u32(s.features), a::u32(s.outputs), a::u32(s.feature.size()), a::u32(s.roots.size()), a::u32(s.native_margin_outputs)};
  }
};
struct DomainStorage {
  a::Buffer<std::int32_t> group, numeric;
  a::Buffer<a::u32> bit, word_offsets, feature_offsets, widths, features;
  a::Buffer<a::u64> masks;
  DomainStorage(a::Budget& b, const d::HostMetadata& h)
      : group(b, h.feature_group.size()), numeric(b, h.feature_numeric.size()), bit(b, h.feature_bit.size()),
        word_offsets(b, h.group_word_offsets.size()), feature_offsets(b, h.group_feature_offsets.size()),
        widths(b, h.group_widths.size()), features(b, h.group_features.size()), masks(b, h.initial_masks.size()) {
    group.upload(h.feature_group); numeric.upload(h.feature_numeric);
    bit.upload(h.feature_bit); word_offsets.upload(h.group_word_offsets);
    feature_offsets.upload(h.group_feature_offsets); widths.upload(h.group_widths);
    features.upload(h.group_features); masks.upload(h.initial_masks);
  }
  d::DomainView view(const d::HostMetadata& h) const {
    return {h.features, h.groups, h.mask_words, h.allow_nan, group.data, bit.data,
            word_offsets.data, feature_offsets.data, widths.data, features.data, masks.data,
            h.numeric_features, numeric.data};
  }
};
a::eta::Sample sample_work(a::Budget& budget,a::EngineView e,WorkEstimateOptions requested,
                          const std::function<void()>& check_gate,
                          const std::function<bool()>& stop_requested) {
  const auto started=std::chrono::steady_clock::now();
  a::eta::Sample result;
  {
    const auto maximum=requested.maximum_decisions?requested.maximum_decisions:e.source.nodes+1;
    a::work::Options sampled{requested.paths,requested.seed,maximum,requested.refinement_visit_budget};
    a::work::Storage paths(budget,e,sampled);a::Buffer<a::work::Summary> output(budget,1);
    paths.initialize();
    a::work::Summary summary{};
    for(a::u64 decisions=0;;) {
      check_gate();summary=a::work::summarize(paths.device_results(),requested.paths,output);
      if(summary.done==requested.paths||summary.errors||summary.overflows||decisions>=maximum||
         (stop_requested&&stop_requested())||
         std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>=requested.maximum_seconds)break;
      const auto chunk=a::u32(std::min<a::u64>(requested.decisions_per_chunk,maximum-decisions));
      paths.advance(chunk);decisions+=chunk;
    }
    result.paths=summary.paths;result.done=summary.done;result.censored=summary.censored;
    result.errors=summary.errors+summary.invalid_status;result.overflows=summary.overflows;
    result.valid_total=summary.valid_total!=0;
    result.mean=summary.decisions.mean;result.standard_error=summary.decisions.standard_error;
    result.standard_error_available=summary.decisions.standard_error_available!=0;
  }
  result.sampling_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  return result;
}
a::u64 committed_decisions(const a::Status& s) {
  a::require(!(s.expansions&1),"ETA committed split-edge count is not even");
  a::u64 total=s.expansions/2;
  for(auto count:{s.class_pruned_states,s.native_terminals,s.terminal_hits}) {
    a::require(count<=UINT64_MAX-total,"ETA committed decision count overflow");total+=count;
  }
  return total;
}
} // namespace

// Both ordinary and composed sources enter this single adaptive constructor.
// A prepared descriptor is already parsed base topology, never a forged booster.
static ConvertedModel convert_prepared_source(const ConversionSource& input, const NativeOracle& native,
                             const ConversionOptions& options,
                             std::optional<SourceData> prepared_source = std::nullopt,
                             std::uint32_t output_classes = 0,
                             const json& provenance = json::object()) {
  const bool composed = prepared_source.has_value();
  auto started = std::chrono::steady_clock::now();
  json statistics = {{"format", "resident-adaptive-conversion-1"}, {"complete", false},
                     {"CPU_predictions", false}, {"input_file_reads", 0}, {"output_file_writes", 0}};
  a::Budget budget{options.gpu_byte_budget};
  auto publish = [&](const char* phase) {
    statistics["phase"] = phase;
    statistics["owned_GPU_bytes"] = budget.used;
    statistics["peak_owned_GPU_bytes_excluding_native_Runtime_and_driver"] = budget.peak;
    statistics["elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (options.progress) options.progress(statistics);
  };
  try {
    ConversionGuard guard;
    a::require(std::isfinite(options.completion_estimate_start_seconds)&&options.completion_estimate_start_seconds>=0,
               "ETA sampling delay must be finite and nonnegative");
    if(options.completion_estimate_enabled) {
      const auto& requested=options.completion_estimate;
      a::require(requested.paths&&requested.decisions_per_chunk&&std::isfinite(requested.maximum_seconds)&&requested.maximum_seconds>0,
                 "ETA sampling resources must be positive and finite");
    }
    a::require(options.batch_size <= 65536 && options.max_batch_size <= 65536 &&
                   (!options.max_batch_size || options.batch_size <= options.max_batch_size),
               "resident adaptive batch options invalid");
    a::require(!options.admission_threads || !(options.admission_threads&(options.admission_threads-1)),
               "resident adaptive admission threads must be zero or a power of two");
    a::require(!options.draft_threads || !(options.draft_threads&(options.draft_threads-1)),
               "resident adaptive draft threads must be zero or a power of two");
    a::require(options.split_policy=="source_order"||options.split_policy=="widest_residual"||options.split_policy=="aggregate_residual"||options.split_policy=="contracting_residual",
               "resident adaptive split policy invalid");
    a::require(native.predict && input.features && input.classes >= 2,
               "resident adaptive source/oracle declaration");
    a::require(input.model_json.size() <= 4ull * 1024 * 1024 * 1024, "resident source transport extent");
    auto source = prepared_source ? std::move(*prepared_source) :
        class_conversion_native::read_source_bytes(std::string(input.model_json));
    a::require(source.features == input.features && source.outputs == input.classes &&
                   native.features == input.features && native.classes == input.classes,
               "resident source/oracle shape differs");
    a::require(input.expected_source_sha256.empty() || input.expected_source_sha256 == source.identity,
               "resident source identity differs");
    a::require(native.source_sha256.empty() || native.source_sha256 == source.identity,
               "resident oracle source identity differs");
    a::require(native.objective == class_conversion_native::native_objective_name(source.objective),
               "resident native objective differs");
    const auto decision_rule = calibrated_decision_rule(native);
    const bool calibrated = !decision_rule.is_null();
    a::require(!composed || !calibrated, "calibrated composed conversion is not supported");
    const bool native_direct_classes = calibrated || native.objective == "multi:softmax";
    a::require(native.library_sha256 == native_softprob_gap::library_sha,
               "resident pinned native library differs");
    a::require(options.initial_states && options.initial_nodes &&
                   options.initial_states <= options.max_states && options.initial_nodes <= options.max_nodes &&
                   options.max_states <= 0x3fffffffu && options.max_nodes <= 0x3fffffffu && budget.limit,
               "resident adaptive resource declaration");
    a::require(source.roots.size() <= UINT32_MAX && source.feature.size() <= INT32_MAX,
               "resident source index representation");
    const auto F = a::u32(source.features), K = a::u32(source.outputs), T = a::u32(source.roots.size());
    const auto C = output_classes ? output_classes : K;
    a::require(K >= 2 && K <= 1024 && source.native_margin_outputs >= 0 &&
                   a::u32(source.native_margin_outputs) <= K && C >= 2 && C <= 1024 && (composed || C == K),
               "resident output class extent differs");
    auto domain = d::prepare_domain_metadata(F, options.domain.one_hot_groups, options.domain.allow_nan);
    auto grid_catalog=a::grid::make_catalog(domain,source.feature,source.cut);
    const auto grid_total=class_grid_report::product(grid_catalog.base_axis_bins);
    statistics["source_grid"]={{"cells_decimal",grid_total.decimal()},
      {"axis_bins",grid_catalog.base_axis_bins},{"materialized",false},
      {"measure","source threshold bins and categorical predicate signatures; separate missing atom"},
      {"runtime_completion_fraction",false}};
    const auto N = domain.numeric_features;
    statistics["source_model_sha256"] = source.identity;
    statistics["native_library_sha256"] = native.library_sha256;
    statistics["features"] = F; statistics["classes"] = C;
    if (composed) {
      statistics.update(provenance);
      statistics["score_channels"] = source.native_margin_outputs ? a::u32(source.native_margin_outputs) : K;
      statistics["composed_source"] = true;
      statistics["native_margin_audit_scope"] = "ordered concatenation of native base-model margins";
      statistics["class_pruning_disabled_reason"] = "base-score winners do not imply composed output classes";
    }
    statistics["numeric_domain_coordinates"] = N;
    statistics["source_trees"] = T - source.structural_guard_trees;
    statistics["source_nodes"] = source.feature.size() - 3ull*source.structural_guard_trees;
    if (source.native_margin_outputs) {
      statistics["native_margin_audit_columns"] = source.native_margin_outputs;
      statistics["authority_channels"] = K;
      statistics["structural_guard_trees"] = source.structural_guard_trees;
      statistics["structural_guard_nodes"] = 3ull*source.structural_guard_trees;
      statistics["authority_trees"] = T;
      statistics["authority_nodes"] = source.feature.size();
      statistics["source_topology_counts_exclude_structural_guards"] = true;
    }
    statistics["domain"] = {{"finite_FP32", true}, {"allow_nan", options.domain.allow_nan},
                             {"one_hot_groups", options.domain.one_hot_groups}, {"excludes_infinities", true}};
    statistics["native_objective"] = native.objective;
    if (calibrated) {
      statistics["decision_rule"] = decision_rule;
      statistics["calibrated_probability_threshold"] = true;
      statistics["class_pruning_disabled_reason"] = "native probability threshold requires source-constant terminal qualification; argmax gap rules are inapplicable";
    }
    statistics["source_bytes_hashed_once"] = true;
    statistics["native_model_loads"] = 0; statistics["endtrial_file_hashes"] = 0;
    statistics["construction"] = "original topology, ordered RN32 prefix, lazy exact residual-state cache";
    statistics["Cartesian_partition_materialized"] = false;
    statistics["support_projection_enabled"] = true;
    statistics["split_policy"] = options.split_policy;
    statistics["oldest_ready_jobs"] = options.oldest_ready_jobs;
    statistics["ready_selection_scope"] = "oldest quota then newest ready jobs; scheduling only, no class authority";
    statistics["split_priority_scope"] = "construction heuristic only; original ordered FP32 prefix and class acceptance unchanged";
    statistics["full_native_witness_context_retained"] = true;
    statistics["frontier_schedule"] = "bounded parallel CUDA drafts, ordered block-cooperative admission, dependency completion";
    statistics["cooperative_state_copy"] = true;
    statistics["admission_thread_mode"] = options.admission_threads ? "fixed" : "dynamic";
    statistics["draft_thread_mode"] = options.draft_threads ? "fixed" : "dynamic";
    statistics["completed_cache_mode"] = options.completed_cache_limit ? (*options.completed_cache_limit ? "fixed" : "disabled") : "dynamic";
    statistics["completed_cache_requested_limit"] = options.completed_cache_limit ? json(*options.completed_cache_limit) : json(nullptr);
    statistics["completed_cache_policy_scope"] = "idle completed-state retention target; within-batch/final counts can exceed target; slot eviction does not shrink allocated arenas";
    statistics["refinement_effort_mode"] = options.refinement_visit_budget ? (*options.refinement_visit_budget ? "fixed" : "disabled") : "dynamic";
    statistics["refinement_requested_visit_budget"] = options.refinement_visit_budget ? json(*options.refinement_visit_budget) : json(nullptr);
    statistics["cover_effort_mode"] = options.cover_visit_budget ? (*options.cover_visit_budget ? "fixed" : "disabled") : "dynamic";
    statistics["cover_requested_visit_budget"] = options.cover_visit_budget ? json(*options.cover_visit_budget) : json(nullptr);
    statistics["cover_proof_scope"] = "same native-qualified winner on every feasible side of selected source predicate; proof splits are not published";
    statistics["autotune_work_equivalent_scope"] = "scheduling proxy: jobs + 2*additional nonterminal prunes + completed cache edges; separate from raw jobs/s";
    const auto initial_batch = options.batch_size ? options.batch_size :
        std::min<std::uint32_t>(32, options.max_batch_size ? options.max_batch_size : 32);
    statistics["frontier_batch_mode"] = options.batch_size ? "fixed" : "dynamic";
    statistics["frontier_initial_batch_capacity"] = initial_batch;
    statistics["frontier_batch_capacity"] = initial_batch;
    statistics["completed_state_recycling"] = true;
    statistics["gpu_byte_budget"] = budget.limit;
    statistics["max_states"] = options.max_states; statistics["max_nodes"] = options.max_nodes;
    std::vector<a::Node> graph;
    a::u32 root = a::none;
    a::u64 state_growths = 0, node_growths = 0, margin_rows = 0, public_rows = 0;
    std::string gate_binding;
    const bool gap_enabled = !composed && !calibrated && source.identity == native_softprob_gap::source_sha && K == 7 &&
                             native.softprob_gap_gate && native.softprob_gap_gate->enabled() &&
                             native.source_sha256 == source.identity &&
                             native.softprob_gap_gate->matches_source(source.identity,K,native.objective,native.library_sha256);
    if (gap_enabled) gate_binding = native.softprob_gap_gate->binding_sha256();
    statistics["qualified_native_gap_pruning_enabled"] = gap_enabled;
    statistics["qualified_native_gap_binding"] = gate_binding;
    auto check_gate = [&] {
      if (gap_enabled)
        a::require(native.softprob_gap_gate->enabled() &&
                       native.softprob_gap_gate->binding_sha256() == gate_binding,
                   "resident same-process native gate lost");
    };
    {
      SourceStorage source_storage(budget, source);
      DomainStorage domain_storage(budget, domain);
      auto selection_policy=options.split_policy=="contracting_residual"?a::split::Policy::contracting_residual:
          options.split_policy=="aggregate_residual"?a::split::Policy::aggregate_residual:
          options.split_policy=="widest_residual"?a::split::Policy::widest_residual:a::split::Policy::source_order;
      std::unique_ptr<a::split::Storage> selection_storage;
      if(selection_policy==a::split::Policy::aggregate_residual)
        selection_storage=std::make_unique<a::split::Storage>(budget,
            a::split::make_metadata(source.feature,source.cut,source.missing_left,domain));
      auto selection_view=selection_storage?selection_storage->view(selection_policy):a::split::View{nullptr,0,selection_policy};
      statistics["split_predicate_groups"] = selection_view.groups;
      auto states = std::make_unique<a::StateStorage>(budget, a::u32(options.initial_states), N, K, T, domain.mask_words);
      auto nodes = std::make_unique<a::NodeStorage>(budget, a::u32(options.initial_nodes));
      a::Buffer<a::Status> status(budget, 1); status.zero();
      a::Buffer<a::u32> words(budget, K), positions(budget, K), blocked(budget, K);
      a::Buffer<std::int32_t> residual(budget, T);
      a::Buffer<a::u32> lower(budget, N), upper(budget, N), missing(budget, N);
      a::Buffer<a::u64> allowed(budget, domain.mask_words);
      a::Buffer<a::u32> witness_lower(budget, N), witness_upper(budget, N), witness_missing(budget, N);
      a::Buffer<a::u64> witness_allowed(budget, domain.mask_words);
      const a::u32 support_words = (F + 63ull) / 64;
      a::Buffer<a::u64> support(budget, a::multiply(source.feature.size(), support_words));
      a::Buffer<a::u64> active_support(budget, support_words);
      a::Buffer<float> range_lower(budget, K), range_upper(budget, K);
      a::Buffer<a::u32> minimum(budget, source.feature.size()), maximum(budget, source.feature.size());
      a::Buffer<a::u32> audit_stack(budget, a::multiply(source.feature.size(), 3));
      a::Buffer<a::u32> walk_shape(budget, 2);
      a::EngineView e{};
      e.source = source_storage.view(source); e.domain = domain_storage.view(domain);
      e.output_classes = C;
      e.draft_words = words.data; e.draft_positions = positions.data; e.blocked = blocked.data;
      e.draft_residual = residual.data;
      e.draft_region = {lower.data, upper.data, missing.data, allowed.data};
      e.draft_witness = {witness_lower.data, witness_upper.data, witness_missing.data, witness_allowed.data};
      e.support = support.data; e.active_support = active_support.data; e.support_words = support_words;
      e.minimum = minimum.data; e.maximum = maximum.data;
      e.range_lower = range_lower.data; e.range_upper = range_upper.data;
      e.qualified_gap = gap_enabled; e.status = status.data;
      a::bind(e, *states, *nodes);
      publish("source_upload");
      a::subtree_extrema<<<1, 1>>>(e.source, minimum.data, maximum.data, audit_stack.data,
                                  support.data, support_words, walk_shape.data);
      a::synchronize();
      const auto source_walk_shape = walk_shape.download(2);
      const bool effort_enabled = gap_enabled && (!options.refinement_visit_budget || *options.refinement_visit_budget != 0);
      const bool cover_enabled = gap_enabled && (!options.cover_visit_budget || *options.cover_visit_budget != 0);
      e.refinement_stack_capacity = (effort_enabled || cover_enabled) ? source_walk_shape[0] : 0;
      e.refinement_maximum_visits = (effort_enabled || cover_enabled) ? source_walk_shape[1] : 0;
      const auto cover_maximum_visits = std::uint32_t(std::min<std::uint64_t>(UINT32_MAX,std::uint64_t(e.refinement_maximum_visits)*2));
      statistics["source_traversal_frontier_words"] = source_walk_shape[0];
      statistics["source_original_node_visits"] = source_walk_shape[1];
      statistics["refinement_authority_available"] = gap_enabled;
      statistics["refinement_effective_enabled"] = effort_enabled && e.refinement_maximum_visits > 0;
      statistics["refinement_stack_capacity"] = e.refinement_stack_capacity;
      statistics["refinement_maximum_visits"] = e.refinement_maximum_visits;
      statistics["cover_effective_enabled"] = cover_enabled && cover_maximum_visits > 0;
      statistics["cover_maximum_visits"] = cover_maximum_visits;
      a::initialize<<<1, 1>>>(e); a::synchronize();
      auto h = status.download(1).front();
      a::require(!h.error, "resident adaptive initialization refusal");
      a::eta::Reporter eta_reporter;
      const auto construction_started=std::chrono::steady_clock::now();
      double eta_sampling_seconds=0,progress_seconds=0;
      bool eta_attempted=false;
      const bool eta_enabled=options.completion_estimate_enabled&&bool(options.progress);
      const bool eta_policy_compatible=options.split_policy=="source_order"&&
        options.proof_module_directory.empty()&&options.proof_module_request.empty()&&
        (!gap_enabled||((options.refinement_visit_budget.has_value()||!effort_enabled)&&
                       options.cover_visit_budget.has_value()&&*options.cover_visit_budget==0));
      const std::string eta_policy_reason="The root-path sampler requires source_order, fixed effective refinement, zero cover effort and no hotloaded proof strategies; dynamic/alternative regimes need a matching frontier sampler.";
      const auto eta_refinement=effort_enabled?
        std::min(options.refinement_visit_budget.value_or(0),e.refinement_maximum_visits):0;
      if(!eta_enabled)eta_reporter.refuse("disabled","Construction ETA reporting is disabled or no progress callback is installed.");
      auto report_eta=[&](const a::frontier::Snapshot& snapshot) {
        const auto completed=committed_decisions(snapshot.status);
        auto elapsed=std::max(0.,std::chrono::duration<double>(std::chrono::steady_clock::now()-construction_started).count()-eta_sampling_seconds-progress_seconds);
        const bool compatible=eta_policy_compatible&&snapshot.refinement_visit_budget==eta_refinement&&snapshot.cover_visit_budget==0;
        if(eta_enabled&&compatible&&!eta_attempted&&!snapshot.status.complete&&!snapshot.stopped&&
           elapsed>=options.completion_estimate_start_seconds) {
          eta_attempted=true;
          auto requested=options.completion_estimate;requested.refinement_visit_budget=eta_refinement;
          const auto sampling_started=std::chrono::steady_clock::now();
          try {eta_reporter.set_sample(sample_work(budget,e,requested,check_gate,options.stop_requested));}
          catch(const a::AllocationRefusal&) {eta_reporter.refuse("sampling_allocation_refused","Reporting scratch did not fit; construction ownership and search remain unchanged.");}
          eta_sampling_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-sampling_started).count();
        }
        eta_reporter.observe(completed,elapsed);
        auto report=eta_reporter.report(completed,snapshot.status.complete!=0,
          eta_enabled&&compatible,!eta_enabled?"ETA reporting is disabled.":eta_policy_reason);
        if(!snapshot.status.complete&&(snapshot.stopped||snapshot.status.error)) {
          report["available"]=false;report["seconds"]=nullptr;report["lower_seconds"]=nullptr;report["upper_seconds"]=nullptr;
          report["status"]=snapshot.stopped?"stopped":"construction_refused";
          report["reason"]="Construction is stopped or refused; no continuing-run forecast is published.";
        }
        report.update({{"source_model_sha256",source.identity},{"native_library_sha256",native.library_sha256},
          {"qualified_native_gap_binding",gate_binding},{"qualified_native_gap_pruning_enabled",gap_enabled},
          {"split_policy",options.split_policy},{"refinement_visit_budget",eta_refinement},{"cover_visit_budget",0},
          {"at_elapsed_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()},
          {"construction_observation_seconds_excluding_reporting",elapsed},
          {"sampling_attempted",eta_attempted},{"sampling_seconds",eta_sampling_seconds},
          {"sample_seed",options.completion_estimate.seed},{"sample_requested_paths",options.completion_estimate.paths},
          {"sample_maximum_seconds",options.completion_estimate.maximum_seconds},
          {"sample_maximum_decisions",options.completion_estimate.maximum_decisions?
            options.completion_estimate.maximum_decisions:e.source.nodes+1},
          {"source_root_sampling",true},{"live_frontier_sampling",false},
          {"native_queries_for_sampling",0},{"reporting_changes_class_authority",false}});
        statistics["eta"]=std::move(report);
      };
      auto report_frontier = [&](const a::frontier::Snapshot& snapshot) {
        const auto reporting_started=std::chrono::steady_clock::now();
        const auto sampling_before=eta_sampling_seconds;
        report_eta(snapshot);
        statistics["checkpoint"]=snapshot.checkpoint_status;
        statistics["resumed_from_checkpoint"]=snapshot.resumed;
        statistics["stopped_at_idle_boundary"]=snapshot.stopped;
        statistics["proof_modules"]=snapshot.proof_module_status;
        const auto& coverage=snapshot.grid_coverage;
        statistics["grid_coverage"]={{"available",coverage.available},
          {"samples",snapshot.grid_coverage_samples},{"at_state_creations",snapshot.grid_coverage_state_creations},
          {"covered_fraction_lower",coverage.available?json(coverage.lower):json(nullptr)},
          {"covered_fraction_upper",coverage.available?json(coverage.upper):json(nullptr)},
          {"measurement_seconds",coverage.seconds},{"scratch_GPU_bytes",coverage.scratch_bytes},
          {"catalog_GPU_bytes",coverage.catalog_bytes},
          {"waves",coverage.waves},{"error",coverage.error},{"timed_out",coverage.timed_out},
          {"allocation_refused",coverage.allocation_refused},
          {"scope","outward-rounded bounds on settled source-grid coverage; not runtime progress"}};
        auto& grid_report=statistics["grid_coverage"];
        grid_report["settled_cells_lower_decimal"]=nullptr;
        grid_report["settled_cells_upper_decimal"]=nullptr;
        grid_report["remaining_cells_lower_decimal"]=nullptr;
        grid_report["remaining_cells_upper_decimal"]=nullptr;
        grid_report["cell_counts_available"]=false;
        grid_report["completion_eta_seconds"]=statistics["eta"]["seconds"];
        grid_report["eta_status"]=statistics["eta"]["status"];
        if(coverage.available) {
          try {
            const auto cells=class_grid_report::bounds(grid_total,coverage.lower,coverage.upper);
            grid_report["settled_cells_lower_decimal"]=cells.settled_lower;
            grid_report["settled_cells_upper_decimal"]=cells.settled_upper;
            grid_report["remaining_cells_lower_decimal"]=cells.remaining_lower;
            grid_report["remaining_cells_upper_decimal"]=cells.remaining_upper;
            grid_report["cell_counts_available"]=true;
          } catch(const std::invalid_argument& error) {
            grid_report["cell_count_error"]=error.what();
          }
        }
        statistics["counts"] = counters(snapshot.status);
        statistics["priority_split_changes"] = snapshot.priority_split_changes;
        statistics["oldest_selected_jobs"] = snapshot.oldest_selected_jobs;
        statistics["state_capacity"] = states->capacity; statistics["node_capacity"] = nodes->capacity;
        statistics["state_growths"] = snapshot.state_growths;
        statistics["node_growths"] = snapshot.node_growths;
        statistics["pending_request"] = snapshot.status.request;
        statistics["frontier_batch_capacity"] = snapshot.active_batch_capacity;
        statistics["frontier"] = {{"ready_states", snapshot.ready_states},
          {"completion_events", snapshot.completion_events}, {"native_pending", snapshot.native_pending},
          {"batch_jobs", snapshot.batch_jobs}, {"native_batches", snapshot.native_batches},
          {"maximum_batch_jobs", snapshot.maximum_batch_jobs},
          {"active_batch_capacity", snapshot.active_batch_capacity},
          {"allocated_batch_capacity", snapshot.allocated_batch_capacity},
          {"batch_adjustments", snapshot.batch_adjustments},
          {"batch_growth_refusals", snapshot.batch_growth_refusals},
          {"batch_memory_backoffs", snapshot.batch_memory_backoffs},
          {"batch_returned_jobs", snapshot.batch_returned_jobs},
          {"autotune_samples", snapshot.autotune_samples},
          {"autotune_probes", snapshot.autotune_probes},
          {"autotune_upward_probes", snapshot.autotune_upward_probes},
          {"autotune_downward_probes", snapshot.autotune_downward_probes},
          {"admission_threads", snapshot.admission_threads},
          {"maximum_admission_threads", snapshot.maximum_admission_threads},
          {"best_admission_threads", snapshot.best_admission_threads},
          {"admission_thread_adjustments", snapshot.admission_thread_adjustments},
          {"admission_thread_probes", snapshot.admission_thread_probes},
          {"draft_threads", snapshot.draft_threads},
          {"maximum_draft_threads", snapshot.maximum_draft_threads},
          {"best_draft_threads", snapshot.best_draft_threads},
          {"draft_thread_adjustments", snapshot.draft_thread_adjustments},
          {"draft_thread_probes", snapshot.draft_thread_probes},
          {"completed_cache_limit", snapshot.completed_cache_limit},
          {"refinement_visit_budget", snapshot.refinement_visit_budget},
          {"cache_limit_adjustments", snapshot.cache_limit_adjustments},
          {"cache_limit_probes", snapshot.cache_limit_probes},
          {"cache_policy_evictions", snapshot.cache_policy_evictions},
          {"refinement_adjustments", snapshot.refinement_adjustments},
          {"refinement_probes", snapshot.refinement_probes},
          {"refinement_attempts", snapshot.refinement_attempts},
          {"refinement_visits", snapshot.refinement_visits},
          {"refinement_tightened_roots", snapshot.refinement_tightened_roots},
          {"refinement_rejected_roots", snapshot.refinement_rejected_roots},
          {"refinement_fallback_frontiers", snapshot.refinement_fallback_frontiers},
          {"refinement_additional_prunes", snapshot.refinement_additional_prunes},
          {"cover_visit_budget", snapshot.cover_visit_budget},
          {"cover_adjustments", snapshot.cover_adjustments},
          {"cover_probes", snapshot.cover_probes},
          {"cover_attempts", snapshot.cover_attempts},
          {"cover_visits", snapshot.cover_visits},
          {"cover_feasible_cases", snapshot.cover_feasible_cases},
          {"cover_certified_cases", snapshot.cover_certified_cases},
          {"cover_additional_prunes", snapshot.cover_additional_prunes},
          {"cover_failures", snapshot.cover_failures},
          {"retune_backoff_batches", snapshot.retune_backoff_batches},
          {"retune_cooldown_batches", snapshot.retune_cooldown_batches},
          {"retune_drift_resets", snapshot.retune_drift_resets},
          {"autotune_last_work_equivalents_per_second", snapshot.autotune_last_work_equivalents_per_second},
          {"autotune_best_work_equivalents_per_second", snapshot.autotune_best_work_equivalents_per_second},
          {"state_growth_required_capacity", snapshot.state_growth_required_capacity},
          {"state_growth_preferred_capacity", snapshot.state_growth_preferred_capacity},
          {"state_growth_selected_capacity", snapshot.state_growth_selected_capacity},
          {"state_growth_additional_bytes", snapshot.state_growth_additional_bytes},
          {"node_growth_required_capacity", snapshot.node_growth_required_capacity},
          {"node_growth_preferred_capacity", snapshot.node_growth_preferred_capacity},
          {"node_growth_selected_capacity", snapshot.node_growth_selected_capacity},
          {"node_growth_additional_bytes", snapshot.node_growth_additional_bytes},
          {"growth_allocation_refusals", snapshot.growth_allocation_refusals},
          {"growth_transaction_rollbacks", snapshot.growth_transaction_rollbacks},
          {"autotune_best_capacity", snapshot.autotune_best_capacity},
          {"autotune_last_jobs_per_second", snapshot.autotune_last_jobs_per_second},
          {"autotune_best_jobs_per_second", snapshot.autotune_best_jobs_per_second},
          {"autotune_last_batch_seconds", snapshot.autotune_last_batch_seconds},
          {"autotune_timed_jobs", snapshot.autotune_timed_jobs},
          {"autotune_excluded_growth_batches", snapshot.autotune_excluded_growth_batches},
          {"draft_batches", snapshot.draft_batches}, {"resolved_edges", snapshot.resolved_edges},
          {"pending_cache_merges", snapshot.pending_cache_merges},
          {"completed_cache_edges", snapshot.completed_cache_edges},
          {"completed_cached_states", snapshot.completed_cached_states},
          {"completion_growth_retries", snapshot.completion_growth_retries}};
        publish("adaptive_construction");
        progress_seconds+=std::max(0.,std::chrono::duration<double>(std::chrono::steady_clock::now()-reporting_started).count()-(eta_sampling_seconds-sampling_before));
      };
      a::frontier::Limits limits;
      limits.checkpoint_path=options.checkpoint_path;limits.resume_from=options.resume_from;
      limits.checkpoint_interval_seconds=options.checkpoint_interval_seconds;
      limits.checkpoint_on_completion=options.checkpoint_on_completion;
      limits.checkpoint_host_byte_budget=options.checkpoint_host_byte_budget;
      limits.stop_requested=options.stop_requested;
      limits.checkpoint_requested=options.checkpoint_requested;
      limits.checkpoint_publication=options.checkpoint_publication;
      limits.proof_module_directory=options.proof_module_directory;
      limits.proof_module_request=options.proof_module_request;
      // Stable construction contract only. Process-local native authority and
      // plugin handles are created afresh before any restored state is used.
      limits.checkpoint_identity={{"format","adaptive-frontier-checkpoint-1"},
        {"source_sha256",source.identity},{"features",F},{"classes",K},{"trees",T},
        {"numeric_features",N},{"source_nodes",source.feature.size()},
        {"allow_nan",options.domain.allow_nan},{"one_hot_groups",options.domain.one_hot_groups},
        {"native_objective",native.objective},{"native_library_sha256",native.library_sha256},
        {"qualified_gap",gap_enabled},{"native_rule_contract","native-softprob-gap-1"}};
      // Preserve the ordinary checkpoint schema while preventing a calibrated
      // resume from reusing labels produced with another rule or FP32 threshold.
      if (calibrated) {
        limits.checkpoint_identity["native_rule_contract"] = "native-positive-probability-threshold-1";
        limits.checkpoint_identity["decision_rule"] = decision_rule;
      }
      limits.split_selection=selection_view;
      limits.oldest_ready_jobs=options.oldest_ready_jobs;
      limits.batch_capacity = initial_batch; limits.max_states = options.max_states;
      limits.max_nodes = options.max_nodes; limits.max_expansions = options.max_expansions;
      limits.dynamic_batching = !options.batch_size; limits.maximum_batch_capacity = options.max_batch_size;
      limits.admission_threads = options.admission_threads; limits.draft_threads = options.draft_threads;
      limits.dynamic_cache = !options.completed_cache_limit.has_value();
      limits.completed_cache_limit = options.completed_cache_limit.value_or(UINT32_MAX);
      limits.dynamic_refinement = effort_enabled && !options.refinement_visit_budget.has_value();
      limits.refinement_visit_budget = effort_enabled ? std::min(options.refinement_visit_budget.value_or(0), e.refinement_maximum_visits) : 0;
      statistics["refinement_resolved_fixed_visit_budget"] = options.refinement_visit_budget ? json(limits.refinement_visit_budget) : json(nullptr);
      limits.dynamic_cover = cover_enabled && !options.cover_visit_budget.has_value();
      limits.cover_visit_budget = cover_enabled ? std::min(options.cover_visit_budget.value_or(0),cover_maximum_visits) : 0;
      statistics["cover_resolved_fixed_visit_budget"] = options.cover_visit_budget ? json(limits.cover_visit_budget) : json(nullptr);
      auto final_frontier = a::frontier::run(e, states, nodes, budget, limits,
          native.predict, native_direct_classes, report_frontier, check_gate,
          options.progress?&grid_catalog:nullptr,&status);
      h = final_frontier.status;
      state_growths = final_frontier.state_growths; node_growths = final_frontier.node_growths;
      margin_rows = final_frontier.native_margin_rows; public_rows = final_frontier.native_public_rows;
      report_frontier(final_frontier);
      if(final_frontier.stopped)throw ConversionFailure("conversion stopped at an idle checkpoint boundary",statistics);
      a::require(h.complete && !h.error && h.root < h.nodes, "resident adaptive final graph refusal");
      check_gate();
      statistics["counts"] = counters(h); statistics["state_growths"] = state_growths;
      statistics["node_growths"] = node_growths;
      statistics["construction_nodes_before_root_collection"] = h.nodes;
      a::Buffer<a::u32> remap(budget, h.nodes);
      a::collect_root<<<1, 1>>>(e, remap.data, a::u32(h.nodes)); a::synchronize();
      h = status.download(1).front();
      a::require(!h.error, "adaptive root collection topology refusal");
      statistics["root_only_collection"] = true;
      graph = nodes->nodes.download(h.nodes); root = h.root;
      publish("canonical_transport");
    }
    ConvertedModel result;
    result.canonical_bytes = encode(graph, root, F, C, source.identity);
    result.runtime = class_runtime::Runtime::load(result.canonical_bytes, sha256(result.canonical_bytes), source.identity);
    if (result.runtime->metadata().compact_resident) result.compact_bytes = result.runtime->compact_bytes();
    if (options.residency != class_runtime::Residency::dual) result.runtime->retain(options.residency);
    const auto& metadata = result.runtime->metadata();
    statistics["complete"] = true; statistics["CUDA_executed"] = true;
    statistics["root"] = root; statistics["nodes"] = graph.size();
    statistics["canonical_bytes"] = result.canonical_bytes.size();
    statistics["compact_bytes"] = result.compact_bytes.size();
    statistics["canonical_sha256"] = metadata.canonical_sha256;
    statistics["compact_sha256"] = metadata.compact_sha256;
    statistics["native_margin_query_rows"] = margin_rows; statistics["native_public_query_rows"] = public_rows;
    statistics["native_margin_word_mismatches"] = 0;
    statistics["retained_graph_device_bytes"] = metadata.device_bytes;
    statistics["full_cross_layout_words_equal"] = metadata.cross_layout_validated;
    statistics["deployment_inference_engine"] = "shared class_runtime";
    statistics["construction_graph_released_before_Runtime_load"] = true;
    statistics["Runtime_internal_sort_scratch_in_converter_counter"] = false;
    statistics["native_public_terminal_contract"] = calibrated ? "native_positive_probability_threshold_direct_class_index"
        : (native_direct_classes ? "direct_class_index" : "softprob_firstargmax");
    statistics["exact_native_class_conversion"] = false;
    statistics["accepted_native_class_root"] = false;
    statistics["acceptance_scope"] = composed
        ? "Complete composed adaptive candidate; ordered native base-margin words and direct composed classes; independent root acceptance remains separate."
        : calibrated
          ? "Complete calibrated adaptive candidate; ordered native terminal words and native full-round probability-threshold classes on source-constant regions; independent root acceptance remains separate."
          : "Complete adaptive candidate; ordered native terminal words and source-specific authorized gap rules; independent root acceptance remains separate.";
    statistics["dataset_payload_read"] = false; statistics["FIT_VALID_TEST_labels_used"] = false;
    statistics["TEST_read"] = false; statistics["global_minimum_claim"] = false;
    publish("complete"); result.metrics = std::move(statistics); return result;
  } catch (const std::exception& error) {
    statistics["complete"] = false; statistics["failure"] = error.what();
    statistics["accepted_native_class_root"] = false;
    statistics["peak_owned_GPU_bytes_excluding_native_Runtime_and_driver"] = budget.peak;
    throw ConversionFailure(error.what(), std::move(statistics));
  }
}
ConvertedModel convert_model(const ConversionSource& input, const NativeOracle& native,
                             const ConversionOptions& options) {
  return convert_prepared_source(input, native, options);
}

ConvertedModel convert_composed_model(const ComposedConversionSource& input,
    const ComposedNativeOracle& native, const ConversionOptions& options) {
  try {
    a::require(native.base_margins && native.predict_classes,
               "composed native margin/class callbacks required");
    a::require(options.checkpoint_path.empty() && options.resume_from.empty() &&
                   options.checkpoint_interval_seconds == 0 && !options.checkpoint_requested &&
                   !options.checkpoint_publication && options.proof_module_directory.empty() &&
                   options.proof_module_request.empty(),
               "composed conversion currently requires in-memory execution without checkpoints or proof modules");
    a::require(!input.bases.empty(), "composed conversion requires base models");
    auto parse = [](const ConversionSource& declaration) {
      a::require(declaration.features && declaration.classes >= 2 &&
                     declaration.model_json.size() <= 4ull * 1024 * 1024 * 1024,
                 "composed source declaration/transport extent");
      auto parsed = class_conversion_native::read_source_bytes(std::string(declaration.model_json));
      a::require(parsed.features == declaration.features && parsed.outputs == declaration.classes,
                 "composed source declared shape differs");
      a::require(declaration.expected_source_sha256.empty() ||
                     declaration.expected_source_sha256 == parsed.identity,
                 "composed source identity differs");
      return parsed;
    };
    std::vector<SourceData> bases;
    bases.reserve(input.bases.size());
    for (const auto& declaration : input.bases) bases.push_back(parse(declaration));
    auto meta = parse(input.meta);
    auto source = class_conversion_native::compose_source_forests(bases, meta, input.meta_input_feature_indices);
    a::require(input.expected_source_sha256.empty() || input.expected_source_sha256 == source.identity,
               "composed descriptor identity differs");
    a::require(native.source_sha256.empty() || native.source_sha256 == source.identity,
               "composed oracle identity differs");
    json provenance = {{"composition_source", json::parse(source.bytes)},
      {"base_models", bases.size()}, {"meta_model_sha256", meta.identity},
      {"meta_features", meta.features}, {"meta_source_trees", meta.roots.size()},
      {"meta_predicates_add_raw_boundaries", source.structural_guard_trees != 0},
      {"checkpoint_support", false}, {"class_proof_modules_enabled", false}};
    if (!input.meta_input_feature_indices.empty()) {
      provenance["meta_input_mapping"] = {{"version", 1}, {"raw_feature_indices", input.meta_input_feature_indices}};
      provenance["native_probability_columns"] = source.native_margin_outputs;
      provenance["native_meta_input_columns"] = meta.features;
      provenance["structural_guard_policy"] = "packed_raw_predicate_bits24_v1";
    }
    // Only the private prepared-source entrypoint consumes this authority shape
    // binding. The native callback still exposes only original base margins.
    // No synthetic ordinary XGBoost JSON or native booster is manufactured.
    ConversionSource shape{{}, a::u32(source.features), a::u32(source.outputs), source.identity};
    NativeOracle bridge;
    bridge.features = shape.features; bridge.classes = shape.classes;
    bridge.objective = "multi:softmax";
    bridge.library_sha256 = native.library_sha256;
    bridge.source_sha256 = source.identity;
    bridge.predict = [&native](const float* values, std::uint64_t rows, bool margins) {
      return margins ? native.base_margins(values, rows) : native.predict_classes(values, rows);
    };
    // Structural normalization and exact caching remain active; all class-gap
    // authority is explicitly absent for this different deterministic transform.
    auto conservative = options;
    conservative.refinement_visit_budget = 0;
    conservative.cover_visit_budget = 0;
    return convert_prepared_source(shape, bridge, conservative, std::move(source),
                                   a::u32(meta.outputs), provenance);
  } catch (const ConversionFailure&) {
    throw;
  } catch (const std::exception& error) {
    throw ConversionFailure(error.what(), {{"format", "resident-adaptive-conversion-1"},
      {"composed_source", true}, {"complete", false}, {"accepted_native_class_root", false},
      {"CPU_predictions", false}, {"input_file_reads", 0}, {"output_file_writes", 0},
      {"failure", error.what()}});
  }
}
nlohmann::json estimate_model_work(const ConversionSource& input,const NativeOracle& native,
    const ConversionOptions& options,const WorkEstimateOptions& requested) {
  const auto started=std::chrono::steady_clock::now();
  json statistics={{"format","resident-adaptive-work-estimate-1"},{"reporting_only",true},
    {"complete",false},{"all_paths_finished",false},{"model_conversion_performed",false},
    {"accepted_native_class_root",false},{"CPU_predictions",false},{"input_file_reads",0},
    {"output_file_writes",0},{"native_margin_query_rows",0},{"native_public_query_rows",0},
    {"live_arena_accessed",false},{"cache_mutations",false},{"completion_eta_seconds",nullptr},
    {"eta_status","unavailable: sampled unshared work has not been matched to live throughput"}};
  a::Budget budget{options.gpu_byte_budget};
  auto publish=[&](const char* phase) {
    statistics["phase"]=phase;statistics["owned_GPU_bytes"]=budget.used;
    statistics["peak_owned_GPU_bytes_excluding_native_Runtime_and_driver"]=budget.peak;
    statistics["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    if(options.progress)options.progress(statistics);
  };
  try {
    ConversionGuard guard;
    a::require(requested.paths&&requested.decisions_per_chunk&&std::isfinite(requested.maximum_seconds)&&
        requested.maximum_seconds>0&&budget.limit,"work estimate resource declaration");
    a::require(input.features&&input.classes>=2&&input.model_json.size()<=4ull*1024*1024*1024,
        "work estimate source declaration");
    auto source=class_conversion_native::read_source_bytes(std::string(input.model_json));
    a::require(source.features==input.features&&source.outputs==input.classes&&
        native.features==input.features&&native.classes==input.classes,"work estimate source/oracle shape differs");
    a::require(input.expected_source_sha256.empty()||input.expected_source_sha256==source.identity,
        "work estimate source identity differs");
    a::require(native.source_sha256.empty()||native.source_sha256==source.identity,
        "work estimate native source identity differs");
    a::require(native.objective==class_conversion_native::native_objective_name(source.objective)&&
        native.library_sha256==native_softprob_gap::library_sha,"work estimate native binding differs");
    a::require(source.roots.size()<=UINT32_MAX&&source.feature.size()<=INT32_MAX,
        "work estimate source index representation");
    const auto decision_rule=calibrated_decision_rule(native);
    const bool calibrated=!decision_rule.is_null();
    if(calibrated) {
      statistics["decision_rule"]=decision_rule;
      statistics["calibrated_probability_threshold"]=true;
      statistics["class_pruning_disabled_reason"]="native probability threshold requires source-constant terminal qualification; argmax gap rules are inapplicable";
    }
    const auto F=a::u32(source.features),K=a::u32(source.outputs);
    auto domain=d::prepare_domain_metadata(F,options.domain.one_hot_groups,options.domain.allow_nan);
    const bool gap_enabled=!calibrated&&source.identity==native_softprob_gap::source_sha&&K==7&&
        native.softprob_gap_gate&&native.softprob_gap_gate->enabled()&&
        native.source_sha256==source.identity&&
        native.softprob_gap_gate->matches_source(source.identity,K,native.objective,native.library_sha256);
    const auto gate_binding=gap_enabled?native.softprob_gap_gate->binding_sha256():std::string{};
    auto check_gate=[&] {
      if(gap_enabled)a::require(native.softprob_gap_gate->enabled()&&
          native.softprob_gap_gate->binding_sha256()==gate_binding,"work estimate same-process native gate lost");
    };
    statistics.update({{"source_model_sha256",source.identity},{"native_library_sha256",native.library_sha256},
      {"features",F},{"classes",K},{"source_trees",source.roots.size()},{"source_nodes",source.feature.size()},
      {"native_objective",native.objective},{"qualified_native_gap_pruning_enabled",gap_enabled},
      {"qualified_native_gap_binding",gate_binding},{"split_policy","source_order"},
      {"requested_conversion_split_policy",options.split_policy},{"seed",requested.seed},
      {"paths_requested",requested.paths},{"maximum_seconds",requested.maximum_seconds},
      {"maximum_seconds_scope","sampling at synchronized chunk boundaries; setup excluded"},
      {"refinement_requested_visit_budget",requested.refinement_visit_budget},
      {"cover_visit_budget",0},
      {"cover_scope","disabled in this frozen unshared-work scenario; live cover pruning can reduce work"},
      {"domain",{{"finite_FP32",true},{"allow_nan",options.domain.allow_nan},
        {"one_hot_groups",options.domain.one_hot_groups},{"excludes_infinities",true}}},
      {"scope","Monte Carlo observations of the unshared source-order construction tree; no runtime confidence interval"},
      {"live_expansion_counter_relation","live committed expansion edges equal twice committed split vertices; excludes uncommitted draft retries"}});
    SourceStorage source_storage(budget,source);DomainStorage domain_storage(budget,domain);
    const a::u32 support_words=(a::u64(F)+63)/64;
    a::Buffer<a::u64> support(budget,a::multiply(source.feature.size(),support_words));
    a::Buffer<a::u32> minimum(budget,source.feature.size()),maximum(budget,source.feature.size());
    a::Buffer<a::u32> audit_stack(budget,a::multiply(source.feature.size(),3)),walk_shape(budget,2);
    a::EngineView e{};e.source=source_storage.view(source);e.domain=domain_storage.view(domain);
    e.support=support.data;e.support_words=support_words;e.minimum=minimum.data;e.maximum=maximum.data;
    e.qualified_gap=gap_enabled;
    a::subtree_extrema<<<1,1>>>(e.source,minimum.data,maximum.data,audit_stack.data,
        support.data,support_words,walk_shape.data);a::synchronize();
    const auto shape=walk_shape.download(2);
    e.refinement_stack_capacity=gap_enabled&&requested.refinement_visit_budget?shape[0]:0;
    e.refinement_maximum_visits=gap_enabled?shape[1]:0;
    const auto maximum_decisions=requested.maximum_decisions?requested.maximum_decisions:
        a::u32(source.feature.size()+1);
    a::work::Options sampled{requested.paths,requested.seed,maximum_decisions,requested.refinement_visit_budget};
    a::work::Storage paths(budget,e,sampled);a::Buffer<a::work::Summary> output(budget,1);
    statistics["maximum_decisions_effective"]=maximum_decisions;
    statistics["maximum_decisions_default_scope"]="source node count plus one; decisions include the terminal/proof visit";
    statistics["refinement_effective_visit_budget"]=paths.options.refinement_visit_budget;
    statistics["refinement_stack_capacity"]=paths.stack_capacity;
    statistics["decisions_per_chunk"]=requested.decisions_per_chunk;
    publish("work_estimate_source_upload");check_gate();
    const auto sampling_started=std::chrono::steady_clock::now();paths.initialize();
    auto metric=[](const a::work::Metric& m) {
      const bool valid=m.count&&!m.numeric_overflow;
      return json{{"count",m.count},{"invalid_values",m.invalid_values},{"numeric_overflow",m.numeric_overflow!=0},
        {"mean",valid?json(m.mean):json(nullptr)},
        {"standard_error",valid&&m.standard_error_available?json(m.standard_error):json(nullptr)},
        {"minimum",valid?json(m.minimum):json(nullptr)},{"maximum",valid?json(m.maximum):json(nullptr)},
        {"max_fraction_of_total",valid&&m.max_fraction_available?json(m.max_fraction_of_total):json(nullptr)}};
    };
    auto metrics=[&](const auto& m) {
      return json{{"weighted_decisions",metric(m.decisions)},
        {"weighted_expanding_vertices",metric(m.expanding_vertices)},
        {"weighted_class_prunes",metric(m.class_prunes)},
        {"weighted_native_terminals",metric(m.native_terminals)},
        {"weighted_refinement_visits",metric(m.refinement_visits)}};
    };
    auto report=[&](const a::work::Summary& s) {
      statistics["sampling"]={{"paths",s.paths},{"done",s.done},{"censored",s.censored},
        {"errors",s.errors},{"overflows",s.overflows},{"invalid_status",s.invalid_status}};
      statistics["all_paths_finished"]=s.paths&&s.done==s.paths&&!s.censored&&!s.errors&&!s.overflows&&!s.invalid_status;
      statistics["valid_unshared_total_work_estimate"]=s.valid_total!=0;
      statistics["completed_means_selection_conditioned"]=s.completed_selection_conditioned!=0;
      statistics["observed_work"]=metrics(s);
      statistics["completed_path_work"]=metrics(s.completed);
      statistics["partial_censored_path_work"]=metrics(s.partial_censored);
      statistics["observed_work_scope"]=s.valid_total?
        "finished-path unshared-work sampling statistics; no runtime confidence interval":
        "partial observations; censored/error paths prevent a valid total-work estimate; completed subset may be selection-conditioned";
      statistics["depth"]={{"count",s.depth.count},{"minimum",s.depth.minimum},{"maximum",s.depth.maximum},{"mean",s.depth.mean}};
      statistics["sampling_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-sampling_started).count();
    };
    a::u64 decision_rounds=0;bool timed_out=false;
    for(;;) {
      check_gate();const auto summary=a::work::summarize(paths.device_results(),requested.paths,output);
      report(summary);publish("work_estimate_sampling");
      if(summary.done==requested.paths||!summary.decisions.count||decision_rounds>=maximum_decisions)break;
      if(std::chrono::duration<double>(std::chrono::steady_clock::now()-sampling_started).count()>=requested.maximum_seconds) {
        timed_out=true;break;
      }
      const auto chunk=a::u32(std::min<a::u64>(requested.decisions_per_chunk,maximum_decisions-decision_rounds));
      paths.advance(chunk);decision_rounds+=chunk;
    }
    statistics["decision_rounds"]=decision_rounds;statistics["timed_out"]=timed_out;
    statistics["complete"]=true;publish("work_estimate_complete");return statistics;
  } catch(const ConversionFailure&) { throw; }
  catch(const std::exception& error) {
    statistics["complete"]=false;statistics["failure"]=error.what();
    statistics["peak_owned_GPU_bytes_excluding_native_Runtime_and_driver"]=budget.peak;
    statistics["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    throw ConversionFailure(error.what(),std::move(statistics));
  }
}
} // namespace class_study
