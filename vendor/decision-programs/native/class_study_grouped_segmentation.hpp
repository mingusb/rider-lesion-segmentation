#pragma once
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace class_study::grouped_segmentation {
// Patient-grouped nested evaluation of the maintained native XGBoost trainer.
// Descriptor: grouped-segmentation-dataset-1, rows, features, classes=2,
// values_path (row-major FP32), train_eligible_rows (exact count), labels_path/patient_ids_path/scan_ids_path/
// train_eligible_path (uint32), domain_description, and scans containing
// patient_id, scan_id and truth_positive_outside_rows. Every declared row is
// evaluated; train_eligible restricts fitting only. Patient/scan IDs need not
// be consecutive. Paths resolve relative to the descriptor's directory in the
// calling frontend. Pixels outside the declared ROI contribute supplied FN.
// Optional plan outer_model_summaries=true records outer-winner structural
// split/gain/Hessian/threshold metadata after scoring, with separate per-use
// raw/teacher input mappings. Default off; no model files, FITs or predictions.
// These are training statistics, not heldout feature contributions or SHAP.
// Optional descriptor missing_feature_indices is a nonempty array of unique
// source columns allowed to contain NaN. All infinities and undeclared NaNs
// remain invalid; configured reference mask/probability columns cannot be
// listed. Omission preserves the finite-only audit. Native training/prediction
// retain their existing NaN missing-branch semantics; no imputation is applied.
// Plan: native_library_path/native_library_sha256, base_hyperparameters and
// meta_hyperparameters arrays, optional inner_folds=3, oof_folds=3,
// gpu_byte_budget and host_byte_budget. Optional outer_folds selects grouped
// K-fold evaluation; omission retains LOPO. outer_patient_ids restricts LOPO
// to a pilot subset and is mutually exclusive with outer_folds. Every level
// keeps all scans from a patient together and must retain enough FIT groups
// for the nested patient-grouped OOF responses.
// include_all_fit_positives=true retains every positive row belonging to FIT
// patients plus existing eligible negatives; it never samples evaluation rows.
// This requires exact per-scan truth_positive_rows for memory preflight.
// fit_class_balanced=true assigns inverse-frequency row weights computed from
// each FIT matrix, normalized to mean one (single-class FIT weights one).
// fit_sampling_weights="inverse_retention" instead restores full FIT class
// prevalence after sampling, using audit-verified counts of only FIT patients.
// It is mutually exclusive with fit_class_balanced, and defaults to none.
// fit_group_weights="equal_patient" additionally assigns every FIT patient the
// same total weight n/G, using original row IDs in raw and meta FIT matrices.
// With inverse_retention, each retained class-c row of patient g receives
// (n/G)*(full_gc/full_g)/retained_gc; missing retained support is refused.
// With class balancing it receives n/(G*present_classes_g*retained_gc), so a
// single-class patient assigns all its mass to its present class. Otherwise
// every retained patient-g row receives n/(G*retained_g). Mean row weight is one.
// Full/retained counts use only FIT patients; outside-ROI positives do not enter
// FIT weights. Group weighting defaults to "none", preserving prior behavior.
// Optional fit_sampling={version:1,method:"binary_mask_hard_negatives",
// mask_feature_index:F,normalization:"legacy_retained_mass"} additionally keeps
// every negative FIT row whose immutable binary mask is1. Background negatives
// keep existing eligibility. Requires all positives/equal_patient/inverse_retention.
// Descriptor fit_sampling_counts version1 carries full/new-retained3-stratum
// counts and legacy n0 per scan; actual arrays are recounted on CUDA at load.
// Strata are positive, negative_mask1, negative_mask0. FIT weights are
// (n0/G)*(full_gs/full_g)/retained_gs in ordered FP64 then FP32; nonempty strata
// without retained support refuse. Total target mass is old n0, not expanded n;
// mean row weight need not be1. Original IDs identify raw features for meta rows.
// fit_sampling_receipts saves one audit per initialized raw/OOF/meta FIT matrix,
// independent of hyperparameter count. Evaluation and native prediction remain
// unchanged. Initial mode supports ordinary native nested studies only; combined
// custom loss/aggregation/raw-meta, summaries and final/conversion requests refuse.
// The default ignores a descriptor count extension and preserves original behavior.
// Optional training_loss={kind:"group_mean_ce_dice",version:1,beta:[0,1],
// loss_scale:"FIT_rows"} uses the shared trainer's bounded-step group-mean
// CE/Dice gradients for every raw and response FIT. Requires all FIT positives,
// equal_patient + inverse_retention weights, no class balancing, subsample=1
// and explicit positive max_delta_step in each candidate. Native DMatrix cut
// weights stay fixed; group gradients are applied once. Outside-ROI positives
// enter only their FIT patients' Dice denominators. beta=0 is the matched custom
// CE control; beta=1 selects Dice (background-only FIT groups retain CE).
// Omission keeps the native objective. No held-out groups enter this loss.
// Both boolean options default false. reference_predictor contains mask_feature_index
// and optional probability_feature_index/thresholds. Its fixed binary mask is
// always eligible; threshold candidates and fitted families are selected only
// by inner patient groups. The fixed default and selected reference are also
// reported independently; no outer-score improvement is guaranteed.
// native_prediction_thresholds optionally declares finite [0,1] thresholds.
// Every fitted model retains exact native argmax as prediction candidate 0;
// the remaining candidates classify native full-round FP32 p(class 1)>=threshold.
// Fixed-grid CUDA histograms score all cutoffs from one probability pass,
// retaining their declared candidate order, FP32 equality and tie priority.
// Model and prediction rule are jointly selected by inner full-ROI patient Dice,
// frozen for outer scoring and recorded in any requested final native pipeline.
// Omit final_deployment to perform outer evaluation without an all-patient FIT.
// If a final pipeline is requested, explicit final_deployment.convert=false is
// required. Calibration conflicts with every training-time conversion mode. Saved binary baseline pipelines support
// calibrated replay through the recorded native FP32 prediction rule.
// Omission preserves native argmax behavior and performs no extra model FITs.
// Optional rule_search_views=[{id, native_prediction_candidate_indices:[...]}]
// adds named nonempty subsets of the same canonical prediction-rule bank.
// Indices must be distinct, increasing and in range; "full" is reserved for
// the implicit existing primary search. Every view uses the same inner count
// arrays, exact score/byte/order ties, reference candidates, full outer base
// bank and OOF responses. Only the union of selected meta HPs is refitted;
// shared tiled probabilities score all frozen views. Additional outer metrics,
// all base hashes, union meta hashes and actual FIT/prediction-call receipts
// are reported under rule_search_views / paired_outer_execution. Outer scores
// never choose a view. Omission preserves the original single-view path.
// Views currently support outer evaluation only; final_deployment and every
// conversion mode are explicitly refused, including native-only final FITs.
// Optional base_candidate_aggregation={version:1,method:"uniform_probability_mean"}
// adds one fixed equal-weight ensemble of every declared full-refit base model.
// Each row averages both native probability columns in canonical model order,
// using explicit RN64 multiply/add/divide then one FP32 cast per class. The
// ensemble candidate0 compares rounded p1>p0 (tie0); remaining rules threshold
// p1>=cut. Ordinary native argmax, model choices and all outputs stay unchanged.
// New base_mean_only and augmented_inner_selected outcomes are selected from
// inner counts only; the mean is last on exact family ties. All bank hashes and
// summed required bytes are retained, without deduplication or a fake single HP.
// No learned coefficients, extra native FITs/calls or prediction files. Extra
// tile state is one FP32 pair (8 bytes/row), using the existing response bank.
// Combined aggregation/blend/bootstrap/custom-loss/raw-meta modes, structural
// summaries and all final FIT/export/conversion requests are refused. Default off.
// Optional teacher_probability_blend={version:1,method:"oof_brier_convex",
// probability_feature_index:F} adds one convex teacher/base probability mixture
// for every base HP. Its coefficient uses only enclosing FIT patient-OOF base
// probabilities, binary labels and existing FIT weight words (once). Fixed256
// pairwise FP64 reductions fit clipped weighted Brier coefficients; alpha32 is
// the inference authority, exact endpoints copy the corresponding FP32 input,
// and interior arithmetic uses explicit RN FP64 operations then one FP32 cast.
// Candidate0 is scalar q>0.5 (tie0), other rules q>=cut. Ordinary native argmax
// remains unchanged. All ordinary candidates stay eligible in the augmented
// policy; the blend wins only on strict inner-Dice improvement. Blend-only and
// augmented outer metrics plus every FIT coefficient are saved separately.
// Effective alpha32 endpoint tags do not imply the inner coefficients matched.
// No extra native FITs or prediction calls, prediction files or coefficient HPO.
// Missing teacher probabilities, combined averaging/bootstrap/custom-loss/raw
// meta modes, final FIT/export/conversion and structural summaries are refused.
// Omission preserves the existing native study path.
// Optional meta_candidate_aggregation={version:1,method:"uniform_probability_mean"}
// compares the legacy ordinary meta winner with one fixed ensemble containing
// every ordinary meta HP in declared order. Inner probability calls and model
// objects are shared; outer FIT trains all M once. The new arm selects only its
// cutoff/family and saves ordered combiner_indices, never a fictitious HP0.
// Ordered FP64 probability accumulation casts once to FP32, with strict > argmax
// (tie0), and retains native-vs-probability control diagnostics. Extra tile state
// is24 bytes/row. Other averaging/bootstrap modes, custom loss, final fitting,
// conversion/export and structural summaries are refused. Default off.
// Optional meta_patient_bootstrap={version:1,replicates:R,seed:U64} compares
// ordinary meta FITs with R weighted patient-bootstrap meta FITs on exactly the
// same OOF response words and full-refit teachers. It requires equal_patient,
// native CE, at most 2^24 FIT patients, and fresh immutable weighted DMatrices.
// Whole-patient multiplicities multiply existing FP32 weights; zero-weight rows
// remain. Draws use the declared portable policy and common seeds across HPs.
// No OOB scoring, label-dependent redraw, custom loss, teacher-bank averaging,
// final fitting, export/conversion or structural summaries are supported.
// Each mode/view chooses on inner scores only; ensemble byte ties sum all member
// JSON bytes. Ordered FP64 probability means cast once to FP32; argmax uses >
// with ties class0. Legacy native argmax is unchanged and disagreement counts
// against its probability argmax are recorded without changing selection.
// Additional selections/replica receipts/metrics use meta_patient_bootstrap.
// Optional oof_teacher_aggregation={version:1,method:"patient_weighted_probability_mean"}
// adds a paired inference mode retaining the OOF teacher banks already fitted.
// The same fitted meta model predicts each complete bank's response vector;
// probabilities are averaged AFTER the nonlinear meta model, in fold order,
// with integer held-patient weights, explicit FP64 RN multiply/add/divide and
// one RN cast per class to FP32. Argmax is p1>p0 (tie class0). Requires equal
// patient FIT weighting. Baseline/reference counts are shared; each mode and
// rule view selects independently on inner groups, with a shared outer meta
// refit union. The existing full-refit mode and outputs remain the control.
// Additional metrics/selection/bank pins are under oof_teacher_aggregation.
// No prediction files or response cube is retained. Final FIT/export and every
// conversion mode are refused; this averaging contract is not yet convertible.
// Structural summaries are currently refused with this mode. Omission leaves
// the original inference path unchanged.
// Optional inner_result_encoding="scan_counts_v1" stores every inner candidate's
// exact saved macro Dice and four integer ROI counts per scan, sharing ordered
// scan/patient/outside-FN metadata once per selection. Outer metrics stay full.
// Omission or "full" retains the existing full inner records. The host-only
// class_study_inner_metrics.hpp decoder reconstructs legacy metric records;
// serialization never changes candidate ordering, selection or native FITs.
// Evaluation uses stable CUDA row-ID batches and all 2*base_count response
// columns per row. Capacity follows live owned bytes, actual free CUDA memory
// after native model registration and known output-buffer reserves. Optional
// positive evaluation_batch_rows only lowers that capacity. Opaque native
// allocations remain additional. FIT-only OOF matrices keep every base model;
// evaluation stores no full probability matrix or prediction files. Models are
// reloaded from authenticated RAM buffers using two inference-only owners.
// Optional meta_input_feature_indices appends distinct raw input columns in
// the declared order after all 2*base_count native probability columns. The
// nonempty array must contain only valid raw feature indices. FIT values are
// copied from the exact same gathered row IDs as the OOF response matrix;
// held-out raw values are appended to the same tiled inference rows. Teacher
// exclusion/coverage and patient grouping are unchanged. Saved combined
// pipelines carry meta_input_mapping={version:1,raw_feature_indices:[...]};
// pruning remaps response columns while preserving this ordered raw suffix.
// Native margin callbacks retain only teacher margins; native class callbacks
// construct the mixed input. Omission preserves probability-only behavior.
// Optional conversion_gate selects a gate-only workflow instead of nested CV:
// withheld_patient_ids, a fixed hyperparameters array, and explicit positive
// max_rounds/max_depth/max_trees/max_source_nodes/max_predicates/
// max_partition_cells, max_states/max_nodes/max_expansions, gpu_byte_budget,
// maximum_seconds, batch_size and qualification_batch_rows. Defaults target
// one-round sources; max_predicates counts feature/threshold/missing-direction
// signatures, while threshold counts and partition cells are separate fields.
// Alternatively, composed_conversion_gate uses base_hyperparameters and fixed
// meta_indices (default: all meta configurations) with grouped OOF responses.
// Its limit fields match conversion_gate; at least two FIT patients remain.
// conversion_on_outer_deployments applies the same adapter to the frozen
// baseline and nonlinear stack after every nested outer evaluation.
// final_deployment performs fresh internal selection over all patients, refits
// the selected pipeline on all patients and converts it once. Outer scores do
// not select this deployment, which makes no additional held-out accuracy claim.
// It uses the same converter limits and may coexist with outer conversions.
// final_deployment.convert=false saves the selected pipeline without conversion.
// An external-reference winner saves its upstream predictor and threshold recipe;
// it performs no all-patient native refit or fabricated predictor conversion.
// Gate-only modes conflict with outer conversion and final deployment.
// Conversion qualification_mode defaults to none. This bypasses Cartesian
// partition preflight and exhaustive verification while retaining maintained
// converter construction checks. A completed construction exports artifacts
// with accepted_native_class_root=false. Explicit exhaustive mode additionally
// requires every finite/NaN source-partition cell to match native classes.
// maximum_seconds=0 means no conversion deadline; batch_size=0 selects adaptive
// batching. The public CLI enables exhaustive verification only with --verify
// and enables a conversion deadline only with a positive --timeout value.
// Conversion refusals preserve native metrics and model selection. Native
// evaluation completion and requested conversion completion are separate.
// final_deployment.complete requires a saved pipeline and completion of any
// requested native conversion; native_only and conversion_deferred are explicit.
// attempt_complete
// records a finished attempt even if conversion was refused. Qualification is
// recorded separately and an unverified artifact has no exhaustive certificate.
// Signed-zero cuts are one semantic predicate; NaNs form a separate source atom.
// All numerical fitting, prediction, gathering, Dice and selection run on CUDA.
// Models and trials remain in memory; completed outer folds are published.
nlohmann::json validate(const nlohmann::json& plan,
                        const nlohmann::json& descriptor);
int run(const nlohmann::json& plan, const nlohmann::json& descriptor,
        const std::filesystem::path& output,
        volatile std::sig_atomic_t& stopping);
// GPU differential fixture against the former per-cutoff scoring loop.
// Optional synthetic benchmark uses resident data only; never fits a model.
nlohmann::json threshold_histogram_checks(const nlohmann::json& plan,
    const nlohmann::json& descriptor, std::uint32_t benchmark_rows = 0);
nlohmann::json evaluation_probability_checks(const nlohmann::json& plan,
    const nlohmann::json& descriptor);
// Focused CUDA chooser checks; no fitting or conversion.
nlohmann::json rule_search_view_checks();
// Reconvert an immutable final native pipeline without reading training data
// or fitting. Uses plan.conversion limits and the same maintained converter;
// exact unused-base removal and original source lineage are recorded.
// Supported binary baseline probability cutoffs retain the native FP32 rule;
// unsupported calibrated compositions are refused.
nlohmann::json validate_replay(const nlohmann::json& plan,
                              const nlohmann::json& pipeline);
int replay(const nlohmann::json& plan, const nlohmann::json& pipeline,
           const std::filesystem::path& output,
           volatile std::sig_atomic_t& stopping);
} // namespace class_study::grouped_segmentation
