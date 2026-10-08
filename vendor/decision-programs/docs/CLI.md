# Decision Programs command reference

`decision-programs` is the public entrypoint. Run `decision-programs --help` or
`decision-programs COMMAND --help` for arguments. Linux and WSL are supported.
Numerical training, conversion, simplification, validation, prediction and path
tracing run in C++23/CUDA. The frontend handles arguments, hashes and input-format
transport.

## Common workflow

```sh
decision-programs doctor --gpu
decision-programs demo

decision-programs train --data examples/quickstart.csv --target label \
  --rounds 2 --depth 2 --output trained
decision-programs convert --model trained/model.json --output compiled
decision-programs predict --model compiled --data examples/predict.csv
decision-programs inspect --model compiled
decision-programs explain --model compiled --data examples/predict.csv --output paths.json
decision-programs export --model compiled --output equations.json
```

Use a fresh output path for every run. `--output` and `--out` are equivalent.
`demo` creates a fresh temporary directory and prints its location when no output
is supplied. Model directories automatically select `model.canonical` and its
compact companion. A directory with multiple numbered trials requires an explicit
model file.

Training and native-source conversion require CUDA-enabled XGBoost **3.4.1**.
Library discovery uses `--library PATH`, then `XGBOOST_LIBRARY`, then the active
Python installation's `xgboost` package. Saved plans may already contain an exact
library binding. Runtime prediction/inspection/export does not require XGBoost.

### Input formats

CSV training uses `--target HEADER_NAME`, a zero-based column index, or `last`.
The target can contain nonnegative integer class IDs or strings. String labels
receive stable first-occurrence class IDs; the output's `input-transport.json`
retains the mapping. Conversion carries this mapping forward. CSV prediction
contains feature columns only, in training order. Headers are detected
automatically; `--header yes|no|auto` overrides detection. Quoted fields, escaped
quotes and CRLF are accepted. Empty, `NaN`, `NA` and `null` features become NaN;
infinity and malformed features are rejected.

Raw FP32 training avoids CSV transport:

```sh
decision-programs train --values inputs.fp32 --labels labels.u32 \
  --rows 1000 --features 4 --classes 3 --rounds 10 --depth 4 --output trained
decision-programs predict --model compiled --values inputs.fp32 --rows 1000
```

Files use little-endian FP32 features and uint32 labels. `--row-stride N` supports
padding. Alternatively pass `--data DESCRIPTOR.json` using the maintained
`dense-fp32-u32-class-labels-1` descriptor. JSON prediction input is
`{"rows":[[1,2],[null,3]]}`; pass it with `--input FILE.json` or `--data FILE.json`.

CSV transport retains immutable input files under
`$XDG_CACHE_HOME/decision-programs/inputs`, or `$HOME/.cache/decision-programs/inputs`.
`DECISION_PROGRAMS_CACHE_DIR` overrides the cache root. Source content, target,
header mode and label mapping determine the identity. Existing cached bytes must
match exactly. These stable paths let checkpoint declarations survive a new
process. Retained CSV files are input artifacts, not CUDA model working memory.

## Commands and backend scope

| Command | Behavior | Maintained backend |
| --- | --- | --- |
| `train` | FIT-only native CUDA training; rounds, depth, regularization and sampling flags | `class_model_train` |
| `convert` | Saved multiclass native-model conversion with explicit resource limits | `class_model_convert` |
| `import-tree` | Structural CLSTREE1 import with a bound origin/domain contract | `class_tree_adapter` |
| `study` | Fixed trials: train, convert, simplify, then evaluate | `class_study` |
| `hpo` | Declared native trials selected on VALID | `class_study`, `native-accuracy-search-1` |
| `combine` / `nonlinear` | OOF nonlinear teacher composition, VALID selection and optional nested holdout confirmation | `class_study`, `native-nonlinear-combination-1` |
| `rl` | Qualified experimental regional equivalent-encoding policy search | `rl_session` |
| `evaluate` | Compare runtime/native class IDs and FIT/VALID scores | `class_model_evaluate` |
| `simplify` | Exact adjacent-predicate shared-DAG rewrites | `class_model_simplify` |
| `predict`, `inspect`, `explain`, `export`, `demo` | Generic persisted-model tools through the shared Runtime | `decision_programs_model_tools` |
| `checkpoint`, `resume` | Checkpoint signaling, metadata inventory and backend recovery | Configured live backend |
| `proofs` | Formal source inventory, maintained Lean elaboration and kernel replay | Portable `CheckLean.cmake` |
| `profile` | One installed profiler/sanitizer/debugger around a public command | Selected external tool |

Internal backend names are implementation details. The installed frontend finds
them beside its binary or under `../libexec/decision-programs`.
`DECISION_PROGRAMS_BACKEND_DIR` provides an explicit alternate backend directory.

### Conversion and simplification

```sh
decision-programs convert --model trained/model.json --output compiled \
  --max-nodes 1048576 --max-states 1048576 --gpu-byte-budget 1073741824
decision-programs simplify --model compiled --source trained/model.json \
  --max-passes 4 --output simplified
```

Conversion supports `--batch-size`, `--max-expansions`, `--split-policy`, runtime
residency, checkpoint and proof-module flags. A bounded or interrupted conversion
can remain incomplete; its receipt determines completion. A completed conversion's
native-class preservation applies to the **declared conversion domain**, including
any one-hot restrictions. Runtime tools report the supported input representation
separately and establish no new native-source equivalence outside that domain.

Simplification retains the source binding. It may stop before a fixed point when
the pass budget is exhausted. Native comparison is a separate evaluation:

```sh
decision-programs evaluate --model simplified --source trained/model.json \
  --data evaluation.csv --target label --fit-rows 100 --output evaluation
```

Evaluation CSV string labels reuse the source/model's `input-transport.json`, or
an explicit `--label-map FILE`. Missing or unseen mappings are rejected. Numeric
labels remain direct class IDs when the trained mapping is numeric; numeric-looking
tokens in a categorical mapping retain their original mapped IDs. Evaluation needs
both FIT and VALID rows and performs no training, selection or TEST access.

### Studies and composition

```sh
decision-programs hpo --data evaluation.csv --target label --fit-rows 100 \
  --rounds 10 --depth 3 --trial '{"max_depth":2}' --trial '{"max_depth":4}' \
  --output native-search
```

Trials override the base hyperparameters. HPO selection orders candidates by VALID
errors, native model bytes, then declared order. Reused VALID scores do not measure
independent generalization.

Fixed `study` accepts `--train-data FIT_DESCRIPTOR.json` plus `--data
EVAL_DESCRIPTOR.json`, or an advanced `--plan PLAN.json`. Its construction finishes
before deferred VALID evaluation.

Composition requires completed HPO source checkpoints. The following small
workflow uses the shipped synthetic CSV, with 24 FIT rows and six VALID rows.
First prepare both teacher sources and retain their complete checkpoint. Then
select the one-round prefix baseline from those exact sources:

```sh
decision-programs hpo --data examples/quickstart.csv --target label --fit-rows 24 \
  --rounds 2 --depth 2 --trial '{"max_depth":2}' --trial '{"max_depth":3}' \
  --checkpoint teacher-checkpoint --output teacher-sources
decision-programs hpo --data examples/quickstart.csv --target label --fit-rows 24 \
  --rounds 1 --depth 2 --trial '{"max_depth":2}' --trial '{"max_depth":3}' \
  --training-checkpoint teacher-checkpoint --output teacher-prefixes
decision-programs combine --data examples/quickstart.csv --target label --fit-rows 24 \
  --teachers '[{"rounds":1,"max_depth":2},{"rounds":1,"max_depth":3}]' \
  --meta '[{"rounds":1,"max_depth":2}]' --baseline teacher-prefixes/selected-model.json \
  --training-checkpoint teacher-checkpoint --folds 2 --output composition
```

The second command reuses the completed sources instead of fitting new models.
Both it and `combine` slice the same source models to the declared teacher rounds,
so the selected prefix baseline is retained in the teacher bank. An unrelated
`train` output is not a compatible baseline merely because its parameters match.

`--training-checkpoint DIRECTORY` is repeatable for `hpo` and `combine`. Source
checkpoints must be complete native HPO checkpoints with exactly matching dataset,
FIT/VALID split, native-library binding and normalized training parameters; their
round count must cover the requested prefix. The backend verifies these bindings
and the baseline's exact identity. `--checkpoint` names a new workflow's output
checkpoint, while `--training-checkpoint` supplies completed input models.

OOF settings, baseline provenance, prefix reuse and frozen refit/final-evaluation
workflows are also available through advanced plans. These tiny commands exercise
the interface; composition results do not imply improved TEST accuracy.

### Nested holdout confirmation

`combine` optionally confirms its frozen inner VALID winner against the required,
predeclared baseline using a separate fresh holdout dataset:

```sh
decision-programs combine --data development.csv --target label --fit-rows 100 \
  --teachers '[{"rounds":1,"max_depth":2},{"rounds":1,"max_depth":3}]' \
  --meta '[{"rounds":1,"max_depth":2}]' --baseline teacher-prefixes/selected-model.json \
  --training-checkpoint teacher-checkpoint --folds 2 \
  --nested-holdout --holdout-depth 2 --holdout-data fresh-holdout.csv \
  --output confirmed-composition
```

The source checkpoint and baseline must still match the development dataset and
declared teacher prefixes as described above. `--nested-holdout` defaults to depth
one. Supplying `--holdout-depth N` or `--holdout-data FILE` also enables
confirmation. Every enabled invocation requires `--holdout-data`; depth must be a
positive integer no larger than the holdout row count. These options apply only to
`combine` and its `nonlinear` alias; `hpo` and fixed `study` reject them.

The backend divides the fresh holdout rows into N disjoint, balanced, contiguous
blocks in input order, with block sizes differing by at most one row. At each gate
the same frozen inner winner must make **strictly fewer errors** than the same
required baseline. A tie or loss selects the baseline and stops confirmation. The
candidate is retained only when it passes every gate. Holdout labels never fit or
retrain a model, rank another candidate, or trigger a retry. Numerical evaluation
of both models runs on the GPU. Existing FIT/VALID HPO selection and OOF fitting
retain their existing behavior.

This is nested holdout confirmation, not full nested cross-validation or a final
TEST evaluation. Reserve a separate final TEST dataset for evaluating the selected
model once. Freshness and appropriate group, patient, or institution separation
remain the caller's responsibility. Repeated search consumes the holdout: reusing
its labels for later search decisions does not provide independent confirmation.
Increasing depth leaves fewer samples per gate. Passing the gates is neither a
statistical significance test nor a guarantee of improved TEST performance.
The interface rejects shared development/holdout source paths, reused dense input
paths, and copies with identical feature contents; those checks cannot establish
that two datasets have independent provenance.

CSV holdout transport uses the same `--target` and `--header` settings as the
development CSV. Feature names and order must match exactly. Labels reuse the
development class mapping and full class count, including classes absent from the
holdout. A numeric-looking token in a categorical development mapping keeps its
original mapped ID. Unknown categorical labels or numeric IDs outside the
development class range are rejected. For non-CSV development inputs, supply the
original `--label-map INPUT_TRANSPORT.json` when the holdout CSV uses mapped string
labels. The frontend converts input formats on the host without model evaluation.

`--holdout-data HOLDOUT_DESCRIPTOR.json` accepts little-endian FP32 features and
uint32 class IDs already in development order. The dedicated descriptor declares
the role explicitly:

```json
{
  "format": "dense-fp32-u32-holdout-labels-1",
  "role": "HOLDOUT",
  "training_allowed": false,
  "selection_allowed": true,
  "TEST_read": false,
  "features": 4,
  "classes": 3,
  "rows": 40,
  "row_stride": 4,
  "values_path": "/absolute/fresh-holdout.fp32",
  "values_sha256": "SHA256_OF_EXACT_FEATURE_FILE",
  "labels_path": "/absolute/fresh-holdout.u32",
  "labels_sha256": "SHA256_OF_EXACT_LABEL_FILE",
  "preprocessing": "Same input representation as development; class IDs preserve its mapping"
}
```

Replace the SHA placeholders with the actual 64-character hexadecimal hashes.
Feature and class dimensions must equal the development descriptor. No `FIT_rows`
or `VALID_rows` fields are allowed, including zero counts. The CUDA backend
validates file pins, extents and class-ID bounds before evaluation.

The generated reviewable plan contains
`nested_holdout: {"depth": N, "dataset": HOLDOUT_DESCRIPTOR,
"acceptance_rule": "strictly_fewer_errors_at_every_level"}`. Advanced plans use
the same object; no alternative acceptance rule is supported. Use `--dry-run` or
`--save-plan` to inspect and retain it before numerical execution.

For a holdout arranged into contiguous groups, an advanced plan can declare
`nested_holdout.partition_rows` to place gate boundaries between groups. For
example, a depth-two holdout with 40 rows can use
`--set '/nested_holdout/partition_rows=[18,22]'`. Supply one positive count per
level, summing to the holdout row count. The backend preserves the declared row
order and boundaries; arranging independent groups remains the caller's job.
Omitting this setting retains the balanced row blocks described above.

With `--checkpoint`, final saves, explicit checkpoint requests, and stop saves
retain frozen model identities, the gate cursor, and committed results. Resuming
committed decisions does not rescore those gates. A crash before checkpoint
publication may repeat evaluation of the same frozen pair on an uncommitted gate.

### Prediction and scientific interrogation

Generic tools accept canonical CLSGDAG1 models, with an optional CLSG64B1 compact
companion. `predict` returns class IDs. `explain` returns the exact visited shared
nodes, raw threshold words, NaN direction, branches and terminal. The default path
storage limit is 4096 nodes per row; `--max-path-nodes N` changes it. An insufficient
capacity is rejected rather than presenting a partial path as complete.

`export --kind equations` preserves shared subfunctions, exact raw threshold bits,
strict FP32 comparisons and stored NaN directions. `export --kind compact` produces
GPU-packed runtime bytes. Equation export establishes exact representation of the
bound runtime; it does not newly prove source equivalence or human clarity.

Specialized regional `explain/export --format regional` accepts CLSRMDL1 models
under the existing 54-feature rank/category contract. Regional explanations require
the original ten-temperature input schema. Fuzzy scores/gradients describe a
distinct mathematical model and are not native probabilities or calibrated
uncertainty. Generic fuzzy gradients, counterfactuals, causal effects and global
importance remain pending; `capabilities` lists their status.

### RL, checkpoints and profiling

```sh
decision-programs rl --model forest-teacher.json --episodes 32 --output policy-search
decision-programs convert --model trained/model.json --checkpoint converter-checkpoint \
  --output first-attempt
decision-programs resume convert --model trained/model.json \
  --resume converter-checkpoint --output resumed
decision-programs checkpoint --pid LIVE_PUBLIC_COMMAND_PID
decision-programs checkpoint --directory converter-checkpoint
```

RL is specialized to ten numeric features, wilderness4 and soil40 one-hot groups,
and seven classes. Its policy orders equivalent encodings; it does not grant class
authority. `--learning-rate` must lie in `(0,1]`. `rl --resume POLICY.json` is a
**policy-word warm start** into a fresh session, table, schedule and warmup. It is
not optimizer/session checkpoint continuation.

Conversion and experiment checkpoints require unchanged semantic declarations
and a fresh final output. `--checkpoint` enables saving; `--resume` supplies an
existing generation. Whole fixed studies, native HPO and nonlinear OOF workflows
support their implemented stage-boundary recovery. Native fitting recovery covers
completed calls, not an in-flight XGBoost update. A SIGUSR1 request or checkpoint
file inventory is not evidence that a generation committed or resumed successfully.

```sh
decision-programs proofs --check --output proof-reports
decision-programs profile --tool memcheck -- demo --output sanitized-demo
decision-programs profile --tool nsys --output trace -- \
  predict --model compiled --data examples/predict.csv
```

Proof checks require Lean 4.34.1, `leanchecker` and CMake, then compile the maintained
module suite in dependency order and replay the kernel artifacts in a fresh report
directory. Abstract Lean theorems are not a CUDA implementation-refinement proof.

Profiling supports Nsight Systems/Compute, all four Compute Sanitizer modes and
CUDA-GDB. Other installed profiling routes can use an absolute `--tool` executable
and repeated `--tool-arg ARG`. Exactly one tool wraps the command. Availability is
not collection success; inspect exit status and retained reports. Instrumented
timings do not establish an uninstrumented speedup.

## Advanced, reviewable plans

`--dry-run` prints the exact backend argv and generated JSON without numerical
execution. `--save-plan FILE` retains the exact generated plan; CSV artifact paths
remain stable. `--set /json/pointer=JSON_VALUE` exposes all supported backend
settings while their existing validators reject unknown or retired settings.
Strings need JSON quotes, for example:

```sh
decision-programs convert --model trained/model.json --output compiled \
  --set '/split_policy="widest_residual"' --save-plan conversion.json --dry-run
```

The host-only interface checks run without CUDA:

```sh
python3 native/decision_programs_cli_checks.py build/cuda-release/bin/decision-programs
```

They verify command help, installed relocation, automatic hashes, paths containing
shell metacharacters, CSV conversion/mappings, stable saved/checkpoint declarations,
model-directory discovery, existing-output preservation, corrupt-cache refusal,
invalid metadata refusal and RL argument forwarding. Numerical qualification uses
the real CUDA demo and maintained backend tests.

## Patient-grouped segmentation comparison

```sh
decision-programs prepare-segmentation --manifest scans.json --out prepared
decision-programs segmentation-study --data prepared/dataset.json \
  --plan search.json --library /path/to/libxgboost.so --out comparison
```

`prepare-segmentation` is an optional DCMTK/CUDA adapter for parallel-plane signed-16
CT and binary DICOM SEG. A scan manifest supplies `patient_id`, `scan_id`,
`ct_dir`, `seg_path`, `seed_x_pixel`, `seed_y_pixel`,
`seed_z_imageposition_mm` and explicit `seed_pixel_index_base` (0 or 1).
Coordinates must come from the declared inference input, never a reference-mask
centroid. Defaults are a fixed160mm cube,2mm isotropic grid and deterministic
label-independent training sampling of roughly one voxel in32. All crop voxels
are scored; missed reference volume outside the crop contributes false negatives.
It is a known-location segmentation task, not autonomous lesion detection.

To compare previously normalized and resampled CT arrays without changing the
reference endpoint, use the same preparation backend:

```sh
decision-programs prepare-segmentation \
  --prepared-images processed/prepared-dataset.json \
  --reference-data prepared/dataset.json --export exported/cases.json \
  --preprocessing-plans released-model/plans.json --out normalized-features
decision-programs segmentation-study --data normalized-features/dataset.json \
  --plan search.json --library /path/to/libxgboost.so --out normalized-comparison
```

This mode maps single-channel float32 CZYX arrays onto the original physical crop
and applies the same twenty CUDA feature operators. The reference labels,
patient groups, scan identities, sampling and outside-crop reference accounting
are retained. Image normalization and resampling geometry come from the supplied
preprocessing metadata and plans. Reference masks and rounded model prompts do
not define the new feature grid. Cropped-away interior data is rejected rather than reconstructed from labels. `--manifest` and `--set` cannot be combined with
this mode.

`segmentation-study` consumes a `grouped-segmentation-dataset-1` descriptor.
Its row-major FP32 features and uint32 labels/patient IDs/scan IDs/training flags
are staged once. Training, gathering, probabilities, confusion counts and model
selection use CUDA. Base/meta candidate fits stay in memory; a result is
published at completed outer-fold boundaries.

Feature values must be finite by default. A dataset descriptor can explicitly
allow native missing-value handling for selected columns:

```json
"missing_feature_indices": [4, 7]
```

This nonempty list contains unique, in-range integer feature indices. Only NaN
is allowed as a missing value in those columns; infinity is always rejected.
Configured reference mask and probability columns must remain finite and cannot
appear in the list. Labels and training eligibility rules are unchanged.
Declare this field in the dataset descriptor, not the search plan. The declared
columns are recorded with the study and saved pipeline; native missing branches
are used without imputing a sentinel value. Omit the field to keep the existing
finite-only behavior.
The search plan declares `base_hyperparameters` and `meta_hyperparameters`,
`inner_folds` and `oof_folds` (default 3 each), and memory budgets. Set
`outer_folds: 5` for five deterministic patient-grouped outer folds. Each patient
is evaluated exactly once, and all repeat scans from that patient remain in the
same fold. Omitting `outer_folds` retains leave-one-patient-out evaluation.
`outer_patient_ids` restricts that LOPO workflow to an explicit pilot subset and
cannot be combined with `outer_folds`. Fold counts must leave enough patient
groups for the nested inner validation and meta-model cross-fit.

Evaluation processes rows in GPU batches sized from the configured memory budget
and available device memory. Every row still receives all base-model outputs
before nonlinear combination; batching does not prune the model bank or sample
the evaluation set. Counts are accumulated across batches before patient Dice is
calculated. Prediction matrices are not written to disk. An optional positive
`evaluation_batch_rows` limits the batch size; omit it for automatic
sizing. Smaller batches reduce evaluation workspace at the cost of more model
loads and prediction calls. Training still requires its FIT and out-of-fold
matrices, and opaque native-library allocations remain additional memory costs.

Every outer test patient is excluded from both stages of fitting and selection.
Each inner validation fold is also excluded from both stages, including the
additional patient-grouped cross-fit used to train its meta model. Selection uses
patient-macro Dice. The raw baseline, raw nonlinear combiner and inner-selected
family are reported separately. Results record completed outer folds and patients;
`full_cohort_outer_evaluation` requires every declared patient to be evaluated.
A pilot cannot claim complete cohort evaluation. `--check` validates metadata
without fitting.

The five-fold example also requests one final deployment:

```sh
decision-programs segmentation-study --data prepared/dataset.json \
  --plan examples/grouped-segmentation-search.json \
  --library /path/to/libxgboost.so --out five-fold-study
```

FIT weighting is optional. `fit_group_weights: "equal_patient"` gives each
training patient the same total weight, including in the out-of-fold models.
With `fit_class_balanced: true`, that weight is divided equally between the
classes present within each patient. With `fit_sampling_weights:
"inverse_retention"`, it instead restores each patient's original class
proportions after sampling. Class balancing and inverse-retention correction
are mutually exclusive; without either, retained rows within a patient share
its weight uniformly. All policies normalize the retained weights to mean one
and calculate them only from the current FIT patients. These weighting policies
optimize a voxel loss; they are not direct optimization of patient Dice.

For example, these search-plan fields enable equal patient and class influence:

```json
{
  "fit_group_weights": "equal_patient",
  "fit_class_balanced": true,
  "fit_sampling_weights": "none"
}
```

The optional group-mean CE/Dice training objective uses the same native trainer
for both source models and their nonlinear combiner. Enable it in the plan:

```json
{
  "include_all_fit_positives": true,
  "fit_group_weights": "equal_patient",
  "fit_sampling_weights": "inverse_retention",
  "training_loss": {
    "kind": "group_mean_ce_dice",
    "version": 1,
    "beta": 0.5,
    "loss_scale": "FIT_rows"
  }
}
```

Each base and combiner hyperparameter candidate must also declare a positive
`max_delta_step` and use `subsample: 1`. Class balancing must be disabled.
`beta` mixes group-mean cross entropy and soft Dice: `0` is the custom
cross-entropy control, `1` uses Dice, and intermediate values mix the two.
Patients without any foreground inside the region use background cross entropy.
Only the current FIT patients contribute counts, including their foreground
outside the region. Native weights used for threshold construction stay fixed;
they are not applied twice to the custom gradients. The training objective is
recorded separately from the native prediction objective. Omitting
`training_loss` preserves ordinary native training. This experimental objective
does not establish improved held-out accuracy or a floating-point descent proof.

The optional `training_loss.curvature_policy` selects `uniform_v1` (the
default) or `class_row_sum_v2`. The latter uses a tighter class-dependent
diagonal bound for the same Dice objective and the same capped update range;
it does not change labels, weights, or the loss mixture. Its real-arithmetic
bound needs no additional group reductions. Both policies leave the `beta: 0`
cross-entropy calculation unchanged. The selected policy is recorded in each
completed FIT receipt; neither policy implies improved validation performance.
The study result also records the effective policy when the plan omits a default.
Custom training carries forward GPU margins through the native predictor's proxy
interface, evaluating the newly appended trees each round. The typed trainer
retains a full-ensemble reference mode. Group moments use keyed warp aggregation
in FP64; this changes reduction order, so identity with earlier kernel versions
is not asserted. Ordinary native training is unaffected.

`meta_input_feature_indices` optionally appends original input columns to the
nonlinear combiner after all base-model probability columns. For example:

```json
"meta_input_feature_indices": [2, 0]
```

The indices must be valid, unique and nonempty; their declared order is kept.
With B binary base models and P selected original columns, the combiner receives
2B + P columns. Each original value is taken from the same row as its predicted
probabilities. Patient-excluding cross-fit predictions remain required for
training the combiner. Omit this field to retain probability-only inputs.

Saved pipelines carry a versioned `meta_input_mapping` through prediction,
unused-model pruning and conversion. When a combiner tests an original input,
the same converter preserves that boundary with exact internal predicate guards.
Those guards are bookkeeping for conversion, not extra inputs to the native
model. A combiner using only original inputs, or a constant combiner, currently
retains one genuine base model to carry the input domain; this is not a claim of
minimum representation size. The option also works with paired rule-search
views; the restrictions on final deployment with those views still apply.
`native_prediction_thresholds` adds a declared list of class-one probability
cutoffs to the native argmax candidate. Inner patient folds jointly select the
model and its full-round FP32 decision rule. Omit `final_deployment` to evaluate
held-out patients without fitting a final model on everyone. If final fitting
is wanted with probability-cutoff selection, use `final_deployment.convert:
false`; the saved baseline pipeline can subsequently be converted with
`convert --pipeline`. Calibrated training-time conversion modes remain
unsupported.

For a paired comparison of decision-rule searches, declare additional subsets:

```json
"rule_search_views": [
  {"id": "small_rule_bank", "native_prediction_candidate_indices": [0, 1, 2]}
]
```

Index 0 is native argmax; indices 1 onward follow the declared probability
cutoffs. Each subset must be nonempty, unique and increasing in canonical
index order. IDs must be unique and cannot be `full`. The implicit `full` view
remains the primary result. Every view selects using the same inner scores,
with the existing exact score, native-byte and declaration-order tie rules.
All views reuse one outer base-model bank and fit each distinct selected
combiner only once. Extra views therefore add no base-model fits; additional
combiner winners may add fits. Results record per-view outcomes, common model
identities, actual fit counts and outer prediction calls. Outer outcomes never
choose the winning view. This option currently supports held-out evaluation
only; omit final-deployment and conversion options. Omitting the field retains
the normal single-view workflow.
Set `inner_result_encoding` to `"scan_counts_v1"` to reduce inner-result JSON
size. Every candidate retains its exact saved `macro_patient_Dice` and a flat
`scan_counts` array of four unsigned integer counts per scan, ordered as TP, FP,
FN inside the ROI, and TN inside the ROI. One versioned `inner_metric_encoding`
object per selection supplies the ordered patient and scan IDs and outside-ROI
false negatives. Candidate order and selection are unchanged; outer metrics
remain full records. Omission or `"full"` retains the full inner records.
The host codec in `native/class_study_inner_metrics.hpp` expands compact records
for count audits without fitting or prediction. A grouped study still requires
a new output directory; compact output does not introduce study resumption.
Saved native-pipeline replay reads the frozen model and decision rule, so inner
metric encoding does not affect replay or conversion checkpoints.

The optional `final_deployment` object applies converter limits after evaluation. A fresh
patient-grouped inner search over all patients selects the baseline or nonlinear
combiner using only those inner results. The selected pipeline is refitted on all
patients and converted once. Outer evaluation scores never choose that final
pipeline. Its all-patient training artifact is a deployment model; its training
results do not constitute an additional held-out performance estimate. The result
retains the final selection, FIT and cross-fit patient groups, native pipeline
identity and conversion receipt separately from outer evaluation completeness.
`final_deployment.complete` requires the final conversion to finish. A refused
conversion records `attempt_complete: true` and `conversion_complete: false`;
the completed native evaluation and native pipeline remain available.

To establish a small, completely converted starting point before a larger search:

```sh
decision-programs segmentation-study --data prepared/dataset.json \
  --plan examples/grouped-segmentation-compact-gate.json \
  --library /path/to/libxgboost.so --out compact-gate \
  --set '/conversion_gate/withheld_patient_ids=[0]'
```

Choose withheld IDs that exist in your descriptor. `conversion_gate` runs only
its declared source-model fits and the maintained converter; it does not run
nested model selection. Structural, search and GPU-memory limits remain
configurable. There is no automatic conversion deadline. Add `--timeout SECONDS`
to request one; `--timeout 0` also means unlimited time. The CLI clears saved-plan
deadlines unless `--timeout` is supplied. A requested limit is checked at batch
boundaries and excludes native fitting. A stopped run is incomplete; it does not
establish that conversion is impossible. The converter
memory limit does not cover the resident dataset or opaque XGBoost allocations.
This establishes conversion feasibility, not useful held-out accuracy.

Exhaustive post-conversion comparison is **off by default**. Add `--verify` to
compare every finite/NaN source-partition cell against native XGBoost:

```sh
decision-programs segmentation-study --data prepared/dataset.json \
  --plan examples/grouped-segmentation-compact-gate.json \
  --out verified-gate --verify
```

The flag applies to the conversion mode declared by the plan: `conversion_gate`,
`composed_conversion_gate`, `conversion_on_outer_deployments`, or
`final_deployment`. The CLI sets
`qualification_mode: "none"` unless `--verify` is supplied, even if a saved plan
previously requested verification. When `--timeout` is supplied, the verification
sweep shares that time budget with construction. It can be much more expensive
than construction.
Its partition-cell limit applies only when verification is enabled.

Without verification, the Cartesian partition preflight and exhaustive sweep are
skipped. The converter's built-in construction rules and internal consistency
checks remain. Completed artifacts record `construction_complete: true`,
`qualification_performed: false`, `all_finite_FP32_and_NaN_cells_verified: false`,
and `accepted_native_class_root: false`; `complete: true` describes completion
of the requested construction. An unverified result does not claim an exhaustive
native-class certificate. `batch_size: 0` enables adaptive converter batching;
the gate default remains 64. Training, construction and verification times are
reported separately in source-model gates, including the failing stage and
partial work when a limit stops a trial.

The matching `examples/grouped-segmentation-composed-gate.json` plan trains a
nonlinear combiner from patient-grouped out-of-fold probabilities and converts
the complete composition into raw-feature decisions. Use the same command with
that plan and `--set '/composed_conversion_gate/withheld_patient_ids=[0]'`.
The `composed_conversion_gate` configuration chooses fixed `meta_indices` and
has the same explicit resource limits. Separate base score channels and native
probability responses are preserved; class-only base artifacts are not used as
combiner inputs. With `--verify`, every cell in the union of the base partitions must pass the
native-composition comparison before an artifact is accepted.

For the complete grouped comparison, use
`examples/grouped-segmentation-compact-study.json`. Its
`conversion_on_outer_deployments` object applies these limits to each frozen
baseline and combined model after inner selection. All native evaluation
outcomes remain in the report, including folds where conversion reaches a
limit. Conversion acceptance does not choose models or patients. These options
reuse the same adaptive converter as ordinary single-source conversion.
`final_deployment` uses that adapter once after its separate all-patient refit;
it does not convert every candidate or rerun outer evaluation.

## Compare external segmentation models

Export an existing grouped DICOM dataset to full-volume MetaImage files and
supplied-point JSON for an external model:

```sh
decision-programs segmentation-export --data prepared/dataset.json --out exported
```

The exporter preserves the original source identities and supplied lesion
locations, resampling actual CT plane positions and the independent reference
SEG geometry onto a regular physical grid. It records image geometry and the
original study crop in `exported/cases.json`. Reference masks are separate from
model inputs. This command requires the optional DCMTK build.

After an external model produces a binary mask for every case, save each mask
as uncompressed binary MetaImage `<case>.mha` in one directory and score it:

```sh
decision-programs segmentation-score --data prepared/dataset.json \
  --export exported/cases.json --predictions external-masks --out external-score.json
```

Scoring uses the existing prepared labels, physical crop and outside-crop false
negatives. Scan counts are pooled within each patient before averaging patient
Dice, matching `segmentation-study`. Missing predictions or mismatched image
geometry fail explicitly. A valid empty prediction is scored and reported as
empty. This supplies a common evaluation endpoint; it does not certify that
an external pretrained model excluded the evaluation cohort during training.

External model dependencies and pretrained weights remain separate from the
Decision Programs distribution. Their preprocessing, prompting, training data
and software version must be recorded alongside any comparison.

### Convert a saved fitted pipeline

```sh
decision-programs convert --pipeline fitted/final-native-pipeline.json --out compiled
```

This converts the saved native model without loading the training dataset or
refitting. Supported binary baseline pipelines retain their recorded native
probability cutoff, including its FP32 equality rule. Upstream feature-producing
models remain required. Calibrated nonlinear compositions are currently refused.
Use `--check` for metadata validation, `--verify` for independent exhaustive
comparison, or `--timeout SECONDS` for an optional deadline. Verification and
automatic deadlines are disabled by default. The converter uses the same
maintained adaptive backend and records its decision rule in the receipt.

### Outer model summaries

Grouped segmentation plans may set `outer_model_summaries: true` to record structural summaries of each outer-fold baseline/combiner winner and the teachers used by its combiner. The default is off. Summaries are computed from native model JSON already held in memory after scoring; they add no fits, predictions, row-data reads or model files. Numeric-index statistics are deduplicated by model SHA, while per-use mappings distinguish original inputs from teacher/class probability channels and any ordered raw meta suffix. Split counts, saved loss-change and Hessian aggregates, and exact threshold-word/missing-direction histograms describe training structure; they are not heldout contributions, SHAP values or causal importance. Infinite model thresholds retain their FP32 words, and the input domain is unchanged.

Teacher hashes in these mappings identify the outer-refit models used for outer scoring. Combiner split statistics were fitted on FIT-only out-of-fold teacher responses, whose fold-specific models are not archived by this option.

An optional grouped-study plan field `oof_teacher_aggregation: {"version": 1, "method": "patient_weighted_probability_mean"}` compares the existing full-refit inference with predictions from the already fitted OOF teacher banks. It requires `fit_group_weights: "equal_patient"`. Each bank supplies the same complete response layout to the same fitted combiner; its two probabilities are averaged **after** the nonlinear combiner, weighted by the number of patients held out of that bank. The implementation uses declared fold order and separate round-to-nearest FP64 multiplication, addition and division, then casts each probability once to FP32. Binary argmax uses `p1 > p0`, with exact ties assigned class 0. Each inference mode and rule-search view freezes its own inner choice; the outer refit shares the union of selected combiner configurations. No outer score chooses a mode. Extra buffers scale with the evaluation tile and combiner count, while OOF model JSON remains in host memory. The default path and existing result fields remain the full-refit control; additional selections, model-bank identities and metrics appear under `oof_teacher_aggregation`. This option currently refuses final fitting, export/conversion requests and `outer_model_summaries`; the banked averaging rule has no qualified converter yet.

An optional grouped-study block `meta_patient_bootstrap: {"version": 1, "replicates": 4, "seed": 20261008}` compares ordinary combiner fits with a fixed number of patient-bootstrap fits on the same OOF response matrix and full-refit teachers. The replica count and seed are supplied prospectively. It requires `fit_group_weights: "equal_patient"` and ordinary native CE; it refuses custom training loss, OOF-teacher aggregation, final fitting/export/conversion and structural summaries. Each replica samples the enclosing FIT patients with replacement, keeps all repeat scans together, multiplies the existing FP32 weights by patient multiplicity, and creates a fresh weighted DMatrix before native cut construction. Zero-weight rows remain; this is a weighted conditional bootstrap, not an assertion of identical cuts to physically duplicated/dropped rows. Existing weights are not renormalized. The result records multiplicities, supported classes, actual mass and native model identities. It does not use bootstrap out-of-bag scores because the fixed OOF teachers can contain omitted patients in other training folds.

Bootstrap members produce separate native two-class probabilities from the same teacher response. Their ordered equal mean uses explicit FP64 round-to-nearest arithmetic and one final FP32 cast, with strict `p1 > p0` argmax and class0 ties. Each method and rule-search view selects its own HP/cut/family solely within inner validation; outer refits share the selected HP union within each method and the teacher bank across methods. Ensemble byte ties sum every required member without deduplication. The legacy control keeps native argmax; exact native-versus-probability-argmax disagreement counts are saved as diagnostics, not used for selection. Additional results appear under `meta_patient_bootstrap`. Buffers scale with tile rows and logical combiner settings, and replica model JSON stays in host memory. This mode currently has no qualified export/conversion contract.

Grouped plans may opt into `meta_candidate_aggregation: {"version": 1, "method": "uniform_probability_mean"}` to compare the ordinary selected combiner with a fixed equal-probability mean of every declared ordinary meta candidate. Members retain their declared order; no subset, coefficient or extra hyperparameter is selected. Both policies use the same OOF feature matrix and full-refit teacher bank. Inner fits and probability calls are shared; the outer FIT trains every meta candidate once and shares those model objects with the control. The mean policy selects only its cutoff and family from inner data. Its result uses an explicit ordered ensemble identity and a single bank of cutoff scores, rather than identifying the mean as one member's hyperparameters.

The mean uses the existing ordered FP64 round-to-nearest sum/divide and one final FP32 cast per class. Its argmax is strict p1 > p0, with class0 ties; ordinary native argmax remains unchanged and current-run disagreements with probability argmax are recorded. Every member contributes its serialized bytes to the ensemble cost without deduplication. One accumulator and completed mean add 24 bytes per evaluation row, plus metric/status storage. No prediction files are written. This option defaults off and initially refuses bootstrap/OOF-teacher averaging, custom loss, final fitting/export/conversion and structural summaries. The existing single-meta converter format does not yet describe this ensemble.

Grouped plans may opt into `base_candidate_aggregation: {"version": 1, "method": "uniform_probability_mean"}` to add one fixed equal-weight mean of every declared full-refit base model. The original base, nonlinear combiner, reference and selected outcomes remain unchanged. Additional `base_mean_only` and `augmented_inner_selected` outcomes use inner-only cutoff and family choices; the mean cannot displace an ordinary family on an equal inner score. No weights or subsets are learned.

Both native probability columns are accumulated in declared base order with explicit FP64 rounding, divided by the member count and rounded once to FP32. Mean candidate zero uses strict comparison of the rounded pair (`p1 > p0`, with ties assigned class zero); other candidates use `p1 >= threshold`. Ordinary native argmax is unchanged. All required member identities and summed serialized bytes are recorded, without deduplication. The complete response bank is reused, so this option adds no native FITs or prediction calls and writes no prediction files. It adds one FP32 pair per evaluation tile row. Other aggregation/blend/bootstrap modes, custom losses, raw meta inputs, structural summaries and final FIT/export/conversion are currently refused. The option is off by default.

The optional grouped-study setting `teacher_probability_blend` accepts
`{"version":1,"method":"oof_brier_convex","probability_feature_index":F}`.
It fits one nonnegative convex coefficient per base candidate from the current
FIT patients' out-of-fold base probabilities and the declared input probability,
using the existing sample weights exactly once. A fixed 256-entry hierarchical
FP64 reduction fits weighted Brier loss; the saved FP32 coefficient word governs
inference. Endpoint coefficients copy the selected probability word exactly.
Interior mixtures use explicit round-to-nearest FP64 subtraction, multiplication
and addition followed by one FP32 cast. Candidate zero is strict `q > 0.5`;
threshold candidates use `q >= cutoff`. Existing native argmax is unchanged.

All ordinary base, combiner and reference candidates remain eligible. The added
blend wins the augmented inner selector only on strictly greater patient Dice;
blend-only and augmented outer results are reported separately. Every inner and
outer coefficient and effective endpoint is retained. A teacher endpoint can
use a finer cutoff bank than the reference, and an outer endpoint does not imply
that all inner coefficients were endpoints. Brier fitting is a training
surrogate, not a guarantee of better held-out Dice. This default-off option adds
no native model FITs or prediction calls. It currently refuses other averaging
or bootstrap policies, custom loss, raw meta passthrough, structural summaries,
all-patient final fitting, export and conversion. Input teacher probabilities
must be finite in `[0,1]` and cannot declare missing values.

The optional grouped-study `fit_sampling: {version: 1, method: "binary_mask_hard_negatives", mask_feature_index: F, normalization: "legacy_retained_mass"}` keeps all FIT positives, every negative row with the declared immutable binary mask equal to1, and the existing eligible background negatives. It requires `include_all_fit_positives`, `equal_patient` and `inverse_retention`, with no class balancing. A complete version1 `fit_sampling_counts` descriptor extension declares per-scan full and retained counts for positive/negative-mask1/negative-mask0 strata and the legacy retained count; the loader recounts the actual arrays before trusting its memory bound. Each FIT patient's stratum weight is `(n0/G)*(N_gs/N_g)/n_gs`, computed in ordered FP64 then stored FP32, where `n0` is that enclosing FIT's OLD retained count. Empty population strata have zero weight; nonempty strata without a retained row refuse. Preserving total target mass `n0` keeps the original loss normalization; mean row weight is no longer necessarily1. Original row IDs carry the same policy into OOF-response meta FITs. `fit_sampling_receipts` records every matrix weight initialization, including OOF children, without per-row outputs. Full-ROI scoring and native prediction are unchanged. The initial mode supports ordinary nested native studies; combined custom losses, aggregation, raw-meta inputs, structural summaries and final/conversion requests are refused. Omission preserves the existing path, even if the descriptor carries unused sampling counts.
