# Procedure results

The [earlier summary](earlier-study-summary.json) and [current summary](latest-study-summary.json) contain the aggregate evidence for the two procedures shown in the paper. Both evaluate all 32 patient groups and 62 scans, with five outer folds, five inner folds and three folds for the out-of-fold predictions used to train the nonlinear combiner.

Dice is computed after pooling repeat-scan counts within each patient, then averaged equally over patients. Every declared ROI row is evaluated. Truth-positive voxels outside the ROI remain false negatives for every method.

| Procedure | Earlier macro patient Dice | Current macro patient Dice |
|---|---:|---:|
| XGBoost, inner-selected model and rule | 0.757527 | 0.764912 |
| Nonlinear combiner, inner-selected model and rule | 0.735891 | 0.756757 |
| Inner-selected complete pipeline | 0.742741 | 0.764268 |
| Frozen CNN default mask | 0.734324 | 0.734324 |
| Frozen CNN, inner-selected rule | 0.727460 | 0.727460 |

The current XGBoost procedure improves the earlier point estimate by 0.007385 and the frozen CNN default by 0.030588. The nonlinear combiner and complete pipeline selector do not exceed the current XGBoost procedure. The current outer selector chose XGBoost in three folds and the combiner in two.

Both runs use the same 54-feature representation and eight base/four combiner configurations. The current recipe uses equal-patient FIT weighting and 161 native probability cutoffs; the earlier recipe uses global FIT weighting and 19 cutoffs. Native argmax remains eligible, and both use inverse-retention weighting. This comparison therefore changes more than one part of the procedure. It does not isolate a single cause of improvement.

The current paired patient difference versus the frozen CNN has a descriptive 95% bootstrap interval of [0.010352, 0.051504], conditional on the saved predictions (10,000 patient resamples, seed 20261007). It does not correct for exploratory procedure selection or substitute for a new confirmatory cohort. The same cohort was used in preceding experiments, and overlap with the external checkpoint's training population is unverified.

The current run completed 1,277 native FITs: 1,065 for the nested evaluation and 212 for final selection/refitting on all patients. Its recorded total was 726.062 seconds, excluding neural feature preparation. The final refits and their 21-node/181-node expanded trees have **no additional held-out score**. See the [model comparison](../models/README.md) and [paper](../paper/rider-neural-decision-programs.pdf).

The JSON summaries preserve full-precision means, aggregate counts, source hashes and scope statements. They contain no clinical arrays or per-patient records.
