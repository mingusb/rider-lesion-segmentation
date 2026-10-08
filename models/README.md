# Released XGBoost-stage models

These are the earlier and current all-patient XGBoost refits and their converted class decision programs. Both require the same ordered 54-feature input, including frozen LesionLocator outputs. The CNN is still required upstream; these artifacts do not replace it.

| Property | Earlier | Current |
|---|---:|---:|
| Corresponding procedure's macro patient Dice | 0.757527 | 0.764912 |
| Source XGBoost trees / nodes | 32 / 224 | 32 / 480 |
| Ordinary binary-tree nodes | 21 | 181 |
| Shared decision-program nodes | 11 | 63 |
| Maximum ordinary-tree decision depth | 8 | 15 |
| Canonical artifact bytes | 240 | 1,072 |
| Compact artifact bytes | 248 | 768 |
| Positive-probability cutoff, before FP32 storage | 0.35 | 0.325 |

The scores evaluate patient-grouped nested cross-validation procedures on 32 patients and 62 scans. They are **not held-out scores for these final refits**. The current procedure uses equal-patient FIT weights and 161 probability cutoffs; the earlier procedure uses global weighting and 19 cutoffs. Both use inverse-retention weighting. The cohort was reused for exploratory comparisons, so the difference cannot be attributed to one change or treated as a new confirmatory result. See the [complete result comparison](../results/README.md).

Each directory contains:

- `model.json`: the native XGBoost source model.
- `decision-program.canonical.bin`: the shared decision program.
- `decision-program.compact.bin`: its compact runtime representation.
- `binary-tree.json`: the complete ordinary binary tree, with shared continuations expanded.
- `conversion-receipt.json` and `conversion-summary.json`: source hashes, the selected prediction rule, construction evidence and verification scope.

The deployed class rule is `p(class 1) >= cutoff`, using the native full-round FP32 probability. The stored cutoffs are exactly `0.3499999940395355` (earlier) and `0.32499998807907104` (current). Applying a native model's default argmax instead would change the requested rule. Conversion preserves the class decision on the declared input domain; it does not preserve probability outputs.

Both constructions completed and their built-in converter checks passed. Independent exhaustive native-class qualification was **not requested** (`qualification_mode: none`); the receipts do not claim it. A separate [runtime smoke check](../protocol/model-runtime-smoke.json) loaded both formats and found zero canonical/compact differences on ten synthetic feature rows per model. That small format check is not an exhaustive equivalence test or clinical evaluation.

The shipped CLI can [inspect and predict with these artifacts](../README.md#inspect-and-predict-with-a-supplied-decision-program). It needs the declared 54-column features, not raw CT images. Feature meanings and the preparation chain are documented in [data and features](../docs/DATA_AND_FEATURES.md).

## Model license and attribution

All assets in `models/earlier/` and `models/current/` are released under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/). The repository's MIT grant for original code does not cover these model assets.

Attribute the XGBoost analysis and conversion to **Brian Mingus**, the upstream neural features to **LesionLocator and its authors** ([checkpoint record, Zenodo 15174217](https://zenodo.org/records/15174217)), and the source dataset to **TCIA RIDER-LUNG-CT** ([dataset DOI](https://doi.org/10.7937/k9/tcia.2015.u1x8a5nr)). No LesionLocator checkpoint or clinical image/segmentation payload is bundled. Those upstream assets retain their own licenses and terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).
