# Inputs and fixed feature preparation

The source collection is [TCIA RIDER-LUNG-CT](https://www.cancerimagingarchive.net/collection/rider-lung-ct/). The released cohort metadata identifies the exact CT and DICOM-SEG series plus the independent spreadsheet point. It does not include DICOM files, masks or array data. Download the matching series using TCIA/NBIA and retain each series' own LICENSE/citation material. TCIA assets carry their individual terms; this repository does not assign one blanket dataset license.

Place each CT series under `<dicom-root>/<SeriesInstanceUID>/` and its corresponding single SEG object under the SEG-series UID directory, using a `.dcm` extension. `bind-cohort` verifies those directories and one SEG path, then writes absolute paths only into your ignored local workspace. The C++ loader performs the substantive image, source-reference and geometry checks. Directory existence alone is not data validation.

The 62 scans comprise 30 repeat pairs and two single scans across 32 patients. Two unresolved/invalid supplied-point cases were excluded prospectively from these prepared inputs, as recorded in `cohort.json`. Coordinate interpretation and fallback series choices are preserved in that metadata. No SEG-derived replacement point or per-patient outcome repair is part of the recipe. Patient/acquisition grouping is retained throughout all folds.

Obtain the frozen point-optimized checkpoint from [Zenodo record 15174217](https://zenodo.org/records/15174217), preserving its CC-BY-NC-SA 4.0 license. Required files are `plans.json`, `dataset.json` and `fold_0` through `fold_4/checkpoint_final.pth` under `LesionLocatorSeg/point_optimized`. `protocol/checkpoints.json` records the five expected SHA-256 hashes. The adapter refuses differing checkpoints before neural inference. The upstream GPLv3 source is pinned to `ef073baaa0dcba8dd036ec5a29f130db18c9d580`; no upstream file is patched.

The preparation chain is:

1. C++ DICOM preparation defines the independent 160 mm cube, with 80³ rows at 2 mm, labels, stable row IDs and deterministic stride-32 eligibility; it retains positives outside the crop in scoring metadata.
2. C++ export writes full CT and the supplied independent point. The official CNN adapter normalizes/resamples CT and performs five-fold/eight-mirror point inference.
3. The same C++ twenty-feature kernel runs on the official normalized image, retaining the original physical scoring grid and original label/ID/eligibility arrays. Historical operator names containing `hu` do not mean these values are raw HU.
4. The unchanged frozen binary mask is sampled nearest-neighbor. Continuous foreground probability is obtained by averaging the official logits over five folds/eight mirrors, sampling logits with the fixed FP32 physical grid, then applying the official softmax. All 32 final decoder activations come from fold 0's first unmirrored forward; no channel fitting or outcome-based selection occurs.
5. The C++ append path preserves the first 20 FP32 words and appends mask, probability and 32 activations in the declared order. Outside neural-patch support, the inherited continuous fields are zero-filled with separate extraction support metadata. Those zeros are not evidence of observed background.

CNN extraction reads CT and supplied points, not reference labels. Fitting/validation labels remain solely in the grouped learner/scorer. The CNN feature stage itself remains necessary when applying the converted XGBoost model to a new scan. No clinical deployment claim is made.
