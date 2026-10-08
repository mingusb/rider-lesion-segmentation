# RIDER lesion segmentation with XGBoost and exact decision programs

Reproducible research for the selected 54-feature RIDER experiment: 32 patient groups, 62 CT scans, frozen LesionLocator features, eight XGBoost base configurations and four nonlinear combiner configurations. Patient-grouped nested validation selects hyperparameters and probability cutoffs. The saved XGBoost stage can be converted to an exact class decision program and expanded as an ordinary binary tree.

[Read the paper](paper/rider-neural-decision-programs.pdf) · [Compare results](results/README.md) · [Model cards](models/README.md)

**The CNN is still required to produce the inputs.** This repository does not convert or distill the neural network. Exact conversion concerns the selected XGBoost classifier on its declared feature domain; it does not preserve probabilities, establish clinical validity, or give the final all-patient model a new held-out score.

The cohort was reused for exploratory comparisons. The released recipe is not a newly untouched confirmatory test. Whether RIDER appeared in the external pretrained checkpoint's training population is unverified. Results, paper and the selected deployment artifact are kept distinct.

## Contents

- `plans/latest54-cv.json`: selected recipe, evaluation only.
- `plans/latest54-final.json`: same nested evaluation followed by all-patient selection/refit (the paper's latest run).
- `plans/cv-b8-dense161.json`: paired 161-cutoff versus original 19-cutoff analysis, with shared fitted models.
- `protocol/cohort.json`: public acquisition identifiers, supplied prompts, declared exclusions and coordinate provenance; no pixels or contours.
- `protocol/features.json`: exact 54-column meaning and inherited CNN support convention.
- `vendor/decision-programs/`: hash-pinned snapshot of the same maintained C++23/CUDA implementation used here, with its MIT license and notices. The public base commit lacks the local grouped-study additions, so a clone of that commit alone is insufficient.
- `scripts/`: orchestration, external-CNN adapters and small result summarizer.
- `paper/`, `results/`, `models/`: release manuscript, evidence and selected XGBoost-stage artifacts.

## Build and run

Use Linux or WSL2 with NVIDIA CUDA 13.4+, a CUDA-compatible C++23 compiler, CMake 3.30+, Ninja, OpenSSL, nlohmann/json and DCMTK development packages. The checked environment used GCC 15.2, NVCC 13.4.59, CMake 4.2.3 and an SM 86 GPU. The plan reserves 12 GiB for explicit CUDA arrays; opaque native allocations and CNN preparation need additional headroom. See [environment details](docs/ENVIRONMENT.md).

```sh
# Set CUDACXX when nvcc is not on PATH (adjust for your installation).
export CUDACXX=/usr/local/cuda/bin/nvcc
python3 scripts/reproduce.py build --architecture 86
```

If you already have the corresponding locally prepared 54-feature dataset and CUDA-enabled XGBoost 3.4.1 library, evaluation needs only:

```sh
python3 scripts/reproduce.py study --data /path/to/dataset.json \
  --library /path/to/libxgboost.so --out work/preflight --check
python3 scripts/reproduce.py study --data /path/to/dataset.json \
  --library /path/to/libxgboost.so --out work/cv
python3 scripts/summarize_result.py --result work/cv/result.json --out work/summary
```

For the complete paper procedure, select `--plan plans/latest54-final.json` and a fresh output directory. That plan includes an all-patient deployment fit after evaluation. Saved plans bind the installed native-library hash at execution. Numerical results can differ across native/toolchain/GPU implementations and reruns; pinned source and recipe are not a bitwise training reproducibility guarantee.

## Recreate the inputs

Acquire the public RIDER-LUNG-CT series through TCIA under the applicable asset terms, and the external point-optimized LesionLocator checkpoints under their separate noncommercial/share-alike license. No data or CNN checkpoints are included. [Data and preparation instructions](docs/DATA_AND_FEATURES.md) explain the source identifiers, directory layout and excluded scans.

```sh
bash scripts/setup_cnn.sh
python3 scripts/reproduce.py bind-cohort --dicom-root /path/to/series-by-UID \
  --out work/cohort-local.json
python3 scripts/reproduce.py prepare --manifest work/cohort-local.json \
  --model /path/to/LesionLocatorSeg/point_optimized --python .venv/bin/python
```

This writes the local descriptor `work/prepared54/dataset.json`. Preparation verifies the five checkpoint hashes, creates independent physical scoring grids, runs the official frozen network, and appends its mask, continuous probability and 32 fixed decoder channels. It retains the original labels, patient identities and sampling eligibility. The normalized CT feature index is image-only; aligned CNN-training label arrays are not needed. Use a fresh workspace for a new preparation; existing receipts are preserved.

## Convert the selected deployment

After `latest54-final.json` finishes, use its saved pipeline through the same converter:

```sh
python3 scripts/reproduce.py convert \
  --pipeline work/final-study/final-native-pipeline.json \
  --library /path/to/libxgboost.so --out work/converted
```

A complete conversion receipt is required before using the artifact. A bounded or interrupted construction is not a complete classifier. Expanded tree size can exceed the compact shared decision-program size substantially. The supplied final tree represents one all-patient refit, while the accuracy table describes the nested evaluation procedure.

## Inspect and predict with a supplied decision program

The build includes the existing `decision_programs_model_tools` runtime. Use explicit files for the supplied canonical and compact layouts:

```sh
work/core/bin/decision-programs inspect \
  --model models/current/decision-program.canonical.bin \
  --compact models/current/decision-program.compact.bin
work/core/bin/decision-programs predict \
  --model models/current/decision-program.canonical.bin \
  --compact models/current/decision-program.compact.bin \
  --layout compact --data /path/to/features54.csv --out work/predictions.json
```

Prediction inputs must contain exactly 54 columns in `protocol/features.json` order, with no target column. These are feature rows, not raw CT pixels. The runtime returns class IDs, not CNN or XGBoost probabilities. Inspection and inference require CUDA, but do not require XGBoost or CNN weights when the feature rows already exist. Substitute `models/earlier/` to inspect the earlier artifact. Supplying both files validates their shared model identity; `--layout canonical` selects the other runtime layout.

## Validation and licenses

The source snapshot's focused host/GPU tests and the original study receipts are distinct from release packaging checks. The portable wrapper changes only path/configuration orchestration; no new clinical rerun is claimed by packaging. Run the relevant hardware tests on a new environment before interpreting newly generated results.

Original analysis/orchestration and the maintained core use MIT except explicitly marked components. `scripts/lesionlocator/` is GPL-3.0-or-later and interacts with the separately obtained pinned GPLv3 upstream project. The released model assets and external checkpoints have separate CC-BY-NC-SA 4.0 scope; the MIT code license does not cover them. Read [third-party scope](THIRD_PARTY_NOTICES.md). Cite the RIDER collection DOI [10.7937/k9/tcia.2015.u1x8a5nr](https://doi.org/10.7937/k9/tcia.2015.u1x8a5nr), LesionLocator and the paper accompanying this repository.
