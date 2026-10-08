# Decision Programs

Decision Programs turns trained XGBoost classifiers into compact, executable decision programs. It also provides GPU training and model-combination workflows, exact rule extraction, simplification, conversion checkpoints, and a research integration of reinforcement learning.

The numerical implementation is C++23 and CUDA. Lean sources formalize the mathematical construction and rewrite rules. Brian Mingus develops this project as research software under the MIT license.

[Read the paper](docs/paper/decision-programs-paper.pdf) · [Paper source and build](docs/paper/) · [Build instructions](docs/building.md) · [CLI reference](docs/CLI.md) · [License](LICENSE)

## Start here

Build the project, then run:

```sh
decision-programs --help
decision-programs doctor
decision-programs demo
```

The CLI is the public entrypoint. Its subcommand help explains the inputs and output formats. The guide below distinguishes common commands from advanced study configuration and specialized research formats.

## Build and run

On Linux or WSL2 with the [required compiler and libraries](docs/building.md):

```sh
git clone https://github.com/mingusb/decision-programs.git
cd decision-programs
cmake --preset cuda-release
cmake --build --preset cuda-release
cmake --install build/cuda-release --prefix "$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"

decision-programs doctor --gpu
decision-programs demo
```

The default CUDA preset detects the local GPU architecture. Training and native-source conversion use a CUDA-enabled XGBoost 3.4.1 library; the CLI discovers it from the active installation or accepts `--library /path/to/libxgboost.so`.

### Your first model

The included CSV files are small synthetic **interface examples**, not research benchmarks:

```sh
decision-programs train --data examples/quickstart.csv --target label \
  --rounds 2 --depth 2 --output trained
decision-programs convert --model trained/model.json --output compiled
decision-programs predict --model compiled --data examples/predict.csv
decision-programs explain --model compiled --data examples/predict.csv --output paths.json
decision-programs export --model compiled --output equations.json
```

Use your own CSV and target-column name in the training command. Prediction CSVs contain feature columns in the same order, without the target. Use a new output path for each run; completed results are preserved.
## What the compiler does

An XGBoost classifier combines scores from many trees. This project constructs a decision program that returns the same predicted class within a declared input domain. It uses region reasoning, exact simplification, and shared substructures to avoid expanding every possible combination of source-tree paths.

A shared decision program is a directed acyclic graph: several decisions can reuse the same continuation. Expanding every reference into a separate copy would produce a conventional tree, often a much larger one. The compact representation retains those shared references.

Class preservation is the target. Equal predicted classes do not imply equal probabilities, SHAP values, or internal score representations. Supported objectives, native-library qualification, floating-point behavior and declared categorical constraints determine which guarantees a particular conversion can establish. Unsupported inputs must fail explicitly.

The compiler reports complete, bounded, interrupted and failed construction separately. A checkpoint or a large amount of settled coverage is not a complete classifier.

## Capabilities

| Task | Public interface | Scope |
| --- | --- | --- |
| Environment diagnostics | `doctor` | Reports installed backends and required dependencies |
| Train and convert | `train`, `convert` | Uses the maintained native training and adaptive conversion components |
| Predict and inspect | `predict`, `inspect` | Executes or describes a supported saved decision program |
| Extract rules and explain decisions | `export`, `explain` | Explicit decision equations and individual decision traces |
| Simplify | `simplify` | Sound supported rewrites; no universal minimum-size claim |
| Hyperparameter search and combination | `hpo`, `combine`, `study` | Native study workflows with explicit training and selection partitions |
| Conversion persistence | `checkpoint`, `resume` | Resumes saved construction state; supported study workflows also save their controller state |
| Formal results | `proofs` | Lean proof sources, distinct from runtime qualification |
| Performance diagnosis | `profile` | Entry to supported external diagnostic tools |
| Learned construction choices | `rl` | Specialized qualified research integration; see its format and qualification requirements |

Run `decision-programs capabilities` to see what is available in the installed build. A host-only build supports help and metadata tools; numerical backends require NVIDIA CUDA.

## Learning and combining models

The study system supports XGBoost training, hyperparameter search, and nonlinear model combination. Models used to produce training features for a later learning stage must exclude those rows during fitting. This prevents a later model from learning from unrealistically optimistic in-sample predictions.

Use a separate validation partition to select among candidates, retaining the best eligible model as the search grows. Evaluate the selected model once on a held-out test partition. A larger search can discover useful alternatives; it does not guarantee a better test result.

The reinforcement-learning integration learns construction choices within its supported search representation. Qualification and exact acceptance rules remain responsible for correctness. Learned proposals do not acquire authority merely because the policy predicts they are good.

## Explanations

A decision trace shows which tests the program executed for an input and which class it returned. An exported decision equation describes a supported program's actual branching behavior. Specialized regional exports also expose binary and fuzzy expressions where the format supports them.

These are useful alongside feature attribution. SHAP describes attribution relative to its chosen background and value function; an executable rule describes a condition under which the decision program takes a branch or returns a class. An exact rule does not by itself establish causality, fairness, or a property of the real world.

## Progress on large conversions

Coverage accounting tracks source-threshold cells settled by construction and proof. Work sampling separately estimates remaining computation. The two answer different questions: how much of the input partition has been covered, and how much processing may remain.

The historical large Forest experiment spans about 100 quintillion source cells and demonstrated settled coverage of about 43 quadrillion cells in a bounded run. **The full 448-tree Forest conversion is not complete.** Estimates are conditional forecasts, not completion guarantees. Worst-case expansion can remain exponential.

Checkpoints are optional and use safe construction boundaries. A configured persistence cadence is separate from the GPU batch cadence. Hot-loaded search modules can change supported proof-search strategies; they must satisfy the existing acceptance checks.

## Reported research results

These examples are documented in the accompanying manuscript and describe particular runs, not universal compression ratios or promises for every dataset.

- A complete 256-tree classifier trained on the UCI Internet Firewall dataset was reduced from roughly 120,000 source nodes to roughly 1,600 decision nodes, about 52 kB, in about one second in the measured environment.
- A complete 35-tree Covertype example covered roughly 300 million source-induced cells with a shared representation of about 12 kB.
- A separate completed example shrank from about 6.3 kB to 4.2 kB through exact simplification.
- The RL component was integrated and exercised with qualification checks. Its presence is a software result; the reported transfer comparison did not demonstrate an advantage over its uniform control.
- The original supervised-combination study improved validation selection but not its test result. A refitted follow-up made 691 test errors with the combiner versus 700 with the baseline. One or two confirmation holdouts retained the combiner; four or eight rejected it. These are retrospective comparisons on previously observed data, not a guarantee that more holdouts improve generalization.

## Source map

- `native/`: C++23/CUDA implementations and public CLI.
- `native/class_conversion/`: components of the shared adaptive converter.
- `native/rl/`: the qualified RL research integration.
- `formal/`: Lean source modules.
- `cmake/`: portable build and test support.
- `docs/`: installation, CLI usage and release details.

The current release focuses on the supported decision-program workflow; private datasets, experiment outputs, local environments and checkpoints are not distributed.

## Contributing

Small reproducible examples are particularly useful. Include the command, software versions, expected behavior and actual result. For conversion issues, report whether the search completed and which input domain was declared. Do not attach private data or credentials.

New optimizations should preserve the supported semantics and come with an appropriate comparison against the existing implementation. Changes to upstream XGBoost are outside this repository's scope.

## License and citation

Brian Mingus's original project code is MIT licensed. External dependencies and historical third-party code retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Citation metadata is available in [CITATION.cff](CITATION.cff).

The README poster was created with OpenAI's built-in image-generation tool to match the author's pinned-project collection. Its text has an accessible equivalent in the image's alt text and this guide.