# Building Decision Programs

The [Decision Programs repository](https://github.com/mingusb/decision-programs)
provides the `decision-programs` entry point. Its CUDA backends are compiled
from an explicit source list in the root `CMakeLists.txt`; the historical
experiment programs and archived build directories are not build inputs.
The supported native environment is Linux, including Windows through WSL2.
Training, conversion, simplification, inference and numerical explanations
require a CUDA build and an NVIDIA GPU. A host-only build provides the CLI,
structural tools, capability reporting and metadata tests.

## Dependencies

Install dependencies through your operating system or the upstream provider:

* A compiler supporting C++23; GCC 15.2 is the qualified local compiler.
* CMake 3.30 or newer and Ninja for the supplied presets.
* nlohmann/json 3.11 or newer, with its CMake package configuration.
* OpenSSL development headers and the Crypto library, plus POSIX threads and
  dynamic loading support.
* For numerical backends, NVIDIA CUDA Toolkit 13.4 or newer, including NVCC,
  CUDA Runtime development files, driver development files and CCCL headers.
  The GPU driver must support the selected toolkit and GPU architecture.
* For native training and teacher comparisons, a CUDA-enabled XGBoost **3.4.1**
  shared library. This is loaded at runtime using the absolute
  `native_library_path` and SHA-256 declared in a plan. The direct CLI computes
  the required hashes for common workflows. It is neither vendored
  nor patched by this project. Other versions are rejected by the maintained
  native adapter because their numerical behavior has not been qualified.
* For proof replay, Lean must match the `lean-toolchain` in the selected proof
  directory. Installing Lean is separate from building the C++ executable.

On a Debian-derived system the host libraries are commonly provided by
`libssl-dev`, `nlohmann-json3-dev` and `ninja-build`. Install a C++23 compiler and
CMake appropriate for your distribution. NVIDIA's Toolkit installation is
separate. A Python XGBoost wheel can supply `libxgboost.so`; Python is not part
of the native execution path. Advanced saved plans retain the exact library
identity automatically; no manual hashing is required for the direct CLI.

The local toolchain used to check this release is CMake 4.2.3, NVCC 13.4.59,
GCC 15.2, nlohmann/json 3.11.3 and OpenSSL 3.5.5, on an NVIDIA RTX A5000 Laptop
GPU (compute capability 8.6). These versions document the
tested configuration; they do not establish support for every newer compiler
or GPU.

## Configure and build

From a fresh clone:

```sh
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release-host
build/cuda-release/bin/decision-programs --help
```

The preset selects `native`, which detects the GPU visible at configuration
time. For the reproducible architecture setting used in the existing SM86
qualifications, configure explicitly:

```sh
cmake --preset cuda-release -DCMAKE_CUDA_ARCHITECTURES=86
```

Set `CMAKE_CUDA_ARCHITECTURES` explicitly when compiling for another machine,
when no GPU is visible, or when packaging for multiple architectures. Changing
the architecture changes the artifact and requires fresh GPU verification.
If CUDA is installed outside the normal
search path, pass `-DCUDAToolkit_ROOT=/absolute/toolkit` and, if needed,
`-DCMAKE_CUDA_COMPILER=/absolute/toolkit/bin/nvcc`. Do not mix headers and
libraries from unrelated Toolkit versions.

The default build uses two concurrent compiler processes. CUDA conversion
translation units are large; reduce concurrency with `cmake --build
build/cuda-release --parallel 1` when host memory is limited. For profiling
builds, configure with `-DDP_CUDA_LINEINFO=ON`; measured release timings should
use the intended uninstrumented build.

Without CUDA:

```sh
cmake --preset host-tools
cmake --build --preset host-tools
ctest --preset host-tools
build/host-tools/bin/decision-programs --help
```

The host tests validate metadata roles, descriptor bounds, conversion options,
checkpoint transport, structural adaptation, symbolic laws and regional
equation export. They do not run numerical predictions on the CPU. GPU tests
are separate and must be scheduled on a suitable GPU:

```sh
ctest --test-dir build/cuda-release -L gpu --output-on-failure
```

The shared runtime fixture exercises CUDA layout validation and prediction
against independently specified decisions. Host checks do not substitute for
this test or for end-to-end training and conversion qualification.

## Install and relocate

```sh
cmake --install build/cuda-release --prefix /absolute/install/prefix
/absolute/install/prefix/bin/decision-programs --help
```

The frontend installs in `bin`; private backend executables install in
`libexec/decision-programs`. The frontend locates that directory relative to its
own executable, so the installation tree can be moved as a unit. It also
accepts `DECISION_PROGRAMS_BACKEND_DIR` for an explicit backend directory.
Lean source and toolchain declarations install in
`share/decision-programs/proofs` when `DP_INSTALL_PROOFS=ON`.

The binary package does not include NVIDIA driver/Toolkit libraries,
XGBoost, OpenSSL or Lean. The host dynamic loader must be able to locate the
required shared libraries, using the provider's normal loader configuration
or `LD_LIBRARY_PATH`. Saved plans bind dataset files, model bytes, executable
bytes and native-library bytes with SHA-256; use the hashes of the files in
your actual installation; the direct CLI computes these pins for you. CMake
embeds the training source manifest in the
training executable, so installed training does not require the source clone.

A binary tarball can be produced after a successful build:

```sh
cpack --config build/cuda-release/CPackConfig.cmake
```

This packages only the install tree. Source releases come from the clean Git
repository; local research artifacts, private data and build directories are
not CPack source-package inputs. Installation and compilation alone do not
establish that a GPU workflow has passed its correctness checks.

## Replay the Lean proofs

Install the declared Lean 4.34.1 toolchain, then configure the executable path
and request the proof target:

```sh
cmake --preset host-tools -DDP_LEAN_EXECUTABLE=/absolute/lean/bin/lean
cmake --build build/host-tools --target check-proofs
```

The target elaborates the maintained modules in their import order, then runs
`leanchecker` on each compiled module. It retains source hashes, compiler and
checker hashes, logs and a receipt under the build directory. Playground
`TacticProbe` files are excluded. This replays the mathematical models and
theorems; it does not certify the C++/CUDA implementation as a formal refinement.

## Optional CT segmentation study

Build the DICOM preparation adapter with DCMTK 3.6.9 or compatible headers and
libraries installed (`libdcmtk-dev` and `dcmtk-data` on Debian-derived systems):

```sh
cmake --preset cuda-release -DDP_BUILD_DICOM=ON
cmake --build --preset cuda-release
```

A non-system DCMTK installation can be located with `CMAKE_PREFIX_PATH`; its
runtime shared libraries and DICOM dictionary must also be available. The
study itself uses the existing XGBoost CUDA trainer. It does not require DCMTK
when a prepared grouped dataset is supplied. See the
[CLI reference](CLI.md) for preparation, patient-grouped evaluation and plan settings.

Focused study checks can be run independently of a real dataset:

```sh
ctest --test-dir build/cuda-release -R class_study_grouped_segmentation_metadata --output-on-failure
build/cuda-release/bin/class_study_grouped_segmentation_integration_checks \
  /path/to/libxgboost.so /fresh/check-output
```

The GPU integration fixture perturbs only held-out labels, verifies unchanged
model identities and selections, checks full evaluation coverage, and tests
outside-region false negatives and empty-prediction repeatability handling.
The optional preparation GPU check is `class_segmentation_prepare --self-test`; `--self-test-host` checks prepared-array metadata, NPY input and image geometry without launching CUDA kernels.
