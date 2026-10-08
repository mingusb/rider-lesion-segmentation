# Third-party notices

The MIT license in this repository applies to Brian Mingus's original code and documentation. It does not replace licenses attached to third-party material, including material in older Git revisions.

- **NVIDIA reference code in repository history:** the original NVIDIA copyright, redistribution conditions and disclaimer are retained with those files. That code is not part of the current default build.
- **CUDA, CUB, Thrust and cuBLAS:** supplied by the user's NVIDIA toolkit or packages, under their respective licenses. They are dependencies, not relicensed by this project.
- **XGBoost:** an external Apache-2.0 dependency, loaded through its C API. This release does not modify or redistribute the XGBoost library.
- **nlohmann/json:** an external MIT-licensed dependency.
- **OpenSSL:** an external dependency, with its upstream license applying to the installed version.
- **zlib:** an external dependency under the zlib license.
- **Lean and its standard library:** external dependencies under their upstream licenses.
- **ICML 2026 LaTeX style:** embedded in the paper source under `docs/paper/`. Upstream provenance and attribution comments are retained. See the [official ICML author instructions](https://icml.cc/Conferences/2026/AuthorInstructions). The repository MIT license does not relicense this third-party style.

Public datasets are not bundled. Users provide datasets and are responsible for the applicable dataset terms.