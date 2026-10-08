# Contributing to Decision Programs

Read PROJECT_GUIDE.md and docs/building.md before changing the build or public interface.

- Keep one maintained conversion backend. New commands should call the shared implementation rather than introduce another converter.
- Do not modify upstream XGBoost. Integrate through the supported public API.
- Preserve exact class semantics within each declared domain. Distinguish a complete conversion from a bounded or interrupted search.
- Make the common workflow available through decision-programs and keep its help, examples and documentation consistent.
- Keep data, checkpoints, credentials, generated experiment results and local machine paths out of commits.
- Use C++23 and CUDA for the implementation; host code may manage arguments, metadata and I/O.
- Run the relevant CPU tests and GPU checks for changed behavior. State which checks require NVIDIA hardware.
- Preserve third-party notices and report limitations without turning historical research observations into universal guarantees.