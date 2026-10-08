# Source-qualified categorical reinforcement learning

This directory packages the existing Resident6 integration for Decision Programs as original C++23/CUDA source. It samples categorical test orders with a Plackett-Luce policy and updates its 44 logits with REINFORCE, using the difference between completed encoded runtime bytes and the frozen default baseline as reward. The resident loop keeps current sampling, gradients, exact key matching, and updates on CUDA between chunk boundaries.

Support is specialized to the declared `forest_valid_v1` domain: ten finite FP32 numeric features (including signed zeros), four exactly-one wilderness indicators, forty exactly-one soil indicators, and seven classes. The source adapter must accept the model shape, ordered score fold, native objective contract, and resource limits. This is an encoding search; it does not train a new classifier or establish improved predictive accuracy.

The Session constructs and independently audits a complete default runtime against the native predictor before admitting any reward. Policy actions change the categorical lowering order only. Internally qualified costs are sealed into a same-process table; an unseen effective key terminates the schedule before credit for that episode. Only a completed schedule and successful final identity checks publish accepted learned state. Imported policy words provide initial preferences and never supply model authority or cost-table entries.

## Build and checks

Configure Decision Programs with CUDA enabled and build `rl_session` and `rl_resident_checks`. The public `decision-programs rl` command supplies model hashes and the positional backend arguments automatically; `decision-programs rl --help` describes its flags. Dependencies are the repository's CUDA toolchain, nlohmann/json and OpenSSL, plus an explicitly supplied native XGBoost library at runtime. These sources preserve the qualified native-library pin `462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4` (XGBoost 3.4.1). A different build requires separate native-contract qualification.

`rl_resident_checks --cpu` checks metadata, resource-plan rejection, and opaque-token properties. `rl_resident_checks --gpu` exercises current traces, order/state words, chunk boundaries, imported continuation, table mutation, stale schedule, incomplete publication, and missing-key rejection. Run the GPU checks and source-model qualification on the rebuilt release before treating a run as verified.

The numerical source is preserved from the schema1 policy and Resident6 cooperative sampler. Schema1 has a documented commutative XOR alias between episode and policy-version counters; no RNG-quality or unique-counter guarantee is claimed. Public packaging changes only build provenance: source hashes are embedded by CMake, and the current executable hash is rechecked. No private frozen objects, proof receipts, local source checkout, model, dataset, or run log is required to establish the build identity.

## Backend interface

```text
rl_session MODEL SOURCE_SHA LIB FRESH_OUT - TRIALS SEED0 EPISODE0 RATE_BITS MAX_NATIVE_CELLS NATIVE_BATCH_ROWS CHUNK_RECORDS WARMUP_EPISODES [POLICY_STATE_JSON] [--diagnostic-trajectories]
```

`SOURCE_SHA` is SHA256 of the exact input model bytes. `FRESH_OUT` must not already exist. The `-` position is a compatibility token; build identity is embedded. `RATE_BITS` is the unsigned integer word encoding of a positive finite FP64 learning rate no larger than one. Set explicit budgets appropriate to the source. A uniform policy performs the declared warmup episodes without updates to populate qualified costs before training. The optional policy state contains exact logit words and version metadata.

Completed outputs include `incumbent/model.bin`, `learned-policy.json`, `episode-qualification.json`, and `result.json`. Check `passed`, `accepted_learned_state`, and `whole_source_conversion_complete`; an output directory or a partial episode journal does not establish completion. Training source and numerical code are MIT under the repository license. CUDA, XGBoost and other dependencies retain their own licenses.
