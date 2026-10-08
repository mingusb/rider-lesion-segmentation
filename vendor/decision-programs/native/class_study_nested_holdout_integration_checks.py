#!/usr/bin/env python3
"""Synthetic nested holdout integration fixture; model numerics run in CUDA.

Usage: python3 class_study_nested_holdout_integration_checks.py CLI FRESH_OUTPUT_ROOT
Requires a CUDA build, an NVIDIA GPU, and XGBOOST_LIBRARY pointing to the supported
CUDA-enabled XGBoost library. Missing XGBOOST_LIBRARY returns the test skip code 77.
The output root must not exist. Logs and generated artifacts are retained there.
This interface fixture is not a research benchmark or evidence about real data.
"""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_synthetic_data(root):
    # These deliberately simple targets define the synthetic fixture itself.
    # No model is implemented, fitted, predicted, or scored in Python.
    rng = random.Random(741)
    for name, rows in (("development.csv", 450), ("unseen.csv", 300)):
        lines = ["x,y,label"]
        for _ in range(rows):
            x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
            label = "west" if x < -0.1 else "south" if y < 0.1 else "north"
            lines.append(f"{x:.6f},{y:.6f},{label}")
        (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    shifted = random.Random(855)
    lines = ["x,y,label"]
    for _ in range(300):
        x, y = shifted.uniform(-1, 1), shifted.uniform(-1, 1)
        lines.append(f"{x:.6f},{y:.6f},west")
    (root / "shifted-labels.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_gate(gate, state):
    require(gate["CUDA_computed"] is True, "holdout scoring must run in CUDA")
    require(gate["role"] == "HOLDOUT", "gate dataset role changed")
    require(gate["training_allowed"] is False, "holdout gate allowed training")
    require(gate["selection_allowed"] is True, "holdout confirmation was disabled")
    require(gate["TEST_read"] is False, "holdout gate read TEST")
    require(gate["candidate_source_sha256"] == state["inner_selected"]["model_sha256"],
            "gate did not score the frozen inner winner")
    require(gate["baseline_source_sha256"] == state["baseline"]["model_sha256"],
            "gate did not score the predeclared baseline")


def exercise(cli, root, env):
    write_synthetic_data(root)
    receipts = []

    def run(name, arguments):
        print(json.dumps({"step": name, "status": "running"}), flush=True)
        start = time.monotonic()
        result = subprocess.run([str(cli), *arguments], cwd=root, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
        (root / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        receipt = {"step": name, "exit_code": result.returncode,
                   "seconds": round(time.monotonic() - start, 3)}
        receipts.append(receipt)
        (root / "receipt.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt), flush=True)
        if result.returncode != 0:
            print(result.stdout[-6000:], file=sys.stderr)
            raise RuntimeError(f"{name} command failed with exit code {result.returncode}")

    development = ["--data", "development.csv", "--target", "label", "--fit-rows", "300"]
    combine = ["combine", *development,
               "--teachers", '[{"rounds":1,"max_depth":1},{"rounds":1,"max_depth":4}]',
               "--meta", '[{"rounds":1,"max_depth":2},{"rounds":2,"max_depth":3}]',
               "--baseline", "baseline/selected-model.json",
               "--training-checkpoint", "teachers-checkpoint", "--folds", "3"]
    run("teachers", ["hpo", *development, "--rounds", "2", "--depth", "1",
                     "--trial", '{"max_depth":1}', "--trial", '{"max_depth":4}',
                     "--checkpoint", "teachers-checkpoint", "--output", "teachers"])
    run("baseline", ["hpo", *development, "--rounds", "1", "--depth", "1",
                     "--training-checkpoint", "teachers-checkpoint", "--output", "baseline"])
    run("without-holdout", [*combine, "--output", "without-holdout"])
    gates = ["--nested-holdout", "--holdout-depth", "2", "--holdout-data", "unseen.csv",
             "--checkpoint", "nested-checkpoint"]
    run("nested", [*combine, *gates, "--output", "nested"])
    run("resumed", [*combine, *gates, "--resume", "nested-checkpoint", "--output", "resumed"])
    run("rejected", [*combine, "--holdout-depth", "2", "--holdout-data", "shifted-labels.csv",
                     "--checkpoint", "rejected-checkpoint", "--output", "rejected"])

    plain = read_json(root / "without-holdout/result.json")
    nested = read_json(root / "nested/result.json")
    resumed = read_json(root / "resumed/result.json")
    rejected = read_json(root / "rejected/result.json")
    require("nested_holdout" not in plain, "feature-off result acquired holdout state")
    for name, result in (("nested", nested), ("rejected", rejected)):
        require(result["inner_selected"] == plain["selected"], f"{name} changed the inner winner")
        require(result["candidates"] == plain["candidates"], f"{name} changed the inner candidates")

    accepted = nested["nested_holdout"]
    require(accepted["candidate_frozen_before_outer_scoring"] is True, "inner winner was not frozen")
    require(accepted["TEST_read"] is False, "confirmation accessed TEST")
    require(accepted["status"] == "candidate_accepted", "synthetic fixture should pass both gates")
    require(len(accepted["gates"]) == 2 and accepted["next_gate"] == 2,
            "passing confirmation did not consume exactly two gates")
    require(nested["holdout_gates_computed_this_process"] == 2, "passing gates were not both evaluated")
    require(accepted["final_selected_index"] == accepted["inner_selected_index"], "accepted winner changed")
    for level, gate in enumerate(accepted["gates"], 1):
        verify_gate(gate, accepted)
        require(gate["level"] == level and gate["accepted"] is True, "passing gate order/decision changed")
        # Validate the reported CUDA result; never compute predictions or errors.
        require(gate["candidate_errors"] < gate["baseline_errors"], "passing gate was not strict")

    require(resumed["nested_holdout"] == accepted, "resume changed committed gate decisions")
    require(resumed["holdout_gates_computed_this_process"] == 0, "resume rescored committed gates")
    require(resumed["candidates"] == nested["candidates"], "resume changed the frozen candidates")
    require((root / "nested/selected-bundle.cbor").read_bytes() ==
            (root / "resumed/selected-bundle.cbor").read_bytes(), "resumed deployment bundle changed")

    fallback = rejected["nested_holdout"]
    require(fallback["status"] == "baseline_retained", "shifted synthetic labels should retain baseline")
    require(len(fallback["gates"]) == 1 and fallback["next_gate"] == 1,
            "a rejected first gate did not stop later evaluation")
    require(rejected["holdout_gates_computed_this_process"] == 1, "rejection evaluated a later gate")
    first = fallback["gates"][0]
    verify_gate(first, fallback)
    require(first["accepted"] is False and first["candidate_errors"] == first["baseline_errors"],
            "shifted-label fixture should reject an exact first-gate tie")
    require(fallback["final_selected_index"] == fallback["baseline_index"], "fallback selected another model")
    require((root / "rejected/selected-model.json").read_bytes() ==
            (root / "baseline/selected-model.json").read_bytes(), "fallback did not preserve exact baseline bytes")

    summary = {"fixture": "synthetic-nested-holdout-integration-1", "passed": True,
               "inner_candidates_unchanged": True, "passing_gates": 2,
               "resumed_gate_evaluations": 0, "resumed_bundle_identical": True,
               "first_gate_tie_stops_confirmation": True, "baseline_bytes_preserved": True}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path, help="built or installed decision-programs executable")
    parser.add_argument("output_root", type=Path, help="fresh directory for synthetic inputs and retained results")
    arguments = parser.parse_args()
    library = os.environ.get("XGBOOST_LIBRARY", "").strip()
    if not library:
        print("SKIP: XGBOOST_LIBRARY is required for the CUDA integration fixture", flush=True)
        return 77
    try:
        cli, root = arguments.cli.resolve(), arguments.output_root.resolve()
        require(cli.is_file() and os.access(cli, os.X_OK), "CLI executable is missing or not executable")
        library_path = Path(library).expanduser().resolve()
        require(library_path.is_file(), "configured XGBOOST_LIBRARY does not exist")
        require(not root.exists(), "output root already exists; use a fresh directory")
        root.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, XGBOOST_LIBRARY=str(library_path),
                   DECISION_PROGRAMS_CACHE_DIR=str(root / "input-cache"))
        exercise(cli, root, env)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
