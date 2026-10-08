#!/usr/bin/env python3
"""Public interface checks using harmless backend stubs; no GPU execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = Path(sys.argv.pop(1)).resolve()
STUB = r'''#!/usr/bin/env python3
import json, pathlib, sys
name=pathlib.Path(sys.argv[0]).name
if name=='class_model_train' and sys.argv[1]=='source-manifest':
    print(json.dumps({'sha256':'0'*64})); sys.exit(0)
if name=='class_model_train' and sys.argv[1]=='--check-plan':
    print(json.dumps({'checked':True})); sys.exit(0)
if name=='class_model_convert': out=pathlib.Path(sys.argv[3]); plan=json.loads(pathlib.Path(sys.argv[2]).read_text())
elif name=='class_model_train': out=pathlib.Path(sys.argv[4]); plan=json.loads(pathlib.Path(sys.argv[2]).read_text())
elif name=='class_study':
    out=pathlib.Path(sys.argv[3]); plan=json.loads(pathlib.Path(sys.argv[1]).read_text())
    data=json.loads(pathlib.Path(sys.argv[2]).read_text())
    if plan.get('validate_only'):
        print(json.dumps({'checked':True,'plan':plan,'data':data,'CUDA_executed':False})); sys.exit(0)
elif name=='class_segmentation_prepare' and sys.argv[1]=='--prepared-images':
    print(json.dumps({'stub':name,'args':sys.argv[1:]})); sys.exit(0)
elif name in ('class_segmentation_prepare','class_segmentation_export'): out=pathlib.Path(sys.argv[2]); plan=json.loads(pathlib.Path(sys.argv[1]).read_text())
elif name in ('class_model_evaluate','class_model_simplify'): out=pathlib.Path(sys.argv[4]); plan=json.loads(pathlib.Path(sys.argv[2]).read_text())
elif name=='decision_programs_model_tools' and sys.argv[1]=='demo': out=pathlib.Path(sys.argv[2]); plan={}
else:
    print(json.dumps({'stub':name,'args':sys.argv[1:]})); sys.exit(0)
if out.exists(): print('output exists',file=sys.stderr); sys.exit(1)
out.mkdir(parents=True)
(out/'stub-plan.json').write_text(json.dumps(plan))
if name=='class_study': (out/'stub-data.json').write_text(json.dumps(data))
print(json.dumps({'stub':name,'output':str(out)}))
'''


class CliChecks(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="decision-programs-cli-checks-")
        self.root = Path(self.scratch.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.cli = self.bin / "decision-programs"
        shutil.copyfile(BINARY, self.cli)
        self.cli.chmod(0o755)
        self.backends = self.root / "libexec" / "decision-programs"
        self.backends.mkdir(parents=True)
        for name in ("class_model_train", "class_model_convert", "class_study", "class_model_evaluate",
                     "class_model_simplify", "decision_programs_model_tools", "rl_session", "class_tree_adapter",
                     "class_segmentation_prepare", "class_segmentation_export", "class_segmentation_score"):
            path = self.backends / name
            path.write_text(STUB)
            path.chmod(0o755)
        self.library = self.root / "libxgboost.so"
        self.library.write_bytes(b"host interface test library, never loaded")
        self.env = dict(os.environ, XGBOOST_LIBRARY=str(self.library),
                        DECISION_PROGRAMS_CACHE_DIR=str(self.root / "cache"))
        self.env.pop("DECISION_PROGRAMS_BACKEND_DIR", None)
        self.source = self.root / "source $(touch injected).json"
        self.source.write_text(json.dumps({"learner": {"learner_model_param": {"num_class": "2"}}}))
        self.model_dir = self.root / "compiled"
        self.model_dir.mkdir()
        self.model = self.model_dir / "model.canonical"
        header = b"CLSGDAG1" + struct.pack("<6I", 1, 2, 2, 2, 3, 1) + bytes(32)
        self.model.write_bytes(header + struct.pack("<12I", 0xffffffff, 0, 0, 0,
                                                  0xffffffff, 1, 0, 0, 0, 0, 0, 1))
        (self.model_dir / "model.compact").write_bytes(b"stub compact bytes")
        self.csv = self.root / "training.csv"
        self.csv.write_text('a,b,label\n-1,2,cat\n0,3,dog\n1,4,cat\n')

    def tearDown(self):
        self.scratch.cleanup()

    def call(self, *args, status=0):
        result = subprocess.run([str(self.cli), *map(str, args)], cwd=self.root,
                                env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        return result

    def dry(self, *args):
        return json.loads(self.call(*args, "--dry-run").stdout)

    def combine_args(self, *, data=None, target="label", extra=()):
        checkpoint = self.root / "completed-teachers"
        checkpoint.mkdir(exist_ok=True)
        return ("combine", "--data", data or self.csv, "--target", target, "--fit-rows", "2",
                "--teachers", '[{"rounds":1,"max_depth":2}]',
                "--meta", '[{"rounds":1,"max_depth":2}]', "--baseline", self.source,
                "--training-checkpoint", checkpoint, "--out", self.root / "composition", *extra)

    def holdout_descriptor(self):
        values, labels = self.root / "holdout.fp32", self.root / "holdout.u32"
        values.write_bytes(struct.pack("<4f", 8, 9, 10, 11))
        labels.write_bytes(struct.pack("<2I", 1, 1))
        return {"format": "dense-fp32-u32-holdout-labels-1", "role": "HOLDOUT",
                "training_allowed": False, "selection_allowed": True, "TEST_read": False,
                "features": 2, "classes": 2, "rows": 2, "row_stride": 2,
                "values_path": str(values), "values_sha256": hashlib.sha256(values.read_bytes()).hexdigest(),
                "labels_path": str(labels), "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                "preprocessing": "caller-supplied FP32 features in development order"}

    def segmentation_inputs(self):
        directory = self.root / "segmentation inputs"
        directory.mkdir(exist_ok=True)
        manifest = directory / "scans.json"
        manifest.write_text(json.dumps({"crop_width_mm": 160, "scans": [
            {"patient_id": 17, "scan_id": 1, "ct_dir": "dicom/../ct series",
             "seg_path": "reference $(touch injected).dcm", "seed_x_pixel": 12,
             "seed_y_pixel": 34, "seed_z_imageposition_mm": -12.5},
            {"patient_id": 17, "scan_id": 2, "ct_dir": str(self.root / "absolute CT"),
             "seg_path": str(self.root / "absolute SEG.dcm")}
        ]}))
        data = directory / "dataset.json"
        data.write_text(json.dumps({
            "format": "grouped-segmentation-dataset-1", "rows": 100, "features": 2,
            "classes": 2, "train_eligible_rows": 50, "values_path": "arrays/../x.fp32",
            "labels_path": "labels.u32", "patient_ids_path": "patients.u32",
            "scan_ids_path": "scans.u32", "train_eligible_path": "eligible.u32",
            "domain_description": "fixed image-only ROI for interface transport checks",
            "scans": [{"patient_id": p, "scan_id": p, "truth_positive_outside_rows": 0}
                      for p in range(5)]}))
        plan = directory / "search.json"
        plan.write_text(json.dumps({
            "base_hyperparameters": [{"rounds": 2, "max_depth": 2}],
            "meta_hyperparameters": [{"rounds": 1, "max_depth": 2}],
            "outer_patient_ids": [0, 3]}))
        return directory, manifest, data, plan

    def test_external_segmentation_export_preserves_supplied_seed_and_resolves_paths(self):
        directory, manifest, data, plan = self.segmentation_inputs()
        dataset = json.loads(data.read_text())
        dataset['manifest'] = json.loads(manifest.read_text())
        data.write_text(json.dumps(dataset))
        out = self.root / 'export with spaces'
        self.call('segmentation-export', '--data', data, '--out', out)
        got = json.loads((out/'stub-plan.json').read_text())
        scan = got['manifest']['scans'][0]
        self.assertEqual(scan['ct_dir'], str(directory/'ct series'))
        self.assertEqual(scan['seg_path'], str(directory/'reference $(touch injected).dcm'))
        self.assertEqual(scan['seed_x_pixel'], 12)
        self.assertFalse((self.root/'injected').exists())

    def test_external_segmentation_score_passes_declared_endpoint_and_predictions(self):
        directory, manifest, data, plan = self.segmentation_inputs()
        cases = directory/'cases.json'
        cases.write_text('{}')
        predictions = self.root/'external predictions'
        predictions.mkdir()
        out = self.root/'external result.json'
        got = json.loads(self.call('segmentation-score', '--data', data, '--export', cases,
                                  '--predictions', predictions, '--out', out).stdout)
        self.assertEqual(got['stub'], 'class_segmentation_score')
        self.assertEqual(got['args'], list(map(str, (data, cases, predictions, out))))

    def test_help_is_available_without_backends(self):
        shutil.rmtree(self.backends)
        for command in ("doctor", "demo", "train", "convert", "study", "hpo", "combine", "rl", "evaluate",
                        "simplify", "predict", "inspect", "explain", "export", "checkpoint", "resume", "proofs", "profile", "import-tree",
                        "prepare-segmentation", "segmentation-study", "segmentation-export", "segmentation-score"):
            self.assertIn("Usage:", self.call(command, "--help").stdout)
        self.assertEqual(self.call("--version").stdout.strip(), "decision-programs 0.1.0")

    def test_prepare_segmentation_resolves_manifest_relative_paths_and_dry_run(self):
        directory, manifest, _, _ = self.segmentation_inputs()
        out = self.root / "prepared"
        result = self.dry("prepare-segmentation", "--manifest", manifest, "--out", out,
                          "--set", "/crop_width_mm=120")
        prepared = result["generated_inputs"]["plan.json"]
        self.assertEqual(result["command"][0], str(self.backends / "class_segmentation_prepare"))
        self.assertEqual(result["command"][2], str(out))
        self.assertEqual(prepared["crop_width_mm"], 120)
        self.assertEqual(prepared["scans"][0]["ct_dir"], str(directory / "ct series"))
        self.assertEqual(prepared["scans"][0]["seg_path"], str(directory / "reference $(touch injected).dcm"))
        self.assertEqual(prepared["scans"][1]["ct_dir"], str(self.root / "absolute CT"))
        self.assertEqual(prepared["scans"][1]["seg_path"], str(self.root / "absolute SEG.dcm"))
        self.assertFalse(result["numerical_execution"])
        self.assertFalse(out.exists())
        self.assertFalse((self.root / "injected").exists())

    def test_prepare_segmentation_backend_transport_and_saved_manifest(self):
        _, manifest, _, _ = self.segmentation_inputs()
        out, saved = self.root / "prepared", self.root / "saved-prepare.json"
        args = ("prepare-segmentation", "--manifest", manifest, "--output", out)
        expected = self.dry(*args)["generated_inputs"]["plan.json"]
        self.call(*args, "--save-plan", saved)
        self.assertEqual(json.loads((out / "stub-plan.json").read_text()), expected)
        self.assertEqual(json.loads(saved.read_text()), expected)
        self.assertFalse((self.root / "injected").exists())

    def test_prepared_images_transport_preserves_reference_paths(self):
        directory, manifest, data, _ = self.segmentation_inputs()
        prepared = directory / "prepared $(touch injected).json"
        exported = directory / "export cases.json"
        plans = directory / "preprocessing plans.json"
        for path in (prepared, exported, plans):
            path.write_text('{}')
        out = self.root / "normalized image features"
        args = ("prepare-segmentation", "--prepared-images", prepared,
                "--reference-data", data, "--export", exported,
                "--preprocessing-plans", plans, "--out", out)
        dry = self.dry(*args)
        got = json.loads(self.call(*args).stdout)
        self.assertEqual(got['stub'], 'class_segmentation_prepare')
        self.assertEqual(got['args'], dry['command'][1:])
        for flag, path in (("--prepared-images", prepared), ("--reference-data", data),
                           ("--export", exported), ("--preprocessing-plans", plans), ("--out", out)):
            self.assertEqual(got['args'][got['args'].index(flag)+1], str(path))
        self.assertFalse((self.root / 'injected').exists())
        self.assertIn('cannot be combined', self.call(*args, '--manifest', manifest, status=2).stderr)
        self.assertIn('reference grid remains fixed', self.call(*args, '--set', '/grid_spacing_mm=1', status=2).stderr)

    def test_prepared_images_require_explicit_reference_and_plans(self):
        directory, manifest, data, _ = self.segmentation_inputs()
        args = ('prepare-segmentation', '--prepared-images', data, '--out', self.root/'out')
        self.assertIn('missing --reference-data', self.call(*args, status=2).stderr)
        self.assertIn('missing --preprocessing-plans', self.call(*args, '--reference-data', data, '--export', data, status=2).stderr)
        self.assertIn('require --prepared-images', self.call('prepare-segmentation', '--manifest', manifest,
                     '--reference-data', data, '--out', self.root/'out', status=2).stderr)

    def test_segmentation_study_resolves_data_and_binds_plan_library(self):
        directory, _, data, plan = self.segmentation_inputs()
        out = self.root / "study output"
        result = self.dry("segmentation-study", "--data", data, "--plan", plan,
                          "--out", out, "--set", "/inner_folds=3")
        self.assertEqual(result["command"][0], str(self.backends / "class_study"))
        self.assertEqual(result["command"][3], str(out))
        settings = result["generated_inputs"]["plan.json"]
        descriptor = result["generated_inputs"]["data.json"]
        self.assertEqual(settings["workflow"], "native-grouped-segmentation-1")
        self.assertEqual(settings["native_library_path"], str(self.library))
        self.assertEqual(settings["native_library_sha256"], hashlib.sha256(self.library.read_bytes()).hexdigest())
        self.assertEqual(settings["inner_folds"], 3)
        self.assertEqual(settings["outer_patient_ids"], [0, 3])
        self.assertNotIn("validate_only", settings)
        for key, name in (("values_path", "x.fp32"), ("labels_path", "labels.u32"),
                          ("patient_ids_path", "patients.u32"), ("scan_ids_path", "scans.u32"),
                          ("train_eligible_path", "eligible.u32")):
            self.assertEqual(descriptor[key], str(directory / name))
            self.assertFalse(Path(descriptor[key]).exists())  # transport must not read pixel arrays
        self.assertFalse(result["numerical_execution"])
        self.assertFalse(out.exists())
        self.call("segmentation-study", "--data", data, "--plan", plan, "--out", out,
                  "--set", "/inner_folds=3")
        self.assertEqual(json.loads((out / "stub-plan.json").read_text()), settings)
        self.assertEqual(json.loads((out / "stub-data.json").read_text()), descriptor)

    def test_segmentation_verification_requires_explicit_request(self):
        _, _, data, plan = self.segmentation_inputs()
        original = json.loads(plan.read_text())
        for mode in ('conversion_gate', 'composed_conversion_gate', 'conversion_on_outer_deployments', 'final_deployment'):
            plan.write_text(json.dumps({**original, mode: {'qualification_mode': 'exhaustive', 'maximum_seconds': 30}}))
            args = ('segmentation-study', '--data', data, '--plan', plan, '--out', self.root/'verify-out')
            default = self.dry(*args)['generated_inputs']['plan.json'][mode]
            self.assertNotEqual(default.get('qualification_mode', 'none'), 'exhaustive')
            self.assertEqual(default['maximum_seconds'], 0)
            timed = self.dry(*args, '--timeout', '60')['generated_inputs']['plan.json'][mode]
            self.assertEqual(timed['maximum_seconds'], 60)
            explicit = self.dry(*args, '--verify')['generated_inputs']['plan.json'][mode]
            self.assertEqual(explicit['qualification_mode'], 'exhaustive')
        plan.write_text(json.dumps(original))
        self.assertIn('--verify requires', self.call(*args, '--verify', '--dry-run', status=2).stderr)
    def test_segmentation_grouped_folds_and_final_deployment_transport(self):
        _, _, data, plan = self.segmentation_inputs()
        settings = json.loads(plan.read_text())
        settings.pop('outer_patient_ids')
        settings.update(outer_folds=5, final_deployment={})
        plan.write_text(json.dumps(settings))
        args = ('segmentation-study', '--data', data, '--plan', plan, '--out', self.root/'five-fold-output')
        default = self.dry(*args)['generated_inputs']['plan.json']
        self.assertEqual(default['outer_folds'], 5)
        self.assertNotIn('outer_patient_ids', default)
        self.assertEqual(default['final_deployment']['qualification_mode'], 'none')
        self.assertEqual(default['final_deployment']['maximum_seconds'], 0)
        verified = self.dry(*args, '--verify', '--timeout', '120')['generated_inputs']['plan.json']
        self.assertEqual(verified['final_deployment']['qualification_mode'], 'exhaustive')
        self.assertEqual(verified['final_deployment']['maximum_seconds'], 120)
        self.assertFalse((self.root/'five-fold-output').exists())

    def test_segmentation_timeout_refuses_invalid_values(self):
        _, _, data, plan = self.segmentation_inputs()
        settings = json.loads(plan.read_text())
        settings['final_deployment'] = {}
        plan.write_text(json.dumps(settings))
        args = ('segmentation-study', '--data', data, '--plan', plan, '--out', self.root/'timeout-output')
        for value in ('-1', 'true', '"60"', 'null', '{}', '1e999'):
            self.assertIn('--timeout must be a finite nonnegative number',
                          self.call(*args, '--timeout', value, '--dry-run', status=2).stderr)
        unlimited = self.dry(*args, '--timeout', '0')['generated_inputs']['plan.json']
        self.assertEqual(unlimited['final_deployment']['maximum_seconds'], 0)

    def test_segmentation_study_check_preserves_no_gpu_validation(self):
        _, _, data, plan = self.segmentation_inputs()
        out = self.root / "checked"
        result = self.dry("segmentation-study", "--data", data, "--plan", plan, "--out", out, "--check")
        self.assertTrue(result["generated_inputs"]["plan.json"]["validate_only"])
        checked = json.loads(self.call("segmentation-study", "--data", data, "--plan", plan,
                                       "--out", out, "--check").stdout)
        self.assertTrue(checked["checked"])
        self.assertFalse(checked["CUDA_executed"])
        self.assertEqual(checked["plan"], result["generated_inputs"]["plan.json"])
        self.assertFalse(out.exists())

    def test_segmentation_study_preserves_or_explicitly_rebinds_library(self):
        _, _, data, plan = self.segmentation_inputs()
        second = self.root / "other library.so"
        second.write_bytes(b"different harmless library fixture")
        settings = json.loads(plan.read_text())
        settings.update(native_library_path=str(second),
                        native_library_sha256=hashlib.sha256(second.read_bytes()).hexdigest())
        plan.write_text(json.dumps(settings))
        args = ("segmentation-study", "--data", data, "--plan", plan, "--out", self.root / "out")
        preserved = self.dry(*args)["generated_inputs"]["plan.json"]
        self.assertEqual(preserved["native_library_path"], str(second))
        self.assertEqual(preserved["native_library_sha256"], settings["native_library_sha256"])
        rebound = self.dry(*args, "--library", self.library)["generated_inputs"]["plan.json"]
        self.assertEqual(rebound["native_library_path"], str(self.library))
        self.assertEqual(rebound["native_library_sha256"], hashlib.sha256(self.library.read_bytes()).hexdigest())

    def test_segmentation_study_refuses_incompatible_workflows(self):
        _, _, data, plan = self.segmentation_inputs()
        args = ("segmentation-study", "--data", data, "--plan", plan, "--out", self.root / "out")
        settings = json.loads(plan.read_text())
        for value in ("native-accuracy-search-1", "native-nonlinear-combination-1", 17):
            plan.write_text(json.dumps(dict(settings, workflow=value)))
            self.assertIn("incompatible segmentation workflow", self.call(*args, "--dry-run", status=2).stderr)
        plan.write_text(json.dumps(settings))
        self.assertIn("incompatible segmentation workflow",
                      self.call(*args, "--set", '/workflow="native-accuracy-search-1"',
                                "--dry-run", status=2).stderr)
        self.assertFalse((self.root / "out").exists())

    def test_relocated_conversion_auto_hashes_and_output_alias(self):
        result = self.dry("convert", "--model", self.source, "--output", self.root / "new output",
                          "--batch-size", "32", "--set", '/domain/allow_nan=false')
        self.assertEqual(result["command"][0], str(self.backends / "class_model_convert"))
        plan = result["generated_inputs"]["plan.json"]
        self.assertEqual(plan["native_library_sha256"], hashlib.sha256(self.library.read_bytes()).hexdigest())
        self.assertEqual(plan["batch_size"], 32)
        self.assertEqual(plan["domain"], {"allow_nan": False})
        self.assertEqual(result["command"][1], str(self.source))
        self.assertFalse((self.root / "injected").exists())

    def test_csv_training_cache_and_saved_plan_are_stable(self):
        saved = self.root / "saved.json"
        args = ("train", "--data", self.csv, "--target", "label", "--output", self.root / "trained")
        first = self.dry(*args, "--save-plan", saved)
        second = self.dry(*args)
        plan = first["generated_inputs"]["plan.json"]
        self.assertEqual(plan, second["generated_inputs"]["plan.json"])
        self.assertEqual(plan, json.loads(saved.read_text()))
        self.assertTrue(Path(plan["dataset"]["values_path"]).is_file())
        self.assertTrue(Path(plan["dataset"]["labels_path"]).is_file())
        self.assertEqual(plan["dataset"]["features"], 2)
        self.assertEqual(plan["dataset"]["classes"], 2)
        self.call(*args)
        mapping = json.loads((self.root / "trained" / "input-transport.json").read_text())
        self.assertEqual(mapping["class_label_mapping"], [{"label": "cat", "class": 0}, {"label": "dog", "class": 1}])

    def test_checkpoint_csv_semantics_survive_new_process(self):
        args = ("hpo", "--data", self.csv, "--target", "label", "--fit-rows", "2",
                "--out", self.root / "search", "--checkpoint", self.root / "checkpoint", "--trial", '{"rounds":2}')
        first = self.dry(*args)
        second = self.dry(*args, "--resume", self.root / "checkpoint")
        self.assertEqual(first["generated_inputs"]["plan.json"], second["generated_inputs"]["plan.json"])
        self.assertEqual(first["generated_inputs"]["data.json"], second["generated_inputs"]["data.json"])
        self.assertEqual(second["command"][-2:], ["--resume", str(self.root / "checkpoint")])

    def test_csv_model_directory_predict_discovers_layout(self):
        inputs = self.root / "inputs.csv"
        inputs.write_text('a,b\n-1,2\n,3\n')
        result = self.dry("predict", "--model", self.model_dir, "--data", inputs)
        plan = result["generated_inputs"]["plan.json"]
        self.assertEqual(plan["canonical_path"], str(self.model))
        self.assertEqual(plan["layout"], "compact")
        self.assertEqual(plan["rows"], 2)
        self.assertEqual(plan["row_stride"], 2)
        self.assertTrue(Path(plan["values"]).is_file())

    def test_hpo_and_combine_bind_completed_source_checkpoint_paths(self):
        sources = [self.root / "teacher-checkpoint", self.root / "other-teachers"]
        for source in sources:
            source.mkdir()
        args = ("--data", self.csv, "--target", "label", "--fit-rows", "2",
                "--output", self.root / "new-study")
        hpo = self.dry("hpo", *args, "--rounds", "1", "--training-checkpoint", sources[0],
                       "--training-checkpoint", sources[1])
        self.assertEqual(hpo["generated_inputs"]["plan.json"]["training_source_checkpoints"],
                         [str(source) for source in sources])
        combine_args = (*args, "--teachers", '[{"rounds":1,"max_depth":2}]',
                        "--meta", '[{"rounds":1,"max_depth":2}]', "--baseline", self.source)
        self.assertIn("combine requires --training-checkpoint", self.call("combine", *combine_args, status=2).stderr)
        combine = self.dry("combine", *combine_args, "--training-checkpoint", sources[0])
        plan = combine["generated_inputs"]["plan.json"]
        self.assertEqual(plan["training_source_checkpoints"], [str(sources[0])])
        self.assertEqual(plan["required_baseline_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertFalse((self.root / "new-study").exists())
        self.assertIn("duplicate", self.call("hpo", *args, "--training-checkpoint", sources[0],
                                              "--training-checkpoint", sources[0], status=2).stderr)
        self.assertIn("directory does not exist", self.call("hpo", *args, "--training-checkpoint",
                                                            self.root / "missing", status=2).stderr)

    def test_evaluation_reuses_mapping_for_one_class_subset(self):
        mapping = {"class_label_mapping": [{"label": "cat", "class": 0}, {"label": "dog", "class": 1}]}
        (self.model_dir / "input-transport.json").write_text(json.dumps(mapping))
        self.csv.write_text('a,b,label\n-1,2,dog\n0,3,dog\n')
        result = self.dry("evaluate", "--model", self.model_dir, "--source", self.source,
                          "--data", self.csv, "--target", "label", "--fit-rows", "1", "--out", self.root / "evaluated")
        data = result["generated_inputs"]["plan.json"]["dataset"]
        self.assertEqual(data["classes"], 2)
        self.assertEqual(Path(data["labels_path"]).read_bytes(), struct.pack("<2I", 1, 1))

    def test_nested_holdout_default_mapping_and_backend_transport(self):
        holdout = self.root / "fresh.csv"
        holdout.write_text('a,b,label\n8,9,dog\n10,11,dog\n')
        args = self.combine_args(extra=("--nested-holdout", "--holdout-data", holdout))
        first = self.dry(*args)
        plan = first["generated_inputs"]["plan.json"]
        nested = plan["nested_holdout"]
        self.assertEqual(nested["depth"], 1)
        self.assertEqual(nested["acceptance_rule"], "strictly_fewer_errors_at_every_level")
        data = nested["dataset"]
        self.assertEqual(data["format"], "dense-fp32-u32-holdout-labels-1")
        self.assertEqual(data["role"], "HOLDOUT")
        self.assertFalse(data["training_allowed"])
        self.assertTrue(data["selection_allowed"])
        self.assertFalse(data["TEST_read"])
        self.assertNotIn("FIT_rows", data)
        self.assertNotIn("VALID_rows", data)
        self.assertEqual(data["classes"], 2)
        self.assertEqual(Path(data["labels_path"]).read_bytes(), struct.pack("<2I", 1, 1))
        for key in ("values", "labels"):
            self.assertEqual(data[key + "_sha256"], hashlib.sha256(Path(data[key + "_path"]).read_bytes()).hexdigest())
        self.assertEqual(plan, self.dry(*args)["generated_inputs"]["plan.json"])
        self.call(*args)
        self.assertEqual(plan, json.loads((self.root / "composition" / "stub-plan.json").read_text()))
        mapping = json.loads((self.root / "composition" / "input-transport.json").read_text())
        self.assertEqual(mapping["class_label_mapping"], [{"label": "cat", "class": 0}, {"label": "dog", "class": 1}])

    def test_nested_holdout_implicit_enable_and_reordered_label_subset(self):
        holdout = self.root / "fresh.csv"
        holdout.write_text('a,b,label\n8,9,dog\n10,11,cat\n')
        implicit = self.dry(*self.combine_args(extra=("--holdout-data", holdout)))
        self.assertEqual(implicit["generated_inputs"]["plan.json"]["nested_holdout"]["depth"], 1)
        result = self.dry(*self.combine_args(extra=("--holdout-depth", "2", "--holdout-data", holdout)))
        nested = result["generated_inputs"]["plan.json"]["nested_holdout"]
        self.assertEqual(nested["depth"], 2)
        self.assertEqual(Path(nested["dataset"]["labels_path"]).read_bytes(), struct.pack("<2I", 1, 0))

    def test_nested_holdout_numeric_looking_categorical_label_keeps_training_id(self):
        self.csv.write_text('a,b,label\n-1,2,1\n0,3,unknown\n1,4,1\n')
        holdout = self.root / "fresh.csv"
        holdout.write_text('a,b,label\n8,9,1\n10,11,1\n')
        result = self.dry(*self.combine_args(extra=("--holdout-data", holdout)))
        data = result["generated_inputs"]["plan.json"]["nested_holdout"]["dataset"]
        self.assertEqual(data["classes"], 2)
        self.assertEqual(Path(data["labels_path"]).read_bytes(), struct.pack("<2I", 0, 0))

    def test_nested_holdout_explicit_partition_rows_transport_and_bounds(self):
        holdout = self.root / "fresh.csv"
        holdout.write_text('a,b,label\n8,9,dog\n10,11,cat\n12,13,dog\n')
        args = self.combine_args(extra=("--holdout-depth", "2", "--holdout-data", holdout))
        result = self.dry(*args, "--set", '/nested_holdout/partition_rows=[1,2]')
        self.assertEqual(result["generated_inputs"]["plan.json"]["nested_holdout"]["partition_rows"], [1, 2])
        for counts in ([], [3], [0, 3], [-1, 4], [1, 1], [2, 2], [1.5, 1.5], [2**64 - 1, 3]):
            self.call(*args, "--set", '/nested_holdout/partition_rows=' + json.dumps(counts), status=2)

    def test_nested_holdout_numeric_subset_keeps_full_class_count(self):
        self.csv.write_text('a,b,label\n-1,2,0\n0,3,2\n1,4,0\n')
        holdout = self.root / "fresh.csv"
        holdout.write_text('a,b,label\n8,9,0\n10,11,0\n')
        result = self.dry(*self.combine_args(extra=("--holdout-data", holdout)))
        data = result["generated_inputs"]["plan.json"]["nested_holdout"]["dataset"]
        self.assertEqual(data["classes"], 3)
        self.assertEqual(Path(data["labels_path"]).read_bytes(), struct.pack("<2I", 0, 0))

    def test_nested_holdout_rejects_missing_data_bad_depth_and_hpo(self):
        for options in (("--nested-holdout",), ("--holdout-depth", "2")):
            self.assertIn("missing --holdout-data", self.call(*self.combine_args(extra=options), status=2).stderr)
        holdout = self.root / "fresh.csv"
        holdout.write_text('a,b,label\n8,9,cat\n10,11,dog\n')
        for depth in ("0", "-1", "1.5", "true", "3", str(2**32)):
            self.call(*self.combine_args(extra=("--holdout-depth", depth, "--holdout-data", holdout)), status=2)
        self.assertIn("only by combine", self.call("hpo", "--data", self.csv, "--target", "label",
                      "--fit-rows", "2", "--out", self.root / "search", "--holdout-data", holdout, status=2).stderr)
        self.assertIn("only by combine", self.call("study", "--nested-holdout", status=2).stderr)

    def test_nested_holdout_rejects_feature_schema_and_unseen_labels(self):
        holdout = self.root / "fresh.csv"
        for content, message in (("b,a,label\n8,9,cat\n10,11,dog\n", "feature names/order"),
                                 ("a,c,label\n8,9,cat\n10,11,dog\n", "feature names/order"),
                                 ("a,b,label\n8,9,bird\n10,11,dog\n", "trained label map")):
            holdout.write_text(content)
            self.assertIn(message, self.call(*self.combine_args(extra=("--holdout-data", holdout)), status=2).stderr)
        self.csv.write_text('a,b,label\n-1,2,0\n0,3,1\n1,4,0\n')
        holdout.write_text('8,9,0\n10,11,1\n')
        self.assertIn("feature names/order", self.call(*self.combine_args(target="last", extra=("--holdout-data", holdout)), status=2).stderr)

    def test_nested_holdout_rejects_reused_source_and_feature_content(self):
        self.assertIn("separate source", self.call(*self.combine_args(extra=("--holdout-data", self.csv)), status=2).stderr)
        holdout = self.root / "copied.csv"
        holdout.write_bytes(self.csv.read_bytes())
        self.assertIn("CSV contents", self.call(*self.combine_args(extra=("--holdout-data", holdout)), status=2).stderr)
        holdout.write_text('a,b,label\n-1,2,dog\n0,3,cat\n1,4,dog\n')
        self.assertIn("feature contents", self.call(*self.combine_args(extra=("--holdout-data", holdout)), status=2).stderr)
        holdout.unlink()
        holdout.symlink_to(self.csv)
        self.assertIn("separate source", self.call(*self.combine_args(extra=("--holdout-data", holdout)), status=2).stderr)

    def test_nested_holdout_descriptor_transport_and_metadata_refusal(self):
        descriptor = self.holdout_descriptor()
        path = self.root / "holdout.json"
        path.write_text(json.dumps(descriptor))
        result = self.dry(*self.combine_args(extra=("--holdout-depth", "2", "--holdout-data", path)))
        self.assertEqual(result["generated_inputs"]["plan.json"]["nested_holdout"]["dataset"], descriptor)
        for update in ({"features": 3}, {"classes": 3}, {"role": "TEST"}, {"training_allowed": True},
                       {"selection_allowed": False}, {"TEST_read": True}, {"FIT_rows": 0}, {"VALID_rows": 0}, {"rows": 0}):
            path.write_text(json.dumps(dict(descriptor, **update)))
            self.call(*self.combine_args(extra=("--holdout-data", path)), status=2)

    def test_nested_holdout_non_csv_development_uses_explicit_label_map(self):
        development = self.dry("hpo", "--data", self.csv, "--target", "label", "--fit-rows", "2",
                              "--out", self.root / "search")["generated_inputs"]["plan.json"]["dataset_source"]
        dev_path, mapping = self.root / "development.json", self.root / "mapping.json"
        dev_path.write_text(json.dumps(development))
        mapping.write_text(json.dumps([{"label": "cat", "class": 0}, {"label": "dog", "class": 1}]))
        holdout = self.root / "fresh.csv"
        holdout.write_text('a,b,label\n8,9,dog\n10,11,dog\n')
        self.assertIn("requires --label-map", self.call(*self.combine_args(data=dev_path, extra=("--holdout-data", holdout)), status=2).stderr)
        result = self.dry(*self.combine_args(data=dev_path, extra=("--holdout-data", holdout, "--label-map", mapping)))
        data = result["generated_inputs"]["plan.json"]["nested_holdout"]["dataset"]
        self.assertEqual(data["classes"], 2)
        self.assertEqual(Path(data["labels_path"]).read_bytes(), struct.pack("<2I", 1, 1))

    def test_refused_existing_output_stays_byte_identical(self):
        output = self.root / "existing"
        output.mkdir()
        keep = output / "keep.txt"
        keep.write_bytes(b"keep exactly")
        self.call("train", "--data", self.csv, "--target", "label", "--out", output, status=1)
        self.assertEqual(list(output.iterdir()), [keep])
        self.assertEqual(keep.read_bytes(), b"keep exactly")

    def test_corrupt_cached_input_is_refused(self):
        args = ("train", "--data", self.csv, "--target", "label", "--out", self.root / "trained")
        plan = self.dry(*args)["generated_inputs"]["plan.json"]
        Path(plan["dataset"]["values_path"]).write_bytes(b"corrupt")
        self.assertIn("changed/corrupt", self.call(*args, status=2).stderr)

    def test_saved_pipeline_preserves_rule_and_opt_in_limits(self):
        pipeline = self.root / "pipeline.json"
        rule = {"kind": "positive_probability_threshold", "threshold": 0.55,
                "probability_class": 1, "comparison": ">=", "threshold_precision": "FP32"}
        source = {"format": "grouped-segmentation-final-native-pipeline-1",
                  "native_library_path": str(self.library),
                  "native_library_sha256": hashlib.sha256(self.library.read_bytes()).hexdigest(),
                  "native_prediction_rule": rule}
        pipeline.write_text(json.dumps(source))
        options = self.root / "conversion-options.json"
        options.write_text(json.dumps({"maximum_seconds": 1, "qualification_mode": "exhaustive"}))
        result = self.dry("convert", "--pipeline", pipeline, "--options", options,
                          "--out", self.root / "replayed")
        self.assertEqual(result["command"][0], str(self.backends / "class_study"))
        generated = result["generated_inputs"]
        self.assertEqual(generated["pipeline.json"], source)
        self.assertEqual(generated["plan.json"]["workflow"], "native-grouped-segmentation-replay-1")
        limits = generated["plan.json"]["conversion"]
        self.assertEqual(limits["maximum_seconds"], 0)
        self.assertEqual(limits["max_expansions"], 0)
        self.assertEqual(limits["qualification_mode"], "none")
        requested = self.dry("convert", "--pipeline", pipeline, "--verify", "--timeout", "12",
                             "--out", self.root / "verified")["generated_inputs"]["plan.json"]
        self.assertEqual(requested["conversion"]["qualification_mode"], "exhaustive")
        self.assertEqual(requested["conversion"]["maximum_seconds"], 12)
        self.call("convert", "--pipeline", pipeline, "--model", self.source,
                  "--out", self.root / "ambiguous", status=2)
        self.call("convert", "--pipeline", pipeline, "--timeout", "-1",
                  "--out", self.root / "bad", status=2)

    def test_malformed_options_and_csv_are_refused(self):
        self.call("convert", "--misspelled", "3", status=2)
        self.call("checkpoint", "--pid", "0", status=2)
        self.csv.write_text('a,b,label\n1,2,cat\n3,dog\n')
        self.assertIn("different column", self.call("train", "--data", self.csv, "--target", "label",
                                                    "--out", self.root / "bad", status=2).stderr)
        self.assertFalse((self.root / "bad").exists())

    def test_invalid_label_maps_do_not_truncate_or_merge(self):
        for mapping in ([{"label": "cat", "class": 0.5}, {"label": "dog", "class": 1}],
                        [{"label": "cat", "class": 0}, {"label": "dog", "class": 0}]):
            path = self.root / "mapping.json"
            path.write_text(json.dumps(mapping))
            self.call("train", "--data", self.csv, "--target", "label", "--label-map", path,
                      "--out", self.root / "bad", status=2)

    def test_rl_schema_and_resume_scope(self):
        result = self.dry("resume", "rl", "--model", self.source, "--output", self.root / "rl",
                          "--resume", self.root / "policy.json", "--episodes", "2", "--diagnostic-trajectories")
        self.assertEqual(result["command"][0], str(self.backends / "rl_session"))
        self.assertEqual(result["command"][5], "-")
        self.assertEqual(result["command"][6], "2")
        self.assertEqual(result["command"][-1], "--diagnostic-trajectories")
        self.assertIn("warm start only", result["generated_inputs"]["RL_semantics"]["resume_scope"])

    def test_demo_no_arguments_selects_new_output(self):
        result = self.dry("demo")
        self.assertEqual(result["command"][1], "demo")
        self.assertIn("decision-programs-demo-", result["command"][2])
        self.assertFalse(Path(result["command"][2]).exists())

    def test_rl_rejects_existing_output_and_invalid_learning_rates(self):
        output = self.root / "prior-rl"
        output.mkdir()
        result = output / "result.json"
        result.write_bytes(b"existing result stays exact")
        self.call("rl", "--model", self.source, "--out", output, status=2)
        self.assertEqual(result.read_bytes(), b"existing result stays exact")
        for rate in ("0", "-0.1", "1.1"):
            self.assertIn("(0,1]", self.call("rl", "--model", self.source, "--out", self.root / "new-rl",
                                            "--learning-rate", rate, status=2).stderr)


if __name__ == "__main__":
    unittest.main()
