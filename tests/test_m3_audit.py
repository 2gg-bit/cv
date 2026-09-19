"""CPU/NumPy audit tests; no Torch, MMCV, CUDA, checkpoint, or dataset needed."""
import ast
import copy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_m3_targets", ROOT / "tools/audit_m3_targets.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def annotation(count=186):
    return dict(images=[dict(id=i, file_name="{}.jpg".format(i)) for i in range(count)],
                categories=[dict(id=1, name="ship")],
                annotations=[dict(id=10, image_id=0, category_id=1, bbox=[0, 0, 10, 10])])


def candidate():
    return dict(anchor_index=0, anchor_bbox=[1., 1., 9., 9.],
                teacher1_bbox=[0., 0., 10., 10.], teacher2_bbox=[2., 2., 8., 8.],
                selected_bbox=[0., 0., 10., 10.], pseudo_score=.9123456789123,
                baseline_uncertainty=[.015] * 4, teacher1_uncertainty=[.005] * 4,
                teacher2_uncertainty=[.025] * 4, source_id=1,
                teacher1_valid=True, teacher2_valid=True,
                branch1_original_reg_eligible=True, branch2_original_reg_eligible=True)


def payload(count=186):
    return dict(schema_version=1, images=[dict(image_id=i, file_name="{}.jpg".format(i),
                                             candidates=[candidate()] if i == 0 else []) for i in range(count)])


def collection(path):
    path.mkdir()
    audit.write_json(path / "candidates.json", payload())
    audit.write_json(path / "dev.json", annotation())
    (path / "resolved_config.py").write_text("model = dict()\n")
    metadata = dict(collection_complete=True, fixed_dev_relative_path="ssdd_dev_protocol/data/dev.json",
                    candidates_sha256=audit.sha256_file(path / "candidates.json"),
                    dev_sha256=audit.sha256_file(path / "dev.json"),
                    resolved_config_sha256=audit.sha256_file(path / "resolved_config.py"),
                    image_ids=list(range(186)), seed=678, reg_pseudo_threshold=.02)
    audit.write_json(path / "metadata.json", metadata)


class AuditGeometryTests(unittest.TestCase):
    def test_original_pixels_use_one_scale_for_every_box(self):
        transform = audit.ensure_transform(dict(scale_factor=[2, 3, 2, 3], flip=False))
        boxes = np.array([[2., 3., 10., 21.], [4., 6., 16., 24.]])
        np.testing.assert_array_equal(audit.original_boxes(boxes, transform), [[1, 1, 5, 7], [2, 2, 8, 8]])
        self.assertEqual(audit.original_boxes([], transform).shape, (0, 4))

    def test_explicit_transform_and_invalid_transform_guards(self):
        matrix = np.diag([2., 2., 1.])
        np.testing.assert_array_equal(audit.ensure_transform(dict(transform_matrix=matrix)), matrix)
        for meta in (dict(scale_factor=2, flip=True), dict(scale_factor=[1, 2]),
                     dict(scale_factor=0), dict(transform_matrix=np.zeros((3, 3))),
                     dict(transform_matrix=[[1, 0, 2], [0, 1, 0], [0, 0, 1]])):
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                audit.ensure_transform(meta)

    def test_invalid_geometry_is_not_repaired(self):
        result = audit.original_boxes([[8, 2, 2, 6]], np.diag([2, 2, 1]))
        np.testing.assert_array_equal(result, [[4, 1, 1, 3]])
        self.assertEqual(audit.pairwise_iou(result, [[0, 0, 10, 10]])[0, 0], 0)
        self.assertEqual(audit.pairwise_iou([[None, 0, 1, 1]], [[0, 0, 10, 10]])[0, 0], 0)

    def test_continuous_iou(self):
        actual = audit.pairwise_iou([[0, 0, 10, 10], [0, 0, 5, 10]], [[0, 0, 10, 10]])
        np.testing.assert_array_equal(actual, [[1.], [.5]])


class AuditAnalysisTests(unittest.TestCase):
    def test_empty_images_are_counted_and_floats_remain_exact(self):
        predictions = payload()
        report, details = audit.analyze_candidates(predictions, annotation())
        self.assertEqual(report["num_images"], 186)
        self.assertEqual(report["num_empty_images"], 185)
        self.assertEqual(report["source_counts"], {"0": 0, "1": 1, "2": 0})
        baseline = report["cohorts"]["all_anchors_with_gt"]["baseline"]
        selected = report["cohorts"]["all_anchors_with_gt"]["selected"]
        self.assertAlmostEqual(baseline["mean_iou"], .64)
        self.assertAlmostEqual(selected["mean_iou_delta"], .36)
        self.assertEqual(selected["thresholds"]["0.85"]["unique_gt_covered"], 1)
        self.assertEqual(details[0]["gt_id"], 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            audit.write_json(path, predictions)
            self.assertEqual(audit.read_json(path), predictions)

    def test_all_methods_keep_anchor_gt_instead_of_rematching(self):
        predictions, gt = payload(), annotation()
        gt["annotations"].append(dict(id=11, image_id=0, bbox=[20, 0, 10, 10]))
        row = predictions["images"][0]["candidates"][0]
        row.update(teacher2_bbox=[20, 0, 30, 10], teacher2_valid=False)
        _, details = audit.analyze_candidates(predictions, gt)
        self.assertEqual(details[0]["gt_id"], 10)
        self.assertEqual(details[0]["ious"]["teacher2"], 0)

    def test_original_eligibility_uses_average_for_both_branches(self):
        report, details = audit.analyze_candidates(payload(), annotation())
        self.assertTrue(details[0]["branch1_original_reg_eligible"])
        self.assertTrue(details[0]["branch2_original_reg_eligible"])
        self.assertTrue(details[0]["teacher1_uncertainty_eligible"])
        self.assertFalse(details[0]["teacher2_uncertainty_eligible"])
        self.assertEqual(report["cohorts"]["branch1_original_reg_eligible"],
                         report["cohorts"]["branch2_original_reg_eligible"])

    def test_regression_eligibility_uses_strict_less_than(self):
        predictions = payload()
        predictions["images"][0]["candidates"][0].update(
            baseline_uncertainty=[.02] * 4, branch1_original_reg_eligible=False,
            branch2_original_reg_eligible=False)
        report, _ = audit.analyze_candidates(predictions, annotation())
        self.assertEqual(report["cohorts"]["branch1_original_reg_eligible"]["selected"]["count"], 0)

    def test_random_control_does_not_consume_rng_and_filters_invalid_sources(self):
        python_before, numpy_before = random.getstate(), np.random.get_state()
        chosen = [audit.random_source(7, i, 678, True, True) for i in range(30)]
        self.assertEqual(chosen, [audit.random_source(7, i, 678, True, True) for i in range(30)])
        self.assertEqual(set(chosen), {1, 2})
        self.assertEqual(audit.random_source(7, 0, 678, False, False), 0)
        self.assertEqual(audit.random_source(7, 0, 678, False, True), 2)
        self.assertEqual(audit.random_source(7, 0, 678, True, False), 1)
        self.assertEqual(random.getstate(), python_before)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_before[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_before[1])
        self.assertEqual(current_numpy[2:], numpy_before[2:])

    def test_oracle_is_diagnostic_and_cannot_change_candidates(self):
        predictions = payload()
        before = copy.deepcopy(predictions)
        row = predictions["images"][0]["candidates"][0]
        row.update(teacher1_valid=False, selected_bbox=row["anchor_bbox"].copy(), source_id=0)
        frozen = copy.deepcopy(predictions)
        report, details = audit.analyze_candidates(predictions, annotation())
        self.assertEqual(predictions, frozen)
        self.assertEqual(details[0]["oracle_diagnostic_source_id"], 0)
        self.assertAlmostEqual(details[0]["ious"]["oracle_diagnostic_only"], .64)
        self.assertIn("diagnostic", report["oracle_rule"])
        self.assertNotEqual(before, frozen)

    def test_multiple_anchors_cover_one_gt_once_and_no_gt_is_explicit(self):
        predictions = payload()
        duplicate = candidate()
        duplicate["anchor_index"] = 1
        predictions["images"][0]["candidates"].append(duplicate)
        predictions["images"][1]["candidates"].append(candidate())
        report, _ = audit.analyze_candidates(predictions, annotation())
        self.assertEqual(report["candidates_without_gt"], 1)
        self.assertEqual(report["cohorts"]["all_anchors_with_gt"]["selected"]["thresholds"]["0.5"]["unique_gt_covered"], 1)

    def test_no_candidates_produces_null_not_nan_statistics(self):
        predictions = payload()
        predictions["images"][0]["candidates"] = []
        report, details = audit.analyze_candidates(predictions, annotation())
        self.assertEqual(details, [])
        self.assertIsNone(report["cohorts"]["all_anchors_with_gt"]["selected"]["mean_iou"])
        json.dumps(report, allow_nan=False)


class AuditIntegrityTests(unittest.TestCase):
    def test_output_must_be_new_and_files_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = audit.prepare_output_dir(Path(directory) / "new")
            with self.assertRaises(ValueError):
                audit.prepare_output_dir(path)
            audit.write_json(path / "result.json", {"value": 1})
            with self.assertRaises(FileExistsError):
                audit.write_json(path / "result.json", {"value": 2})
            self.assertEqual(audit.read_json(path / "result.json"), {"value": 1})

    def test_missing_duplicate_wrong_ids_or_filenames_fail(self):
        for case in ("missing", "duplicate", "wrong", "filename"):
            predictions = payload()
            if case == "missing":
                predictions["images"].pop()
            elif case == "duplicate":
                predictions["images"][-1]["image_id"] = 0
            elif case == "wrong":
                predictions["images"][-1]["image_id"] = 999
            else:
                predictions["images"][0]["file_name"] = "wrong.jpg"
            with self.subTest(case=case), self.assertRaises(ValueError):
                audit.analyze_candidates(predictions, annotation())

    def test_source_guard_and_gt_pipeline_guard(self):
        for field, value in (("source_id", 4), ("teacher1_valid", False),
                             ("selected_bbox", [1, 1, 9, 9]), ("anchor_index", 2)):
            predictions = payload()
            predictions["images"][0]["candidates"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.analyze_candidates(predictions, annotation())
        for pipeline in ([dict(type="LoadAnnotations")], [dict(type="Collect", keys=["img", "gt_bboxes"])],
                         [dict(type="MultiScaleFlipAug", flip=True, transforms=[])],
                         [dict(type="MultiScaleFlipAug", img_scale=[(100, 100), (200, 200)], transforms=[])]):
            with self.subTest(pipeline=pipeline), self.assertRaises(ValueError):
                audit.validate_pipeline(pipeline)
        with self.assertRaises(ValueError):
            audit.validate_dev_path(ROOT / "data/ssdd/annotations/test.json")

    def test_collection_hashes_and_analysis_readonly(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "collection", Path(directory) / "analysis"
            collection(source)
            before = {p.name: audit.sha256_file(p) for p in source.iterdir()}
            audit.run_analysis(source, audit.prepare_output_dir(target))
            self.assertEqual(before, {p.name: audit.sha256_file(p) for p in source.iterdir()})
            self.assertTrue((target / "analysis.json").is_file())
            (source / "candidates.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                audit.load_collection(source)

    def test_nonfinite_diagnostics_are_explicit_json_null(self):
        self.assertEqual(audit.safe_list([np.nan, np.inf, 1.234567890123]), [None, None, 1.234567890123])

    def test_help_has_no_gpu_dependency(self):
        for suffix in (["--help"], ["collect", "--help"], ["analyze", "--help"]):
            result = subprocess.run([sys.executable, str(ROOT / "tools/audit_m3_targets.py")] + suffix,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        tree = ast.parse((ROOT / "tools/audit_m3_targets.py").read_text())
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertNotIn("torch", [alias.name for node in top_imports if isinstance(node, ast.Import) for alias in node.names])
        collect = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "collect")
        calls = [node for node in ast.walk(collect) if isinstance(node, ast.Call)]
        loader = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "load_checkpoint")
        self.assertTrue(next(keyword.value.value for keyword in loader.keywords if keyword.arg == "strict"))
        extract = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "extract_teacher_info")
        self.assertEqual([argument.id for argument in extract.args], ["img", "metas"])
        self.assertEqual(extract.keywords, [])

    def test_fixed_config_is_validated_without_mutation(self):
        config = dict(type="DualTeacher", train_cfg=dict(m3_enabled=True,
                      m3_target_mode="lower_uncertainty", m3_min_anchor_iou=.5))
        original = copy.deepcopy(config)
        audit.validate_m3_config(config)
        self.assertEqual(config, original)
        for key, value in (("m3_enabled", False), ("m3_enabled", 1),
                           ("m3_target_mode", "teacher1"), ("m3_target_mode", "original"),
                           ("m3_min_anchor_iou", .4), ("m3_min_anchor_iou", float("nan")),
                           ("m3_min_anchor_iou", "0.5"), ("m3_min_anchor_iou", None)):
            changed = copy.deepcopy(config)
            changed["train_cfg"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                audit.validate_m3_config(changed)
        for value in (.4, .6, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.validate_m3_config(config, value)

    def test_cli_requires_command_and_rejects_threshold_sweep(self):
        collect_args = ["collect", "config.py", "checkpoint.pth", "--fold", "6", "--out-dir", "new"]
        self.assertEqual(audit.parse_args(collect_args).min_anchor_iou, .5)
        for suffix in ([], collect_args + ["--min-anchor-iou", ".4"],
                       collect_args + ["--min-anchor-iou", "nan"],
                       collect_args + ["--min-anchor-iou", "inf"]):
            with self.subTest(args=suffix), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                audit.parse_args(suffix)
            self.assertEqual(raised.exception.code, 2)

    def test_python36_and_old_numpy_api_guards(self):
        tree = ast.parse((ROOT / "tools/audit_m3_targets.py").read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in ("check_output", "run", "Popen"):
                self.assertNotIn("text", [keyword.arg for keyword in node.keywords])
            if node.func.attr == "add_subparsers":
                self.assertNotIn("required", [keyword.arg for keyword in node.keywords])
            self.assertNotIn(node.func.attr, ("default_rng", "broadcast_shapes", "sliding_window_view"))


if __name__ == "__main__":
    unittest.main()
