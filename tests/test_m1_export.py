"""CPU/standard-library checks for export integrity and prediction equivalence."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "tools" / (name + ".py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_tool("eval_teacher2_export")
comparator = load_tool("compare_prediction_exports")


def annotation(count=232):
    return dict(images=[dict(id=i) for i in range(count)],
                categories=[dict(id=7, name="ship")], annotations=[])


def prediction(image_id=0, x=1.0, score=0.8):
    return dict(image_id=image_id, category_id=7, bbox=[x, 2.0, 3.0, 4.0], score=score)


def write_export(directory, predictions, ground_truth=None):
    directory.mkdir()
    ground_truth = ground_truth or annotation()
    ids = [item["id"] for item in ground_truth["images"]]
    (directory / "test.json").write_text(json.dumps(ground_truth))
    metadata = dict(image_ids=ids, num_test_images=len(ids), category_ids=[7],
                    test_ann_snapshot="test.json",
                    test_ann_sha256=comparator.file_hash(directory / "test.json"))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "predictions.bbox.json").write_text(json.dumps(predictions))
    (directory / "metrics.json").write_text(json.dumps({"bbox_mAP": 0.459}))


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class ExportIntegrityTests(unittest.TestCase):
    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "export"
            exporter.prepare_output_dir(str(target))
            exporter.prepare_output_dir(str(target))
            (target / "old.json").write_text("preserve")
            with self.assertRaises(ValueError):
                exporter.prepare_output_dir(str(target))
            self.assertEqual((target / "old.json").read_text(), "preserve")

    def test_complete_manifest_accepts_different_order(self):
        exporter.validate_test_manifest(annotation(), list(reversed(range(232))))

    def test_manifest_rejects_duplicate_missing_and_wrong_ids(self):
        for ids in [list(range(231)), list(range(231)) + [0], list(range(1, 233))]:
            with self.subTest(ids=ids[-2:]), self.assertRaises(ValueError):
                exporter.validate_test_manifest(annotation(), ids)

    def test_metadata_records_actual_revision_and_scoring_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "model.pth"
            checkpoint.write_bytes(b"fixture checkpoint")
            ann_file = temporary / "test.json"
            ann_file.write_text(json.dumps(annotation()))
            roi = AttrDict(quality_enabled=True, quality_inference=True)
            inner = AttrDict(roi_head=roi, test_cfg=AttrDict(rcnn=AttrDict(
                score_thr=0.05, nms=dict(iou_threshold=0.5), max_per_img=100)))
            cfg = AttrDict(filename="m1.py", model=AttrDict(model=inner),
                           data=AttrDict(test=AttrDict(ann_file=str(ann_file))), fp16=dict(loss_scale="dynamic"))
            with mock.patch.object(exporter, "git_revision", return_value=("abc123", False)):
                result = exporter.build_metadata(cfg, str(checkpoint), 6, list(range(232)), "manual-label")
                self.assertEqual(result["git_revision"], "abc123")
                self.assertEqual(result["version"], "abc123")
                self.assertEqual(result["version_label"], "manual-label")
                self.assertEqual(result["checkpoint_sha256"], comparator.file_hash(checkpoint))
                self.assertEqual(result["eval_params"]["candidate_rule"], "p_ship > score_thr")
                self.assertEqual(result["eval_params"]["ranking_and_export_score"], "p_ship * sigmoid(quality_logit)")
                self.assertFalse(result["eval_params"]["second_joint_score_threshold"])
                roi["quality_enabled"] = False
                disabled = exporter.build_metadata(cfg, str(checkpoint), 6, list(range(232)))
                self.assertFalse(disabled["eval_params"]["quality_inference"])
                self.assertEqual(disabled["eval_params"]["ranking_and_export_score"], "p_ship")

    def test_export_routing_and_strict_loading_are_preserved(self):
        source = (ROOT / "tools/eval_teacher2_export.py").read_text()
        tree = ast.parse(source)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        strict_load = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "load_checkpoint"]
        self.assertEqual(len(strict_load), 1)
        self.assertTrue(next(item.value.value for item in strict_load[0].keywords if item.arg == "strict"))
        self.assertIn('model.inference_on = "teacher2"', source)
        self.assertIn('MMDataParallel(model.cuda(0), device_ids=[0])', source)
        self.assertLess(source.index("cfg.merge_from_dict(args.cfg_options)"), source.index("cfg = patch_config(cfg)"))
        self.assertLess(source.index("prepare_output_dir(args.out_dir)"), source.index("single_gpu_test(modelx"))
        self.assertIn("dataset.format_results(outputs", source)
        self.assertNotIn("resume_from(", source)


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def exports(self, left_predictions, right_predictions):
        left, right = self.root / "a", self.root / "b"
        write_export(left, left_predictions)
        write_export(right, right_predictions)
        return comparator.load_export(left), comparator.load_export(right)

    def test_order_insensitive_and_empty_images_count(self):
        rows = [prediction(), prediction(image_id=3, x=6.0)]
        left, right = self.exports(rows, list(reversed(rows)))
        report = comparator.compare_exports(left, right)
        self.assertTrue(report["predictions_equal"])
        self.assertEqual(report["num_images"], 232)
        self.assertEqual(len(report["empty_images_a"]), 230)

    def test_bbox_score_and_multiplicity_differences_fail(self):
        rows = [prediction()]
        left, right = self.exports(rows, [prediction(score=0.800001)])
        self.assertFalse(comparator.compare_exports(left, right)["predictions_equal"])
        self.assertTrue(comparator.compare_exports(left, right, score_atol=0.00001)["predictions_equal"])
        right["grouped"][(0, 7)][0] = (1.01, 2, 3, 4, 0.8)
        self.assertFalse(comparator.compare_exports(left, right)["predictions_equal"])
        right["grouped"][(0, 7)] = left["grouped"][(0, 7)] * 2
        self.assertFalse(comparator.compare_exports(left, right)["predictions_equal"])

    def test_tolerance_uses_one_to_one_not_greedy_matching(self):
        # First A can match both B; second A can only match the first B.
        a = [(0, 0, 1, 1, 0.5), (0.15, 0, 1, 1, 0.5)]
        b = [(0.05, 0, 1, 1, 0.5), (-0.05, 0, 1, 1, 0.5)]
        self.assertTrue(comparator._matched_rows(a, b, 0.11, 0))
        self.assertFalse(comparator._matched_rows(a, [b[1], b[1]], 0.11, 0))

    def test_invalid_predictions_fail_closed(self):
        invalid = []
        for key, value in [("image_id", 999), ("category_id", 0), ("score", float("nan")),
                           ("score", float("inf")), ("score", True), ("score", 1.1),
                           ("bbox", [0, 0, -1, 3]), ("bbox", [0, 0, 1, float("nan")])]:
            row = prediction()
            row[key] = value
            invalid.append(row)
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                comparator.validate_predictions([row], list(range(232)), [7])

    def test_missing_image_and_gt_hash_fail(self):
        left, right = self.exports([], [])
        right["image_ids"].remove(1)
        with self.assertRaises(ValueError):
            comparator.compare_exports(left, right)
        right["image_ids"].add(1)
        right["annotation_sha256"] = "different"
        with self.assertRaises(ValueError):
            comparator.compare_exports(left, right)

    def test_annotation_tampering_is_detected(self):
        target = self.root / "tampered"
        write_export(target, [])
        changed = annotation()
        changed["annotations"] = [dict(id=1, image_id=0, category_id=7, bbox=[1, 2, 3, 4])]
        (target / "test.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            comparator.load_export(target)

    def test_partial_manifest_is_rejected_even_if_predictions_match(self):
        target = self.root / "partial"
        write_export(target, [])
        metadata = comparator.read_json(target / "metadata.json")
        metadata["image_ids"] = metadata["image_ids"][:-1]
        (target / "metadata.json").write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            comparator.load_export(target)

    def test_old_exports_use_adjacent_annotation_without_mutation(self):
        target = self.root / "old"
        write_export(target, [])
        metadata = comparator.read_json(target / "metadata.json")
        metadata.pop("test_ann_snapshot")
        metadata.pop("test_ann_sha256")
        (target / "metadata.json").write_text(json.dumps(metadata))
        (target / "test.json").rename(self.root / "test.json")
        self.assertEqual(comparator.load_export(target)["count"], 0)
        self.assertFalse((target / "test.json").exists())

    def test_metric_differences_are_reported_but_ap_not_used_as_equivalence(self):
        left, right = self.exports([], [])
        right["metrics"]["bbox_mAP"] = 0.46
        result = comparator.compare_exports(left, right)
        self.assertTrue(result["predictions_equal"])
        self.assertIn("bbox_mAP", result["metric_differences"])

    def test_cli_exit_codes(self):
        self.exports([prediction()], [prediction()])
        command = [sys.executable, str(ROOT / "tools/compare_prediction_exports.py"),
                   str(self.root / "a"), str(self.root / "b")]
        self.assertEqual(subprocess.run(command, stdout=subprocess.PIPE).returncode, 0)
        (self.root / "b/predictions.bbox.json").write_text("[]")
        self.assertEqual(subprocess.run(command, stdout=subprocess.PIPE).returncode, 1)
        (self.root / "b/predictions.bbox.json").write_text('[{"score": NaN}]')
        self.assertEqual(subprocess.run(command, stdout=subprocess.PIPE).returncode, 2)


if __name__ == "__main__":
    unittest.main()
