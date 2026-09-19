"""M3 input preflight tests using tiny local files and only the standard library."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_m3_inputs", str(ROOT / "tools/check_m3_inputs.py"))
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class Config(dict):
    """The dictionary/attribute behavior used by the preflight, without MMCV."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def merge_from_dict(self, values):
        self.update(values)

    def dump(self, path):
        Path(path).write_text("model = dict(type='DualTeacher')\n")


def config(**values):
    return Config(values)


def annotation(names, with_gt=False):
    # IDs deliberately restart in each COCO file. Only per-file IDs need to
    # be unique; isolation uses SAR image names, not cross-file integer IDs.
    images = [dict(id=index, file_name=name + ".jpg") for index, name in enumerate(names)]
    annotations = ([dict(id=1, image_id=0, category_id=1, bbox=[0, 0, 2, 2])]
                   if with_gt else [])
    return dict(images=images, annotations=annotations, categories=[dict(id=1, name="ship")])


class M3InputsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        previous = os.getcwd()
        os.chdir(temporary.name)
        self.addCleanup(os.chdir, previous)
        self.paths = dict(dev="ssdd_dev_protocol/data/dev.json", sup2="data/sup2.json",
                          unsup="data/unlabeled-dev.json", sup1="data/optical.json",
                          phase1="weights/phase1.pth")
        self.annotations = dict(
            dev=annotation(["dev{:06d}".format(i) for i in range(186)], True),
            sup2=annotation(sorted(preflight.ORIGINAL_SELECTIONS[6]), True),
            unsup=annotation(["train{:06d}".format(i) for i in range(739)]),
            # Numeric optical/SAR filenames may legitimately overlap because
            # these are separate image roots and domains.
            sup1=annotation(["000020"], True))
        for key, value in self.annotations.items():
            self.write_annotation(key, value)
            self.create_images(value, "optical" if key == "sup1" else "sar")
        for names in preflight.ORIGINAL_SELECTIONS.values():
            self.create_images(annotation(sorted(names)), "sar")
        Path("weights").mkdir()
        Path(self.paths["phase1"]).write_bytes(b"small fake phase1 checkpoint")
        for fold in (6, 7, 8):
            checkpoint = self.phase2(fold)
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"small fake phase2 checkpoint")
        self.cfg = config(
            model=config(type="DualTeacher", train_cfg=config(
                m2_enabled=True, m3_enabled=True, m2_force_weight_one=False,
                m3_target_mode="lower_uncertainty", m3_min_anchor_iou=0.5,
                load1_from=self.paths["phase1"], load2_from=str(self.phase2(6)),
                use_teacher_proposal=False, pseudo_label_initial_score_thr=0.5,
                rpn_pseudo_threshold=0.9, cls_pseudo_threshold=0.9, reg_pseudo_threshold=0.02,
                jitter_times=10, jitter_scale=0.06, min_pseduo_box_size=0, unsup_weight=2.0),
                model=config(roi_head=config(type="M3RoIHead")),
                test_cfg=config(inference_on="teacher2")),
            auto_resume=False, resume_from=None, load_from=None,
            runner=config(type="IterBasedRunner", max_iters=32000),
            optimizer=config(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001),
            optimizer_config=config(grad_clip=config(max_norm=35, norm_type=2)),
            lr_config=config(policy="step", warmup="linear", warmup_iters=500,
                             warmup_ratio=0.001, step=[120000, 160000]),
            fp16=config(loss_scale="dynamic"),
            custom_hooks=[config(type="NumClassCheckHook"), config(type="WeightSummary"),
                          config(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0)],
            data=config(samples_per_gpu=3,
                train=config(**{key: config(ann_file=self.paths[key],
                                             img_prefix="optical" if key == "sup1" else "sar")
                                for key in ("sup1", "sup2", "unsup")}),
                val=config(ann_file=self.paths["dev"], img_prefix="sar"),
                test=config(ann_file=self.paths["dev"], img_prefix="sar"),
                sampler=config(train=config(type="SemiCrossBalanceSampler", by_prob=False,
                                            epoch_length=7330, sample_ratio=[1, 1, 1]))))
        patcher = patch.object(preflight, "DEV_SHA256", preflight.sha256(self.paths["dev"]))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def phase2(fold):
        return Path("work_dirs/dev_ssdd/phase2_pretrain_optical_sar/3/{}/iter_11200.pth".format(fold))

    def write_annotation(self, key, value):
        path = Path(self.paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    @staticmethod
    def create_images(value, prefix):
        Path(prefix).mkdir(exist_ok=True)
        for image in value["images"]:
            (Path(prefix) / image["file_name"]).touch()

    def input_hashes(self):
        paths = list(self.paths.values()) + [str(self.phase2(fold)) for fold in (6, 7, 8)]
        return {path: preflight.sha256(path) for path in paths}

    def test_existing_protocol_accepts_all_original_folds_and_preserves_inputs(self):
        for fold in (6, 7, 8):
            with self.subTest(fold=fold):
                self.write_annotation("sup2", annotation(sorted(preflight.ORIGINAL_SELECTIONS[fold]), True))
                self.cfg.model.train_cfg["load2_from"] = str(self.phase2(fold))
                before = self.input_hashes()
                result = preflight.inspect_inputs(self.cfg, fold)
                self.assertTrue(result["PASS"])
                self.assertEqual(result["original_selection"], sorted(preflight.ORIGINAL_SELECTIONS[fold]))
                self.assertEqual(result["counts"], dict(dev=186, sup2=3, unsup=739, sup1=1))
                self.assertEqual(result["developer_image_ids"], list(range(186)))
                self.assertEqual(result["inputs"]["phase2"]["sha256"], before[str(self.phase2(fold))])
                self.assertEqual(before, self.input_hashes())
                self.assertIn("not performed", result["checkpoint_loading"])

    def test_key_frozen_settings_reject_drift(self):
        cases = [
            ("model.train_cfg.m3_target_mode", "teacher1"),
            ("model.train_cfg.m3_min_anchor_iou", 0.4),
            ("model.train_cfg.m2_force_weight_one", True),
            ("model.train_cfg.m2_enabled", False),
            ("model.train_cfg.m3_enabled", False),
            ("model.train_cfg.pseudo_label_initial_score_thr", 0.4),
            ("model.train_cfg.rpn_pseudo_threshold", 0.8),
            ("model.train_cfg.cls_pseudo_threshold", 0.8),
            ("model.train_cfg.reg_pseudo_threshold", 0.5),
            ("model.train_cfg.jitter_times", 5),
            ("model.train_cfg.jitter_scale", 0.1),
            ("model.train_cfg.unsup_weight", 4.0),
            ("model.train_cfg.min_pseduo_box_size", 1),
            ("model.train_cfg.use_teacher_proposal", True),
            ("model.model.roi_head.quality_enabled", True),
            ("model.test_cfg.inference_on", "student1"),
            ("auto_resume", True), ("load_from", "old.pth"), ("resume_from", "old.pth"),
            ("runner.max_iters", 1000), ("data.samples_per_gpu", 6),
            ("optimizer.lr", 0.01), ("lr_config.warmup_iters", 100),
            ("lr_config.step", [16000]), ("optimizer_config.grad_clip.max_norm", 10),
            ("fp16.loss_scale", 512), ("data.sampler.train.sample_ratio", [1, 1, 2]),
            ("custom_hooks", [config(type="MeanTeacher", momentum=0.99, interval=1, warm_up=0)]),
        ]
        for dotted, value in cases:
            with self.subTest(setting=dotted):
                cfg = copy.deepcopy(self.cfg)
                target, parts = cfg, dotted.split(".")
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
                with self.assertRaises(ValueError):
                    preflight.inspect_inputs(cfg, 6)

    def test_developer_path_and_checksum_are_frozen(self):
        self.cfg.data.test["ann_file"] = "data/official_test.json"
        with self.assertRaisesRegex(ValueError, "existing SSDD developer"):
            preflight.inspect_inputs(self.cfg, 6)
        self.cfg.data.test["ann_file"] = self.paths["dev"]
        self.cfg.data.val["ann_file"] = "data/official_test.json"
        with self.assertRaisesRegex(ValueError, "val/test"):
            preflight.inspect_inputs(self.cfg, 6)
        self.cfg.data.val["ann_file"] = self.paths["dev"]
        with patch.object(preflight, "DEV_SHA256", "wrong hash"):
            with self.assertRaisesRegex(ValueError, "Developer annotation changed"):
                preflight.inspect_inputs(self.cfg, 6)

    def test_sar_selection_counts_gt_and_isolation_guards(self):
        cases = [("sup2", "selection", "original three"),
                 ("unsup", "count", "739"), ("dev", "count", "186"),
                 ("unsup", "dev_overlap", "leakage"),
                 ("unsup", "labeled_overlap", "leakage"),
                 ("unsup", "gt", "GT annotations")]
        for key, mutation, message in cases:
            with self.subTest(mutation=mutation, key=key):
                value = copy.deepcopy(self.annotations[key])
                if mutation == "selection":
                    value["images"][0]["file_name"] = "replacement.jpg"
                elif mutation == "count":
                    value["images"].pop()
                elif mutation == "dev_overlap":
                    value["images"][0]["file_name"] = self.annotations["dev"]["images"][0]["file_name"]
                elif mutation == "labeled_overlap":
                    value["images"][0]["file_name"] = self.annotations["sup2"]["images"][0]["file_name"]
                else:
                    value["annotations"] = [dict(image_id=0, bbox=[0, 0, 1, 1])]
                self.write_annotation(key, value)
                # Test count/leakage independently of the fixed-byte hash gate.
                with patch.object(preflight, "DEV_SHA256", preflight.sha256(self.paths["dev"])):
                    with self.assertRaisesRegex(ValueError, message):
                        preflight.inspect_inputs(self.cfg, 6)
                self.write_annotation(key, self.annotations[key])

    def test_duplicate_names_and_ids_within_one_annotation_are_rejected(self):
        for field in ("file_name", "id"):
            value = annotation(["first", "second"])
            value["images"][1][field] = value["images"][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Duplicate"):
                preflight.image_names(value)

    def test_missing_inputs_images_and_wrong_phase2_selection_are_rejected(self):
        self.cfg.model.train_cfg["load1_from"] = "missing.pth"
        with self.assertRaisesRegex(FileNotFoundError, "Missing phase1"):
            preflight.inspect_inputs(self.cfg, 6)
        self.cfg.model.train_cfg["load1_from"] = self.paths["phase1"]
        self.cfg.data.train.sup1["img_prefix"] = "missing-optical"
        with self.assertRaisesRegex(FileNotFoundError, "sup1 missing images"):
            preflight.inspect_inputs(self.cfg, 6)
        self.cfg.data.train.sup1["img_prefix"] = "optical"
        self.cfg.model.train_cfg["load2_from"] = str(self.phase2(7))
        with self.assertRaisesRegex(ValueError, "wrong original selection"):
            preflight.inspect_inputs(self.cfg, 6)

    def test_existing_output_file_or_directory_is_never_overwritten(self):
        for output in (Path("existing-file"), Path("existing-dir")):
            if output.name.endswith("dir"):
                output.mkdir()
                sentinel = output / "preflight.json"
            else:
                sentinel = output
            sentinel.write_bytes(b"keep existing experiment")
            with self.subTest(output=str(output)), patch.object(sys, "argv", [
                    "check_m3_inputs.py", "unused.py", "--fold", "6", "--out-dir", str(output)]):
                with self.assertRaises(FileExistsError):
                    preflight.main()
            self.assertEqual(sentinel.read_bytes(), b"keep existing experiment")

    def test_cli_writes_new_report_only_after_validation_and_preserves_inputs(self):
        mmcv, ssod, utils = ModuleType("mmcv"), ModuleType("ssod"), ModuleType("ssod.utils")
        mmcv.Config = SimpleNamespace(fromfile=lambda path: self.cfg)
        utils.patch_config = lambda cfg: cfg
        commands = []

        def git_output(command):
            commands.append(command)
            return b"test-revision\n" if command[1] == "rev-parse" else b" M file.py\n"

        before = self.input_hashes()
        arguments = ["check_m3_inputs.py", "fixture.py", "--fold", "6", "--out-dir", "new-report"]
        with patch.dict(sys.modules, {"mmcv": mmcv, "ssod": ssod, "ssod.utils": utils}), \
                patch.object(sys, "argv", arguments), \
                patch.object(preflight.subprocess, "check_output", side_effect=git_output), \
                patch("sys.stdout", new_callable=io.StringIO):
            preflight.main()
            report = json.loads(Path("new-report/preflight.json").read_text())
            self.assertTrue(report["PASS"])
            self.assertEqual(report["git_revision"], "test-revision")
            self.assertEqual(report["resolved_config_sha256"],
                             preflight.sha256("new-report/resolved_config.py"))
            self.assertEqual(self.cfg["percent"], 3)
            self.assertEqual(self.cfg["fold"], 6)
            self.assertEqual(before, self.input_hashes())
            self.assertEqual(len(commands), 2)
            self.cfg.model.train_cfg["m3_enabled"] = False
            arguments[-1] = "failed-report"
            with self.assertRaises(ValueError):
                preflight.main()
            self.assertFalse(Path("failed-report").exists())


if __name__ == "__main__":
    unittest.main()
