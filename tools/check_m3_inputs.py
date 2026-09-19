"""Read-only M3 preflight: existing split, resolved config and input hashes.

Run from the repository root in the original conda dt environment. This never
creates a split, trains, loads a GPU, or overwrites experiment outputs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


DEV_SHA256 = "b7fc7c9cdc482f062bc09d55783237149fb2141e630488a100b5ceddc88b0479"
ORIGINAL_SELECTIONS = {
    6: {"000020", "000613", "000893"},
    7: {"000255", "000312", "000678"},
    8: {"000050", "000664", "001014"},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_names(annotation):
    images = annotation["images"]
    names = [Path(x["file_name"]).stem for x in images]
    ids = [x["id"] for x in images]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate image filename or ID within an annotation")
    return set(names)


def _expect_values(section, expected, name):
    """Check named experiment controls, not full configuration equivalence."""
    for key, value in expected.items():
        if section.get(key) != value:
            raise ValueError("Frozen {}.{} must remain {!r}".format(name, key, value))


def _check_frozen_settings(cfg):
    # Values come from the existing dev config and dual_teacher/base.py.  Keep
    # this explicit: these are key controls, not a claim that every config
    # option has been compared to a resolved baseline configuration.
    _expect_values(cfg.model.train_cfg, dict(
        use_teacher_proposal=False, pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9, cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02, jitter_times=10, jitter_scale=0.06,
        min_pseduo_box_size=0, unsup_weight=2.0), "train_cfg")
    _expect_values(cfg.runner, dict(type="IterBasedRunner", max_iters=32000), "runner")
    _expect_values(cfg.data, dict(samples_per_gpu=3), "data")
    _expect_values(cfg.optimizer, dict(type="SGD", lr=0.0025, momentum=0.9,
                                       weight_decay=0.0001), "optimizer")
    _expect_values(cfg.lr_config, dict(policy="step", warmup="linear",
                                       warmup_iters=500, warmup_ratio=0.001), "lr_config")
    if list(cfg.lr_config.get("step", [])) != [120000, 160000]:
        raise ValueError("Frozen learning-rate steps must remain [120000, 160000]")
    _expect_values(cfg.optimizer_config.get("grad_clip") or {},
                   dict(max_norm=35, norm_type=2), "grad_clip")
    _expect_values(cfg.get("fp16") or {}, dict(loss_scale="dynamic"), "fp16")
    sampler = cfg.data.sampler.train
    _expect_values(sampler, dict(type="SemiCrossBalanceSampler", by_prob=False,
                                 epoch_length=7330), "sampler")
    if list(sampler.get("sample_ratio", [])) != [1, 1, 1]:
        raise ValueError("Frozen sampler ratio must remain 1:1:1")
    ema = [hook for hook in cfg.get("custom_hooks", []) if hook.get("type") == "MeanTeacher"]
    if len(ema) != 1:
        raise ValueError("Expected exactly one frozen MeanTeacher EMA hook")
    _expect_values(ema[0], dict(momentum=0.999, interval=1, warm_up=0), "MeanTeacher")


def inspect_inputs(cfg, fold):
    if cfg.model.type != "DualTeacher":
        raise ValueError("Expected DualTeacher wrapper")
    train = cfg.model.train_cfg
    if not train.get("m2_enabled") or not train.get("m3_enabled"):
        raise ValueError("M3 experiment must enable M2 and M3")
    if train.get("m2_force_weight_one", False):
        raise ValueError("Test-only all-one weights must be OFF")
    if train.get("m3_target_mode") != "lower_uncertainty" or train.get("m3_min_anchor_iou") != 0.5:
        raise ValueError("M3 v1 routing and geometric guard must stay frozen")
    if cfg.model.model.roi_head.type != "M3RoIHead":
        raise ValueError("Expected parameter-free M3RoIHead")
    if cfg.model.model.roi_head.get("quality_enabled", False):
        raise ValueError("Do not combine M1 with M3 v1")
    if cfg.model.test_cfg.inference_on != "teacher2":
        raise ValueError("Inference must use teacher2 only")
    if cfg.get("auto_resume", False) or cfg.get("resume_from") or cfg.get("load_from"):
        raise ValueError("Fresh experiment must not resume/load a Phase3 checkpoint")
    _check_frozen_settings(cfg)
    files = {"dev": cfg.data.test.ann_file,
             "sup2": cfg.data.train.sup2.ann_file,
             "unsup": cfg.data.train.unsup.ann_file,
             "sup1": cfg.data.train.sup1.ann_file,
             "phase1": train.load1_from, "phase2": train.load2_from}
    if Path(files["dev"]).resolve() != Path("ssdd_dev_protocol/data/dev.json").resolve():
        raise ValueError("Use the existing SSDD developer split, not the official test set")
    if Path(cfg.data.val.ann_file).resolve() != Path(files["dev"]).resolve():
        raise ValueError("val/test must refer to the same developer split")
    for name, path in files.items():
        if not Path(path).is_file():
            raise FileNotFoundError("Missing {}: {}".format(name, path))
    if sha256(files["dev"]) != DEV_SHA256:
        raise ValueError("Developer annotation changed: restore the frozen split; do not re-split")
    annotations = {}
    for key in ("dev", "sup2", "unsup", "sup1"):
        with open(files[key]) as handle:
            annotations[key] = json.load(handle)
    names = {k: image_names(v) for k, v in annotations.items()}
    if len(names["dev"]) != 186 or len(names["unsup"]) != 739:
        raise ValueError("Expected 186 developer images and 739 unlabeled training images")
    if names["sup2"] != ORIGINAL_SELECTIONS[fold]:
        raise ValueError("The original three labeled SAR images changed")
    if names["dev"] & (names["sup2"] | names["unsup"]) or names["sup2"] & names["unsup"]:
        raise ValueError("Developer/labeled/unlabeled SAR image leakage")
    if annotations["unsup"].get("annotations"):
        raise ValueError("The unlabeled training file must not contain GT annotations")
    for key, prefix in (("sup2", cfg.data.train.sup2.img_prefix),
                        ("unsup", cfg.data.train.unsup.img_prefix),
                        ("dev", cfg.data.test.img_prefix),
                        ("sup1", cfg.data.train.sup1.img_prefix)):
        missing = [x["file_name"] for x in annotations[key]["images"]
                   if not (Path(prefix) / x["file_name"]).is_file()]
        if missing:
            raise FileNotFoundError("{} missing images: {}".format(key, missing[:5]))
    expected_p2 = Path("work_dirs/dev_ssdd/phase2_pretrain_optical_sar/3/{}/iter_11200.pth".format(fold))
    if Path(files["phase2"]).resolve() != expected_p2.resolve():
        raise ValueError("Phase2 points to the wrong original selection")
    return {"PASS": True, "fold": fold, "original_selection": sorted(names["sup2"]),
            "developer_image_ids": [x["id"] for x in annotations["dev"]["images"]],
            "counts": {k: len(v) for k, v in names.items()},
            "inputs": {k: {"path": str(Path(v).resolve()), "sha256": sha256(v)}
                       for k, v in files.items()},
            "checkpoint_loading": "not performed here; run check_dual_teacher_init.py next"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--fold", type=int, choices=sorted(ORIGINAL_SELECTIONS), required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError("Use a new preflight directory: {}".format(out))
    from mmcv import Config
    from ssod.utils import patch_config
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(fold=args.fold, percent=3))
    cfg = patch_config(cfg)
    result = inspect_inputs(cfg, args.fold)
    result["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    result["git_status"] = subprocess.check_output(["git", "status", "--porcelain"]).decode()
    source = Path(__file__).resolve().parents[1] / "ssod/models/dual_teacher.py"
    result["dual_teacher_sha256"] = sha256(source)
    out.mkdir(parents=True)
    cfg.dump(str(out / "resolved_config.py"))
    result["resolved_config_sha256"] = sha256(out / "resolved_config.py")
    with open(out / "preflight.json", "w") as handle:
        json.dump(result, handle, indent=2)
    print("PASS: original selections/developer isolation/key frozen settings/input hashes; no training was run.")


if __name__ == "__main__":
    main()
