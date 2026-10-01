"""Build isolated experiments from a fully resolved B0, without framework IO."""

import copy
import math
from pathlib import Path


EXPERIMENTS = ("b0", "m2", "fg", "pg_sup2", "pg_both")
ALL_EXPERIMENTS = EXPERIMENTS + ("reweight_l1", "sup2_giou")


def differences(before, after, prefix=""):
    rows = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            path = prefix + "." + key if prefix else key
            if key not in before or key not in after:
                rows.append(dict(path=path, before=before.get(key), after=after.get(key)))
            else:
                rows.extend(differences(before[key], after[key], path))
    elif before != after:
        rows.append(dict(path=prefix, before=before, after=after))
    return rows


def make_suite(baseline, work_root, seed, fg_weight=0.1, start_ratio=0.5,
               experiments=None, giou_weight=1.0):
    """Preserve B0's dataset/schedule/optimizer/evaluation; never choose a split."""
    cfg = copy.deepcopy(baseline)
    names = tuple(EXPERIMENTS if experiments is None else experiments)
    if (not names or len(set(names)) != len(names) or
            any(name not in ALL_EXPERIMENTS for name in names)):
        raise ValueError("Choose unique, supported experiments")
    if cfg.get("semi_wrapper"):
        raise ValueError("Resolve and patch_config the baseline before building the suite")
    model = cfg.get("model", {})
    if model.get("type") != "DualTeacher":
        raise ValueError("B0 must use DualTeacher")
    tc = model.get("train_cfg", {})
    for flag in ("m2_enabled", "m2_force_weight_one", "m3_enabled", "mvdt_enabled", "pg"):
        if tc.get(flag):
            raise ValueError("B0 already enables an experimental feature: " + flag)
    head = model.get("model", {}).get("roi_head", {})
    if head.get("type") != "StandardRoIHead" or head.get("quality_enabled") or head.get("foreground_enabled"):
        raise ValueError("B0 must have the original StandardRoIHead")
    if head.get("bbox_head", {}).get("num_classes") != 1:
        raise ValueError("This suite is for the single ship class")
    if head.get("bbox_head", {}).get("reweight", {}).get("enable"):
        raise ValueError("B0 must not enable background reweighting")
    if "reweight_l1" in names and head["bbox_head"].get("type") != "Shared2FCBBoxHead":
        raise ValueError("reweight_l1 requires the B0 Shared2FCBBoxHead")
    if cfg.get("runner", {}).get("type") != "IterBasedRunner":
        raise ValueError("This suite requires IterBasedRunner")
    total = cfg["runner"].get("max_iters")
    if type(total) is not int or total < 2:
        raise ValueError("B0 runner.max_iters must be >= 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("Use the explicit nonnegative B0 seed")
    if cfg.get("seed") is not None and cfg["seed"] != seed:
        raise ValueError("Requested seed differs from the supplied B0 config")
    if any(not math.isfinite(float(v)) or v <= 0 for v in (fg_weight, start_ratio)):
        raise ValueError("Auxiliary weight and PG start ratio must be finite and positive")
    if not math.isfinite(float(giou_weight)) or giou_weight <= 0:
        raise ValueError("GIoU experiment weight must be finite and positive")
    hooks = cfg.get("custom_hooks", [])
    if any(h.get("type") in ("ProgressiveGammaHook", "GammaScheduler", "ZigzagSchedulerHook", "Weighter") for h in hooks):
        raise ValueError("B0 already contains a loss-weight scheduling hook")
    if not tc.get("load1_from") or not tc.get("load2_from"):
        raise ValueError("B0 must name its Phase1/2 initialization files")
    root = Path(work_root).resolve()
    if cfg.get("work_dir") and root == Path(cfg["work_dir"]).resolve():
        raise ValueError("Use a new experiment root, not the baseline work_dir")
    cfg.update(auto_resume=False, resume_from=None, load_from=None, seed=seed)
    suite = {}
    for name in names:
        item = copy.deepcopy(cfg)
        item["work_dir"] = str(root / name)
        item["ablation_experiment"] = name
        train = item["model"]["train_cfg"]
        train.update(m2_enabled=(name == "m2"), m2_force_weight_one=False)
        if name == "fg":
            item["model"]["model"]["roi_head"].update(
                type="ForegroundRoIHead", foreground_enabled=True,
                foreground_hidden_channels=32, foreground_loss_weight=fg_weight)
        if name.startswith("pg_"):
            train["pg"] = dict(mode=name[3:], total_iters=total,
                               start_ratio=start_ratio, end_ratio=1.0)
        if name in ("reweight_l1", "sup2_giou"):
            if item.get("custom_imports") is None:
                item["custom_imports"] = {}
            imports = item.setdefault("custom_imports", {})
            previous_imports = imports.get("imports", [])
            modules = [previous_imports] if isinstance(previous_imports, str) else list(previous_imports)
            module = "ssod.models.roi_heads." + (
                "small_bkg_reweight" if name == "reweight_l1" else "sup2_giou")
            if module not in modules:
                modules.append(module)
            imports.update(imports=modules, allow_failed_imports=False)
            roi = item["model"]["model"]["roi_head"]
            if name == "reweight_l1":
                roi["type"] = "SmallBkgReweightRoIHead"
                roi["bbox_head"].update(type="SmallBkgReweightBBoxHead",
                    reweight=dict(enable=True, lambda_=1.0, max_area=1024.0, tag="sup2"))
            else:
                roi.update(type="Sup2GIoURoIHead", sup2_giou_enabled=True,
                           sup2_giou_weight=float(giou_weight))
        suite[name] = item
    return suite
