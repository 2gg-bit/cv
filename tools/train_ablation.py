"""Pin each ablation worker to this checkout before importing ssod.

Training args are forwarded to train.py. --check-init and --check-step instead
dispatch the CPU initialization check or the one-batch CUDA acceptance check.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODULES = {
    "ssod": "ssod/__init__.py",
    "ssod.models.dual_teacher": "ssod/models/dual_teacher.py",
    "ssod.models.progressive_gamma": "ssod/models/progressive_gamma.py",
    "ssod.models.roi_heads.foreground_head": "ssod/models/roi_heads/foreground_head.py",
    "ssod.models.roi_heads.foreground_roi_head": "ssod/models/roi_heads/foreground_roi_head.py",
    "ssod.utils.checkpoint": "ssod/utils/checkpoint.py",
    "ssod.utils.hooks.mean_teacher": "ssod/utils/hooks/mean_teacher.py",
    "ssod.utils.hooks.progressive_gamma": "ssod/utils/hooks/progressive_gamma.py",
    "ssod.apis.train": "ssod/apis/train.py",
}


def pin_repository(root=ROOT):
    root = str(Path(root).resolve())
    sys.path.insert(0, root)
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = root + (os.pathsep + previous if previous else "")


def source_manifest(root=ROOT):
    sources = {}
    for name, relative in SOURCE_MODULES.items():
        loaded = Path(importlib.import_module(name).__file__).resolve()
        expected = (root / relative).resolve()
        if loaded != expected:
            raise RuntimeError("Worker imported {} from {}, expected {}".format(name, loaded, expected))
        sources[name] = dict(path=str(loaded), sha256=hashlib.sha256(loaded.read_bytes()).hexdigest())
    return dict(repo_root=str(root), rank=os.environ.get("RANK", "0"),
                local_rank=os.environ.get("LOCAL_RANK"), sources=sources)


def main():
    pin_repository()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from this checkout root to preserve relative data/weight paths")
    # tee the worker's stdout into launcher.log to retain this actual import
    # receipt, including when the distributed launcher runs another process.
    print("[Ablation source] " + json.dumps(source_manifest(), sort_keys=True), flush=True)
    if "--check-source-only" in sys.argv[1:]:
        return
    target = "train.py"
    modes = [flag for flag in ("--check-init", "--check-step", "--eval") if flag in sys.argv[1:]]
    if len(modes) > 1:
        raise ValueError("Select one check mode")
    if modes:
        sys.argv.remove(modes[0])
        target = {"--check-init": "check_dual_teacher_init.py",
                  "--check-step": "check_ablation_step.py",
                  "--eval": "eval_teacher2_export.py"}[modes[0]]
    else:
        from mmcv import Config
        from ssod.utils import patch_config
        # train.py otherwise overwrites cfg.seed with its None default.
        parsed = runpy.run_path(str(ROOT / "tools/train.py"))["parse_args"]()
        if parsed.seed is None:
            raise ValueError("Pass --seed explicitly, using the correct B0 seed")
        cfg = Config.fromfile(parsed.config)
        if parsed.cfg_options:
            cfg.merge_from_dict(parsed.cfg_options)
        if parsed.work_dir:
            cfg.work_dir = parsed.work_dir
        cfg = patch_config(cfg)
        if cfg.get("seed") is not None and cfg.seed != parsed.seed:
            raise ValueError("CLI seed differs from the frozen experiment config")
        if cfg.get("auto_resume", False) or cfg.get("load_from"):
            raise ValueError("Disable auto_resume/load_from; resume explicitly if needed")
        if (os.environ.get("RANK", "0") == "0" and not parsed.resume_from
                and not cfg.get("resume_from") and Path(cfg.work_dir).exists()):
            raise ValueError("Training work_dir already exists; choose a fresh suite or explicitly resume")
    runpy.run_path(str(ROOT / "tools" / target), run_name="__main__")


if __name__ == "__main__":
    main()
