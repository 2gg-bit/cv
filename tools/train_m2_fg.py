"""Pin each M2+foreground worker to this checkout before importing ssod.

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
    "ssod.models.roi_heads.foreground_head": "ssod/models/roi_heads/foreground_head.py",
    "ssod.models.roi_heads.foreground_roi_head": "ssod/models/roi_heads/foreground_roi_head.py",
    "ssod.utils.checkpoint": "ssod/utils/checkpoint.py",
    "ssod.utils.hooks.mean_teacher": "ssod/utils/hooks/mean_teacher.py",
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
    # tee the worker's stdout into launcher.log to retain this actual import
    # receipt, including when the distributed launcher runs another process.
    print("[Foreground source] " + json.dumps(source_manifest(), sort_keys=True), flush=True)
    if "--check-source-only" in sys.argv[1:]:
        return
    target = "train.py"
    modes = [flag for flag in ("--check-init", "--check-step") if flag in sys.argv[1:]]
    if len(modes) > 1:
        raise ValueError("Select one check mode")
    if modes:
        sys.argv.remove(modes[0])
        target = {"--check-init": "check_dual_teacher_init.py",
                  "--check-step": "check_m2_fg_step.py"}[modes[0]]
    runpy.run_path(str(ROOT / "tools" / target), run_name="__main__")


if __name__ == "__main__":
    main()
