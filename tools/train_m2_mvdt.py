"""Launch M2+MVDT with each training worker pinned to this checkout.

Accepts the same arguments as tools/train.py. Use --check-source-only to print
the imported implementation paths and SHA256 values without starting training.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]


def pin_repository(root=ROOT):
    root = str(Path(root).resolve())
    sys.path.insert(0, root)
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = root + (os.pathsep + previous if previous else "")


def source_manifest(root=ROOT):
    sources = {}
    for name, relative in (
            ("ssod", "ssod/__init__.py"),
            ("ssod.models.dual_teacher", "ssod/models/dual_teacher.py"),
            ("ssod.models.mvdt", "ssod/models/mvdt.py")):
        module = importlib.import_module(name)
        loaded = Path(module.__file__).resolve()
        expected = (root / relative).resolve()
        if loaded != expected:
            raise RuntimeError("Training worker imported {} from {}, expected {}"
                               .format(name, loaded, expected))
        sources[name] = dict(path=str(loaded), sha256=hashlib.sha256(loaded.read_bytes()).hexdigest())
    return dict(repo_root=str(root), rank=os.environ.get("RANK", "0"),
                local_rank=os.environ.get("LOCAL_RANK"), sources=sources)


def main():
    pin_repository()
    print("[MVDT source] " + json.dumps(source_manifest(), sort_keys=True), flush=True)
    if "--check-source-only" in sys.argv[1:]:
        return
    runpy.run_path(str(ROOT / "tools/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
