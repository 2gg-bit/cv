"""Source pinning without importing the training machine's MMDetection stack."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mvdt_launcher", ROOT / "tools/train_m2_mvdt.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_repo_path_precedes_editable_install_for_worker_and_children(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher.sys, "path", [str(tmp_path / "wrong_repo")])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "another_repo"))
    launcher.pin_repository()
    assert Path(launcher.sys.path[0]) == ROOT
    assert Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) == ROOT


def test_manifest_records_actual_module_files_and_rejects_wrong_root(monkeypatch, tmp_path):
    paths = {"ssod": ROOT / "ssod/__init__.py",
             "ssod.models.dual_teacher": ROOT / "ssod/models/dual_teacher.py",
             "ssod.models.mvdt": ROOT / "ssod/models/mvdt.py"}
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: SimpleNamespace(__file__=paths[name]))
    manifest = launcher.source_manifest()
    assert set(manifest["sources"]) == set(paths)
    assert all(len(value["sha256"]) == 64 for value in manifest["sources"].values())
    paths["ssod.models.dual_teacher"] = tmp_path / "wrong.py"
    with pytest.raises(RuntimeError, match="expected"):
        launcher.source_manifest()
