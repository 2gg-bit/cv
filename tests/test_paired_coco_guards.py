"""Artifact/provenance guards; COCO numerical evaluation is mocked, not rerun."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycocotools", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pycocotools.coco", SimpleNamespace(COCO=None))
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", SimpleNamespace(COCOeval=None))
    spec = importlib.util.spec_from_file_location("paired_guard_test", str(ROOT / "tools/recompute_paired_coco_delta.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "check_provenance", lambda paths: {})
    def compute(pred, gt, cfg):
        value = json.loads(Path(pred).read_text())[0]["score"]
        return dict(raw={k: value for k in module.METRIC_ITEMS}, cat_ids=[0],
                    img_ids=list(range(232)), params=dict(maxDets=[100, 300, 1000]),
                    num_predictions=1, summarize_text="mock COCO numerical evaluation")
    monkeypatch.setattr(module, "compute_stats", compute)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def artifacts(root, module, value):
    root.mkdir()
    (root / "weights.bin").write_bytes(str(value).encode())
    (root / "test.json").write_text("{}")
    (root / "resolved_config.py").write_text("model = dict()")
    write_json(root / "predictions.bbox.json", [dict(score=value)])
    write_json(root / "metrics.json", {"bbox_"+k: module.mmdet_round(value) for k in module.METRIC_ITEMS})
    write_json(root / "metadata.json", dict(
        checkpoint=str(root / "weights.bin"), checkpoint_sha256=module.sha256_file(root / "weights.bin"),
        test_ann_sha256=module.sha256_file(root / "test.json"),
        resolved_config_sha256=module.sha256_file(root / "resolved_config.py"),
        num_test_images=232, num_predictions=1))
    module.FROZEN_TEST_ANN_SHA256 = module.sha256_file(root / "test.json")
    return root


def make_pair(tmp_path, module, suffix=""):
    return dict(label="fold"+suffix,
                baseline_dir=str(artifacts(tmp_path / ("b"+suffix), module, .483)),
                variant_dir=str(artifacts(tmp_path / ("v"+suffix), module, .493)))


@pytest.mark.parametrize("fault", ["ann_metadata", "config", "count", "checkpoint", "same_weight", "frozen"])
def test_failed_side_check_cannot_report_consistent(tmp_path, module, fault):
    pair = make_pair(tmp_path, module)
    assert module.compare_pair(pair, True)["consistent"]
    root = Path(pair["variant_dir"])
    meta = json.loads((root / "metadata.json").read_text())
    if fault == "ann_metadata": meta["test_ann_sha256"] = "0"*64
    elif fault == "config": (root / "resolved_config.py").write_text("changed = True")
    elif fault == "count": meta["num_predictions"] = 2
    elif fault == "checkpoint": (root / "weights.bin").write_bytes(b"changed")
    elif fault == "same_weight":
        (root / "weights.bin").write_bytes((Path(pair["baseline_dir"]) / "weights.bin").read_bytes())
        meta["checkpoint_sha256"] = module.sha256_file(root / "weights.bin")
    elif fault == "frozen": module.FROZEN_TEST_ANN_SHA256 = "0"*64
    write_json(root / "metadata.json", meta)
    assert not module.compare_pair(pair)["consistent"]


def test_missing_checkpoint_policy_is_explicit(tmp_path, module):
    pair = make_pair(tmp_path, module)
    (Path(pair["variant_dir"]) / "weights.bin").unlink()
    result = module.compare_pair(pair)
    assert result["consistent"]
    assert result["variant"]["checkpoint_rehash"]["matches_metadata"] is None
    assert not module.compare_pair(pair, require_checkpoint=True)["consistent"]


@pytest.mark.parametrize("kind", ["complete", "incomplete", "inconsistent"])
def test_cli_status_exit_and_mean_follow_all_checks(tmp_path, module, monkeypatch, kind):
    pair = make_pair(tmp_path, module)
    if kind == "inconsistent": module.FROZEN_TEST_ANN_SHA256 = "0"*64
    out = tmp_path / "out.json"
    argv = ["recompute", "--pair", pair["label"], pair["baseline_dir"], pair["variant_dir"],
            "--expect-pairs", "2" if kind == "incomplete" else "1", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    if kind == "complete": module.main()
    else:
        with pytest.raises(SystemExit) as exc: module.main()
        assert exc.value.code == 1
    result = json.loads(out.read_text())
    assert result["status"] == kind
    assert (result["aggregate"] is not None) == (kind == "complete")
    before = out.read_bytes()
    with pytest.raises(SystemExit): module.main()
    assert out.read_bytes() == before


def test_duplicate_pairs_cannot_satisfy_fold_count(tmp_path, module, monkeypatch):
    pair = make_pair(tmp_path, module)
    argv = ["recompute", "--out", str(tmp_path / "out.json"), "--expect-pairs", "3"]
    for label in ("fold6", "fold7", "fold8"):
        argv += ["--pair", label, pair["baseline_dir"], pair["variant_dir"]]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit): module.main()
    assert not (tmp_path / "out.json").exists()
