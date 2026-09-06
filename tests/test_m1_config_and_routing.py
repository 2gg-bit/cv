"""Test the actual supervision call routing without MMDetection or a GPU."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_forward_class():
    source = ROOT / "ssod/models/dual_teacher.py"
    tree = ast.parse(source.read_text())
    dual = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DualTeacher")
    method = next(n for n in dual.body if isinstance(n, ast.FunctionDef) and n.name == "forward_train")
    fixture = ast.ClassDef(
        name="RoutingFixture", bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))

    class Base:
        def forward_train(self, *args, **kwargs):
            pass

    def split(data, key):
        return {
            tag: {k: [v[i] for i in range(len(data[key])) if data[key][i] == tag]
                  for k, v in data.items()}
            for tag in set(data[key])
        }

    namespace = dict(Base=Base, dict_split=split, log_every_n=lambda *args: None,
                     weighted_loss=lambda loss, weight: {k: v * weight for k, v in loss.items()})
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["RoutingFixture"]


@pytest.mark.parametrize("enabled", [False, True])
def test_supervision_opt_in_and_existing_weights(enabled):
    calls = []

    class Detector:
        def __init__(self, name):
            self.name = name
            self.roi_head = SimpleNamespace(quality_enabled=enabled)

        def forward_train(self, **kwargs):
            calls.append((self.name, kwargs))
            loss = {"loss_cls": 2.0}
            if kwargs.get("quality_supervised"):
                loss["loss_quality"] = 3.0
            return loss

    model = load_forward_class()()
    model.student1, model.student2 = Detector("s1"), Detector("s2")
    meta = [{"tag": "sup1"}, {"tag": "sup2"}]
    losses = model.forward_train([1, 2], meta, gt_bboxes=[[[0, 0, 1, 1]], [[0, 0, 2, 2]]])
    assert losses["sup1_loss_cls"] == 2.0
    assert losses["sup2_loss_cls"] == 0.4
    assert {name for name, _ in calls} == {"s1", "s2"}
    for _, inputs in calls:
        assert inputs.get("quality_supervised", False) is enabled
        if not enabled:
            assert "quality_supervised" not in inputs
        assert "tag" not in inputs
    if enabled:
        assert losses["sup1_loss_quality"] == 3.0
        assert losses["sup2_loss_quality"] == pytest.approx(0.6)
    else:
        assert not any("quality" in key for key in losses)
    assert meta == [{"tag": "sup1"}, {"tag": "sup2"}]


def test_m1_inherits_baseline_without_training_hyperparameter_overrides():
    values = {}
    config = ROOT / "configs/reproduce/phase3_dual_teacher_ssdd_m1.py"
    exec(compile(config.read_text(), str(config), "exec"), {}, values)
    assert set(values) - {"__doc__"} == {"_base_", "model", "work_dir"}
    assert values["_base_"] == ["phase3_dual_teacher_ssdd.py"]
    assert set(values["model"]) == {"roi_head"}
    assert values["model"]["roi_head"] == dict(
        type="QualityRoIHead", quality_enabled=True, quality_inference=False,
        quality_hidden_channels=64, quality_loss_weight=1.0,
    )
    assert values["work_dir"] == "work_dirs/phase3_dual_teacher_m1_quality/${percent}/${fold}"
