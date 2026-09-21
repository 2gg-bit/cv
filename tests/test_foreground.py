"""CPU math, routing, initialization and restore checks for foreground loss.

Framework fixtures execute the real class/method AST, not CUDA RoI operators.
Use train_m2_fg.py --check-step for the real MMDetection training stack.
"""

import ast
import importlib.util
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_dual_teacher_baseline import actual_model_namespace, TrainConfig
from test_m1_config_and_routing import load_forward_class


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fg = load("foreground_math", "ssod/models/roi_heads/foreground_head.py")
ckpt = load("foreground_checkpoint", "ssod/utils/checkpoint.py")


def metas(n=1, image=(13, 17, 3), pad=(16, 20, 3)):
    return [dict(img_shape=image, pad_shape=pad) for _ in range(n)]


def test_tiny_border_boxes_peak_and_padding_are_correct_without_mutation():
    boxes = torch.tensor([[0., 0., 1., 1.], [16., 12., 17., 13.], [30., 30., 40., 40.]])
    before = boxes.clone()
    target, valid = fg.foreground_targets(torch.zeros(1, 1, 8, 8), [boxes], metas())
    assert target[0, 0, 0, 0] == target[0, 0, 3, 4] == 1
    assert int(target.eq(1).sum()) == 2
    assert int(valid.sum()) == 20
    assert not bool(valid[:, :, 4:, :].any())
    assert not bool(valid[:, :, :, 5:].any())
    assert torch.equal(boxes, before)
    assert not target.requires_grad


def test_gaussian_width_depends_on_box_size_and_duplicates_do_not_add():
    z = torch.zeros(1, 1, 16, 16)
    meta = metas(image=(64, 64, 3), pad=(64, 64, 3))
    small = torch.tensor([[30., 30., 34., 34.]])
    large = torch.tensor([[8., 8., 56., 56.]])
    a, _ = fg.foreground_targets(z, [small], meta)
    b, _ = fg.foreground_targets(z, [large], meta)
    c, _ = fg.foreground_targets(z, [torch.cat((large, large))], meta)
    assert a[0, 0, 8, 8] == b[0, 0, 8, 8] == 1
    assert b[0, 0, 8, 9] > a[0, 0, 8, 9] > 0
    assert torch.equal(b, c)


def test_ignored_regions_and_padding_have_zero_gradient():
    z = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target, valid = fg.foreground_targets(
        z, [torch.tensor([[8., 8., 12., 12.]])], metas(),
        gt_bboxes_ignore=[torch.tensor([[0., 0., 8., 8.]])])
    fg.foreground_focal_loss(z, target, valid).backward()
    assert torch.equal(z.grad[~valid], torch.zeros_like(z.grad[~valid]))
    assert z.grad[0, 0, 2, 2] < 0  # increase confidence at the true center
    assert z.grad[0, 0, 0, 4] > 0  # decrease confidence at real background
    assert not bool(valid[0, 0, :2, :2].any())


def test_empty_annotated_image_trains_background_and_all_ignored_is_zero():
    z = torch.zeros(1, 1, 4, 5, requires_grad=True)
    target, valid = fg.foreground_targets(z, [torch.empty(0, 4)], metas())
    loss = fg.foreground_focal_loss(z, target, valid)
    loss.backward()
    assert loss > 0 and bool((z.grad > 0).all())
    assert fg.foreground_focal_loss(z, target, valid & False) == 0


@pytest.mark.parametrize("value", [-1000., 1000.])
def test_extreme_logits_stay_finite(value):
    z = torch.full((1, 1, 2, 2), value, requires_grad=True)
    t = torch.zeros_like(z)
    t[0, 0, 0, 0] = 1
    loss = fg.foreground_focal_loss(z, t, torch.ones_like(z, dtype=torch.bool))
    loss.backward()
    assert bool(torch.isfinite(loss)) and bool(torch.isfinite(z.grad).all())


@pytest.mark.parametrize("half", [False, True])
def test_head_gradient_reaches_shared_features_and_every_parameter(half):
    head = fg.ForegroundHead(4, 3)
    if half:
        head.half()
    feature = torch.randn(2, 4, 4, 5, dtype=torch.float16 if half else torch.float32,
                          requires_grad=True)
    z = head(feature)
    target, valid = fg.foreground_targets(z, [torch.tensor([[0., 0., 8., 8.]])] * 2, metas(2))
    fg.foreground_focal_loss(z, target, valid).backward()
    assert z.dtype == torch.float32
    for grad in [feature.grad] + [p.grad for p in head.parameters()]:
        assert grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.abs().sum() > 0)


@pytest.mark.parametrize("problem", ["nan", "shape", "stride", "batch"])
def test_invalid_target_inputs_fail(problem):
    boxes = [torch.tensor([[0., 0., 4., 4.]])]
    z, meta, stride = torch.zeros(1, 1, 4, 5), metas(), 4
    if problem == "nan":
        boxes[0][0, 0] = float("nan")
    elif problem == "shape":
        z = torch.zeros(1, 1, 2, 2)
    elif problem == "stride":
        stride = 0
    else:
        boxes = []
    with pytest.raises(ValueError):
        fg.foreground_targets(z, boxes, meta, stride)


class StandardFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.with_bbox, self.with_mask = True, False
        self.bbox_head = nn.Linear(4, 1)
        self.bbox_head.num_classes, self.bbox_head.in_channels = 1, 4
        self.bbox_roi_extractor = SimpleNamespace(featmap_strides=[4, 8, 16, 32])

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None):
        return dict(loss_cls=x[0].mean())

    def simple_test(self, *args, **kwargs):
        return "original inference"


def roi_class():
    tree = ast.parse((ROOT / "ssod/models/roi_heads/foreground_roi_head.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    ns = dict(math=math, StandardRoIHead=StandardFixture,
              HEADS=SimpleNamespace(register_module=lambda: lambda cls: cls),
              ForegroundHead=fg.ForegroundHead, foreground_targets=fg.foreground_targets,
              foreground_focal_loss=fg.foreground_focal_loss)
    exec(compile(tree, "foreground_roi_head.py", "exec"), ns)
    return ns["ForegroundRoIHead"]


RoI = roi_class()


@pytest.mark.parametrize("enabled", [False, True])
def test_only_explicit_supervision_runs_head_and_detection_is_unchanged(enabled):
    head = RoI(foreground_enabled=enabled, foreground_hidden_channels=3)
    x = [torch.randn(1, 4, 4, 5, requires_grad=True)]
    inputs = dict(x=x, img_metas=metas(), proposal_list=[],
                  gt_bboxes=[torch.tensor([[0., 0., 8., 8.]])], gt_labels=[])
    baseline = head.forward_train(**inputs)
    losses = head.forward_train(**inputs, foreground_supervised=True)
    assert torch.equal(baseline["loss_cls"], losses["loss_cls"])
    assert "loss_foreground" not in baseline
    assert ("loss_foreground" in losses) == enabled
    assert RoI.simple_test is StandardFixture.simple_test
    if not enabled:
        assert set(head.state_dict()) == set(StandardFixture().state_dict())
    else:
        head.foreground_head.forward = lambda *_: (_ for _ in ()).throw(AssertionError("called"))
        head.forward_train(**inputs)  # pseudo-box training must never call it
        assert head.simple_test() == "original inference"


@pytest.mark.parametrize("enabled", [False, True])
def test_dual_teacher_routes_real_supervision_and_preserves_point_two_weight(enabled):
    calls = []

    class Detector:
        roi_head = SimpleNamespace(foreground_enabled=enabled)

        def forward_train(self, **inputs):
            calls.append(inputs)
            return dict(loss_cls=2., **({"loss_foreground": 3.} if inputs.get("foreground_supervised") else {}))

    model = load_forward_class()()
    model.mvdt = None
    model.student1, model.student2 = Detector(), Detector()
    losses = model.forward_train([1, 2], [{"tag": "sup1"}, {"tag": "sup2"}],
                                 gt_bboxes=[[], []])
    assert losses["sup1_loss_cls"] == 2. and losses["sup2_loss_cls"] == .4
    assert all(call.get("foreground_supervised", False) == enabled for call in calls)
    if enabled:
        assert losses["sup1_loss_foreground"] == 3.
        assert losses["sup2_loss_foreground"] == pytest.approx(.6)


class Detector(nn.Module):
    def __init__(self, enabled=True, hidden=3):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.roi_head = RoI(foreground_enabled=enabled, foreground_hidden_channels=hidden)


def save(path, state):
    torch.save({"state_dict": state}, str(path))
    return str(path)


def test_phase_initialization_is_strict_and_copies_head_to_pair(tmp_path):
    teacher, student = Detector(), Detector()
    state = Detector(False).state_dict()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    ckpt.load_branch_weights(save(tmp_path / "old.pth", state),
                             [("teacher", teacher), ("student", student)], logging.getLogger())
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, student.state_dict()[key])
        assert torch.equal(value, state[key] if key in state else before[key])
    # Full checkpoint round trip: no fresh auxiliary initialization on resume.
    restored = Detector()
    restored.load_state_dict(teacher.state_dict(), strict=True)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in teacher.state_dict().items())


@pytest.mark.parametrize("problem", ["partial", "backbone_missing", "shape", "whitelist"])
def test_bad_initialization_is_rejected_before_any_branch_is_mutated(tmp_path, problem):
    teacher, student = Detector(), Detector()
    state = dict(Detector(False).state_dict())
    if problem == "partial":
        state["roi_head.foreground_head.out.bias"] = torch.zeros(1)
    elif problem == "backbone_missing":
        del state["backbone.weight"]
    elif problem == "shape":
        student = Detector(hidden=5)
    else:
        teacher.roi_head.foreground_initialization_keys = lambda: ("backbone.weight",)
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in (teacher, student)]
    with pytest.raises(RuntimeError):
        ckpt.load_branch_weights(save(tmp_path / "bad.pth", state),
                                 [("t", teacher), ("s", student)], logging.getLogger())
    for model, original in zip((teacher, student), before):
        assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("enabled", [False, True])
def test_full_restore_cannot_silently_change_head_presence(enabled):
    with pytest.raises(RuntimeError, match="checkpoint/config mismatch"):
        Detector(enabled).load_state_dict(Detector(not enabled).state_dict(), strict=False)


@pytest.mark.parametrize("identical", [False, True])
def test_actual_dual_initialization_ignores_random_head_when_comparing_phases(tmp_path, identical):
    state1 = Detector(False).state_dict()
    state2 = {k: v.clone() for k, v in state1.items()}
    if not identical:
        state2["backbone.weight"].add_(1)
    ns = actual_model_namespace()
    ns["build_detector"] = lambda cfg: Detector()
    model = ns["DualTeacher"]({}, TrainConfig(
        unsup_weight=2., load1_from=save(tmp_path / "p1.pth", state1),
        load2_from=save(tmp_path / "p2.pth", state2)), dict(inference_on="teacher2"))
    if identical:
        with pytest.raises(RuntimeError, match="identical branches"):
            model.init_from_pretrained()
    else:
        model.init_from_pretrained()
        assert all(torch.equal(v, model.student2.state_dict()[k]) for k, v in model.teacher2.state_dict().items())
        assert all(not p.requires_grad for p in model.teacher2.parameters())


def test_actual_ema_includes_foreground_head():
    tree = ast.parse((ROOT / "ssod/utils/hooks/mean_teacher.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    tree.body[0].decorator_list = []
    ns = dict(Hook=object)
    exec(compile(tree, "mean_teacher.py", "exec"), ns)
    models = {name: Detector() for name in ("student1", "student2", "teacher1", "teacher2")}
    with torch.no_grad():
        for name, model in models.items():
            for p in model.parameters():
                p.fill_(6 if name.startswith("student") else 2)
    ns["MeanTeacher"]().momentum_update(SimpleNamespace(**models), .75)
    for name in ("teacher1", "teacher2"):
        assert all(bool((p == 3).all()) for p in models[name].roi_head.foreground_head.parameters())


def test_launcher_checks_actual_paths_and_pins_child_imports(monkeypatch, tmp_path):
    launch = load("fg_launch_test", "tools/train_m2_fg.py")
    monkeypatch.setattr(launch.sys, "path", [str(tmp_path)])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    launch.pin_repository()
    assert Path(launch.sys.path[0]) == ROOT
    assert Path(launch.os.environ["PYTHONPATH"].split(launch.os.pathsep)[0]) == ROOT
    paths = {name: ROOT / relative for name, relative in launch.SOURCE_MODULES.items()}
    monkeypatch.setattr(launch.importlib, "import_module", lambda name: SimpleNamespace(__file__=paths[name]))
    assert len(launch.source_manifest()["sources"]) == 6
    paths["ssod.models.dual_teacher"] = tmp_path / "wrong.py"
    with pytest.raises(RuntimeError, match="expected"):
        launch.source_manifest()
