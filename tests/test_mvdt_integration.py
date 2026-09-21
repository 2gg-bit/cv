"""Exercise the actual DualTeacher methods with small CPU detector fixtures."""

import ast
import copy
import logging
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from test_dual_teacher_baseline import actual_model_namespace, TrainConfig
from test_mvdt import controller, mvdt


ROOT = Path(__file__).resolve().parents[1]


def model_fixture(enabled=True):
    ns = actual_model_namespace()
    ns["MVDTThreshold"] = mvdt.MVDTThreshold
    # Use the REAL box filtering implementation, without importing CUDA ops.
    source = ROOT / "ssod/models/utils/bbox_utils.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "filter_invalid"]
    exec(compile(tree, str(source), "exec"), ns)
    model = ns["DualTeacher"]({}, None, dict(inference_on="teacher2"))
    model.train_cfg = TrainConfig(cls_pseudo_threshold=.9)
    model.unsup_weight = 2.0
    if enabled:
        model.mvdt = controller()
    return model, ns


def test_default_off_matches_original_filter_and_old_checkpoint():
    model, ns = model_fixture(False)
    assert model.mvdt is None
    assert not any(key.startswith("mvdt.") for key in model.state_dict())
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 0., 10.], [1., 1., 2., 2.]])
    labels, scores = torch.zeros(3, dtype=torch.long), torch.tensor([.95, .99, .8])
    actual = model._filter_cls_pseudo(boxes, labels, scores)
    expected = ns["filter_invalid"](boxes, labels, scores, thr=.9)
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
    clone, _ = model_fixture(False)
    clone.load_state_dict(model.state_dict(), strict=True)


def test_dynamic_admission_preserves_geometry_and_input_tensors():
    model, _ = model_fixture()
    model.mvdt.observe(torch.tensor([.55, .56]))
    model.mvdt.observe(torch.tensor([.8, .81]))
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 0., 10.], [1., 1., 2., 2.]])
    labels, scores = torch.zeros(3, dtype=torch.long), torch.tensor([.8, .99, .56])
    before = [x.clone() for x in (boxes, labels, scores)]
    selected, _, _ = model._filter_cls_pseudo(boxes, labels, scores)
    assert torch.equal(selected, boxes[:1])
    assert all(torch.equal(a, b) for a, b in zip(before, (boxes, labels, scores)))


def test_full_checkpoint_requires_mvdt_when_training_but_not_for_inference():
    model, _ = model_fixture()
    model.mvdt.observe(torch.tensor([.6, .65]))
    state = copy.deepcopy(model.state_dict())
    restored, _ = model_fixture()
    restored.load_state_dict(state, strict=True)
    for key in state:
        assert torch.equal(state[key], restored.state_dict()[key])
    disabled, _ = model_fixture(False)
    with pytest.raises(RuntimeError, match="mvdt_enabled"):
        disabled.load_state_dict(state, strict=False)
    inference, _ = model_fixture(False)
    inference.train_cfg = None
    inference.load_state_dict(state, strict=True)
    with pytest.raises(RuntimeError, match="MVDT checkpoint state"):
        model.load_state_dict(inference.state_dict(), strict=False)


def test_actual_forward_counts_fused_candidates_once_and_updates_after_both_students():
    model, ns = model_fixture()
    ns["MultiSteamDetector"].forward_train = lambda *args, **kwargs: None
    groups = {
        "unsup_teacher": dict(img=torch.zeros(1, 3, 8, 8),
                              img_metas=[dict(filename="sar.jpg")], tag=["unsup_teacher"]),
        "unsup_student": dict(img=torch.zeros(1, 3, 8, 8),
                              img_metas=[dict(filename="sar.jpg")], tag=["unsup_student"]),
    }
    ns["dict_split"] = lambda *args: copy.deepcopy(groups)
    ns["weighted_loss"] = lambda loss, weight: loss
    ns["log_every_n"] = lambda *args: None
    candidates = torch.tensor([[0., 0., 5., 5., .55], [1., 1., 6., 6., .8]])
    info = dict(det_bboxes=[candidates])
    model.extract_teacher_info = lambda *args: (info, info)
    thresholds = []

    def unsup(self, *args):
        thresholds.append(self.mvdt.value)
        return {"loss_cls": torch.tensor(1.)}

    model.foward_unsup1_train = MethodType(unsup, model)
    model.foward_unsup2_train = MethodType(unsup, model)
    for _ in range(3):
        model.forward_train(torch.zeros(1), [dict(tag="unused")])
    assert thresholds[:4] == [.9] * 4
    assert thresholds[4:] == pytest.approx([.8, .8])
    assert int(model.mvdt.steps) == 3 and int(model.mvdt.count) == 2


def test_threshold_is_logged_at_info_between_updates_and_labels_boundary(caplog):
    model, ns = model_fixture()
    model.mvdt = controller(warmup_iters=50, update_interval=50)
    ns["MultiSteamDetector"].forward_train = lambda *args, **kwargs: None
    ns["dict_split"] = lambda *args: {}
    # A live wandb session can swallow the old dictionary log. The threshold
    # line must reach the INFO logger without relying on this helper.
    ns["log_every_n"] = lambda *args, **kwargs: None
    with caplog.at_level(logging.INFO, logger="dual_teacher_test"):
        model.forward_train(torch.zeros(1), [dict(tag="unused")])
        model.mvdt.steps.fill_(49)
        model.mvdt.count.fill_(4)
        model.mvdt.scores[:4] = torch.tensor([.55, .56, .8, .81])
        model.forward_train(torch.zeros(1), [dict(tag="unused")])
    threshold_lines = [record.getMessage() for record in caplog.records
                       if "[MVDT threshold]" in record.getMessage()]
    assert threshold_lines == [
        "[MVDT threshold] step=1 mvdt_cls_threshold=0.900000 next_cls_threshold=0.900000",
        "[MVDT threshold] step=50 mvdt_cls_threshold=0.900000 next_cls_threshold=0.800000",
    ]


def test_both_real_classification_methods_use_dynamic_admission():
    model, ns = model_fixture()
    model.mvdt.observe(torch.tensor([.55, .56]))
    model.mvdt.observe(torch.tensor([.8, .81]))
    ns["multi_apply"] = lambda fn, *args, **kwargs: tuple(map(list, zip(
        *(fn(*items, **kwargs) for items in zip(*args)))))
    ns["log_every_n"] = lambda *args: None
    captured = []

    class StopAtSampling(Exception):
        pass

    def sampling(metas, proposals, boxes, labels):
        captured.append(boxes[0].clone())
        raise StopAtSampling()

    model.get_sampling_result1 = sampling
    model.get_sampling_result2 = sampling
    pseudo = torch.tensor([[0., 0., 10., 10., .8], [0., 0., 5., 5., .56]])
    for method in (model.unsup1_rcnn_cls_loss, model.unsup2_rcnn_cls_loss):
        with pytest.raises(StopAtSampling):
            method({}, [], [{}], [], [pseudo], [torch.zeros(2, dtype=torch.long)],
                   [], [], [], [])
    assert len(captured) == 2
    assert all(torch.equal(boxes, pseudo[:1, :4]) for boxes in captured)


def test_config_is_opt_in_isolated_and_keeps_m2():
    values = {}
    exec((ROOT / "configs/reproduce/phase3_dual_teacher_ssdd_dev_m2_mvdt.py").read_text(), {}, values)
    cfg = values["semi_wrapper"]["train_cfg"]
    assert cfg["m2_enabled"] and cfg["mvdt_enabled"] and not cfg["m3_enabled"]
    assert not cfg["m2_force_weight_one"]
    assert "m2_mvdt" in values["work_dir"]
    assert values["load_from"] is None and values["resume_from"] is None
    assert values["auto_resume"] is False
    assert set(cfg["mvdt"]) == {"warmup_iters", "update_interval", "min_samples",
                               "min_group_size", "max_scores"}
