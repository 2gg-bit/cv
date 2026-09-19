"""Real DualTeacher routing methods, with small detector/geometry fixtures.

These CPU tests do not claim equivalence of the training machine's CUDA ops.
"""

import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from test_dual_teacher_baseline import actual_model_namespace


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("m3_integration_math", ROOT / "ssod/models/m3_routing.py")
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


def fixture_model(enabled=False, mode="lower_uncertainty"):
    ns = actual_model_namespace()
    ns["select_regression_targets"] = math.select_regression_targets
    model = ns["DualTeacher"]({}, None, dict(inference_on="teacher2"))
    model.m3_enabled, model.m3_target_mode = enabled, mode
    model.train_cfg = SimpleNamespace(reg_pseudo_threshold=0.02)
    return model


def test_m3_flags_default_off_no_new_parameters():
    model = fixture_model()
    assert model.m2_enabled is False and model.m3_enabled is False
    assert not any("m3" in k or "m2" in k for k in model.state_dict())


def test_teacher_candidates_keep_original_fusion_uncertainty_and_rng():
    model = fixture_model()
    boxes = torch.tensor([[1., 1., 11., 11., .97]])
    labels = torch.tensor([0])

    def get_det(self, name, *args, **kwargs):
        # Empty second side makes the author's exact fusion passthrough easy
        # to inspect, while both teachers still regress the SAME fused anchor.
        b = boxes.clone() if name == "teacher1" else boxes.new_empty((0, 5))
        l = labels.clone() if name == "teacher1" else labels.new_empty((0,))
        return [torch.zeros(1)], [b], [l], [b], {"proposals": [b]}

    def uncertainty(teacher):
        def fn(feat, metas, proposals, labels, return_boxes=False):
            unc = [torch.rand(len(x), 4) * .01 + teacher * .001 for x in proposals]
            refined = [x[:, :4] + teacher * .1 for x in proposals]
            return (unc, refined) if return_boxes else unc
        return fn

    model.get_det_bboxes = MethodType(get_det, model)
    model.compute_uncertainty_with_aug_1 = uncertainty(1)
    model.compute_uncertainty_with_aug_2 = uncertainty(2)
    meta = [{"transform_matrix": np.eye(3), "img_shape": (32, 32, 3)}]
    torch.manual_seed(6)
    original = model.extract_teacher_info(None, meta)
    rng = torch.get_rng_state().clone()
    model.m3_enabled, model.m3_target_mode = True, "original"
    torch.manual_seed(6)
    routed = model.extract_teacher_info(None, meta)
    assert torch.equal(rng, torch.get_rng_state())
    for baseline, new in zip(original, routed):
        assert torch.equal(baseline["det_bboxes"][0], new["det_bboxes"][0])
        assert torch.equal(baseline["det_labels"][0], new["det_labels"][0])
        assert torch.equal(new["reg_target_bboxes"][0], boxes[:, :4])
        assert torch.equal(new["m3_source"][0], torch.zeros(1, dtype=torch.long))
    expected_unc = (routed[0]["teacher1_reg_unc"][0] + routed[0]["teacher2_reg_unc"][0]) * .5
    assert torch.equal(routed[0]["det_bboxes"][0][:, 5:], expected_unc)
    assert torch.equal(routed[0]["raw_det_bboxes"][0], boxes)
    assert routed[1]["raw_det_bboxes"][0].shape == (0, 5)


def test_regression_eligibility_uses_original_average_uncertainty():
    model = fixture_model(True)
    rows = torch.tensor([[0., 0., 10., 10., .95, .01, .01, .01, .01],
                         [0., 0., 10., 10., .95, .02, .02, .02, .02],
                         [0., 0., 0., 10., .95, .01, .01, .01, .01],
                         [0., 0., 10., 10., .95, float('nan'), 0., 0., 0.]])
    chosen = torch.tensor([[1., 1., 9., 9.]]).repeat(4, 1)
    result = model._m3_filter_reg_targets([rows], [chosen])
    assert torch.equal(result[0], chosen[:1])
    assert torch.equal(rows[0, :4], torch.tensor([0., 0., 10., 10.]))
    with pytest.raises(ValueError):
        model._m3_filter_reg_targets([rows], [chosen[:1]])


def test_strong_transform_degenerate_target_falls_back_without_dropping_row():
    model = fixture_model(True)
    anchors = torch.tensor([[0., 0., 10., 10., .9], [0., 0., 10., 10., .9]])
    transformed = torch.tensor([[2., 2., 8., 8.], [10., 0., 10., 10.]])
    model._transform_bbox = lambda boxes, matrix, shapes: [transformed]
    out = model._m3_student_targets(
        {"reg_target_bboxes": [transformed]}, None,
        {"img_metas": [{"img_shape": (10, 10, 3)}]}, [anchors])
    assert torch.equal(out[0][0], transformed[0])
    assert torch.equal(out[0][1], anchors[1, :4])
    model.m3_enabled = False
    assert model._m3_student_targets({}, None, {}, []) is None


def test_m3_config_does_not_enable_inference_or_quality_head():
    values = {}
    exec((ROOT / "configs/reproduce/phase3_dual_teacher_ssdd_dev_m3.py").read_text(), {}, values)
    assert values["model"] == dict(roi_head=dict(type="M3RoIHead"))
    train = values["semi_wrapper"]["train_cfg"]
    assert train["m2_enabled"] and train["m3_enabled"]
    assert train["m2_force_weight_one"] is False
    assert values["auto_resume"] is False
    assert values["resume_from"] is None and values["load_from"] is None
