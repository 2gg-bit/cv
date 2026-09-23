"""Execute each real M2 classification method with synthetic RoIs."""
import ast
from types import SimpleNamespace

import pytest
import torch

from test_dual_teacher_baseline import ROOT, actual_model_namespace


@pytest.mark.parametrize("branch", [1, 2])
@pytest.mark.parametrize("enabled,force,positive", [(False, False, 1.), (True, False, .6), (True, True, 1.)])
@pytest.mark.parametrize("has_pseudo_boxes", [False, True])
def test_positive_weights_and_original_normalizer(branch, enabled, force, positive, has_pseudo_boxes):
    ns = actual_model_namespace()
    boxes = torch.tensor([[1., 2., 3., 4.]]) if has_pseudo_boxes else torch.empty(0, 4)
    labels = torch.zeros(len(boxes), dtype=torch.long)
    visualizations = []
    # Stub only detector geometry/sampling; execute the actual weighting and loss path.
    ns.update(multi_apply=lambda *a, **k: ([boxes], [labels], [None]),
              filter_invalid=None, log_every_n=lambda *a: None,
              log_image_with_boxes=lambda *a, **kw: visualizations.append(kw["class_names"]),
              bbox2roi=lambda boxes: torch.zeros(2, 5))
    scores1 = torch.tensor([[.8, .2], [.3, .7]], requires_grad=True)
    scores2 = torch.tensor([[.6, .4], [.1, .9]], requires_grad=True)
    recorded = {}

    def targets(*args):
        return (torch.tensor([0, 1]), torch.ones(2), torch.zeros(2, 4), torch.ones(2, 4))

    def loss(cls, bbox, rois, labels, weights, bbox_targets, bbox_weights, **kwargs):
        recorded["weights"] = weights
        recorded["bbox_weights"] = bbox_weights
        return dict(loss_cls=weights * torch.tensor([3., 5.]),
                    loss_bbox=torch.ones(2, 4))
    roi = SimpleNamespace(
        _bbox_forward=lambda *a: dict(cls_score=torch.zeros(2, 2), bbox_pred=torch.zeros(2, 4)),
        bbox_head=SimpleNamespace(num_classes=1, get_targets=targets, loss=loss))
    model = SimpleNamespace(
        train_cfg=SimpleNamespace(cls_pseudo_threshold=.9),
        student1=SimpleNamespace(roi_head=roi, train_cfg=SimpleNamespace(rcnn={})),
        student2=SimpleNamespace(roi_head=roi, train_cfg=SimpleNamespace(rcnn={})),
        teacher1=SimpleNamespace(roi_head=SimpleNamespace(simple_test_bboxes=lambda *a, **k: (None, [scores1]))),
        teacher2=SimpleNamespace(roi_head=SimpleNamespace(simple_test_bboxes=lambda *a, **k: (None, [scores2]))),
        _get_trans_mat=lambda *a: None, _transform_bbox=lambda boxes, *a: boxes,
        get_sampling_result1=lambda *a: [SimpleNamespace(bboxes=torch.zeros(2, 4))],
        get_sampling_result2=lambda *a: [SimpleNamespace(bboxes=torch.zeros(2, 4))],
        m2_enabled=enabled, m2_force_weight_one=force, _log_m2_stats=lambda *a: None)
    # Execute the actual checker's dataset setup, then reach the real positive
    # pseudo-label logging path. Removing CLASSES assignment reproduces the bug.
    setup = ast.parse((ROOT / "tools/check_ablation_step.py").read_text(encoding="utf-8"))
    run = next(n for n in setup.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    setup.body = [n for n in run.body if isinstance(n, ast.Assign) and any(
        (isinstance(t, ast.Name) and t.id == "dataset") or
        (isinstance(t, ast.Attribute) and t.attr == "CLASSES") for t in n.targets)]
    exec(compile(setup, "checker_dataset_setup", "exec"), dict(
        model=model, cfg=SimpleNamespace(data=SimpleNamespace(train={})),
        build_dataset=lambda cfg: SimpleNamespace(CLASSES=("ship",))))
    result = getattr(ns["DualTeacher"], "unsup%d_rcnn_cls_loss" % branch)(
        model, dict(backbone_feature=[], img_metas=[]), [], [], [],
        [torch.empty(0, 5)], [torch.empty(0, dtype=torch.long)], [], [],
        [dict(img_shape=(10, 10, 3))], [],
        student_info=dict(img=torch.zeros(1, 3, 10, 10), img_metas=[dict(img_norm_cfg={})]))
    assert visualizations == ([("ship",)] if has_pseudo_boxes else [])
    assert float(recorded["weights"][0]) == pytest.approx(positive)
    assert float(recorded["weights"][1]) == pytest.approx(.8)
    assert not recorded["weights"].requires_grad
    assert torch.equal(recorded["bbox_weights"], torch.ones(2, 4))
    assert float(result["loss_cls"]) == pytest.approx((3 * positive + 5 * .8) / 1.8)
    assert float(result["loss_bbox"]) == 4.
