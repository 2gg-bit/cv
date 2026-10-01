import copy
from unittest.mock import Mock

import pytest
import torch

from roi_fixture import load_heads, ParentBBox, ParentRoI, sample
from routing_fixture import load_forward_class

module = load_heads("ssod/models/roi_heads/sup2_giou.py")
Head = module.Sup2GIoURoIHead


def test_giou_known_geometry_and_nonoverlap_gradients():
    pred = torch.tensor([[0., 0., 2., 2.]] * 3, requires_grad=True)
    gt = torch.tensor([[0., 0., 2., 2.], [1., 0., 3., 2.], [4., 0., 6., 2.]], requires_grad=True)
    loss = module.aligned_giou_loss(pred, gt)
    assert torch.allclose(loss, torch.tensor([0., 2/3, 4/3]))
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[2].abs().sum() > 0
    assert gt.grad is None
    assert module.aligned_giou_loss(pred.half(), gt.half()).dtype == torch.float32


@pytest.mark.parametrize("enabled,beta", [(False, 1.), (True, 0.)])
def test_off_and_zero_beta_are_exact_parent_paths(enabled, beta):
    parent = ParentRoI()
    head = Head(sup2_giou_enabled=enabled, sup2_giou_weight=beta)
    head.load_state_dict(parent.state_dict(), strict=True)
    assert set(head.state_dict()) == set(parent.state_dict())
    assert sum(p.numel() for p in head.parameters()) == sum(p.numel() for p in parent.parameters())
    head.bbox_head.bbox_coder.decode = Mock(side_effect=AssertionError("identity must not decode"))
    x = torch.randn(2, 4, requires_grad=True)
    samples = [sample([[0, 0, 4, 4]], [[0, 0, 1, 1]])]
    args = (x, samples, [], [], [dict(tag="sup2")])
    before = torch.get_rng_state().clone()
    a, b = parent._bbox_forward_train(*args), head._bbox_forward_train(*args)
    assert set(a["loss_bbox"]) == set(b["loss_bbox"])
    assert all(torch.equal(v, b["loss_bbox"][k]) for k, v in a["loss_bbox"].items())
    assert torch.equal(before, torch.get_rng_state())


def test_only_sup2_positive_class_coordinates_receive_auxiliary_gradients():
    bbox_head = ParentBBox(num_classes=2, reg_class_agnostic=False)
    head = Head(bbox_head=bbox_head, sup2_giou_enabled=True, sup2_giou_weight=.7)
    samples = [sample([[0, 0, 4, 4]], [[0, 0, 1, 1]], gt=[[1, 0, 5, 4]], labels=[0]),
               sample([[0, 0, 10, 10]], [[0, 0, 1, 1]], gt=[[2, 0, 12, 10]], labels=[1])]
    x = torch.zeros(4, 8, requires_grad=True)
    state = copy.deepcopy(head.state_dict())
    result = head._bbox_forward_train(x, samples, [], [], [dict(tag="sup1"), dict(tag="sup2")])
    parent = ParentRoI(bbox_head=bbox_head)._bbox_forward_train(x, samples, [], [], [])
    for k, value in parent["loss_bbox"].items():
        assert torch.equal(value, result["loss_bbox"][k])
    # For the second image IoU = 80/120; normalize by its TWO sampled RoIs.
    assert float(result["loss_bbox"]["loss_giou"]) == pytest.approx(.7 * (1 - 80/120) / 2)
    result["loss_bbox"]["loss_giou"].backward()
    assert x.grad[2, 4:].abs().sum() > 0
    assert torch.count_nonzero(x.grad[[0, 1, 3]]) == 0
    assert torch.count_nonzero(x.grad[2, :4]) == 0
    assert all(torch.equal(value, head.state_dict()[k]) for k, value in state.items())


@pytest.mark.parametrize("tag", ["sup1", "unsup_student", "unsup_teacher", None])
def test_non_sup2_paths_have_no_added_loss(tag):
    head = Head(sup2_giou_enabled=True)
    result = head._bbox_forward_train(torch.zeros(1, 4), [sample([[0, 0, 2, 2]], [])],
                                     [], [], [dict(tag=tag)])
    assert "loss_giou" not in result["loss_bbox"]


def test_empty_positives_and_class_agnostic_decoding():
    head = Head(bbox_head=ParentBBox(reg_class_agnostic=True), sup2_giou_enabled=True)
    x = torch.zeros(1, 4, requires_grad=True)
    zero = head._bbox_forward_train(x, [sample([], [[0, 0, 2, 2]])], [], [],
                                    [dict(tag="sup2")])["loss_bbox"]["loss_giou"]
    zero.backward()
    assert zero == 0 and torch.equal(x.grad, torch.zeros_like(x))
    x = torch.zeros(1, 4, requires_grad=True)
    value = head._bbox_forward_train(x, [sample([[0, 0, 10, 10]], [], gt=[[1, 0, 11, 10]])],
                                     [], [], [dict(tag="sup2")])["loss_bbox"]["loss_giou"]
    value.backward()
    assert value > 0 and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_outer_dualteacher_keeps_sup2_gamma_and_unsup_losses():
    model = load_forward_class()()
    model.unsup_weight, model.sup2_weight = 2., .2
    class Detector:
        roi_head = None
        def forward_train(self, img, img_metas, **kwargs):
            losses = {"loss_bbox": torch.tensor(3.)}
            if img_metas[0]["tag"] == "sup2":
                losses["loss_giou"] = torch.tensor(2.)
            return losses
    model.student1 = model.student2 = Detector()
    model.extract_teacher_info = lambda *args: ({}, {})
    model.foward_unsup1_train = model.foward_unsup2_train = lambda *args: {"loss_bbox": torch.tensor(4.)}
    losses = model.forward_train(torch.zeros(4, 3, 2, 2),
        [dict(tag=t, filename="same_sar_image") for t in ("sup1", "sup2", "unsup_student", "unsup_teacher")], gt_bboxes=[[]]*4)
    assert losses["sup2_loss_giou"].item() == pytest.approx(.4)
    assert losses["sup2_loss_bbox"].item() == pytest.approx(.6)
    assert losses["unsup2_loss_bbox"].item() == pytest.approx(1.6)
    assert not any("giou" in k for k in losses if not k.startswith("sup2_"))
