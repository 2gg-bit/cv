"""M1 CPU regression tests plus an optional real MMDetection integration test.

The lightweight head executes unchanged with real PyTorch. The RoI class is
compiled from its source AST with small external-framework fixtures; this does
not substitute for the optional pinned-stack test or server GPU validation.
"""

import ast
import asyncio
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
HEAD_DIR = ROOT / "ssod/models/roi_heads"


def run_async(coroutine):
    # asyncio.run was introduced after the README's Python 3.6 stack.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


spec = importlib.util.spec_from_file_location(
    "quality_math_under_test", str(HEAD_DIR / "localization_quality_head.py"))
quality_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality_math)
LocalizationQualityHead = quality_math.LocalizationQualityHead


def bbox2roi(boxes_per_image):
    return torch.cat([
        torch.cat((boxes.new_full((len(boxes), 1), i), boxes[:, :4]), dim=1)
        for i, boxes in enumerate(boxes_per_image)], dim=0)


def bbox2result(boxes, labels, num_classes):
    return [boxes[labels == index].detach().cpu().numpy()
            for index in range(num_classes)]


class AdditiveCoder:
    def decode(self, proposals, deltas, max_shape=None):
        decoded = proposals + deltas
        if max_shape is not None:
            decoded = decoded.clone()
            decoded[:, 0::2].clamp_(min=0, max=max_shape[1])
            decoded[:, 1::2].clamp_(min=0, max=max_shape[0])
        return decoded


class BBoxHeadFixture(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(1.0))
        self.in_channels = 3
        self.num_classes = num_classes
        self.with_cls = self.with_reg = True
        self.custom_cls_channels = False
        self.bbox_coder = AdditiveCoder()

    def get_bboxes(self, rois, cls, deltas, img_shape, scale_factor,
                   rescale=False, cfg=None):
        assert cfg is None
        boxes = self.bbox_coder.decode(rois[:, 1:], deltas, img_shape)
        if rescale:
            boxes = boxes / boxes.new_tensor(scale_factor)
        return boxes, cls.softmax(dim=1)


class StandardRoIHeadFixture(nn.Module):
    def __init__(self, bbox_head=None, with_mask=False, with_shared_head=False,
                 **kwargs):
        super().__init__()
        self.bbox_head = bbox_head or BBoxHeadFixture()
        self.with_bbox = True
        self.with_mask, self.with_shared_head = with_mask, with_shared_head
        self.test_cfg = SimpleNamespace(
            score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100)
        self.parent_output = object()

    def simple_test(self, *args, **kwargs):
        return self.parent_output

    def simple_test_bboxes(self, *args, **kwargs):
        return self.parent_output

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      **kwargs):
        if getattr(self, "raise_during_forward", False):
            raise RuntimeError("fixture training failure")
        return self._bbox_forward_train(
            x, proposal_list, gt_bboxes, gt_labels, img_metas)["loss_bbox"]

    def _bbox_forward_train(self, *args, **kwargs):
        return dict(self.next_bbox_results, loss_bbox=dict(loss_bbox=torch.tensor(2.0)))

    def _bbox_forward(self, *args, **kwargs):
        return self.next_bbox_results

    def aug_test(self, *args, **kwargs):
        return self.parent_output

    async def async_simple_test(self, *args, **kwargs):
        return self.parent_output

    def onnx_export(self, *args, **kwargs):
        return self.parent_output


def cpu_batched_nms(boxes, scores, labels, cfg):
    """Small deterministic fixture; tests ranking without compiled MMCV ops."""
    assert cfg == dict(type="nms", iou_threshold=0.5)
    order, keep = torch.argsort(scores, descending=True), []
    while order.numel():
        selected = order[0]
        keep.append(selected)
        rest = order[1:]
        if not rest.numel():
            break
        overlaps = quality_math.aligned_iou(
            boxes[selected].expand(len(rest), 4), boxes[rest])
        order = rest[overlaps <= cfg["iou_threshold"]]
    keep = torch.stack(keep)
    return torch.cat((boxes[keep], scores[keep, None]), dim=1), keep


def actual_roi_class():
    source = HEAD_DIR / "quality_roi_head.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(
        __name__="quality_roi_under_test", torch=torch, math=math,
        HEADS=SimpleNamespace(register_module=lambda: lambda cls: cls),
        StandardRoIHead=StandardRoIHeadFixture,
        LocalizationQualityHead=LocalizationQualityHead,
        aligned_iou=quality_math.aligned_iou,
        fp32_quality_context=quality_math.fp32_quality_context,
        quality_candidate_scores=quality_math.quality_candidate_scores,
        bbox2roi=bbox2roi, bbox2result=bbox2result,
        batched_nms=cpu_batched_nms)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["QualityRoIHead"]


QualityRoIHead = actual_roi_class()


def test_quality_head_is_small_and_has_only_four_named_tensors():
    head = LocalizationQualityHead()
    assert tuple(head.state_dict()) == (
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
    assert sum(p.numel() for p in head.parameters()) == 16769


def test_quality_gradients_reach_features_but_not_deltas_or_targets():
    torch.manual_seed(678)
    head = LocalizationQualityHead(in_channels=3)
    features = torch.randn(4, 3, 7, 7, requires_grad=True)
    deltas = torch.randn(4, 4, requires_grad=True)
    target = torch.rand(4, requires_grad=True)
    head.loss(head(features, deltas), target).backward()
    assert features.grad is not None and features.grad.abs().sum() > 0
    assert deltas.grad is None and target.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters())


def test_fp16_inputs_are_computed_in_fp32():
    head = LocalizationQualityHead(in_channels=3)
    logits = head(torch.ones(2, 3, 7, 7).half(), torch.ones(2, 4).half())
    loss = head.loss(logits, torch.tensor([0.0, 1.0]).half())
    assert logits.dtype == loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()


def test_empty_positive_loss_has_zero_finite_gradients_for_every_parameter():
    head = LocalizationQualityHead(in_channels=3)
    features = torch.empty(0, 3, 7, 7, requires_grad=True)
    loss = head.loss(head(features, torch.empty(0, 4)), torch.empty(0))
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0
               for p in head.parameters())


def test_aligned_iou_is_clipped_degenerate_safe_and_detached():
    predicted = torch.tensor([[0., 0., 10., 10.], [0., 0., 5., 10.],
                              [0., 0., 0., 0.]], requires_grad=True)
    target = torch.tensor([[0., 0., 10., 10.]] * 3, requires_grad=True)
    iou = quality_math.aligned_iou(predicted, target)
    assert torch.equal(iou, torch.tensor([1.0, 0.5, 0.0]))
    assert not iou.requires_grad


def test_candidate_mask_is_original_class_threshold_without_joint_refilter():
    probabilities = torch.tensor([0.05, 0.051, 0.9, 0.01])
    logits = torch.tensor([10., -10., 0., 10.])
    valid, joint = quality_math.quality_candidate_scores(probabilities, logits, .05)
    assert valid.tolist() == [False, True, True, False]
    assert 0 < joint[1] < .05
    assert torch.isclose(joint[2], torch.tensor(.45))


@pytest.mark.parametrize("enabled,ranking", [(False, False), (False, True), (True, False)])
def test_baseline_inference_delegates_exactly_to_parent(enabled, ranking):
    head = QualityRoIHead(quality_enabled=enabled, quality_inference=ranking)
    # No override/wrapper can accidentally run the quality head for pseudoboxes.
    assert QualityRoIHead.simple_test_bboxes is StandardRoIHeadFixture.simple_test_bboxes
    if enabled:
        head.quality_head.forward = lambda *args: pytest.fail("Unexpected quality forward")
    assert head.simple_test([], [], []) is head.parent_output
    assert head.simple_test_bboxes([], [], [], None) is head.parent_output


def test_disabled_head_has_exact_baseline_keys_and_strict_load():
    baseline = StandardRoIHeadFixture()
    head = QualityRoIHead(quality_enabled=False)
    assert set(head.state_dict()) == set(baseline.state_dict())
    head.load_state_dict(baseline.state_dict(), strict=True)
    assert head.quality_initialization_keys() == ()


def test_initialization_whitelist_contains_only_quality_parameters():
    head = QualityRoIHead()
    assert set(head.quality_initialization_keys()) == {
        "roi_head.quality_head.fc1.weight", "roi_head.quality_head.fc1.bias",
        "roi_head.quality_head.fc2.weight", "roi_head.quality_head.fc2.bias"}


def sample_fixture(positive, negative, target):
    positive = torch.tensor(positive, dtype=torch.float32).reshape(-1, 4)
    negative = torch.tensor(negative, dtype=torch.float32).reshape(-1, 4)
    return SimpleNamespace(pos_bboxes=positive, neg_bboxes=negative,
                           bboxes=torch.cat((positive, negative)),
                           pos_gt_bboxes=torch.tensor(target).float().reshape(-1, 4))


def training_fixture(empty_positive=False):
    head = QualityRoIHead()
    if empty_positive:
        samples = [sample_fixture([], [[20., 20., 30., 30.]], [])]
    else:
        samples = [sample_fixture([[0., 0., 10., 10.]],
                                  [[20., 20., 30., 30.]], [[0., 0., 10., 10.]]),
                   sample_fixture([[0., 0., 10., 10.]], [], [[0., 0., 5., 10.]])]
    count = sum(len(sample.bboxes) for sample in samples)
    head.next_bbox_results = dict(
        bbox_feats=torch.ones(count, 3, 7, 7, requires_grad=True),
        bbox_pred=torch.zeros(count, 4, requires_grad=True))
    meta = [dict(img_shape=(100, 100, 3)) for _ in samples]
    return head, samples, meta


def test_supervision_is_explicit_and_flag_restored_after_exception():
    head, samples, meta = training_fixture()
    default_losses = head.forward_train([], meta, samples, [], [])
    assert "loss_quality" not in default_losses
    losses = head.forward_train([], meta, samples, [], [], quality_supervised=True)
    assert "loss_quality" in losses and not head._quality_supervised
    assert "loss_quality" not in head.forward_train([], meta, samples, [], [])
    head.raise_during_forward = True
    with pytest.raises(RuntimeError, match="fixture training failure"):
        head.forward_train([], meta, samples, [], [], quality_supervised=True)
    assert not head._quality_supervised


def test_quality_loss_uses_only_positive_rows_and_decoded_box_iou():
    head, samples, meta = training_fixture()
    # The second image's decoded box shrinks to the assigned target (IoU 1),
    # whereas the first image's decoded box shrinks to half its GT (IoU .5).
    with torch.no_grad():
        head.next_bbox_results["bbox_pred"][[0, 2], 2] = -5
    losses = head.forward_train([], meta, samples, [], [], quality_supervised=True)
    features, deltas = (head.next_bbox_results[key]
                        for key in ("bbox_feats", "bbox_pred"))
    expected = F.binary_cross_entropy_with_logits(
        head.quality_head(features[[0, 2]], deltas[[0, 2]]), torch.tensor([.5, 1.]))
    assert torch.equal(losses["loss_quality"], expected)
    losses["loss_quality"].backward()
    assert features.grad[1].abs().sum() == 0
    assert deltas.grad is None


def test_supervised_empty_positive_batch_is_ddp_safe():
    head, samples, meta = training_fixture(empty_positive=True)
    loss = head.forward_train([], meta, samples, [], [], quality_supervised=True)["loss_quality"]
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is not None for p in head.quality_head.parameters())


class FixedQuality(nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        probabilities = torch.tensor(probabilities)
        self.logits = (probabilities / (1 - probabilities)).log()

    def forward(self, features, deltas):
        return self.logits


def test_quality_nms_ranks_and_exports_joint_score_without_changing_geometry():
    head = QualityRoIHead(quality_inference=True)
    # First two boxes overlap: lower-class-score box has higher localization
    # quality and must win. Third survives although its joint score is < .05.
    proposals = [torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.],
                               [20., 20., 30., 30.], [40., 40., 50., 50.]])]
    probability = torch.tensor([.9, .8, .06, .01])
    head.next_bbox_results = dict(
        cls_score=torch.stack((probability.log(), (1 - probability).log()), dim=1),
        bbox_pred=torch.zeros(4, 4), bbox_feats=torch.ones(4, 3, 7, 7))
    head.quality_head = FixedQuality([.1, .9, .1, .999])
    meta = [dict(img_shape=(100, 100, 3), scale_factor=np.array([2.] * 4))]
    result = head.simple_test([], proposals, meta, rescale=True)[0][0]
    assert result.shape == (2, 5)
    np.testing.assert_allclose(result[:, :4], [[.5, .5, 5.5, 5.5], [10., 10., 15., 15.]])
    np.testing.assert_allclose(result[:, 4], [.72, .006], rtol=1e-6)
    head.test_cfg.max_per_img = 1
    assert head.simple_test([], proposals, meta)[0][0].shape == (1, 5)
    # Even with quality ranking enabled, the pseudo-label interface is parent.
    assert head.simple_test_bboxes([], [], [], None) is head.parent_output


def test_empty_proposal_images_are_preserved():
    head = QualityRoIHead(quality_inference=True)
    result = head.simple_test([], [torch.empty(0, 4), torch.empty(0, 4)], [{}, {}])
    assert len(result) == 2 and all(item[0].shape == (0, 5) for item in result)


def test_empty_image_in_nonempty_batch_is_preserved():
    head = QualityRoIHead(quality_inference=True)
    head.next_bbox_results = dict(
        cls_score=torch.tensor([[2., 0.]]), bbox_pred=torch.zeros(1, 4),
        bbox_feats=torch.ones(1, 3, 7, 7))
    proposals = [torch.empty(0, 4), torch.tensor([[0., 0., 10., 10.]])]
    meta = [{}, dict(img_shape=(100, 100, 3), scale_factor=np.ones(4))]
    result = head.simple_test([], proposals, meta)
    assert len(result) == 2 and result[0][0].shape == (0, 5)
    assert result[1][0].shape == (1, 5)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0])
def test_invalid_quality_loss_weight_is_rejected(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        QualityRoIHead(quality_loss_weight=weight)


@pytest.mark.parametrize("kwargs", [dict(with_mask=True), dict(with_shared_head=True),
                                    dict(bbox_head=BBoxHeadFixture(num_classes=2))])
def test_unsupported_head_structures_fail_explicitly(kwargs):
    with pytest.raises(ValueError, match="single-class"):
        QualityRoIHead(**kwargs)


def test_unsupported_ranked_test_paths_fail_instead_of_silent_fallback():
    head = QualityRoIHead(quality_inference=True)
    with pytest.raises(NotImplementedError, match="single-scale"):
        head.aug_test()
    with pytest.raises(NotImplementedError, match="async"):
        run_async(head.async_simple_test())
    with pytest.raises(NotImplementedError, match="ONNX"):
        head.onnx_export()
    head.quality_inference = False
    assert head.aug_test() is head.parent_output
    assert run_async(head.async_simple_test()) is head.parent_output
    assert head.onnx_export() is head.parent_output


def test_real_mmdet_216_disabled_output_and_supervised_forward():
    """Runs only in the actual legacy stack; no fixture substitutes operators."""
    mmdet = pytest.importorskip("mmdet")
    if not mmdet.__version__.startswith("2.16."):
        pytest.skip("This integration test targets the pinned MMDetection 2.16 stack")
    pytest.importorskip("mmcv.ops")
    from mmcv import ConfigDict
    from mmdet.models.builder import build_head
    from ssod.models.roi_heads import QualityRoIHead as RegisteredQualityRoIHead

    assert RegisteredQualityRoIHead.__name__ == "QualityRoIHead"
    train_cfg = ConfigDict(
        assigner=dict(type="MaxIoUAssigner", pos_iou_thr=.5, neg_iou_thr=.5,
                      min_pos_iou=.5, match_low_quality=False),
        sampler=dict(type="RandomSampler", num=8, pos_fraction=.5,
                     neg_pos_ub=-1, add_gt_as_proposals=True), pos_weight=-1,
        debug=False)
    cfg = dict(
        type="StandardRoIHead", train_cfg=train_cfg,
        test_cfg=ConfigDict(score_thr=.05, nms=dict(type="nms", iou_threshold=.5),
                            max_per_img=100),
        bbox_roi_extractor=dict(type="SingleRoIExtractor",
                                roi_layer=dict(type="RoIAlign", output_size=7,
                                               sampling_ratio=0),
                                out_channels=256, featmap_strides=[4]),
        bbox_head=dict(type="Shared2FCBBoxHead", in_channels=256, fc_out_channels=16,
                       roi_feat_size=7, num_classes=1, reg_class_agnostic=False))
    baseline = build_head(cfg)
    disabled = build_head(dict(cfg, type="QualityRoIHead", quality_enabled=False))
    disabled.load_state_dict(baseline.state_dict(), strict=True)
    features = (torch.randn(1, 256, 16, 16, requires_grad=True),)
    proposals = [torch.tensor([[0., 0., 20., 20.], [30., 30., 50., 50.]])]
    meta = [dict(img_shape=(64, 64, 3), scale_factor=np.ones(4))]
    with torch.no_grad():
        expected = baseline.simple_test(features, proposals, meta, rescale=True)
        actual = disabled.simple_test(features, proposals, meta, rescale=True)
    np.testing.assert_array_equal(actual[0][0], expected[0][0])
    enabled = build_head(dict(cfg, type="QualityRoIHead"))
    gt, labels = [torch.tensor([[0., 0., 20., 20.]])], [torch.tensor([0])]
    losses = enabled.forward_train(features, meta, proposals, gt, labels,
                                   quality_supervised=True)
    assert "loss_quality" in losses and torch.isfinite(losses["loss_quality"])
    losses["loss_quality"].backward()
    assert all(parameter.grad is not None for parameter in enabled.quality_head.parameters())
    assert "loss_quality" not in enabled.forward_train(features, meta, proposals, gt, labels)
