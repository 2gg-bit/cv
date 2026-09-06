"""M1: opt-in supervised localization quality without changing pseudo labels."""

import math

import torch
from mmcv.ops import batched_nms
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

from .localization_quality_head import (
    LocalizationQualityHead, aligned_iou, fp32_quality_context,
    quality_candidate_scores)


@HEADS.register_module()
class QualityRoIHead(StandardRoIHead):
    """Single-class quality auxiliary head with an independent test-time switch.

    ``quality_enabled=False`` constructs no new parameters and delegates to
    StandardRoIHead. ``quality_inference=False`` trains the auxiliary head but
    retains baseline class-score inference (M1-A). Turning inference on changes
    only final single-scale detector ranking to p(ship) * q(IoU) (M1-B).

    IMPORTANT: ``simple_test_bboxes`` is intentionally inherited unchanged.
    DualTeacher calls that lower-level API for pseudo labels, background scores
    and jitter estimates; none of those paths may use this ranking branch.
    """

    def __init__(self, quality_enabled=True, quality_inference=False,
                 quality_hidden_channels=64, quality_loss_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.quality_enabled = bool(quality_enabled)
        self.quality_inference = bool(quality_inference)
        self.quality_loss_weight = float(quality_loss_weight)
        self._quality_supervised = False
        if not math.isfinite(self.quality_loss_weight) or self.quality_loss_weight < 0:
            raise ValueError("quality_loss_weight must be finite and nonnegative")
        if self.quality_enabled:
            if (not self.with_bbox or self.with_mask or self.with_shared_head or
                    self.bbox_head.num_classes != 1 or
                    not self.bbox_head.with_cls or not self.bbox_head.with_reg):
                raise ValueError("M1 requires a single-class, bbox-only RoI head "
                                 "without a shared head")
            if self.bbox_head.custom_cls_channels:
                raise ValueError("M1 requires the baseline softmax classifier")
            self.quality_head = LocalizationQualityHead(
                in_channels=self.bbox_head.in_channels,
                hidden_channels=quality_hidden_channels)

    def quality_initialization_keys(self):
        """Only these detector-relative keys may be absent in Phase 1/2 files."""
        if not self.quality_enabled:
            return ()
        return tuple("roi_head.quality_head." + name for name in (
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"))

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None,
                      quality_supervised=False):
        # Explicit opt-in is supplied ONLY by DualTeacher's sup1/sup2 paths.
        # Direct unsupervised roi_head.forward_train calls default to False.
        previous = self._quality_supervised
        self._quality_supervised = bool(quality_supervised)
        try:
            return super().forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks)
        finally:
            self._quality_supervised = previous

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        results = super()._bbox_forward_train(
            x, sampling_results, gt_bboxes, gt_labels, img_metas)
        if not (self.quality_enabled and self._quality_supervised):
            return results
        features, deltas = results["bbox_feats"], results["bbox_pred"]
        positive_indices, targets = [], []
        offset = 0
        # SamplingResult.bboxes concatenates positive boxes before negatives,
        # matching StandardRoIHead's bbox2roi and the bbox target ordering.
        for sample, meta in zip(sampling_results, img_metas):
            count = sample.pos_bboxes.size(0)
            positive_indices.append(torch.arange(
                offset, offset + count, device=features.device, dtype=torch.long))
            if count:
                with torch.no_grad(), fp32_quality_context(deltas):
                    decoded = self.bbox_head.bbox_coder.decode(
                        sample.pos_bboxes.detach().float(),
                        deltas[offset:offset + count].detach().float(),
                        max_shape=meta["img_shape"])
                    targets.append(aligned_iou(decoded, sample.pos_gt_bboxes))
            offset += sample.bboxes.size(0)
        if offset != features.size(0) or len(sampling_results) != len(img_metas):
            raise RuntimeError("Quality sampling results do not align with RoIs")
        indices = (torch.cat(positive_indices) if positive_indices else
                   torch.empty(0, device=features.device, dtype=torch.long))
        target = (torch.cat(targets) if targets else features.new_empty((0,)))
        logits = self.quality_head(features[indices], deltas[indices])
        results["loss_bbox"]["loss_quality"] = (
            self.quality_loss_weight * self.quality_head.loss(logits, target))
        return results

    def simple_test(self, x, proposal_list, img_metas, proposals=None,
                    rescale=False):
        if not (self.quality_enabled and self.quality_inference):
            return super().simple_test(
                x, proposal_list, img_metas, proposals=proposals, rescale=rescale)
        if len(proposal_list) != len(img_metas):
            raise ValueError("Proposal and image metadata counts differ")
        rois = bbox2roi(proposal_list)
        if rois.size(0) == 0:
            return [bbox2result(
                rois.new_zeros((0, 5)),
                rois.new_zeros((0,), dtype=torch.long), 1)
                    for _ in proposal_list]
        results = self._bbox_forward(x, rois)
        logits = self.quality_head(results["bbox_feats"], results["bbox_pred"])
        sizes = tuple(len(proposal) for proposal in proposal_list)
        rois_per_image = rois.split(sizes, 0)
        cls_per_image = results["cls_score"].split(sizes, 0)
        deltas_per_image = results["bbox_pred"].split(sizes, 0)
        logits_per_image = logits.split(sizes, 0)
        output = []
        for i, meta in enumerate(img_metas):
            if sizes[i] == 0:
                output.append(bbox2result(
                    rois.new_zeros((0, 5)),
                    rois.new_zeros((0,), dtype=torch.long), 1))
                continue
            boxes, scores = self.bbox_head.get_bboxes(
                rois_per_image[i], cls_per_image[i], deltas_per_image[i],
                meta["img_shape"], meta["scale_factor"], rescale=rescale,
                cfg=None)
            valid, joint = quality_candidate_scores(
                scores[:, 0], logits_per_image[i], self.test_cfg.score_thr)
            boxes, joint = boxes[valid], joint[valid]
            labels = boxes.new_zeros((boxes.size(0),), dtype=torch.long)
            if boxes.size(0):
                dets, keep = batched_nms(
                    boxes, joint, labels, self.test_cfg.nms)
                if self.test_cfg.max_per_img > 0:
                    dets = dets[:self.test_cfg.max_per_img]
                    keep = keep[:self.test_cfg.max_per_img]
                labels = labels[keep]
            else:
                dets = boxes.new_zeros((0, 5))
            output.append(bbox2result(dets, labels, 1))
        return output

    def aug_test(self, *args, **kwargs):
        if self.quality_enabled and self.quality_inference:
            raise NotImplementedError("M1 quality ranking is single-scale only")
        return super().aug_test(*args, **kwargs)

    async def async_simple_test(self, *args, **kwargs):
        if self.quality_enabled and self.quality_inference:
            raise NotImplementedError("M1 quality ranking does not support async test")
        return await super().async_simple_test(*args, **kwargs)

    def onnx_export(self, *args, **kwargs):
        if self.quality_enabled and self.quality_inference:
            raise NotImplementedError("M1 quality ranking does not support ONNX export")
        return super().onnx_export(*args, **kwargs)
