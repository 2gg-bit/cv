"""Supervised-only P2 auxiliary loss; detection and inference are inherited."""

import math

from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

from .foreground_head import ForegroundHead, foreground_targets, foreground_focal_loss


@HEADS.register_module()
class ForegroundRoIHead(StandardRoIHead):
    def __init__(self, foreground_enabled=False, foreground_hidden_channels=32,
                 foreground_loss_weight=.1, **kwargs):
        super().__init__(**kwargs)
        self.foreground_enabled = bool(foreground_enabled)
        self.foreground_loss_weight = float(foreground_loss_weight)
        if not math.isfinite(self.foreground_loss_weight) or self.foreground_loss_weight < 0:
            raise ValueError("Foreground loss weight must be finite and nonnegative")
        if self.foreground_enabled:
            if not self.with_bbox or self.with_mask or self.bbox_head.num_classes != 1:
                raise ValueError("Foreground supervision requires a single-class bbox detector")
            self.foreground_stride = self.bbox_roi_extractor.featmap_strides[0]
            self.foreground_head = ForegroundHead(
                self.bbox_head.in_channels, foreground_hidden_channels)

    def foreground_initialization_keys(self):
        if not self.foreground_enabled:
            return ()
        return tuple("roi_head.foreground_head." + key for key in (
            "conv.weight", "conv.bias", "out.weight", "out.bias"))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # MMCV full-run restore may use strict=False. Never silently resume an
        # M2 checkpoint with a random head or discard a trained foreground head.
        loaded = {key for key in state_dict if key.startswith(prefix + "foreground_head.")}
        expected = {prefix + key for key in self.state_dict()
                    if key.startswith("foreground_head.")}
        if loaded != expected:
            raise RuntimeError("Foreground checkpoint/config mismatch; initialize a fresh "
                               "Phase3 via load1_from/load2_from or resume the matching FG config")
        for key in expected:
            if state_dict[key].shape != self.state_dict()[key[len(prefix):]].shape:
                raise RuntimeError("Foreground checkpoint shape mismatch: " + key)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None,
                      foreground_supervised=False):
        losses = super().forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks)
        # DualTeacher explicitly opts in only for real labeled sup1/sup2.
        # Direct pseudo-box RoI regression calls retain the default False.
        if self.foreground_enabled and foreground_supervised:
            logits = self.foreground_head(x[0])
            target, valid = foreground_targets(
                logits, gt_bboxes, img_metas, self.foreground_stride, gt_bboxes_ignore)
            losses["loss_foreground"] = self.foreground_loss_weight * foreground_focal_loss(
                logits, target, valid)
        return losses
