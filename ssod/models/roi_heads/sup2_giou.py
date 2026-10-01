"""Optional GIoU auxiliary supervision for positive, labeled SAR RoIs only.

The existing bbox loss, targets, sampling and inference are inherited unchanged.
This head adds no parameters or buffers. Off / beta=0 takes the exact parent path.
"""

import math

import torch
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead


def aligned_giou_loss(pred, target, eps=1e-7):
    """Per-box 1-GIoU for aligned Nx4 xyxy boxes, in FP32 (no IoU matching)."""
    if pred.shape != target.shape or pred.dim() != 2 or pred.size(1) != 4:
        raise ValueError("GIoU expects aligned Nx4 boxes")
    pred, target = pred.float(), target.detach().float()
    overlap_wh = (torch.min(pred[:, 2:], target[:, 2:]) -
                  torch.max(pred[:, :2], target[:, :2])).clamp(min=0)
    intersection = overlap_wh[:, 0] * overlap_wh[:, 1]
    pred_wh = (pred[:, 2:] - pred[:, :2]).clamp(min=0)
    target_wh = (target[:, 2:] - target[:, :2]).clamp(min=0)
    union = (pred_wh[:, 0] * pred_wh[:, 1] +
             target_wh[:, 0] * target_wh[:, 1] - intersection).clamp(min=eps)
    enclosing_wh = (torch.max(pred[:, 2:], target[:, 2:]) -
                    torch.min(pred[:, :2], target[:, :2])).clamp(min=0)
    enclosing = (enclosing_wh[:, 0] * enclosing_wh[:, 1]).clamp(min=eps)
    giou = intersection / union - (enclosing - union) / enclosing
    return 1.0 - giou


@HEADS.register_module()
class Sup2GIoURoIHead(StandardRoIHead):
    def __init__(self, sup2_giou_enabled=False, sup2_giou_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sup2_giou_enabled = bool(sup2_giou_enabled)
        self.sup2_giou_weight = float(sup2_giou_weight)
        if not math.isfinite(self.sup2_giou_weight) or self.sup2_giou_weight < 0:
            raise ValueError("sup2_giou_weight must be finite and nonnegative")

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        # Keep the parent's sampling targets, original classification/regression
        # losses and forward order, including the beta=0 identity endpoint.
        result = super()._bbox_forward_train(
            x, sampling_results, gt_bboxes, gt_labels, img_metas)
        if not self.sup2_giou_enabled or self.sup2_giou_weight == 0:
            return result
        if len(sampling_results) != len(img_metas):
            raise ValueError("Sampling results and image metadata must align")
        eligible = [meta.get("tag") == "sup2" for meta in img_metas]
        if not any(eligible):
            return result
        # Pseudo-label RoI regression also calls this method, but its image tag
        # is unsup_student; no auxiliary loss is added on that path.
        with torch.cuda.amp.autocast(enabled=False):
            loss = self._sup2_giou_loss(result["bbox_pred"].float(),
                                        sampling_results, eligible)
        result["loss_bbox"]["loss_giou"] = self.sup2_giou_weight * loss
        return result

    def _sup2_giou_loss(self, bbox_pred, sampling_results, eligible):
        offset, normalizer = 0, 0
        predicted, targets = [], []
        head = self.bbox_head
        for sample, enabled in zip(sampling_results, eligible):
            count = sample.bboxes.size(0)
            npos = sample.pos_bboxes.size(0)
            if enabled:
                # Match the existing RoI regression scale: divide by all sampled
                # RoIs on eligible images, not just by the number of positives.
                normalizer += count
                if npos:
                    delta = bbox_pred[offset:offset + npos]
                    if head.reg_class_agnostic:
                        delta = delta.reshape(npos, 4)
                    else:
                        rows = torch.arange(npos, device=delta.device)
                        delta = delta.reshape(npos, head.num_classes, 4)[
                            rows, sample.pos_gt_labels.long()]
                    # Use the same bbox coder as the original L1 loss, with the
                    # original regression stds. Never compute IoU on encoded deltas.
                    predicted.append(head.bbox_coder.decode(
                        sample.pos_bboxes.detach().float(), delta))
                    targets.append(sample.pos_gt_bboxes.detach().float())
            offset += count
        if offset != bbox_pred.size(0):
            raise ValueError("RoI order/count does not match bbox predictions")
        if not predicted:
            # Keep a valid zero-gradient connection for empty positive batches.
            return bbox_pred.sum() * 0.0
        return aligned_giou_loss(torch.cat(predicted), torch.cat(targets)).sum() / max(normalizer, 1)
