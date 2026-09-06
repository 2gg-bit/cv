"""Small, FP32 localization-quality predictor and side-effect-free helpers.

This module deliberately has no MMDetection/MMCV dependencies, so its math can
also be regression-tested on CPU. It does not produce pseudo labels.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def fp32_quality_context(tensor):
    # The supported training stack (torch >= 1.7 + MMCV 1.3.9) uses CUDA AMP
    # with FP32 master parameters. Do not depend on the outer autocast state.
    if tensor.is_cuda:
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield


def aligned_iou(predicted_boxes, assigned_boxes):
    """Detached continuous-coordinate IoU for corresponding pairs of boxes."""
    if predicted_boxes.shape != assigned_boxes.shape:
        raise ValueError("Quality IoU expects identically shaped box pairs")
    if predicted_boxes.dim() != 2 or predicted_boxes.size(1) != 4:
        raise ValueError("Quality IoU expects N x 4 boxes")
    with torch.no_grad(), fp32_quality_context(predicted_boxes):
        predicted = predicted_boxes.detach().float()
        assigned = assigned_boxes.detach().float()
        wh = (torch.min(predicted[:, 2:], assigned[:, 2:]) -
              torch.max(predicted[:, :2], assigned[:, :2])).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]
        pred_wh = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0)
        gt_wh = (assigned[:, 2:] - assigned[:, :2]).clamp(min=0)
        union = (pred_wh[:, 0] * pred_wh[:, 1] +
                 gt_wh[:, 0] * gt_wh[:, 1] - intersection)
        return (intersection / union.clamp(min=1e-6)).clamp(0, 1)


def quality_candidate_scores(class_scores, quality_logits, score_thr):
    """Keep the baseline class-score mask, then score by p(ship) * q(IoU).

    The combined score MUST NOT be thresholded a second time. In particular,
    a baseline-eligible box can retain a combined score below ``score_thr``.
    """
    if class_scores.dim() != 1 or quality_logits.shape != class_scores.shape:
        raise ValueError("Quality scoring expects paired one-dimensional scores")
    with fp32_quality_context(class_scores):
        valid = class_scores.float() > score_thr
        joint = class_scores.float() * quality_logits.float().sigmoid()
    return valid, joint


class LocalizationQualityHead(nn.Module):
    """Pooled RoI features + detached regression deltas -> one quality logit."""

    def __init__(self, in_channels=256, hidden_channels=64):
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Quality-head channel counts must be positive")
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels + 4, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
        self.init_weights()

    def init_weights(self):
        for layer in (self.fc1, self.fc2):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, roi_features, bbox_deltas):
        if roi_features.dim() != 4 or roi_features.size(1) != self.in_channels:
            raise ValueError("Quality head expects N x C x H x W RoI features")
        if bbox_deltas.shape != (roi_features.size(0), 4):
            raise ValueError("M1 quality head supports one-class N x 4 deltas only")
        with fp32_quality_context(roi_features):
            pooled = roi_features.float().mean(dim=(2, 3))
            inputs = torch.cat((pooled, bbox_deltas.detach().float()), dim=1)
            return self.fc2(F.relu(self.fc1(inputs))).squeeze(1)

    def loss(self, logits, targets):
        if logits.shape != targets.shape or logits.dim() != 1:
            raise ValueError("Quality loss expects paired one-dimensional tensors")
        with fp32_quality_context(logits):
            if logits.numel() == 0:
                # An empty forward still connects ALL head parameters to the
                # graph, keeping zero-positive supervised batches DDP-safe.
                return logits.float().sum()
            return F.binary_cross_entropy_with_logits(
                logits.float(), targets.detach().float())
