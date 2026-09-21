"""Box-supervised foreground locations, independent of MMCV/MMDetection.

These are coarse position targets, not segmentation masks. Only annotated
images may use this loss; missing pseudo boxes do not establish background.
"""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


@contextmanager
def fp32_foreground_context(tensor):
    if tensor.is_cuda:
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield


@torch.no_grad()
def foreground_targets(logits, gt_bboxes, img_metas, stride=4,
                       gt_bboxes_ignore=None):
    """Max-combine truncated, box-sized Gaussians on the feature grid.

    Centers are quantized with floor(center / stride). Each valid box has a
    unit peak even when smaller than one cell. Sigma is box size / 6, with a
    half-cell minimum; draw to 3 sigma. Padding and ignored regions are masked
    out. The final partial cell of each image is valid. Boxes are already in
    the augmented image coordinates supplied to the detector.
    """
    if logits.dim() != 4 or logits.size(1) != 1:
        raise ValueError("Foreground logits must have shape N x 1 x H x W")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("Foreground stride must be a positive integer")
    n, _, h, w = logits.shape
    if len(gt_bboxes) != n or len(img_metas) != n:
        raise ValueError("Foreground target batch sizes must match")
    ignored = gt_bboxes_ignore if gt_bboxes_ignore is not None else [None] * n
    if len(ignored) != n:
        raise ValueError("Ignored-box batch size must match")
    target = torch.zeros_like(logits, dtype=torch.float32)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    for i, (boxes, meta, ignore) in enumerate(zip(gt_bboxes, img_metas, ignored)):
        ih, iw = meta["img_shape"][:2]
        ph, pw = meta["pad_shape"][:2]
        if not (0 < ih <= ph and 0 < iw <= pw):
            raise ValueError("Invalid image/padding shape")
        # Other images in the batch may make the actual tensor larger than
        # this image's pad_shape. Never rescale targets using that shape.
        if h < math.ceil(ph / stride) or w < math.ceil(pw / stride):
            raise ValueError("Feature map is too small for the configured stride")
        vh, vw = int(math.ceil(ih / stride)), int(math.ceil(iw / stride))
        valid[i, 0, :vh, :vw] = True
        for collection, is_ignore in ((ignore, True), (boxes, False)):
            if collection is None:
                continue
            if collection.dim() != 2 or collection.size(1) != 4:
                raise ValueError("Foreground boxes must be N x 4 xyxy tensors")
            if not bool(torch.isfinite(collection).all()):
                raise ValueError("Foreground boxes contain NaN/Inf")
            for x1, y1, x2, y2 in collection.detach().float().cpu().tolist():
                x1, x2 = max(0., min(iw, x1)), max(0., min(iw, x2))
                y1, y2 = max(0., min(ih, y1)), max(0., min(ih, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                if is_ignore:
                    valid[i, 0, int(y1 // stride):int(math.ceil(y2 / stride)),
                          int(x1 // stride):int(math.ceil(x2 / stride))] = False
                    continue
                cx = min(vw - 1, int((x1 + x2) / (2 * stride)))
                cy = min(vh - 1, int((y1 + y2) / (2 * stride)))
                sx, sy = max((x2 - x1) / (6 * stride), .5), max((y2 - y1) / (6 * stride), .5)
                rx, ry = int(math.ceil(3 * sx)), int(math.ceil(3 * sy))
                left, right = max(0, cx - rx), min(vw, cx + rx + 1)
                top, bottom = max(0, cy - ry), min(vh, cy + ry + 1)
                dx = (torch.arange(left, right, device=logits.device, dtype=torch.float32) - cx) / sx
                dy = (torch.arange(top, bottom, device=logits.device, dtype=torch.float32) - cy) / sy
                gaussian = torch.exp(-.5 * (dy[:, None].pow(2) + dx[None, :].pow(2)))
                patch = target[i, 0, top:bottom, left:right]
                patch.copy_(torch.max(patch, gaussian))
    return target, valid


def foreground_focal_loss(logits, target, valid):
    """Gaussian focal loss (alpha=2, beta=4), per-image peak normalization.

    Average images, not pixels. Empty annotated images contribute background
    loss with denominator 1. Padding/ignore pixels contribute exactly zero.
    """
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("Foreground loss shapes must match")
    with fp32_foreground_context(logits):
        z, target = logits.float(), target.detach().float()
        p = z.sigmoid()
        pos = target.eq(1) & valid
        neg = target.lt(1) & valid
        positive = -F.logsigmoid(z) * (1 - p).pow(2) * pos.float()
        negative = -F.logsigmoid(-z) * p.pow(2) * (1 - target).pow(4) * neg.float()
        count = pos.float().flatten(1).sum(1).clamp(min=1)
        return ((positive + negative).flatten(1).sum(1) / count).mean()


class ForegroundHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=32):
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Foreground channels must be positive")
        self.conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.out = nn.Conv2d(hidden_channels, 1, 1)
        self.init_weights()

    def init_weights(self):
        for layer in (self.conv, self.out):
            nn.init.normal_(layer.weight, std=.01)
            nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.out.bias, math.log(.01 / .99))

    def forward(self, features):
        # Functional FP32 convolutions also tolerate legacy model.half().
        # Casts retain gradients back to both the head and shared FPN features.
        with fp32_foreground_context(features):
            hidden = F.relu(F.conv2d(features.float(), self.conv.weight.float(),
                                    self.conv.bias.float(), padding=1))
            return F.conv2d(hidden, self.out.weight.float(), self.out.bias.float())
