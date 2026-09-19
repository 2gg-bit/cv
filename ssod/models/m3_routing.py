"""Deterministic, detached regression-target routing for M3.

Rows must already correspond to the same fused pseudo-label anchor.  This
module does not match boxes, assign proposals, sample RoIs, or change labels.
It deliberately depends only on PyTorch so its invariants can be tested on CPU.
"""

import math

import torch


def _boxes(name, value, rows=None, device=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError("{} must be a tensor".format(name))
    if value.dim() != 2 or value.size(1) != 4:
        raise ValueError("{} must have shape (N, 4)".format(name))
    if rows is not None and value.size(0) != rows:
        raise ValueError("{} must have {} rows".format(name, rows))
    if device is not None and value.device != device:
        raise ValueError("{} must share the anchors/targets device".format(name))
    if not value.is_floating_point():
        raise ValueError("{} must have a floating-point dtype".format(name))


def _uncertainty(name, value, rows, device):
    if not isinstance(value, torch.Tensor):
        raise ValueError("{} must be a tensor".format(name))
    if not (value.dim() == 1 and value.size(0) == rows) and not (
        value.dim() == 2 and tuple(value.shape) == (rows, 4)
    ):
        raise ValueError("{} must have shape (N,) or (N, 4)".format(name))
    if value.device != device:
        raise ValueError("{} must share the anchors device".format(name))
    value = value.detach().float()
    valid = torch.isfinite(value) & (value >= 0)
    if value.dim() == 2:
        valid = valid.all(dim=1)
        mean = value.mean(dim=1)
        # Preserve the normal FP32 mean (including subnormal inputs). Only an
        # overflowing finite nonnegative row needs the scaled sum fallback.
        overflow = valid & ~torch.isfinite(mean)
        value = torch.where(overflow, (value * 0.25).sum(dim=1), mean)
    return value, valid


def _valid_geometry(boxes):
    sizes = boxes[:, 2:] - boxes[:, :2]
    return (torch.isfinite(boxes).all(dim=1)
            & torch.isfinite(sizes).all(dim=1) & (sizes > 0).all(dim=1))


def _aligned_iou(first, second):
    """Continuous-coordinate IoU of corresponding rows, computed in FP32."""
    overlap = (torch.min(first[:, 2:], second[:, 2:])
               - torch.max(first[:, :2], second[:, :2])).clamp(min=0)
    first_size = first[:, 2:] - first[:, :2]
    second_size = second[:, 2:] - second[:, :2]
    # A common scale on each axis leaves IoU unchanged, while avoiding area
    # overflow/underflow for otherwise valid finite coordinate ranges.
    scale = torch.max(first_size, second_size)
    overlap = overlap / scale
    first_size, second_size = first_size / scale, second_size / scale
    intersection = overlap[:, 0] * overlap[:, 1]
    union = (first_size[:, 0] * first_size[:, 1]
             + second_size[:, 0] * second_size[:, 1] - intersection)
    return intersection / union


@torch.no_grad()
def select_regression_targets(anchors, teacher1_boxes, teacher2_boxes,
                              uncertainty1, uncertainty2, min_anchor_iou=0.5,
                              mode="lower_uncertainty"):
    """Choose a valid teacher box for each *existing* pseudo-label anchor.

    A candidate needs finite coordinates, positive finite width/height, finite
    nonnegative uncertainty, and IoU >= ``min_anchor_iou`` with its own anchor.
    Four-coordinate uncertainty is reduced by its arithmetic mean.  Only a
    strictly smaller mean wins when both candidates are valid; an exact FP32
    tie keeps the anchor.  If only one candidate is valid, that candidate wins.

    ``mode`` may also be ``original``, ``teacher1``, or ``teacher2``.  The latter
    two use only the specified teacher when valid, otherwise retaining the
    anchor.  Source IDs are 0 for the original anchor, 1 for teacher 1, and 2
    for teacher 2.  Outputs are detached; box dtype/device match ``anchors``.
    """
    if mode not in ("lower_uncertainty", "original", "teacher1", "teacher2"):
        raise ValueError("Unsupported M3 routing mode: {}".format(mode))
    try:
        threshold = float(min_anchor_iou)
    except (TypeError, ValueError):
        raise ValueError("min_anchor_iou must be finite and in [0, 1]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("min_anchor_iou must be finite and in [0, 1]")

    _boxes("anchors", anchors)
    count, device = anchors.size(0), anchors.device
    _boxes("teacher1_boxes", teacher1_boxes, count, device)
    _boxes("teacher2_boxes", teacher2_boxes, count, device)
    score1, valid1 = _uncertainty("uncertainty1", uncertainty1, count, device)
    score2, valid2 = _uncertainty("uncertainty2", uncertainty2, count, device)
    anchors_fp32 = anchors.detach().float()
    if not bool(_valid_geometry(anchors_fp32).all()):
        raise ValueError("anchors must have finite coordinates and positive finite sizes")

    selected = anchors.detach().clone()
    source_ids = torch.zeros(count, dtype=torch.long, device=device)
    if mode == "original" or count == 0:
        return selected, source_ids

    boxes1 = teacher1_boxes.detach().float()
    boxes2 = teacher2_boxes.detach().float()
    valid1 = valid1 & _valid_geometry(boxes1) & (
        _aligned_iou(anchors_fp32, boxes1) >= threshold)
    valid2 = valid2 & _valid_geometry(boxes2) & (
        _aligned_iou(anchors_fp32, boxes2) >= threshold)
    if mode == "teacher1":
        choose1, choose2 = valid1, torch.zeros_like(valid2)
    elif mode == "teacher2":
        choose1, choose2 = torch.zeros_like(valid1), valid2
    else:
        choose1 = valid1 & (~valid2 | (score1 < score2))
        choose2 = valid2 & (~valid1 | (score2 < score1))
    selected[choose1] = teacher1_boxes.detach().to(dtype=anchors.dtype)[choose1]
    selected[choose2] = teacher2_boxes.detach().to(dtype=anchors.dtype)[choose2]
    source_ids[choose1], source_ids[choose2] = 1, 2
    return selected, source_ids


@torch.no_grad()
def replace_positive_regression_targets(bbox_head, sampling_results,
                                        bbox_targets, targets_per_image):
    """Replace only changed positive regression targets after original sampling.

    ``targets_per_image[i]`` has the same GT row order used for assignment and
    sampling of image ``i``.  ``pos_assigned_gt_inds`` selects rows from it.
    The standard MMDetection target layout is positives then negatives within
    each image, with images concatenated.  Labels, both weights, the sampling
    objects, and all unchanged target rows are preserved exactly.

    Returns a four-tensor tuple, cloning/detaching only ``bbox_targets[2]``.
    """
    if not isinstance(bbox_targets, (tuple, list)) or len(bbox_targets) != 4:
        raise ValueError("bbox_targets must contain four tensors")
    if len(sampling_results) != len(targets_per_image):
        raise ValueError("sampling_results and targets_per_image must have equal lengths")
    labels, label_weights, original_targets, bbox_weights = bbox_targets
    _boxes("bbox_targets[2]", original_targets)
    count, device = original_targets.size(0), original_targets.device
    _boxes("bbox_targets[3]", bbox_weights, count, device)
    for name, value in (("labels", labels), ("label_weights", label_weights)):
        if (not isinstance(value, torch.Tensor) or value.dim() != 1
                or value.size(0) != count or value.device != device):
            raise ValueError("{} must have shape (N,) on the targets device".format(name))

    replaced = original_targets.detach().clone()
    offset = 0
    for image_index, (result, selected) in enumerate(zip(sampling_results,
                                                        targets_per_image)):
        _boxes("targets_per_image[{}]".format(image_index), selected, device=device)
        _boxes("pos_bboxes", result.pos_bboxes, device=device)
        _boxes("neg_bboxes", result.neg_bboxes, device=device)
        positives = result.pos_bboxes.size(0)
        negatives = result.neg_bboxes.size(0)
        _boxes("pos_gt_bboxes", result.pos_gt_bboxes, positives, device)
        indices = result.pos_assigned_gt_inds
        if (not isinstance(indices, torch.Tensor) or indices.dim() != 1
                or indices.size(0) != positives or indices.dtype != torch.long
                or indices.device != device):
            raise ValueError("pos_assigned_gt_inds must be a LongTensor of shape (num_pos,)")
        if offset + positives + negatives > count:
            raise ValueError("sampling counts exceed the number of bbox target rows")
        if positives:
            if bool((indices < 0).any()) or bool((indices >= selected.size(0)).any()):
                raise ValueError("pos_assigned_gt_inds must index targets_per_image")
            chosen = selected.detach()[indices]
            changed = (chosen != result.pos_gt_bboxes.detach()).any(dim=1)
            if bool(changed.any()):
                chosen = chosen[changed]
                if not bbox_head.reg_decoded_bbox:
                    chosen = bbox_head.bbox_coder.encode(
                        result.pos_bboxes.detach()[changed], chosen)
                _boxes("replacement targets", chosen, int(changed.sum()), device)
                positive_targets = replaced[offset:offset + positives]
                positive_targets[changed] = chosen.detach().to(dtype=replaced.dtype)
        offset += positives + negatives
    if offset != count:
        raise ValueError("sampling counts do not match the number of bbox target rows")
    return labels, label_weights, replaced, bbox_weights
