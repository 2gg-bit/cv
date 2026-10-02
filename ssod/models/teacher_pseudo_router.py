"""Teacher-aware pseudo-label consolidation for the dual-teacher detector.

This module is deliberately NumPy-only so its matching/fusion rules can be
tested without importing MMDetection. Inputs are per-image detections in
``xyxy + score`` format. Matched same-class detections become one
score-weighted box with a conservative geometric-mean score; unmatched
detections from either teacher are retained. The resulting shared target set
is consumed by both student branches, so a detection from one teacher cannot
silently become a background target for the other branch.
"""

import numpy as np


def _as_detections(boxes, labels, name):
    boxes = np.asarray(boxes, dtype=np.float64)
    labels = np.asarray(labels)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 5)
    if labels.size == 0:
        labels = np.zeros((0,), dtype=np.int64)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("{} boxes must have shape (N, 5)".format(name))
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError("{} labels must have shape (N,)".format(name))
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("{} labels must be integers".format(name))
    if not np.isfinite(boxes).all():
        raise ValueError("{} boxes contain NaN or Inf".format(name))
    if boxes.shape[0]:
        if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
            raise ValueError("{} boxes must have positive width and height".format(name))
        if np.any(boxes[:, 4] < 0) or np.any(boxes[:, 4] > 1):
            raise ValueError("{} scores must be in [0, 1]".format(name))
        if np.any(labels < 0):
            raise ValueError("{} labels must be nonnegative".format(name))
    return boxes, labels.astype(np.int64, copy=False)


def _pairwise_iou(boxes1, boxes2):
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    left_top = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = np.minimum(boxes1[:, None, 2:4], boxes2[None, :, 2:4])
    extent = np.maximum(right_bottom - left_top, 0.0)
    intersection = extent[..., 0] * extent[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]) *
             (boxes1[:, 3] - boxes1[:, 1]))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) *
             (boxes2[:, 3] - boxes2[:, 1]))[None, :]
    union = area1 + area2 - intersection
    return intersection / np.maximum(union, np.finfo(np.float64).tiny)


def fuse_teacher_detections(boxes1, labels1, boxes2, labels2, iou_threshold=0.5):
    """Match/fuse teacher detections and retain one-sided candidates.

    Matching is class-aware, greedy by descending IoU, and one-to-one. A
    matched pair is fused using its two detection scores as coordinate weights;
    the fused score is ``sqrt(score1 * score2)`` to reward agreement without
    exceeding the stronger teacher's confidence. Unmatched detections retain
    their original boxes and scores. Output ordering is deterministic and
    descending by score.

    Returns a dict with ``boxes`` (N, 5), ``labels`` (N,), ``sources`` (N,),
    and integer counts. Source IDs are 1 (teacher1), 2 (teacher2), and 3
    (both teachers).
    """
    if not np.isfinite(iou_threshold) or not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be finite and in (0, 1]")
    boxes1, labels1 = _as_detections(boxes1, labels1, "teacher1")
    boxes2, labels2 = _as_detections(boxes2, labels2, "teacher2")

    ious = _pairwise_iou(boxes1[:, :4], boxes2[:, :4])
    candidates = []
    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            if labels1[i] == labels2[j] and ious[i, j] >= iou_threshold:
                candidates.append((float(ious[i, j]), i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used1, used2 = set(), set()
    fused = []
    for _overlap, i, j in candidates:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        score1, score2 = boxes1[i, 4], boxes2[j, 4]
        weight_sum = score1 + score2
        if weight_sum > 0:
            coordinates = (boxes1[i, :4] * score1 + boxes2[j, :4] * score2) / weight_sum
        else:
            coordinates = (boxes1[i, :4] + boxes2[j, :4]) * 0.5
        score = np.sqrt(score1 * score2)
        fused.append((float(score), 3, min(i, j), np.r_[coordinates, score], int(labels1[i])))

    for i in range(len(boxes1)):
        if i not in used1:
            fused.append((float(boxes1[i, 4]), 1, i,
                          boxes1[i].copy(), int(labels1[i])))
    for j in range(len(boxes2)):
        if j not in used2:
            fused.append((float(boxes2[j, 4]), 2, j,
                          boxes2[j].copy(), int(labels2[j])))

    fused.sort(key=lambda item: (-item[0], item[1], item[2]))
    if fused:
        out_boxes = np.stack([item[3] for item in fused]).astype(np.float32, copy=False)
        out_labels = np.asarray([item[4] for item in fused], dtype=np.int64)
        out_sources = np.asarray([item[1] for item in fused], dtype=np.int8)
    else:
        out_boxes = np.zeros((0, 5), dtype=np.float32)
        out_labels = np.zeros((0,), dtype=np.int64)
        out_sources = np.zeros((0,), dtype=np.int8)

    return dict(
        boxes=out_boxes,
        labels=out_labels,
        sources=out_sources,
        matched_pairs=len(used1),
        teacher1_only=len(boxes1) - len(used1),
        teacher2_only=len(boxes2) - len(used2),
        input_teacher1=len(boxes1),
        input_teacher2=len(boxes2),
    )
