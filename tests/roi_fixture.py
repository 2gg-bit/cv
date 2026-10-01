"""Execute production heads with a small external MMDetection interface fixture.

This does not replace the training-machine CUDA acceptance. The parent provides
the MMDet 2.16 positive-first target layout and an independent DeltaXYWH decoder.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def multi_apply(func, *args, **kwargs):
    return tuple(map(list, zip(*(func(*items, **kwargs) for items in zip(*args)))))


def bbox2roi(boxes):
    return torch.cat([torch.cat((b.new_full((b.size(0), 1), i), b), 1)
                      for i, b in enumerate(boxes)])


class DeltaCoder:
    def decode(self, boxes, deltas):
        size = boxes[:, 2:] - boxes[:, :2]
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        shifted = center + deltas[:, :2] * .1 * size
        resized = size * torch.exp(deltas[:, 2:] * .2)
        return torch.cat((shifted - resized / 2, shifted + resized / 2), 1)


class ParentBBox(nn.Module):
    def __init__(self, num_classes=1, reg_class_agnostic=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.fc_reg = nn.Linear(3, 4 if reg_class_agnostic else 4 * num_classes)
        self.bbox_coder = DeltaCoder()

    def _get_target_single(self, pos, neg, gt, labels, cfg):
        n, p = len(pos) + len(neg), len(pos)
        target = pos.new_full((n,), self.num_classes, dtype=torch.long)
        weights = pos.new_ones(n)
        target[:p] = labels
        weights[:p] = 1 if cfg.pos_weight <= 0 else cfg.pos_weight
        reg, reg_weights = pos.new_zeros(n, 4), pos.new_zeros(n, 4)
        reg[:p], reg_weights[:p] = gt - pos, 1
        return target, weights, reg, reg_weights

    def get_targets(self, samples, gt_bboxes, gt_labels, cfg, concat=True):
        values = multi_apply(self._get_target_single,
                             [s.pos_bboxes for s in samples], [s.neg_bboxes for s in samples],
                             [s.pos_gt_bboxes for s in samples], [s.pos_gt_labels for s in samples], cfg=cfg)
        return tuple(torch.cat(v) for v in values) if concat else values


class ParentRoI(nn.Module):
    def __init__(self, bbox_head=None, **kwargs):
        super().__init__()
        self.bbox_head = bbox_head if bbox_head is not None else ParentBBox()
        self.train_cfg = SimpleNamespace(pos_weight=-1)

    def _bbox_forward_train(self, x, samples, gt, labels, metas):
        return dict(bbox_pred=x, loss_bbox=dict(loss_bbox=x.square().sum(),
                                               loss_cls=x.sum() * 0 + 2))


def load_heads(relative, source=None):
    text = source if source is not None else (ROOT / relative).read_text(encoding="utf-8")
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    ns = dict(torch=torch, np=np, math=math, bbox2roi=bbox2roi,
              multi_apply=multi_apply, StandardRoIHead=ParentRoI,
              Shared2FCBBoxHead=ParentBBox,
              HEADS=SimpleNamespace(register_module=lambda: lambda cls: cls))
    exec(compile(tree, relative, "exec"), ns)
    return SimpleNamespace(**ns)


def sample(pos, neg, gt=None, labels=None):
    pos = torch.tensor(pos, dtype=torch.float32).reshape(-1, 4)
    neg = torch.tensor(neg, dtype=torch.float32).reshape(-1, 4)
    return SimpleNamespace(pos_bboxes=pos, neg_bboxes=neg,
                           pos_gt_bboxes=pos.clone() if gt is None else torch.tensor(gt, dtype=torch.float32).reshape(-1, 4),
                           pos_gt_labels=torch.zeros(len(pos), dtype=torch.long) if labels is None else torch.tensor(labels),
                           bboxes=torch.cat((pos, neg)))
