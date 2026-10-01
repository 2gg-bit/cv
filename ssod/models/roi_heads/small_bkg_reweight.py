"""sup2-only multiplicative reweighting of background ROI classification targets.

The classification label weight of a sampled background ROI is multiplied by
``1 + lambda`` when its box area, mapped back to the *original* image scale, is
below ``max_area``::

    w_i *= 1 + lambda   for   label_i == num_classes   and   area_orig_i < max_area

Everything else is left alone: no new parameters, no new loss term, positives,
ignored samples, the sampler, the regression weights and the inference path are
untouched.  With ``lambda == 0`` the produced targets and the resulting loss are
bitwise identical to the unmodified head.

Only the supervised ``sup2`` stream is affected.  Two gates do that:

* ``SmallBkgReweightRoIHead`` forwards the per-image ``tag`` / ``scale_factor``
  from ``img_metas`` (``StandardRoIHead._bbox_forward_train`` drops them);
* ``SmallBkgReweightBBoxHead`` applies the multiplier only to images whose tag
  equals ``reweight.tag``.  When ``get_targets`` is called without the extra
  context -- which is how the pseudo-label branches (``unsup1_rcnn_cls_loss`` /
  ``unsup2_rcnn_cls_loss``) call it -- the parent implementation runs unchanged.

The multiplier reuses the existing ``label_weights`` channel, so this needs no
change to ``BBoxHead.loss`` (including its ``avg_factor`` denominator).
"""

import math

import numpy as np
import torch

from mmdet.core import bbox2roi, multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
# this vendored mmdet keeps one MODELS registry, so HEADS covers roi heads too

DEFAULT_TAG = "sup2"
DEFAULT_LAMBDA = 1.0
DEFAULT_MAX_AREA = 32.0 ** 2


def enable_reweight_diagnostics(model, max_records=16):
    """Opt in from an acceptance tool; never called by the training entry point."""
    for module in model.modules():
        if isinstance(module, SmallBkgReweightBBoxHead):
            module.enable_reweight_diagnostics(max_records)


@HEADS.register_module()
class SmallBkgReweightBBoxHead(Shared2FCBBoxHead):
    """``Shared2FCBBoxHead`` with an optional background-weight multiplier.

    Args:
        reweight (dict | None): ``enable`` (bool), ``lambda_`` (float),
            ``max_area`` (float, original-scale box area in px^2) and ``tag``
            (str).  Disabled by default, in which case the head behaves exactly
            like ``Shared2FCBBoxHead``.
    """

    # tells a RoI head that the extra `reweight_ctx` argument is accepted
    supports_reweight_ctx = True

    def __init__(self, reweight=None, **kwargs):
        super().__init__(**kwargs)
        reweight = dict(reweight or {})
        self.reweight_enabled = bool(reweight.get("enable", False))
        self.reweight_lambda = float(reweight.get("lambda_", DEFAULT_LAMBDA))
        self.reweight_max_area = float(reweight.get("max_area", DEFAULT_MAX_AREA))
        self.reweight_tag = reweight.get("tag", DEFAULT_TAG)
        if self.reweight_enabled:
            if not math.isfinite(self.reweight_lambda) or self.reweight_lambda < 0.0:
                raise ValueError("reweight.lambda_ must be >= 0")
            if not math.isfinite(self.reweight_max_area) or not self.reweight_max_area > 0.0:
                raise ValueError("reweight.max_area must be > 0")
            if not isinstance(self.reweight_tag, str) or not self.reweight_tag:
                raise ValueError("reweight.tag must be a non-empty string")
        # diagnostics only; a plain attribute (never a parameter or buffer), so
        # state_dict keys are identical to the unmodified head
        self.reweight_log = []
        # Training must not retain every sampled ROI for all 32000 iterations.
        # Acceptance tools explicitly enable a bounded diagnostic buffer.
        self.reweight_log_limit = 0

    # ------------------------------------------------------------------ public

    def reset_reweight_log(self):
        self.reweight_log = []

    def enable_reweight_diagnostics(self, max_records=16):
        if type(max_records) is not int or max_records < 0:
            raise ValueError("max_records must be a nonnegative integer")
        self.reweight_log_limit = max_records
        self.reset_reweight_log()

    def _record_reweight(self, record):
        self.reweight_log.append(record)
        del self.reweight_log[:-self.reweight_log_limit]

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    reweight_ctx=None):
        """Same as ``BBoxHead.get_targets`` plus an optional reweight context.

        ``reweight_ctx`` carries the per-image ``tags`` / ``scale_factors``.
        Without it (or with the feature disabled) this defers to the parent.
        """
        if not self.reweight_enabled or reweight_ctx is None:
            return super().get_targets(sampling_results, gt_bboxes, gt_labels,
                                       rcnn_train_cfg, concat)

        contexts = self._contexts(reweight_ctx, len(sampling_results))
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            [res.pos_bboxes for res in sampling_results],
            [res.neg_bboxes for res in sampling_results],
            [res.pos_gt_bboxes for res in sampling_results],
            [res.pos_gt_labels for res in sampling_results],
            contexts,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    # ----------------------------------------------------------------- private

    def _contexts(self, reweight_ctx, num_images):
        scale_factors = reweight_ctx.get("scale_factors") or [None] * num_images
        tags = reweight_ctx.get("tags") or [None] * num_images
        contexts = []
        for index in range(num_images):
            tag = tags[index] if index < len(tags) else None
            contexts.append(
                dict(
                    image_index=index,
                    tag=tag,
                    scale_factor=(scale_factors[index]
                                  if index < len(scale_factors) else None),
                    eligible=(self.reweight_tag is None or tag == self.reweight_tag),
                ))
        return contexts

    def _get_target_single(self,
                           pos_bboxes,
                           neg_bboxes,
                           pos_gt_bboxes,
                           pos_gt_labels,
                           reweight=None,
                           cfg=None):
        labels, label_weights, bbox_targets, bbox_weights = \
            super()._get_target_single(pos_bboxes, neg_bboxes, pos_gt_bboxes,
                                       pos_gt_labels, cfg)
        if self.reweight_enabled and reweight is not None:
            self._apply_reweight(reweight, labels, label_weights, neg_bboxes)
        return labels, label_weights, bbox_targets, bbox_weights

    def _apply_reweight(self, reweight, labels, label_weights, neg_bboxes):
        num_neg = int(neg_bboxes.size(0))
        recording = self.reweight_log_limit > 0
        if recording:
            record = dict(
                image_index=int(reweight["image_index"]),
                tag=reweight["tag"],
                eligible=bool(reweight["eligible"]),
                num_pos=int(labels.numel()) - num_neg,
                num_neg=num_neg,
                scale_factor=self._as_list(reweight["scale_factor"]),
            )
        if not reweight["eligible"]:
            if recording:
                record.update(n_reweighted=0, skipped="tag_mismatch")
                self._record_reweight(record)
            return
        if num_neg == 0:
            if recording:
                record.update(n_reweighted=0, skipped="no_negatives")
                self._record_reweight(record)
            return

        # `_get_target_single` writes the negatives into the *trailing* num_neg
        # slots. Diagnostics record the invariant for the independent checker.
        if recording:
            record["neg_labels_all_bg"] = bool(
                torch.all(labels[-num_neg:] == self.num_classes).item())

        areas = self._original_areas(neg_bboxes, reweight["scale_factor"])
        small = areas < self.reweight_max_area
        factor = torch.ones_like(label_weights[-num_neg:])
        factor[small] = 1.0 + self.reweight_lambda
        weight_before = label_weights[-num_neg:].clone()
        label_weights[-num_neg:] = weight_before * factor

        if recording:
            record.update(
                n_reweighted=int(small.sum().item()),
                neg_boxes=self._as_list(neg_bboxes),
                areas=self._as_list(areas),
                small_mask=self._as_list(small),
                factors=self._as_list(factor),
                weight_before=self._as_list(weight_before),
                weight_after=self._as_list(label_weights[-num_neg:]),
            )
            self._record_reweight(record)

    def _original_areas(self, neg_bboxes, scale_factor):
        """Box areas divided by the per-image scale, i.e. in original px^2."""
        boxes = neg_bboxes.detach().float()
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        if scale_factor is not None:
            scale_x, scale_y = float(scale_factor[0]), float(scale_factor[1])
            if not (scale_x > 0.0 and scale_y > 0.0):
                raise ValueError("scale_factor must be positive")
            widths = widths / scale_x
            heights = heights / scale_y
        return widths * heights

    @staticmethod
    def _as_list(value):
        """Detached plain-python view for the diagnostics log."""
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        return [float(item) for item in value]


@HEADS.register_module()
class SmallBkgReweightRoIHead(StandardRoIHead):
    """``StandardRoIHead`` that forwards ``tag``/``scale_factor`` to the head.

    The only difference from the parent is that ``_bbox_forward_train`` builds a
    reweight context from ``img_metas`` and hands it to ``bbox_head.get_targets``.
    A bbox head without ``supports_reweight_ctx`` keeps the original call.
    """

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        if getattr(self.bbox_head, "supports_reweight_ctx", False):
            reweight_ctx = dict(
                scale_factors=[meta.get("scale_factor") for meta in img_metas],
                tags=[meta.get("tag") for meta in img_metas],
            )
            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg,
                                                      reweight_ctx=reweight_ctx)
        else:
            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results["cls_score"],
                                        bbox_results["bbox_pred"], rois,
                                        *bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
