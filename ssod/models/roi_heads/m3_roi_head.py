"""Training-only target routing, after StandardRoIHead assignment/sampling."""

from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

from ..m3_routing import replace_positive_regression_targets


@HEADS.register_module()
class M3RoIHead(StandardRoIHead):
    """No new parameters and no inference changes.

    StandardRoIHead.forward_train still performs assignment and sampling on the
    ORIGINAL GT boxes. Only already-sampled positive regression targets change.
    Supervised, classification and disabled calls delegate to the original head.
    """

    supports_m3_targets = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.with_bbox or self.with_mask or self.bbox_head.num_classes != 1:
            raise ValueError("M3 v1 supports the existing single-class bbox-only detector")
        self._m3_targets = None

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None, reg_target_bboxes=None):
        if reg_target_bboxes is not None:
            if len(reg_target_bboxes) != len(gt_bboxes):
                raise ValueError("M3 target batch differs from original GT batch")
            for targets, originals in zip(reg_target_bboxes, gt_bboxes):
                if targets.shape != originals.shape:
                    raise ValueError("M3 target rows must match ORIGINAL GT order")
        previous = self._m3_targets
        self._m3_targets = reg_target_bboxes
        try:
            return super().forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks)
        finally:
            self._m3_targets = previous

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        if self._m3_targets is None:
            return super()._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, img_metas)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        results = self._bbox_forward(x, rois)
        targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        targets = replace_positive_regression_targets(
            self.bbox_head, sampling_results, targets, self._m3_targets)
        loss_bbox = self.bbox_head.loss(
            results["cls_score"], results["bbox_pred"], rois, *targets)
        results.update(loss_bbox=loss_bbox)
        return results
