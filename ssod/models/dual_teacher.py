import torch
import numpy as np
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import get_root_logger, log_image_with_boxes, log_every_n
from ssod.utils.checkpoint import load_branch_weights

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid
from .progressive_gamma import ProgressiveGamma

from ssod.utils.ensemble_boxes import nms


def fuse_teacher_proposals(proposals1, labels1, proposals2, labels2):
    """Author-code NMS fusion, without consensus score boosting.

    The released DualTeacher code uses IoU=0 for pseudo-label fusion (not
    the detector's own NMS). Keep that baseline setting and its empty-side
    passthrough behavior. Process each image independently.
    """
    if not (len(proposals1) == len(labels1) == len(proposals2) == len(labels2)):
        raise ValueError("Teacher proposal batch sizes must match")
    fused_proposals, fused_labels = [], []
    for first, first_labels, second, second_labels in zip(
        proposals1, labels1, proposals2, labels2
    ):
        if len(first) == 0:
            proposal, label = second, second_labels
        elif len(second) == 0:
            proposal, label = first, first_labels
        else:
            boxes, scores, labels = nms(
                [first[:, :4].detach().cpu().numpy(), second[:, :4].detach().cpu().numpy()],
                [first[:, 4].detach().cpu().numpy(), second[:, 4].detach().cpu().numpy()],
                [first_labels.detach().cpu().numpy(), second_labels.detach().cpu().numpy()],
                iou_thr=0,
            )
            proposal = first.new_tensor(np.column_stack((boxes, scores)))
            label = first_labels.new_tensor(labels)
        fused_proposals.append(proposal)
        fused_labels.append(label)
    return fused_proposals, fused_labels


@DETECTORS.register_module()
class DualTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(DualTeacher, self).__init__(
            dict(
                teacher1=build_detector(model),
                teacher2=build_detector(model),
                student1=build_detector(model),
                student2=build_detector(model),
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        self._pretrained_initialized = False
        self.m2_enabled = False
        self.m2_force_weight_one = False
        self.pg = None
        self.sup2_weight = 0.2
        if train_cfg is not None:
            self.freeze("teacher1")
            self.freeze("teacher2")
            self.unsup_weight = self.train_cfg.unsup_weight
            self.load1_from = self.train_cfg.get("load1_from")
            self.load2_from = self.train_cfg.get("load2_from")
            for retired in ("m3_enabled", "mvdt_enabled"):
                if self.train_cfg.get(retired, False):
                    raise ValueError("Retired experiment flag: " + retired)
            self.m2_enabled = bool(self.train_cfg.get("m2_enabled", False))
            self.m2_force_weight_one = bool(self.train_cfg.get("m2_force_weight_one", False))
            self.sup2_weight = float(self.train_cfg.get("sup2_weight", 0.2))
            if not np.isfinite(self.sup2_weight) or self.sup2_weight <= 0:
                raise ValueError("sup2_weight must be finite and positive")
            if self.train_cfg.get("pg"):
                self.pg = ProgressiveGamma(
                    base_gamma=self.sup2_weight, unsup_weight=self.unsup_weight,
                    **dict(self.train_cfg.pg))


    def init_from_pretrained(self):
        """Explicitly initialize a fresh run, after generic init_weights.

        train_detector calls this only when not loading/resuming a full-run
        checkpoint. Construction and inference never read Phase 1/2 files.
        """
        if self._pretrained_initialized:
            return
        if self.train_cfg is None:
            raise RuntimeError("Phase initialization requires a training config")
        logger = get_root_logger()
        load_branch_weights(
            self.load1_from,
            [("teacher1", self.teacher1), ("student1", self.student1)],
            logger,
        )
        load_branch_weights(
            self.load2_from,
            [("teacher2", self.teacher2), ("student2", self.student2)],
            logger,
        )
        first, second = self.teacher1.state_dict(), self.teacher2.state_dict()
        # Random auxiliary heads must not hide identical Phase 1/2 detector
        # initialization. The loader already validated this explicit whitelist.
        roi_head = getattr(self.teacher1, "roi_head", None)
        foreground_keys = (set(roi_head.foreground_initialization_keys())
                           if hasattr(roi_head, "foreground_initialization_keys") else set())
        if all(torch.equal(first[key], second[key])
               for key in first if key not in foreground_keys):
            raise RuntimeError(
                "Phase 1/2 initialized identical branches; check load1_from/load2_from"
            )
        self._pretrained_initialized = True
        logger.info(
            "[DualTeacher init] PASS: T1=S1, T2=S2, T1!=T2; fusion=NMS, fusion_iou=0"
        )

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        sup2_weight = getattr(self, "sup2_weight", 0.2)
        unsup2_multiplier = sup2_weight
        pg = getattr(self, "pg", None)
        if pg is not None:
            sup2_weight, unsup2_multiplier = pg.weights()
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup1" in data_groups:
            gt_bboxes = data_groups["sup1"]["gt_bboxes"]
            log_every_n(
                {"sup1_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup1_inputs = dict(data_groups["sup1"])
            if getattr(getattr(self.student1, "roi_head", None), "foreground_enabled", False):
                sup1_inputs["foreground_supervised"] = True
            sup1_loss = self.student1.forward_train(**sup1_inputs)
            sup1_loss = {"sup1_" + k: v for k, v in sup1_loss.items()}
            loss.update(**sup1_loss)
        if "sup2" in data_groups:
            gt_bboxes = data_groups["sup2"]["gt_bboxes"]
            log_every_n(
                {"sup2_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup2_inputs = dict(data_groups["sup2"])
            if getattr(getattr(self.student2, "roi_head", None), "foreground_enabled", False):
                sup2_inputs["foreground_supervised"] = True
            sup2_loss = weighted_loss(self.student2.forward_train(**sup2_inputs), sup2_weight)
            sup2_loss = {"sup2_" + k: v for k, v in sup2_loss.items()}
            loss.update(**sup2_loss)
        if "unsup_teacher" in data_groups and "unsup_student" in data_groups:
            teacher_data = data_groups["unsup_teacher"]
            student_data = data_groups["unsup_student"]
            tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
            snames = [meta["filename"] for meta in student_data["img_metas"]]
            tidx = [tnames.index(name) for name in snames]
            with torch.no_grad():
                teacher1_info, teacher2_info = self.extract_teacher_info(
                    teacher_data["img"][
                        torch.Tensor(tidx).to(teacher_data["img"].device).long()
                    ],
                    [teacher_data["img_metas"][idx] for idx in tidx],
                    [teacher_data["proposals"][idx] for idx in tidx]
                    if ("proposals" in teacher_data)
                       and (teacher_data["proposals"] is not None)
                    else None,
                )

            unsup1_loss = weighted_loss(
                self.foward_unsup1_train(
                    teacher1_info, teacher2_info, data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup1_loss = {"unsup1_" + k: v for k, v in unsup1_loss.items()}
            loss.update(**unsup1_loss)

            unsup2_loss = weighted_loss(
                self.foward_unsup2_train(
                    teacher2_info, teacher1_info, data_groups["unsup_student"]
                ),
                weight=self.unsup_weight * unsup2_multiplier,
            )
            unsup2_loss = {"unsup2_" + k: v for k, v in unsup2_loss.items()}
            loss.update(**unsup2_loss)

            # unsup12_loss = weighted_loss(
            #     self.foward_unsup12_train(
            #         data_groups["unsup1_teacher"], data_groups["unsup1_student"]
            #     ),
            #     weight=self.unsup_weight / 2,
            # )
            # unsup12_loss = {"unsup12_" + k: v for k, v in unsup12_loss.items()}
            # loss.update(**unsup12_loss)
            #
            # unsup21_loss = weighted_loss(
            #     self.foward_unsup21_train(
            #         data_groups["unsup2_teacher"], data_groups["unsup2_student"]
            #     ),
            #     weight=self.unsup_weight / 2,
            # )
            # unsup21_loss = {"unsup21_" + k: v for k, v in unsup21_loss.items()}
            # loss.update(**unsup21_loss)

        return loss

    def get_det_bboxes(self, model, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        if model == 'teacher1':
            feat = self.teacher1.extract_feat(img)
        if model == 'teacher2':
            feat = self.teacher2.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            if model == 'teacher1':
                proposal_cfg = self.teacher1.train_cfg.get(
                    "rpn_proposal", self.teacher1.test_cfg.rpn
                )
                rpn_out = list(self.teacher1.rpn_head(feat))
                proposal_list = self.teacher1.rpn_head.get_bboxes(
                    *rpn_out, img_metas, cfg=proposal_cfg
                )
            if model == 'teacher2':
                proposal_cfg = self.teacher2.train_cfg.get(
                    "rpn_proposal", self.teacher2.test_cfg.rpn
                )
                rpn_out = list(self.teacher2.rpn_head(feat))
                proposal_list = self.teacher2.rpn_head.get_bboxes(
                    *rpn_out, img_metas, cfg=proposal_cfg
                )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        if model == 'teacher1':
            proposal_list, proposal_label_list = self.teacher1.roi_head.simple_test_bboxes(
                feat, img_metas, proposal_list, self.teacher1.test_cfg.rcnn, rescale=False
            )
        if model == 'teacher2':
            proposal_list, proposal_label_list = self.teacher2.roi_head.simple_test_bboxes(
                feat, img_metas, proposal_list, self.teacher2.test_cfg.rcnn, rescale=False
            )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        return feat, proposal_list, proposal_label_list, det_bboxes, teacher_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        feat1, proposal1_list, proposal1_label_list, det1_bboxes, teacher1_info = \
            self.get_det_bboxes('teacher1', img, img_metas, proposals=None, **kwargs)
        feat2, proposal2_list, proposal2_label_list, det2_bboxes, teacher2_info = \
            self.get_det_bboxes('teacher2', img, img_metas, proposals=None, **kwargs)
        proposal_list, proposal_label_list = fuse_teacher_proposals(
            proposal1_list, proposal1_label_list, proposal2_list, proposal2_label_list
        )

        reg1_unc = self.compute_uncertainty_with_aug_1(
            feat1, img_metas, proposal_list, proposal_label_list
        )
        det1_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(proposal_list, reg1_unc)
        ]
        det1_labels = proposal_label_list
        teacher1_info["det_bboxes"] = det1_bboxes
        teacher1_info["det_labels"] = det1_labels
        teacher1_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat1[0][0].device)
            for meta in img_metas
        ]
        teacher1_info["img_metas"] = img_metas

        reg2_unc = self.compute_uncertainty_with_aug_2(
            feat2, img_metas, proposal_list, proposal_label_list
        )
        det2_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(proposal_list, reg2_unc)
        ]
        det2_labels = proposal_label_list
        teacher2_info["det_bboxes"] = det2_bboxes
        teacher2_info["det_labels"] = det2_labels
        teacher2_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat2[0][0].device)
            for meta in img_metas
        ]
        teacher2_info["img_metas"] = img_metas

        reg_unc = [(first + second) * 0.5 for first, second in zip(reg1_unc, reg2_unc)]
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(proposal_list, reg_unc)
        ]
        teacher1_info["det_bboxes"] = det_bboxes
        teacher2_info["det_bboxes"] = det_bboxes

        return teacher1_info, teacher2_info

    def foward_unsup1_train(self, teacher_info, teacher0_info, student_data):
        # # sort the teacher and student input to avoid some bugs
        # tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        # snames = [meta["filename"] for meta in student_data["img_metas"]]
        # tidx = [tnames.index(name) for name in snames]
        # with torch.no_grad():
        #     teacher_info = self.extract_teacher1_info(
        #         teacher_data["img"][
        #             torch.Tensor(tidx).to(teacher_data["img"].device).long()
        #         ],
        #         [teacher_data["img_metas"][idx] for idx in tidx],
        #         [teacher_data["proposals"][idx] for idx in tidx]
        #         if ("proposals" in teacher_data)
        #         and (teacher_data["proposals"] is not None)
        #         else None,
        #     )
        student_info = self.extract_student1_info(**student_data)

        return self.compute_pseudo_label1_loss(student_info, teacher_info, teacher0_info)

    def compute_pseudo_label1_loss(self, student_info, teacher_info, teacher0_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        rpn_loss, proposal_list = self.rpn1_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup1_rcnn_cls_loss(
                teacher0_info,
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        loss.update(
            self.unsup1_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )
        return loss

    def rpn1_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student1.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student1.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student1.train_cfg.get(
                "rpn_proposal", self.student1.test_cfg.rpn
            )
            proposal_list = self.student1.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def _log_m2_stats(self, branch, num_pos, num_neg, Z_0, w_pos=None):
        """Log positive-ROI diagnostics (INFO, every call).

        Always logs pos/neg counts and Z_0 (for both M0 and M2 health checks);
        when w_pos is given (M2 enabled) also logs the retention ratio and the
        weight min/median/max.
        """
        if w_pos is not None and num_pos > 0:
            new_pos_w = float(w_pos.sum())
            get_root_logger().info(
                f"[ROI-{branch}] pos={num_pos} neg={num_neg} retain={new_pos_w / num_pos:.4f} "
                f"w[min={float(w_pos.min()):.4f}, med={float(w_pos.median()):.4f}, "
                f"max={float(w_pos.max()):.4f}] Z0={float(Z_0):.2f}")
        else:
            get_root_logger().info(
                f"[ROI-{branch}] pos={num_pos} neg={num_neg} Z0={float(Z_0):.2f}")


    def unsup1_rcnn_cls_loss(
        self,
        teacher0_info,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result1(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student1.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student1.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student1.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores1 = self.teacher1.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score1 = torch.cat([_score[:, -1] for _score in _scores1])
            _, _scores2 = self.teacher2.roi_head.simple_test_bboxes(
                teacher0_info['backbone_feature'],
                teacher0_info['img_metas'],
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score2 = torch.cat([_score[:, -1] for _score in _scores2])
            bg_score = (bg_score1 + bg_score2) * 0.5
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student1.roi_head.bbox_head.num_classes
            # 1. original background weight assignment (unchanged)
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
            # 2. save the original normalization denominator Z_0
            Z_0 = bbox_targets[1].sum()
            num_pos = int((~neg_inds).sum())
            num_neg = int(neg_inds.sum())
            # 3. M2: modify ONLY positive-ROI classification weights
            if self.m2_enabled:
                pos_inds = ~neg_inds
                p_T1 = torch.cat([_score[:, 0] for _score in _scores1])
                p_T2 = torch.cat([_score[:, 0] for _score in _scores2])
                if self.m2_force_weight_one:
                    w = torch.ones_like(p_T1)
                else:
                    w = torch.min(p_T1, p_T2).detach()
                w = w.float()  # explicit dtype; no learned quality head
                bbox_targets[1][pos_inds] = bbox_targets[1][pos_inds] * w[pos_inds]
                self._log_m2_stats("unsup1", num_pos, num_neg, Z_0, w[pos_inds])
            else:
                self._log_m2_stats("unsup1", num_pos, num_neg, Z_0)
        loss = self.student1.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        if "loss_cls" in loss:
            # 4. normalize by the ORIGINAL Z_0 (never the modified weight sum)
            loss["loss_cls"] = loss["loss_cls"].sum() / max(Z_0, 1.0)
        if "loss_bbox" in loss:
            loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
                bbox_targets[1].size()[0], 1.0
            )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def unsup1_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student1.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}

    def get_sampling_result1(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student1.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student1.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student1_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student1.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student1.with_rpn:
            rpn_out = self.student1.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher1_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher1.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher1.train_cfg.get(
                "rpn_proposal", self.teacher1.test_cfg.rpn
            )
            rpn_out = list(self.teacher1.rpn_head(feat))
            proposal_list = self.teacher1.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher1.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher1.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug_1(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug_1(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher1.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Full-run checkpoint loading must never silently substitute Phase 1/2
        # files or require their presence when evaluating a trained detector.
        if not all(
            any(key.startswith(prefix + name + ".") for key in state_dict)
            for name in self.submodules
        ):
            raise RuntimeError(
                "DualTeacher load/resume requires a full four-branch checkpoint. "
                "For a fresh run, set load_from=None and use load1_from/load2_from."
            )
        pg_keys = [key for key in state_dict if key.startswith(prefix + "pg.")]
        if self.pg is None and pg_keys:
            if self.train_cfg is not None:
                raise RuntimeError("A PG checkpoint requires the matching PG training config")
            for key in pg_keys:
                state_dict.pop(key)
        if any(key.startswith(prefix + "mvdt.") or ".quality_head." in key for key in state_dict):
            raise RuntimeError("Retired experimental checkpoint; use the original archived code")
        for name in self.submodules:
            head_prefix = prefix + name + ".roi_head.foreground_head."
            loaded = {key for key in state_dict if key.startswith(head_prefix)}
            expected = {prefix + name + "." + key for key in getattr(self, name).state_dict()
                        if key.startswith("roi_head.foreground_head.")}
            if loaded != expected:
                raise RuntimeError("Foreground checkpoint/config mismatch for " + name)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def foward_unsup2_train(self, teacher_info, teacher0_info, student_data):
        # # sort the teacher and student input to avoid some bugs
        # tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        # snames = [meta["filename"] for meta in student_data["img_metas"]]
        # tidx = [tnames.index(name) for name in snames]
        # with torch.no_grad():
        #     teacher_info = self.extract_teacher2_info(
        #         teacher_data["img"][
        #             torch.Tensor(tidx).to(teacher_data["img"].device).long()
        #         ],
        #         [teacher_data["img_metas"][idx] for idx in tidx],
        #         [teacher_data["proposals"][idx] for idx in tidx]
        #         if ("proposals" in teacher_data)
        #         and (teacher_data["proposals"] is not None)
        #         else None,
        #     )
        student_info = self.extract_student2_info(**student_data)

        return self.compute_pseudo_label2_loss(student_info, teacher_info, teacher0_info)

    def compute_pseudo_label2_loss(self, student_info, teacher_info, teacher0_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        rpn_loss, proposal_list = self.rpn2_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup2_rcnn_cls_loss(
                teacher0_info,
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        loss.update(
            self.unsup2_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )
        return loss

    def rpn2_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student2.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student2.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student2.train_cfg.get(
                "rpn_proposal", self.student2.test_cfg.rpn
            )
            proposal_list = self.student2.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup2_rcnn_cls_loss(
        self,
        teacher0_info,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result2(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student2.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student2.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student2.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores2 = self.teacher2.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score2 = torch.cat([_score[:, -1] for _score in _scores2])
            _, _scores1 = self.teacher1.roi_head.simple_test_bboxes(
                teacher0_info['backbone_feature'],
                teacher0_info['img_metas'],
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score1 = torch.cat([_score[:, -1] for _score in _scores1])
            bg_score = (bg_score1 + bg_score2) * 0.5
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student2.roi_head.bbox_head.num_classes
            # 1. original background weight assignment (unchanged)
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
            # 2. save the original normalization denominator Z_0
            Z_0 = bbox_targets[1].sum()
            num_pos = int((~neg_inds).sum())
            num_neg = int(neg_inds.sum())
            # 3. M2: modify ONLY positive-ROI classification weights
            if self.m2_enabled:
                pos_inds = ~neg_inds
                p_T1 = torch.cat([_score[:, 0] for _score in _scores1])
                p_T2 = torch.cat([_score[:, 0] for _score in _scores2])
                if self.m2_force_weight_one:
                    w = torch.ones_like(p_T1)
                else:
                    w = torch.min(p_T1, p_T2).detach()
                w = w.float()  # explicit dtype; no learned quality head
                bbox_targets[1][pos_inds] = bbox_targets[1][pos_inds] * w[pos_inds]
                self._log_m2_stats("unsup2", num_pos, num_neg, Z_0, w[pos_inds])
            else:
                self._log_m2_stats("unsup2", num_pos, num_neg, Z_0)
        loss = self.student2.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        if "loss_cls" in loss:
            # 4. normalize by the ORIGINAL Z_0 (never the modified weight sum)
            loss["loss_cls"] = loss["loss_cls"].sum() / max(Z_0, 1.0)
        if "loss_bbox" in loss:
            loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
                bbox_targets[1].size()[0], 1.0
            )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def unsup2_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student2.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}

    def get_sampling_result2(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student2.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student2.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    def extract_student2_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student2.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student2.with_rpn:
            rpn_out = self.student2.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher2_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher2.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher2.train_cfg.get(
                "rpn_proposal", self.teacher2.test_cfg.rpn
            )
            rpn_out = list(self.teacher2.rpn_head(feat))
            proposal_list = self.teacher2.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher2.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher2.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug_2(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug_2(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher2.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    def foward_unsup12_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher1_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student2_info(**student_data)

        return self.compute_pseudo_label2_loss(student_info, teacher_info)

    def foward_unsup21_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher2_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student1_info(**student_data)

        return self.compute_pseudo_label1_loss(student_info, teacher_info)
