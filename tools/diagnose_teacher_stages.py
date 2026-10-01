"""Diagnose where targets are lost in DualTeacher — baseline-consistent version.

Mirrors the frozen-baseline inference path:
    - same model build (build_detector + wrap_fp16_model + load_checkpoint strict)
    - same FP16/GPU execution via torch.cuda.amp.autocast (matches auto_fp16)
    - records real scale_factor / img_shape / dtype / device / jitter seed

Records, per image:
    1. per-teacher RPN proposals  (rpn_head.get_bboxes)
    2. per-teacher ROI post-regression pre-NMS boxes (get_bboxes cfg=None)
    3. per-teacher final detections (get_bboxes cfg=rcnn)
    4. FUSED common proposals (fuse_teacher_proposals, NMS iou=0)
    5. per-teacher jitter mean/std/rel_unc ON THE FUSED common proposals
       (this is what training actually does), with a fixed recorded seed

The formal teacher2 predictions are written separately (rescaled to original
image coords) so AP can be re-verified.
"""
import argparse
import json
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.core import bbox2roi
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

from ssod.datasets import build_dataloader
from ssod.utils import patch_config
from ssod.models.dual_teacher import fuse_teacher_proposals


def _to_list(t):
    if t is None:
        return []
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    return np.asarray(t).tolist()


def _aug_box(boxes, times, frac):
    def _aug(box):
        box_scale = box[:, 2:4] - box[:, :2]
        box_scale = box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
        aug_scale = box_scale * frac
        offset = torch.randn(times, box.shape[0], 4, device=box.device) * aug_scale[None, ...]
        new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
        return torch.cat([new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1)
    return [_aug(box) for box in boxes]


def _forward_roi(teacher, feat, proposal_list, img_metas, cfg, rescale=False):
    img_shapes = tuple(meta['img_shape'] for meta in img_metas)
    scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
    rois = bbox2roi(proposal_list)
    if rois.shape[0] == 0:
        b = len(proposal_list)
        return [rois.new_zeros(0, 5)] * b, [rois.new_zeros(0, dtype=torch.long)] * b
    bbox_results = teacher.roi_head._bbox_forward(feat, rois)
    cls_score = bbox_results['cls_score']
    bbox_pred = bbox_results['bbox_pred']
    num = tuple(len(p) for p in proposal_list)
    rois = rois.split(num, 0)
    cls_score = cls_score.split(num, 0)
    bbox_pred = bbox_pred.split(num, 0) if bbox_pred is not None else (None,) * len(proposal_list)
    det_bboxes, det_labels = [], []
    for i in range(len(proposal_list)):
        db, dl = teacher.roi_head.bbox_head.get_bboxes(
            rois[i], cls_score[i], bbox_pred[i], img_shapes[i], scale_factors[i],
            rescale=rescale, cfg=cfg)
        det_bboxes.append(db)
        det_labels.append(dl)
    return det_bboxes, det_labels


def _jitter_stats(teacher, feat, img_metas, fused_proposals, fused_labels,
                  jitter_times, jitter_scale, seed):
    torch.manual_seed(seed)
    auged = _aug_box(fused_proposals, jitter_times, jitter_scale)
    auged = [a.reshape(-1, a.shape[-1]) for a in auged]
    bboxes, _ = teacher.roi_head.simple_test_bboxes(feat, img_metas, auged, None, rescale=False)
    reg_channel = max([b.shape[-1] for b in bboxes]) // 4
    bboxes = [
        b.reshape(jitter_times, -1, b.shape[-1]) if b.numel() > 0
        else b.new_zeros(jitter_times, 0, 4 * reg_channel).float()
        for b in bboxes
    ]
    std = [b.std(dim=0) for b in bboxes]
    mean = [b.mean(dim=0) for b in bboxes]
    if reg_channel != 1:
        mean = [m.reshape(m.shape[0], reg_channel, 4)[torch.arange(m.shape[0]), l]
                for m, l in zip(mean, fused_labels)]
        std = [s.reshape(s.shape[0], reg_channel, 4)[torch.arange(s.shape[0]), l]
               for s, l in zip(std, fused_labels)]
    box_shape = [(m[:, 2:4] - m[:, :2]).clamp(min=1.0) for m in mean]
    rel_unc = [s / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4) if wh.numel() > 0 else s
               for s, wh in zip(std, box_shape)]
    return mean, std, rel_unc


def diagnose(dual, img, img_metas, jitter_times, jitter_scale, jitter_seed):
    t1 = dual.teacher1
    t2 = dual.teacher2
    rpn_cfg1, rcnn_cfg1 = t1.test_cfg.rpn, t1.test_cfg.rcnn
    rpn_cfg2, rcnn_cfg2 = t2.test_cfg.rpn, t2.test_cfg.rcnn

    out = {}
    feats, proposals, pre_nms, finals = {}, {}, {}, {}
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for name, teacher, rpn_cfg, rcnn_cfg in [("teacher1", t1, rpn_cfg1, rcnn_cfg1),
                                                 ("teacher2", t2, rpn_cfg2, rcnn_cfg2)]:
            feat = teacher.extract_feat(img)
            rpn_out = list(teacher.rpn_head(feat))
            prop = teacher.rpn_head.get_bboxes(*rpn_out, img_metas, cfg=rpn_cfg)
            pn, ps = _forward_roi(teacher, feat, prop, img_metas, None)
            db, dl = _forward_roi(teacher, feat, prop, img_metas, rcnn_cfg)
            feats[name] = feat
            proposals[name] = prop
            pre_nms[name] = (pn, ps)
            finals[name] = (db, dl)

        # fused common proposals (on the FINAL detections, as in training)
        fused_prop, fused_label = fuse_teacher_proposals(
            finals["teacher1"][0], finals["teacher1"][1],
            finals["teacher2"][0], finals["teacher2"][1])

        jit = {}
        for name, teacher in [("teacher1", t1), ("teacher2", t2)]:
            mean, std, rel_unc = _jitter_stats(
                teacher, feats[name], img_metas, fused_prop, fused_label,
                jitter_times, jitter_scale, jitter_seed)
            jit[name] = (mean, std, rel_unc)

    for name in ["teacher1", "teacher2"]:
        prop = proposals[name]
        pn, ps = pre_nms[name]
        db, dl = finals[name]
        mean, std, rel_unc = jit[name]
        out[name] = {
            "rpn_proposals": {"num": int(prop[0].shape[0]), "boxes": _to_list(prop[0])},
            "roi_pre_nms": {"num": int(pn[0].shape[0]), "boxes": _to_list(pn[0]),
                            "scores": _to_list(ps[0])},
            "final_det": {"num": int(db[0].shape[0]), "boxes": _to_list(db[0]),
                          "labels": _to_list(dl[0])},
            "jitter": {"mean": _to_list(mean[0]), "std": _to_list(std[0]),
                       "rel_unc": _to_list(rel_unc[0])},
        }
    out["fused_proposals"] = {"num": int(fused_prop[0].shape[0]),
                              "boxes": _to_list(fused_prop[0]),
                              "labels": _to_list(fused_label[0])}

    # formal teacher2 predictions in ORIGINAL image coords (rescale=True),
    # used to re-verify AP against the baseline
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        db_f, dl_f = _forward_roi(t2, feats["teacher2"], proposals["teacher2"],
                                  img_metas, rcnn_cfg2, rescale=True)
    out["teacher2_formal"] = {"boxes": _to_list(db_f[0]), "labels": _to_list(dl_f[0])}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max-imgs", type=int, default=None)
    parser.add_argument("--jitter-seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(fold=args.fold, percent=3))
    jitter_times = cfg.semi_wrapper.train_cfg.jitter_times
    jitter_scale = cfg.semi_wrapper.train_cfg.jitter_scale
    cfg = patch_config(cfg)

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)

    # baseline-consistent model build
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model.CLASSES = dataset.CLASSES
    dual = model.cuda()
    model.eval()
    dtype = next(dual.teacher2.parameters()).dtype
    device = next(dual.teacher2.parameters()).device

    records = []
    n = len(loader.dataset)
    if args.max_imgs is not None:
        n = min(n, args.max_imgs)
    for i, data in enumerate(loader):
        if i >= n:
            break
        img = data["img"][0].cuda()
        img_metas = data["img_metas"][0].data[0]
        fn = img_metas[0]["ori_filename"]
        d = diagnose(dual, img, img_metas, jitter_times, jitter_scale, args.jitter_seed)
        records.append({
            "idx": i, "filename": fn,
            "scale_factor": _to_list(img_metas[0]["scale_factor"]),
            "img_shape": img_metas[0]["img_shape"],
            "ori_shape": img_metas[0]["ori_shape"],
            **d,
        })
        if (i + 1) % 20 == 0:
            print(f"processed {i+1}/{n}", flush=True)

    out = {
        "checkpoint": args.checkpoint,
        "config": cfg.filename,
        "fold": args.fold,
        "num_images": len(records),
        "dtype": str(dtype), "device": str(device),
        "jitter_seed": args.jitter_seed,
        "jitter_times": jitter_times, "jitter_scale": jitter_scale,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"written {args.out} ({len(records)} images)", flush=True)


if __name__ == "__main__":
    main()
