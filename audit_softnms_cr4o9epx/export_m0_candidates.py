"""Export the M0 teacher2 pre-NMS candidate cache (one forward pass per image).

The correctness requirement for the offline post-processing replay is that the
candidate cache and the reference Hard-NMS output come from the **same**
`_bbox_forward` call.  `tools/diagnose_teacher_stages.py` runs `_bbox_forward`
several times per image (RPN-pre-NMS / final / formal / jitter), and under FP16
autocast those separate calls can produce slightly different regression deltas
(~3 boxes out of ~2880 across 232 images), so its `roi_pre_nms` is not a
bit-exact input for the formal NMS.

This script does, per image, exactly one `_bbox_forward`, then derives from that
single `cls_score`/`bbox_pred`:
  * the pre-NMS candidates  (decoded boxes in resized coords + p_ship), and
  * the reference Hard-NMS output (get_bboxes rescale=True cfg=rcnn).

Both are dumped to JSON.  A separate CPU replay (replay_nms.py) then re-applies
the entry score filter + real mmcv batched_nms + max_per_img on the candidates
and must reproduce the reference per-box, bit-exactly.

No GT / IoU is used anywhere here; this only forwards the frozen teacher2.
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


def _to_list(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    return np.asarray(t).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-ann", type=str, default=None,
                    help="override test annotation (e.g. dev set); default = cfg.data.test")
    ap.add_argument("--img-prefix", type=str, default=None,
                    help="override test image prefix; default = cfg.data.test")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(fold=args.fold, percent=3))
    cfg = patch_config(cfg)
    cfg.data.test.test_mode = True
    if args.test_ann is not None:
        cfg.data.test.ann_file = args.test_ann
    if args.img_prefix is not None:
        cfg.data.test.img_prefix = args.img_prefix
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model.CLASSES = dataset.CLASSES
    dual = model.cuda()
    model.eval()

    from pycocotools.coco import COCO
    coco = COCO(cfg.data.test.ann_file)
    fn2id = {im['file_name']: im['id'] for im in coco.dataset['images']}

    records = []
    for i, data in enumerate(loader):
        img = data["img"][0].cuda()
        img_metas = data["img_metas"][0].data[0]
        meta = img_metas[0]
        fn = meta["ori_filename"]
        teacher = dual.teacher2
        roi = teacher.roi_head
        rcnn_cfg = teacher.test_cfg.rcnn

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            feat = teacher.extract_feat(img)
            rpn_out = list(teacher.rpn_head(feat))
            proposal_list = teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=teacher.test_cfg.rpn)
            rois = bbox2roi(proposal_list)
            if rois.size(0) == 0:
                records.append({"idx": i, "filename": fn, "image_id": fn2id[fn],
                                "candidates": [], "reference_hard_nms": []})
                continue
            bbox_results = roi._bbox_forward(feat, rois)  # ONE call only
            sizes = tuple(len(p) for p in proposal_list)
            rois_i = rois.split(sizes, 0)
            cls_i = bbox_results["cls_score"].split(sizes, 0)
            deltas_i = bbox_results["bbox_pred"].split(sizes, 0)

            # pre-NMS candidates (same cls_score/bbox_pred as the reference)
            boxes, scores = roi.bbox_head.get_bboxes(
                rois_i[0], cls_i[0], deltas_i[0],
                meta["img_shape"], meta["scale_factor"], rescale=False, cfg=None)
            p = scores[:, 0].float()  # ship score (verified: multiclass_nms uses [:, :-1])
            cands = []
            for k in range(boxes.size(0)):
                cands.append({
                    "proposal_id": int(k),
                    "box": _to_list(boxes[k]),      # xyxy, resized coords
                    "p": float(p[k].cpu()),
                })

            # reference Hard-NMS output from the SAME decoded boxes
            ref_boxes, ref_labels = roi.bbox_head.get_bboxes(
                rois_i[0], cls_i[0], deltas_i[0],
                meta["img_shape"], meta["scale_factor"], rescale=True, cfg=rcnn_cfg)
            ref = [{"box": _to_list(ref_boxes[j][:4]),
                    "score": float(ref_boxes[j][4].cpu()),
                    "label": int(ref_labels[j].cpu())}
                   for j in range(ref_boxes.size(0))]

            records.append({
                "idx": i, "filename": fn, "image_id": int(fn2id[fn]),
                "scale_factor": _to_list(meta["scale_factor"]),
                "img_shape": meta["img_shape"],
                "dtype": "float32",
                "coordinate_space": "resized (decode output, pre-rescale)",
                "candidates": cands,
                "reference_hard_nms": ref,
            })
        if (i + 1) % 50 == 0:
            print(f"processed {i+1}/{len(dataset)}", flush=True)

    out = {
        "checkpoint": args.checkpoint,
        "config": cfg.filename,
        "fold": args.fold,
        "num_images": len(records),
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"written {args.out} ({len(records)} images)", flush=True)


if __name__ == "__main__":
    main()
