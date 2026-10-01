"""Export M1 pre-NMS candidates (p, q, boxes, proposal id) for score/NMS split.

For the M1 checkpoint, runs teacher2 up to the point right BEFORE batched_nms,
and dumps per-image: rescaled boxes, classification score p (ship), quality
score q = sigmoid(logit), and the proposal index. This lets the analysis replay
the four score/NMS combinations without re-training or re-inference.
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
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(fold=args.fold, percent=3))
    cfg = patch_config(cfg)

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model.CLASSES = dataset.CLASSES
    dual = model.cuda()
    model.eval()

    records = []
    for i, data in enumerate(loader):
        img = data["img"][0].cuda()
        img_metas = data["img_metas"][0].data[0]
        fn = img_metas[0]["ori_filename"]
        teacher = dual.teacher2
        roi = teacher.roi_head
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            feat = teacher.extract_feat(img)
            rpn_out = list(teacher.rpn_head(feat))
            proposal_list = teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas, cfg=teacher.test_cfg.rpn)
            rois = bbox2roi(proposal_list)
            if rois.size(0) == 0:
                records.append({"idx": i, "filename": fn, "candidates": []})
                continue
            results = roi._bbox_forward(feat, rois)
            logits = roi.quality_head(results["bbox_feats"], results["bbox_pred"])
            sizes = tuple(len(p) for p in proposal_list)
            rois_i = rois.split(sizes, 0)
            cls_i = results["cls_score"].split(sizes, 0)
            deltas_i = results["bbox_pred"].split(sizes, 0)
            logits_i = logits.split(sizes, 0)
            # batch=1, only image 0
            meta = img_metas[0]
            boxes, scores = roi.bbox_head.get_bboxes(
                rois_i[0], cls_i[0], deltas_i[0],
                meta["img_shape"], meta["scale_factor"], rescale=False, cfg=None)
            p = scores[:, 0].float()  # ship score
            q = logits_i[0].float().sigmoid().reshape(-1)
            # proposal index is the original proposal_list[0] order
            cands = []
            for k in range(boxes.size(0)):
                cands.append({
                    "proposal_id": int(k),
                    "box": _to_list(boxes[k]),   # xyxy (input scale, rescale=False)
                    "p": float(p[k].cpu()),
                    "q": float(q[k].cpu()),
                })
            records.append({
                "idx": i, "filename": fn,
                "scale_factor": _to_list(meta["scale_factor"]),
                "img_shape": meta["img_shape"],
                "candidates": cands,
            })
        if (i + 1) % 20 == 0:
            print(f"processed {i+1}/{len(dataset)}", flush=True)

    out = {"checkpoint": args.checkpoint, "fold": args.fold,
           "num_images": len(records), "records": records}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"written {args.out} ({len(records)} images)", flush=True)


if __name__ == "__main__":
    main()
