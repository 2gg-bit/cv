"""Independent offline evaluation + prediction export for DualTeacher.

Loads a Phase-3 baseline-NMS checkpoint, runs inference with **teacher2 only**
(no teacher1 fusion, no shared proposals, no TTA/flip/multi-scale), and exports:

    predictions.bbox.json   COCO-format per-box predictions (original px xywh)
    metrics.json            AP / AP50 / AP75 / APs / APm / APl / AR
    eval.log                full COCO evaluation output
    metadata.json           config / checkpoint / image ids / version / eval params

The test set is fixed to data/ssdd/annotations/test.json (232 images).
Images with no detections simply contribute no rows to the prediction JSON; the
full 232-image coverage is guaranteed by the image-id manifest in metadata.json.

Usage:
    python tools/eval_teacher2_export.py configs/reproduce/phase3_dual_teacher_ssdd.py \
        work_dirs/phase3_dual_teacher_baseline_nms/3/6/iter_32000.pth \
        --fold 6 --out-dir eval_export/fold6
"""
import argparse
import contextlib
import io
import json
import os
import os.path as osp
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import single_gpu_test
from mmdet.models import build_detector
from mmdet.datasets import build_dataset

from ssod.datasets import build_dataloader
from ssod.utils import patch_config


def get_eval_kwargs(cfg):
    """Strip EvalHook-only args from cfg.evaluation, keep pure evaluate kwargs."""
    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["type", "interval", "tmpdir", "start", "gpu_collect",
                "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    return eval_kwargs


def build_metadata(cfg, checkpoint_path, fold, img_ids, version):
    # 真正的 Faster R-CNN 测试配置在 cfg.model.model.test_cfg（patch_config 后
    # cfg.model 是 DualTeacher 包装，内层 model 才是检测器）
    inner = cfg.model.get("model", None)
    test_cfg = inner.test_cfg if inner is not None else {}
    rcnn = test_cfg.get("rcnn", {})
    rpn = test_cfg.get("rpn", {})
    return {
        "version": version,
        "config": cfg.filename,
        "checkpoint": checkpoint_path,
        "fold": fold,
        "percent": 3,
        "test_ann_file": cfg.data.test.ann_file,
        "num_test_images": len(img_ids),
        "image_ids": img_ids,
        "eval_params": {
            "inference_on": "teacher2",
            "score_thr": rcnn.get("score_thr", None),
            "nms_iou_threshold": (rcnn.get("nms", {}) or {}).get("iou_threshold", None),
            "max_per_img": rcnn.get("max_per_img", None),
            "rpn_score_thr": rpn.get("score_thr", None),
            "rpn_nms_iou_threshold": (rpn.get("nms", {}) or {}).get("iou_threshold", None),
            "rpn_max_per_img": rpn.get("max_per_img", None),
            "fp16": cfg.get("fp16", None),
            "metric": "bbox",
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--version", type=str, default="be8dddf")
    args = parser.parse_args()

    # ---- 1. config ----
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(fold=args.fold, percent=3))
    cfg = patch_config(cfg)

    # ---- 2. test dataset (fixed test.json) ----
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    n_imgs = len(dataset)
    assert n_imgs == 232, f"expected 232 test images, got {n_imgs}"
    img_ids = [int(x) for x in dataset.img_ids]
    assert len(img_ids) == 232, "image-id manifest must cover 232 images"

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # ---- 3. model ----
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # ---- 4. strict checkpoint load (no silent skip) ----
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model.CLASSES = dataset.CLASSES
    model.inference_on = "teacher2"  # teacher2 only, no fusion

    # ---- 5. inference ----
    modelx = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(modelx, data_loader, show=False, out_dir=None)
    assert len(outputs) == 232, f"inference returned {len(outputs)} results"

    os.makedirs(args.out_dir, exist_ok=True)
    pred_prefix = osp.join(args.out_dir, "predictions")

    # ---- 6. COCO-format predictions ----
    result_files, _ = dataset.format_results(outputs, jsonfile_prefix=pred_prefix)
    pred_json = result_files["bbox"]

    # ---- 7. evaluate (metric_items includes AR from final detections) ----
    eval_kwargs = get_eval_kwargs(cfg)
    metric_items = [
        "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l",
        "AR@100", "AR@300", "AR@1000", "AR_s@1000", "AR_m@1000", "AR_l@1000",
    ]
    eval_kwargs.update(dict(metric="bbox", metric_items=metric_items))

    eval_log_buf = io.StringIO()
    with contextlib.redirect_stdout(eval_log_buf):
        metrics = dataset.evaluate(
            outputs, logger=None, jsonfile_prefix=pred_prefix, **eval_kwargs
        )
    eval_text = eval_log_buf.getvalue()

    # ---- 8. write outputs ----
    with open(osp.join(args.out_dir, "eval.log"), "w") as f:
        f.write(eval_text)

    metrics_out = {str(k): v for k, v in metrics.items()}
    with open(osp.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    metadata = build_metadata(cfg, args.checkpoint, args.fold, img_ids, args.version)
    with open(osp.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"fold {args.fold}  evaluation complete")
    print(f"  predictions: {pred_json}")
    print(f"  bbox_mAP:    {metrics.get('bbox_mAP')}")
    print(f"  bbox_mAP_50: {metrics.get('bbox_mAP_50')}")
    print(f"  bbox_mAP_75: {metrics.get('bbox_mAP_75')}")
    print(f"  AR@100:      {metrics.get('bbox_AR@100')}")
    print(f"  n_imgs:      {n_imgs} (full 232 coverage via metadata.image_ids)")
    print("=" * 60)


if __name__ == "__main__":
    main()
