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
import hashlib
import io
import json
import os
import os.path as osp
import platform
import shutil
import subprocess
import time


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    """Record this checkout, never a stale hard-coded reproduction revision."""
    repo_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            stderr=subprocess.DEVNULL, universal_newlines=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir,
            stderr=subprocess.DEVNULL, universal_newlines=True).strip())
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", None


def prepare_output_dir(path):
    """Fail before inference rather than overwrite an existing experiment."""
    if osp.lexists(path):
        if not osp.isdir(path) or os.listdir(path):
            raise ValueError("output directory must be new or empty: {}".format(path))
    os.makedirs(path, exist_ok=True)


def validate_test_manifest(annotation, image_ids, expected_count=232):
    annotation_ids = [item["id"] for item in annotation["images"]]
    if any(type(value) is not int for value in annotation_ids + list(image_ids)):
        raise ValueError("image IDs must be integers")
    if len(annotation_ids) != expected_count or len(set(annotation_ids)) != expected_count:
        raise ValueError("test annotation must contain {} unique images".format(expected_count))
    if len(image_ids) != expected_count or len(set(image_ids)) != expected_count:
        raise ValueError("dataset manifest must contain {} unique images".format(expected_count))
    if set(image_ids) != set(annotation_ids):
        raise ValueError("dataset image IDs do not match the fixed test annotation")


def get_eval_kwargs(cfg):
    """Strip EvalHook-only args from cfg.evaluation, keep pure evaluate kwargs."""
    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["type", "interval", "tmpdir", "start", "gpu_collect",
                "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    return eval_kwargs


def build_metadata(cfg, checkpoint_path, fold, img_ids, version=None,
                   software_versions=None):
    # 真正的 Faster R-CNN 测试配置在 cfg.model.model.test_cfg（patch_config 后
    # cfg.model 是 DualTeacher 包装，内层 model 才是检测器）
    inner = cfg.model.get("model", None)
    test_cfg = inner.get("test_cfg", {}) if inner is not None else {}
    rcnn = test_cfg.get("rcnn", {})
    rpn = test_cfg.get("rpn", {})
    roi_head = inner.get("roi_head", {}) if inner is not None else {}
    quality_enabled = bool(roi_head.get("quality_enabled", False))
    quality_ranking = quality_enabled and bool(roi_head.get("quality_inference", False))
    revision, dirty = git_revision()
    return {
        "version": revision,
        "version_label": version,
        "git_revision": revision,
        "git_dirty": dirty,
        "software_versions": software_versions or {},
        "config": cfg.filename,
        "resolved_config": "resolved_config.py",
        "checkpoint": osp.abspath(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "fold": fold,
        "percent": 3,
        "test_ann_file": cfg.data.test.ann_file,
        "test_ann_snapshot": "test.json",
        "test_ann_sha256": sha256_file(cfg.data.test.ann_file),
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
            "quality_enabled": quality_enabled,
            "quality_inference": quality_ranking,
            "candidate_rule": "p_ship > score_thr",
            "ranking_and_export_score": (
                "p_ship * sigmoid(quality_logit)" if quality_ranking else "p_ship"),
            "second_joint_score_threshold": False,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def main():
    # Keep helpers and the export comparator usable without the GPU stack.
    import mmcv
    import mmdet
    import numpy as np
    import torch
    from mmcv import Config, DictAction
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import single_gpu_test
    from mmdet.models import build_detector
    from mmdet.datasets import build_dataset
    from ssod.datasets import build_dataloader
    from ssod.utils import patch_config
    try:
        from .compare_prediction_exports import validate_predictions
    except ImportError:  # normal invocation: python tools/eval_teacher2_export.py
        from compare_prediction_exports import validate_predictions

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--version", type=str, default=None,
                        help="optional user label; actual git revision is always recorded")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction,
                        help="config overrides before patch_config; fold/percent remain fixed")
    args = parser.parse_args()
    prepare_output_dir(args.out_dir)

    # ---- 1. config ----
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        if "fold" in args.cfg_options or "percent" in args.cfg_options:
            parser.error("use --fold for the fold; this exporter fixes percent=3")
        cfg.merge_from_dict(args.cfg_options)
    cfg.merge_from_dict(dict(fold=args.fold, percent=3))
    cfg = patch_config(cfg)

    # ---- 2. test dataset (fixed test.json) ----
    cfg.data.test.test_mode = True
    fixed_annotation_path = osp.realpath("data/ssdd/annotations/test.json")
    if osp.realpath(cfg.data.test.ann_file) != fixed_annotation_path:
        raise ValueError("evaluation is fixed to data/ssdd/annotations/test.json")
    with open(fixed_annotation_path) as handle:
        annotation = json.load(handle)
    dataset = build_dataset(cfg.data.test)
    n_imgs = len(dataset)
    assert n_imgs == 232, f"expected 232 test images, got {n_imgs}"
    img_ids = [int(x) for x in dataset.img_ids]
    validate_test_manifest(annotation, img_ids)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # ---- 3. model ----
    cfg.model.train_cfg = None
    cfg.dump(osp.join(args.out_dir, "resolved_config.py"))
    shutil.copyfile(fixed_annotation_path, osp.join(args.out_dir, "test.json"))
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # ---- 4. strict checkpoint load (no silent skip) ----
    load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model.CLASSES = dataset.CLASSES
    model.inference_on = "teacher2"  # teacher2 only, no fusion

    # ---- 5. inference ----
    modelx = MMDataParallel(model.cuda(0), device_ids=[0])
    outputs = single_gpu_test(modelx, data_loader, show=False, out_dir=None)
    assert len(outputs) == 232, f"inference returned {len(outputs)} results"

    pred_prefix = osp.join(args.out_dir, "predictions")

    # ---- 6. COCO-format predictions ----
    result_files, _ = dataset.format_results(outputs, jsonfile_prefix=pred_prefix)
    pred_json = result_files["bbox"]
    with open(pred_json) as handle:
        predictions = json.load(handle)
    validate_predictions(predictions, img_ids, [item["id"] for item in annotation["categories"]])

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

    metadata = build_metadata(
        cfg, args.checkpoint, args.fold, img_ids, args.version,
        software_versions={
            "python": platform.python_version(), "torch": torch.__version__,
            "mmcv": mmcv.__version__, "mmdet": mmdet.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })
    metadata["resolved_config_sha256"] = sha256_file(osp.join(args.out_dir, "resolved_config.py"))
    metadata["category_ids"] = [item["id"] for item in annotation["categories"]]
    metadata["num_predictions"] = len(predictions)
    metadata["empty_prediction_image_ids"] = sorted(
        set(img_ids) - {item["image_id"] for item in predictions})
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
