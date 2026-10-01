#!/usr/bin/env python
"""训练图诊断：同折 B0 在 sup2 三张训练图上的 teacher2 推理导出。

性质与边界（同时写进产物，避免被误读）：
  * 这是**训练图诊断**，**不是开发集指标**，**不是泛化证据**，不计算任何 COCO 指标、不写 metrics.json。
  * 使用的图像是该折 `sup2` 训练集自己的 3 张图，**模型见过它们**。
  * 本脚本**不修改** tools/train_ablation.py 与 tools/eval_teacher2_export.py；
    并断言后者仍保留"评估固定为 232 图"的断言（正式评估入口的约束不被削弱）。

与正式评估的关系：
  * 复用 `tools/train_ablation.py` 的 `pin_repository()` / `source_manifest()`：先在 checkout 内
    钉住 `ssod`，再导入 torch/mmdet，并记录 9 个 ssod 模块的源码收据。
  * 复用 `tools/eval_teacher2_export.py` 的 `sha256_file` / `git_revision` / `prepare_output_dir`，
    以及 `tools/compare_prediction_exports.py` 的 `validate_predictions`。
  * 数据集只把正式 `data.test` 的 `ann_file` / `img_prefix` 换成该折的 `@3.json` / `JPEGImages/`；
    **pipeline 与 test_mode 原样不动**，后处理沿用 cfg 的 test_cfg（未经任何调整）。
  * 复用冻结误差分析脚本的匹配原语（`iou` / `load_gt` / `legacy_match` / `write_*`），
    使"与已标注 GT 最大 IoU"与报告 §5.6 同一口径。

用法（必须从 checkout 根目录运行）：
    python tools/diagnose_train3shot_export.py --fold 6 --out-dir <新目录>
"""
import argparse
import copy
import json
import os
import os.path as osp
import re
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import train_ablation  # noqa: E402  （纯 stdlib，可安全先导入）

# ---- 先钉住 checkout，再导入任何 GPU 栈 ----
train_ablation.pin_repository()
if Path.cwd().resolve() != Path(train_ablation.ROOT).resolve():
    raise SystemExit(
        "必须从 checkout 根目录运行，以保持相对数据/权重路径一致：\n  %s" % train_ablation.ROOT)

import eval_teacher2_export as official_eval  # noqa: E402

# 冻结误差分析脚本的匹配原语（同一口径，不重写）
FROZEN_ANALYSIS_DIR = REPO_ROOT / "ablation_configs/error_analysis_b0_pgboth/code"
sys.path.insert(0, str(FROZEN_ANALYSIS_DIR))
import error_analysis as ea  # noqa: E402

# ---- 预注册常量 ----
SMALL_WH = 32 * 32           # 小框口径：bbox w*h（与报告 §5.6 一致）
HIGH_SCORE = 0.7             # 高分档下界（与报告 PROD_SCORE_BANDS / §5.6 一致）
UNLABELED_IOU_LT = 0.1       # "与已标注 GT 重叠很低"的切点（同上）
OVERLAY_SCALE = 3
ALLOWED_ANN_RE = re.compile(r"^instances_train2017\.(\d)@3\.json$")
FOLDS = (6, 7, 8)

# 正式评估入口必须保留的断言（不得被本脚本削弱）
OFFICIAL_EVAL_REQUIRED = [
    "data/ssdd/annotations/test.json",
    "expected 232 test images",
    "inference returned",
]

DISCLAIMER = {
    "kind": "训练图诊断",
    "is_dev_set_metric": False,
    "is_generalization_evidence": False,
    "coco_metrics_computed": False,
    "images_were_seen_in_training": True,
    "why_not_a_metric": (
        "这 9 张图是该折 sup2 训练集自身，模型见过；此处只回答"
        "「模型对这些图输出了什么框」，不构成任何指标或泛化证据。"),
    "count_note": (
        "框数与 train.json 一致，只证明**标注文件之间一致**，不能称为现实目标「无遗漏」。"),
    "clean_scene_note": (
        "干净场景中的高分小框，也可能是重复框或尺度不准的船框，"
        "不能自动判成真实背景。"),
    "inference_scope_note": (
        "推理能确认「模型是否对疑似区域给高分」，**不能证明那些区域是不是船、是否漏标**；"
        "判不清的保留「无法判断」。"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="训练图诊断：同折 B0 × sup2 三张训练图")
    p.add_argument("--fold", type=int, required=True, choices=list(FOLDS))
    p.add_argument("--out-dir", required=True, help="输出目录（必须不存在或为空）")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", default=None, help="默认取该折 b0_evaluation/metadata.json 中记录的值")
    p.add_argument("--checkpoint", default=None, help="默认取该折 b0_evaluation/metadata.json 中记录的值")
    p.add_argument("--score-thr", type=float, default=None,
                   help="仅用于探查：覆盖 rcnn.score_thr 以看低分候选分布。"
                        "提供即视为**后处理被改动**，产物必须另存并标注，不得与既定后处理的结果混用。")
    return p.parse_args(argv)


def assert_official_eval_intact():
    """证明正式评估脚本仍强制固定 232 图入口（只读断言，不改动它）。"""
    source = Path(official_eval.__file__).read_text(encoding="utf-8")
    missing = [s for s in OFFICIAL_EVAL_REQUIRED if s not in source]
    if missing:
        raise SystemExit(
            "正式评估脚本 %s 中缺少预期断言片段：%s\n"
            "本诊断不得削弱正式入口的 232 图约束。" % (official_eval.__file__, missing))
    return {
        "path": str(Path(official_eval.__file__).resolve()),
        "sha256": official_eval.sha256_file(official_eval.__file__),
        "required_fragments_present": OFFICIAL_EVAL_REQUIRED,
    }


def plain(obj):
    """把 mmcv ConfigDict 递归降级为内建 dict/list，避免序列化歧义。"""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    return obj


def build_diagnostic_dataset_cfg(cfg, fold):
    """把正式 test 数据集只换 ann_file / img_prefix；pipeline 与 test_mode 原样不动。"""
    diag = copy.deepcopy(cfg.data.test)
    rel_ann = "data/ssdd/annotations/semi_supervised/instances_train2017.%d@3.json" % fold
    m = ALLOWED_ANN_RE.match(osp.basename(rel_ann))
    if not m or int(m.group(1)) != fold:
        raise SystemExit("内部错误：诊断标注文件名与 --fold 不一致：%s" % rel_ann)
    base = osp.basename(rel_ann)
    if base == "test.json" or "unlabeled" in base:
        raise SystemExit("拒绝：诊断只能用该折标注的 3 张训练图，不得用测试集或未标注池：%s" % base)
    if not osp.isfile(rel_ann):
        raise SystemExit("标注文件不存在：%s" % rel_ann)
    diag.ann_file = rel_ann
    diag.img_prefix = "data/ssdd/JPEGImages/"
    diag.test_mode = True
    return diag, rel_ann


def score_band_of(score):
    if score >= 0.9:
        return "ge_0.9"
    if score >= HIGH_SCORE:
        return "0.7_0.9"
    if score >= 0.5:
        return "0.5_0.7"
    if score >= 0.3:
        return "0.3_0.5"
    return "lt_0.3"


def draw_overlay(img_path, preds, gts, dest):
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    im = im.resize((im.width * OVERLAY_SCALE, im.height * OVERLAY_SCALE), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    for gt in gts:
        x1, y1, x2, y2 = [v * OVERLAY_SCALE for v in gt["box"]]
        dr.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
    n_hi = n_mid = n_lo = 0
    for p in preds:
        x, y, w, h = p["bbox"]
        box = [x * OVERLAY_SCALE, y * OVERLAY_SCALE,
               (x + w) * OVERLAY_SCALE, (y + h) * OVERLAY_SCALE]
        s = float(p["score"])
        if s >= 0.9:
            color, width, n_hi = (255, 0, 0), 2, n_hi + 1
        elif s >= HIGH_SCORE:
            color, width, n_mid = (255, 165, 0), 2, n_mid + 1
        else:
            color, width, n_lo = (0, 170, 255), 1, n_lo + 1
        dr.rectangle(box, outline=color, width=width)
    dr.text((6, 6), "GT(green)=%d  preds: >=0.9(red)=%d  [0.7,0.9)(orange)=%d  <0.7(cyan)=%d"
            % (len(gts), n_hi, n_mid, n_lo), fill=(255, 255, 0))
    im.save(dest)


def draw_crop(img_path, pred_box, gts, pad, scale, dest, label):
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    x, y, w, h = pred_box
    x1, y1 = max(0, int(x) - pad), max(0, int(y) - pad)
    x2, y2 = min(im.width, int(x + w) + pad), min(im.height, int(y + h) + pad)
    crop = im.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(crop)
    for gt in gts:
        gx1, gy1, gx2, gy2 = gt["box"]
        dr.rectangle([(gx1 - x1) * scale, (gy1 - y1) * scale,
                      (gx2 - x1) * scale, (gy2 - y1) * scale], outline=(0, 220, 0), width=2)
    dr.rectangle([(x - x1) * scale, (y - y1) * scale,
                  (x + w - x1) * scale, (y + h - y1) * scale], outline=(255, 0, 0), width=2)
    dr.text((4, 4), label, fill=(255, 255, 0))
    crop.save(dest)


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = osp.abspath(args.out_dir)
    official_eval.prepare_output_dir(out_dir)   # 新目录或空目录，否则失败

    receipt = assert_official_eval_intact()
    source_receipt = train_ablation.source_manifest()
    if Path(source_receipt["repo_root"]).resolve() != Path(train_ablation.ROOT).resolve():
        raise SystemExit("源码收据的 repo_root 不是当前 checkout")

    # ---- 该折 B0 的记录（权重路径与 SHA256 以正式评估记录为准）----
    eval_dir = REPO_ROOT / ("ablation_configs/fold%d_seed678/b0_evaluation" % args.fold)
    recorded = json.loads((eval_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg_rel = args.config or recorded["config"]
    ckpt_abs = args.checkpoint or recorded["checkpoint"]
    if not osp.isfile(ckpt_abs):
        raise SystemExit("权重不存在：%s" % ckpt_abs)
    ckpt_sha = official_eval.sha256_file(ckpt_abs)
    sha_matches_record = (ckpt_sha == recorded["checkpoint_sha256"])
    if not sha_matches_record and args.checkpoint is None:
        raise SystemExit("权重 SHA256 与正式评估记录不一致：\n  %s\n  %s" % (ckpt_sha, recorded["checkpoint_sha256"]))

    import mmcv
    import mmdet
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
    try:
        from tools.compare_prediction_exports import validate_predictions
    except ImportError:
        from compare_prediction_exports import validate_predictions

    # ---- 1. config：与正式评估同一条路径 ----
    cfg = Config.fromfile(cfg_rel)
    if cfg.get("ablation_experiment") != "b0":
        raise SystemExit("配置不是 b0：%s" % cfg_rel)
    if cfg.get("fold") != args.fold:
        raise SystemExit("配置 fold=%s 与 --fold=%d 不一致" % (cfg.get("fold"), args.fold))
    cfg.merge_from_dict(dict(fold=args.fold))
    cfg = patch_config(cfg)
    cfg.data.test.test_mode = True
    cfg.data.test_type = "CocoDataset"

    diag_cfg, ann_rel = build_diagnostic_dataset_cfg(cfg, args.fold)
    # 防御：诊断标注绝不等同于正式固定测试标注
    if osp.realpath(diag_cfg.ann_file) == osp.realpath("data/ssdd/annotations/test.json"):
        raise SystemExit("拒绝：诊断标注指向固定测试集 test.json")

    inner = cfg.model.model
    frozen_test_cfg = plain(inner.test_cfg)
    if args.score_thr is not None:
        print("[探查] rcnn.score_thr 由 %s 覆盖为 %s —— 本产物**不是**既定后处理的结果"
              % (frozen_test_cfg["rcnn"].get("score_thr"), args.score_thr), flush=True)
        inner.test_cfg["rcnn"]["score_thr"] = args.score_thr
    effective_test_cfg = plain(inner.test_cfg)

    dataset = build_dataset(diag_cfg)
    if len(dataset) != 3:
        raise SystemExit("该折标注图应为 3 张，实得 %d 张：%s" % (len(dataset), ann_rel))
    img_ids = [int(x) for x in dataset.img_ids]

    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False)

    # ---- 2. 模型：strict 加载 + teacher2 ----
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt_abs, map_location="cpu", strict=True)
    model.CLASSES = dataset.CLASSES
    model.inference_on = "teacher2"

    modelx = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu])
    outputs = single_gpu_test(modelx, data_loader, show=False, out_dir=None)
    if len(outputs) != 3:
        raise SystemExit("推理返回 %d 条，应为 3 条" % len(outputs))

    # ---- 3. 导出 COCO 格式预测（既定后处理已在 test_cfg 内）----
    pred_prefix = osp.join(out_dir, "predictions")
    result_files, _ = dataset.format_results(outputs, jsonfile_prefix=pred_prefix)
    pred_path = result_files["bbox"]
    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(diag_cfg.ann_file, encoding="utf-8") as f:
        ann = json.load(f)
    cat_ids = [item["id"] for item in ann["categories"]]
    validate_predictions(predictions, img_ids, cat_ids)   # 复用正式导出的校验

    # ---- 4. 冻结原语：GT 与贪心匹配 ----
    id2gt, gt_images = ea.load_gt(diag_cfg.ann_file)
    fn_by_id = {im["id"]: im["file_name"] for im in gt_images}
    id_by_fn = {im["file_name"]: im["id"] for im in gt_images}
    preds = ea.load_predictions(pred_path)
    recs, _ = ea.legacy_match(preds, id2gt)
    rec_by_idx = {r["pred_index"]: r for r in recs}
    if len(rec_by_idx) != len(preds):
        raise SystemExit("匹配结果与预测数不一致：%d vs %d" % (len(rec_by_idx), len(preds)))

    # ---- 5. 逐预测明细 ----
    rows = []
    for idx, p in enumerate(preds):
        r = rec_by_idx[idx]
        x, y, w, h = [float(v) for v in p["bbox"]]
        img_id = int(p["image_id"])
        small = (w * h) < SMALL_WH
        high = float(p["score"]) >= HIGH_SCORE
        unlabeled = r["best_iou"] < UNLABELED_IOU_LT
        rows.append({
            "fold": args.fold, "image_id": img_id, "file_name": fn_by_id.get(img_id, ""),
            "pred_index": idx, "score": "%.6f" % float(p["score"]),
            "score_band": score_band_of(float(p["score"])),
            "x": "%.2f" % x, "y": "%.2f" % y, "w": "%.2f" % w, "h": "%.2f" % h,
            "bbox_wh_area": "%.2f" % (w * h),
            "max_gt_iou": "%.6f" % float(r["best_iou"]),
            "kind": r["kind"], "matched_gt_id": r["matched_gt_id"],
            "is_small": int(small), "is_high_score": int(high),
            "is_unlabeled_region": int(unlabeled),
            "n_gt_in_image": len(id2gt.get(img_id, []))})
    rows.sort(key=lambda d: (d["file_name"], -float(d["score"]), d["pred_index"]))
    ea.write_csv(osp.join(out_dir, "predictions_diagnostic.csv"), rows, [
        "fold", "image_id", "file_name", "pred_index", "score", "score_band",
        "x", "y", "w", "h", "bbox_wh_area", "max_gt_iou", "kind", "matched_gt_id",
        "is_small", "is_high_score", "is_unlabeled_region", "n_gt_in_image"])

    # ---- 6. 叠图 + 裁图 ----
    os.makedirs(osp.join(out_dir, "overlay"), exist_ok=True)
    os.makedirs(osp.join(out_dir, "crops_high_small"), exist_ok=True)
    os.makedirs(osp.join(out_dir, "crops_unlabeled"), exist_ok=True)
    crops_hi, crops_unlab = [], []
    for img_id in sorted(id2gt):
        fn = fn_by_id.get(img_id, "")
        src = osp.join("data/ssdd/JPEGImages/", fn)
        if not osp.isfile(src):
            raise SystemExit("图像不存在：%s" % src)
        gts = id2gt[img_id]
        img_preds = [p for p in preds if int(p["image_id"]) == img_id]
        draw_overlay(src, img_preds, gts, osp.join(out_dir, "overlay", fn.replace(".jpg", ".png")))
        for idx, p in enumerate(preds):
            if int(p["image_id"]) != img_id:
                continue
            r = rec_by_idx[idx]
            s, (x, y, w, h) = float(p["score"]), [float(v) for v in p["bbox"]]
            label = "s=%.3f maxIoU=%.3f %s wh=%.0f" % (s, r["best_iou"], r["kind"], w * h)
            if s >= HIGH_SCORE and (w * h) < SMALL_WH:
                rel = "crops_high_small/%s_p%03d.png" % (fn.replace(".jpg", ""), idx)
                dest = osp.join(out_dir, rel)
                os.makedirs(osp.dirname(dest), exist_ok=True)
                draw_crop(src, (x, y, w, h), gts, 40, 6, dest, label)
                crops_hi.append({"image_id": img_id, "file_name": fn, "pred_index": idx,
                                 "score": "%.6f" % s, "bbox_xywh": [x, y, w, h],
                                 "bbox_wh_area": "%.2f" % (w * h),
                                 "max_gt_iou": "%.6f" % r["best_iou"], "kind": r["kind"],
                                 "crop": rel})
            if s >= HIGH_SCORE and r["best_iou"] < UNLABELED_IOU_LT:
                rel = "crops_unlabeled/%s_p%03d.png" % (fn.replace(".jpg", ""), idx)
                dest = osp.join(out_dir, rel)
                os.makedirs(osp.dirname(dest), exist_ok=True)
                draw_crop(src, (x, y, w, h), gts, 60, 4, dest, label)
                crops_unlab.append({"image_id": img_id, "file_name": fn, "pred_index": idx,
                                    "score": "%.6f" % s, "bbox_xywh": [x, y, w, h],
                                    "bbox_wh_area": "%.2f" % (w * h),
                                    "max_gt_iou": "%.6f" % r["best_iou"], "kind": r["kind"],
                                    "crop": rel})
    ea.write_csv(osp.join(out_dir, "crops_high_small.csv"), crops_hi, [
        "image_id", "file_name", "pred_index", "score", "bbox_xywh", "bbox_wh_area",
        "max_gt_iou", "kind", "crop"])
    ea.write_csv(osp.join(out_dir, "crops_unlabeled_high_score.csv"), crops_unlab, [
        "image_id", "file_name", "pred_index", "score", "bbox_xywh", "bbox_wh_area",
        "max_gt_iou", "kind", "crop"])

    # ---- 7. 未标注区域预测（重点：fold6/000020.jpg）----
    unlab_by_img = {}
    for img_id in sorted(id2gt):
        fn = fn_by_id.get(img_id, "")
        items = []
        for idx, p in enumerate(preds):
            if int(p["image_id"]) != img_id:
                continue
            r = rec_by_idx[idx]
            if r["best_iou"] >= UNLABELED_IOU_LT:
                continue
            x, y, w, h = [float(v) for v in p["bbox"]]
            items.append({"pred_index": idx, "score": round(float(p["score"]), 6),
                          "bbox_xywh": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                          "bbox_wh_area": round(w * h, 2),
                          "max_gt_iou_to_annotated_gt": round(float(r["best_iou"]), 6),
                          "score_band": score_band_of(float(p["score"])),
                          "kind": r["kind"]})
        items.sort(key=lambda d: (-d["score"], d["pred_index"]))
        unlab_by_img[fn] = items
        ea.write_json(osp.join(out_dir, "unlabeled_region_%s.json" % fn.replace(".jpg", "")), {
            "fold": args.fold, "file_name": fn,
            "definition": ("该图预测框中，对同图**已标注** GT 的最大 IoU < %g 的那些；"
                           "只说明「与已标注 GT 重叠很低」，**不等于空间上远离 GT**，"
                           "也不代表该处没有船。" % UNLABELED_IOU_LT),
            "annotated_gt": [{"id": g["id"], "box_xyxy": [round(v, 2) for v in g["box"]],
                              "ann_area": round(g["ann_area"], 2)} for g in id2gt[img_id]],
            "n_predictions_in_image": len([1 for p in preds if int(p["image_id"]) == img_id]),
            "n_unlabeled_region": len(items), "items": items})

    # ---- 8. 逐图汇总 ----
    per_image = {}
    for img_id in sorted(id2gt):
        fn = fn_by_id.get(img_id, "")
        img_preds = [p for p in preds if int(p["image_id"]) == img_id]
        per_image[fn] = {
            "image_id": img_id,
            "n_annotated_gt": len(id2gt[img_id]),
            "n_predictions": len(img_preds),
            "n_preds_ge_0.9": sum(1 for p in img_preds if float(p["score"]) >= 0.9),
            "n_preds_ge_0.7": sum(1 for p in img_preds if float(p["score"]) >= HIGH_SCORE),
            "n_small_preds_wh_lt_1024": sum(
                1 for p in img_preds if p["bbox"][2] * p["bbox"][3] < SMALL_WH),
            "n_high_score_small_preds": sum(
                1 for p in img_preds
                if float(p["score"]) >= HIGH_SCORE and p["bbox"][2] * p["bbox"][3] < SMALL_WH),
            "n_unlabeled_region_preds": len(unlab_by_img[fn]),
            "n_unlabeled_region_preds_ge_0.7": sum(
                1 for d in unlab_by_img[fn] if d["score"] >= HIGH_SCORE),
            "max_unlabeled_region_score": (max(d["score"] for d in unlab_by_img[fn])
                                           if unlab_by_img[fn] else None),
            "n_dupe": sum(1 for idx, p in enumerate(preds)
                          if int(p["image_id"]) == img_id and rec_by_idx[idx]["kind"] == "dupe"),
            "n_bkg": sum(1 for idx, p in enumerate(preds)
                         if int(p["image_id"]) == img_id and rec_by_idx[idx]["kind"] == "bkg"),
            "n_tp": sum(1 for idx, p in enumerate(preds)
                        if int(p["image_id"]) == img_id and rec_by_idx[idx]["kind"] == "tp"),
        }

    summary = {
        "disclaimer": DISCLAIMER,
        "post_processing_overridden": args.score_thr is not None,
        "post_processing_override_note": (
            "**本产物是探查运行**：rcnn.score_thr 被显式覆盖为 %s，不是该折正式评估所用的既定后处理。"
            "只用于观察低分候选的分布，不得与既定后处理的产物混用、不得据此比较指标。"
            % args.score_thr) if args.score_thr is not None else None,
        "fold": args.fold,
        "annotation": ann_rel,
        "annotation_sha256": official_eval.sha256_file(diag_cfg.ann_file),
        "image_prefix": diag_cfg.img_prefix,
        "n_images": len(id2gt),
        "n_predictions": len(preds),
        "n_annotated_gt": sum(len(v) for v in id2gt.values()),
        "post_processing": {
            "source": ("cfg.model.model.test_cfg（**rcnn.score_thr 被探查性覆盖**）"
                       if args.score_thr is not None else "cfg.model.model.test_cfg（未经调整）"),
            "effective": effective_test_cfg,
            "frozen_reference": frozen_test_cfg,
            "inference_on": "teacher2",
            "max_per_img_note": ("rcnn.max_per_img 会截断每图预测数；本表计数受该上限影响。"),
        },
        "per_image": per_image,
        "max_unlabeled_region_score_overall": max(
            [d["score"] for v in unlab_by_img.values() for d in v], default=None),
        "max_matched_tp_score_overall": max(
            [float(p["score"]) for idx, p in enumerate(preds)
             if rec_by_idx[idx]["kind"] == "tp"], default=None),
        "unlabeled_region_by_file": {k: [d["pred_index"] for d in v]
                                     for k, v in unlab_by_img.items()},
        "unlabeled_region_counts": {k: {"total": len(v),
                                        "ge_0.7": sum(1 for d in v if d["score"] >= HIGH_SCORE),
                                        "ge_0.9": sum(1 for d in v if d["score"] >= 0.9)}
                                    for k, v in unlab_by_img.items()},
        "reading_rules": [
            "框数与 train.json 一致只证明**标注文件之间一致**，不能称为现实目标「无遗漏」。",
            "干净场景中的高分小框也可能是重复框或尺度不准的船框，不能自动判成真实背景。",
            "推理能确认「模型是否对疑似区域给高分」，**不能证明那些区域是不是船、是否漏标**；"
            "判不清的保留「无法判断」。",
        ],
    }
    ea.write_json(osp.join(out_dir, "summary.json"), summary)

    manifest = {
        "purpose": "训练图诊断（非开发集指标、非泛化证据）",
        "fold": args.fold,
        "repo_root": source_receipt["repo_root"],
        "ssod_source_receipt": source_receipt["sources"],
        "config": cfg_rel,
        "checkpoint": ckpt_abs,
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_sha256_matches_official_eval_record": sha_matches_record,
        "diagnostic_annotation": ann_rel,
        "diagnostic_annotation_sha256": official_eval.sha256_file(diag_cfg.ann_file),
        "official_eval_record": str(eval_dir / "metadata.json"),
        "official_eval_recorded_test_ann_sha256": recorded["test_ann_sha256"],
        "official_eval_script": receipt,
        "train_ablation_script": {
            "path": str(Path(train_ablation.__file__).resolve()),
            "sha256": official_eval.sha256_file(train_ablation.__file__)},
        "frozen_analysis_primitives": {
            "path": str((FROZEN_ANALYSIS_DIR / "error_analysis.py").resolve()),
            "sha256": official_eval.sha256_file(str(FROZEN_ANALYSIS_DIR / "error_analysis.py"))},
        "diagnostic_script": {"path": str(Path(__file__).resolve()),
                              "sha256": official_eval.sha256_file(__file__)},
        "software_versions": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "mmcv": mmcv.__version__, "mmdet": mmdet.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "gpu": args.gpu,
        "image_ids": img_ids,
        "post_processing_overridden": args.score_thr is not None,
        "score_thr_override": args.score_thr,
        "effective_rcnn_score_thr": effective_test_cfg["rcnn"].get("score_thr"),
        "frozen_rcnn_score_thr": frozen_test_cfg["rcnn"].get("score_thr"),
        "test_set_not_used": "本次推理未使用 data/ssdd/annotations/test.json 及其图像",
    }
    ea.write_json(osp.join(out_dir, "manifest.json"), manifest)

    ea.write_json(osp.join(out_dir, "run_metadata.json"), {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "duration_seconds": round(time.time() - t0, 1),
        "out_dir": out_dir,
        "argv": sys.argv[1:],
    })

    # ---- 9. 报告（措辞边界写死在报告里）----
    L = []
    A = L.append
    A("# 训练图诊断：同折 B0 × sup2 三张训练图（teacher2，%s）\n"
      % ("**探查性覆盖后处理**" if args.score_thr is not None else "既定后处理"))
    A("- 性质：**训练图诊断**。**不是开发集指标**，**不是泛化证据**，**不计算任何 COCO 指标**。")
    A("- 图像：该折 `sup2` 训练集自身的 3 张图（模型见过这些图）。")
    A("- 入口：`tools/diagnose_train3shot_export.py`（独立诊断入口；未改动正式评估脚本）。")
    A("- 输入收据：见 `manifest.json`（`ssod` 9 个模块 sha256、权重 sha256、标注 sha256、脚本 sha256）。\n")
    A("## 1. 输入与后处理\n")
    A("- 折：%d；标注：`%s`（sha256 `%s`）" % (args.fold, ann_rel, summary["annotation_sha256"]))
    A("- 权重：`%s`" % ckpt_abs)
    A("- 权重 sha256 与正式评估记录%s" % ("一致" if sha_matches_record else "**不一致**"))
    if args.score_thr is not None:
        A("> **注意：本产物是探查运行** —— rcnn `score_thr` 被显式覆盖为 %s"
          "（该折正式评估用的是 %s）。本文件**不是**既定后处理的结果，"
          "只用于观察低分候选的分布，**不得**与既定后处理的产物混用或据此比较指标。\n"
          % (args.score_thr, frozen_test_cfg["rcnn"].get("score_thr")))
    A("- 后处理（%s）：rcnn `score_thr=%s`、`nms iou=%s`、`max_per_img=%s`；"
      "rpn `nms iou=%s`、`max_per_img=%s`；`inference_on=teacher2`"
      % ("**探查性覆盖**" if args.score_thr is not None else "取自 cfg，未经调整",
         effective_test_cfg["rcnn"].get("score_thr"),
         (effective_test_cfg["rcnn"].get("nms") or {}).get("iou_threshold"),
         effective_test_cfg["rcnn"].get("max_per_img"),
         (effective_test_cfg["rpn"].get("nms") or {}).get("iou_threshold"),
         effective_test_cfg["rpn"].get("max_per_img")))
    A("- 注：`rcnn.max_per_img` 会截断每图预测数，下表计数受该上限影响。\n")
    A("## 2. 逐图汇总\n")
    A("| 图 | 已标注 GT | 预测 | ≥0.9 | ≥0.7 | 小框(w·h<32²) | 高分小框 | "
      "与已标注 GT 最大 IoU<0.1 | 其中≥0.7 | 未标注区最高分 | tp | dupe | bkg |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fn, d in per_image.items():
        A("| `%s` | %d | %d | %d | %d | %d | %d | %d | %d | %s | %d | %d | %d |" % (
            fn, d["n_annotated_gt"], d["n_predictions"], d["n_preds_ge_0.9"], d["n_preds_ge_0.7"],
            d["n_small_preds_wh_lt_1024"], d["n_high_score_small_preds"],
            d["n_unlabeled_region_preds"], d["n_unlabeled_region_preds_ge_0.7"],
            ("%.4f" % d["max_unlabeled_region_score"])
            if d["max_unlabeled_region_score"] is not None else "—",
            d["n_tp"], d["n_dupe"], d["n_bkg"]))
    A("")
    band_counts = {}
    for r in rows:
        band_counts[r["score_band"]] = band_counts.get(r["score_band"], 0) + 1
    A("### 分数分档计数（全部 %d 个预测框）\n" % len(rows))
    A("| 档 | " + " | ".join(["ge_0.9", "0.7_0.9", "0.5_0.7", "0.3_0.5", "lt_0.3"]) + " |")
    A("|---|" + "---|" * 5)
    A("| 个数 | " + " | ".join(
        str(band_counts.get(b, 0)) for b in ["ge_0.9", "0.7_0.9", "0.5_0.7", "0.3_0.5", "lt_0.3"]) + " |")
    A("")
    A("## 3. 高分小框裁图索引（`crops_high_small/`）\n")
    if crops_hi:
        A("| 图 | 分数 | bbox(x,y,w,h) | 面积 | 与已标注 GT 最大 IoU | 类别 | 裁图 |")
        A("|---|---|---|---|---|---|---|")
        for c in crops_hi:
            A("| `%s` | %.3f | %s | %s | %s | %s | `%s` |" % (
                c["file_name"], float(c["score"]),
                ", ".join("%.1f" % v for v in c["bbox_xywh"]), c["bbox_wh_area"],
                c["max_gt_iou"], c["kind"], c["crop"]))
    else:
        A("（本折 3 张图上没有满足「分数 ≥ %g 且 w·h < 32²」的预测框。）" % HIGH_SCORE)
    A("")
    A("## 4. 未标注区域（与已标注 GT 最大 IoU < %g）上的预测\n" % UNLABELED_IOU_LT)
    A("口径：只说明**与已标注 GT 重叠很低**，**不等于空间上远离 GT**（很小的框落在大 GT 内 IoU 也可以很低），")
    A("也**不代表该处没有船**。裁图见 `crops_unlabeled/`（无条目时该目录为空）。\n")
    A("- 本折：与已标注 GT 匹配上的最高分数 = %s；未标注区域上的最高分数 = %s。"
      % (("%.4f" % summary["max_matched_tp_score_overall"])
         if summary["max_matched_tp_score_overall"] is not None else "（无 tp）",
         ("%.4f" % summary["max_unlabeled_region_score_overall"])
         if summary["max_unlabeled_region_score_overall"] is not None else "（无该档预测）"))
    A("- 两者之差是**分数分离度**的观测值；它**只在「模型见过这些图」的前提下成立**，"
      "不能外推到未见图像：同一区域在未见图像上完全可能得到远高于此的分数。\n")
    for fn, items in unlab_by_img.items():
        hi = [d for d in items if d["score"] >= HIGH_SCORE]
        A("### `%s` —— 未标注区域预测 %d 个（其中 ≥%g 的 %d 个，≥0.9 的 %d 个）"
          % (fn, len(items), HIGH_SCORE, len(hi), sum(1 for d in items if d["score"] >= 0.9)))
        gts_here = id2gt[id_by_fn[fn]]
        A("- 同图已标注 GT（%d 个）：%s" % (len(gts_here), json.dumps(
            [{"id": g["id"], "box_xyxy": [round(v, 2) for v in g["box"]],
              "ann_area": round(g["ann_area"], 2)} for g in gts_here], ensure_ascii=False)))
        A("")
        shown = hi if len(hi) >= 10 else items[:10]
        if not shown:
            A("（该图没有与已标注 GT 最大 IoU < %g 的预测框。）\n" % UNLABELED_IOU_LT)
            continue
        if len(shown) > len(hi):
            A("（表中另含 %d 个分数 < %g 的框；该图与已标注 GT 最大 IoU < %g 的预测共 %d 个，"
              "此处按分数降序列出前 %d 个。）\n"
              % (len(shown) - len(hi), HIGH_SCORE, UNLABELED_IOU_LT, len(items), len(shown)))
        A("| # | 分数 | x | y | w | h | 面积(w·h) | 与已标注 GT 最大 IoU | 类别 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for d in shown:
            b = d["bbox_xywh"]
            A("| %d | %.4f | %.1f | %.1f | %.1f | %.1f | %.1f | %.3f | %s |" % (
                d["pred_index"], d["score"], b[0], b[1], b[2], b[3],
                d["bbox_wh_area"], d["max_gt_iou_to_annotated_gt"], d["kind"]))
        A("")
    A("## 5. 判读口径（不得越界）\n")
    for r in summary["reading_rules"]:
        A("- %s" % r)
    A("- 目视判不清的，保留「**无法判断**」，不强行归类。")
    A("- 本产物**不得**用作开发集指标、模型选择依据或泛化证据。")
    A("")
    ea.write_text(osp.join(out_dir, "训练图诊断报告.md"), "\n".join(L))

    print("=" * 60)
    print("训练图诊断完成：fold %d" % args.fold)
    print("  预测数: %d（%d 图）" % (len(preds), len(id2gt)))
    print("  高分小框: %d  未标注区域≥0.7: %d"
          % (len(crops_hi), sum(1 for v in unlab_by_img.values()
                                for d in v if d["score"] >= HIGH_SCORE)))
    print("  权重 sha256 与正式评估记录一致: %s" % sha_matches_record)
    print("  产物: %s" % out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
