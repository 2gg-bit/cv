#!/usr/bin/env python
"""Stage 0 案例抽查抽样器（方案 proposal_small_bkg_roi_reweight_20260929.md §6）。

抽样总体：某划分上「legacy 判为背景 ∧ bbox 面积 < 32² ∧ 与同图所有 GT 最大 IoU < 0.1
∧ 分数 >= 0.7」的预测框（与报告 §5.6 的 high_score_small_bkg_iou_lt_0.1 同口径）。

分层：按分数 [0.7,0.9) 与 >=0.9 各抽 --per-band 个（默认 30）。
轮转：先按图去重取整轮（尽力做到"每图一个"），不足则进入下一轮；实际数量如实报出，不凑数。
确定性：仅用 numpy RandomState(--seed)，所有遍历顺序 sorted()。

匹配逻辑**不重写**，直接复用 error_analysis.py 的冻结原语（iou / load_gt /
load_predictions / legacy_match），避免与冻结口径分叉。

用法：
  python sample_stage0_cases.py --predictions P.json --gt G.json \
      --img-prefix /path/to/images --out-dir /path/to/out
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_analysis as ea  # noqa: E402  （只用其冻结原语，不执行其 CLI）

SMALL_WH = 32 * 32          # bbox 面积口径（w*h），与报告 §5.6 一致
MAXGT_IOU_LT = 0.1          # 只取"与已标注 GT 重叠很低"的档
SCORE_FLOOR = 0.7
BANDS = [("[0.7,0.9)", 0.7, 0.9), ("[0.9,1.0]", 0.9, float("inf"))]
DEFAULT_SEED = 20260929


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 0 案例抽查抽样器")
    p.add_argument("--predictions", required=True, help="predictions.bbox.json")
    p.add_argument("--gt", required=True, help="该划分的 GT 标注 json")
    p.add_argument("--img-prefix", required=True, help="图像目录")
    p.add_argument("--out-dir", required=True, help="输出目录（必须不存在或为空）")
    p.add_argument("--per-band", type=int, default=30, help="每个分数档抽几个（默认 30）")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--crop-pad", type=int, default=40, help="裁剪时四周外扩像素（原图尺度）")
    p.add_argument("--crop-scale", type=int, default=3, help="裁剪图放大倍数")
    p.add_argument("--allow-test-gt", action="store_true",
                   help="显式允许 --gt 指向 test.json（默认拒绝：最终测试集不参与 Stage 0）")
    return p.parse_args(argv)


def guard_test_set(gt_path, allow):
    base = os.path.basename(gt_path)
    if base == "test.json" and not allow:
        raise SystemExit(
            "拒绝：--gt 指向 %s。最终测试集不参与 Stage 0（方案 §4.3）。\n"
            "若确实要用它，请显式加 --allow-test-gt，并在记录中注明它已不是留出集。" % base)


def prepare_out_dir(path):
    if os.path.exists(path):
        existing = sorted(os.listdir(path))
        if existing:
            raise SystemExit("拒绝：输出目录已存在且非空，列出冲突项：\n  " + "\n  ".join(existing))
    else:
        os.makedirs(path)


def build_candidates(preds, id2gt):
    """返回按分数分档的候选列表；每项含抽样与判读所需字段。"""
    recs, _ = ea.legacy_match(preds, id2gt)
    n_gt_by_img = {img: len(id2gt.get(img, [])) for img in id2gt}
    out = {lbl: [] for lbl, _, _ in BANDS}
    for r in recs:
        if r["kind"] != "bkg":
            continue
        if r["score"] < SCORE_FLOOR:
            continue
        x, y, w, h = preds[r["pred_index"]]["bbox"]
        if w * h >= SMALL_WH:
            continue
        # legacy 的 best_iou 即「对该图所有 GT 的最大 IoU」
        if not (r["best_iou"] < MAXGT_IOU_LT):
            continue
        for lbl, lo, hi in BANDS:
            if lo <= r["score"] < hi:
                out[lbl].append({
                    "image_id": r["image_id"], "pred_index": r["pred_index"],
                    "score": float(r["score"]), "bbox": [float(v) for v in (x, y, w, h)],
                    "bbox_wh_area": float(w * h), "max_gt_iou": float(r["best_iou"]),
                    "n_gt_in_image": int(n_gt_by_img.get(r["image_id"], 0))})
                break
    for lbl in out:
        out[lbl].sort(key=lambda d: (d["image_id"], -d["score"], d["pred_index"]))
    return out


def sample_round_robin(cands, n, seed):
    """按图轮转取样：第一轮每图至多 1 个，之后逐轮补。确定性由 seed 决定图的顺序。"""
    by_img = {}
    for c in cands:                       # 已按 (image_id, -score, idx) 排好
        by_img.setdefault(c["image_id"], []).append(c)
    imgs = sorted(by_img)
    order = list(np.random.RandomState(seed).permutation(len(imgs)))
    order = [imgs[i] for i in order]
    picked, rnd = [], 0
    while len(picked) < n:
        added = False
        for img in order:
            if len(picked) >= n:
                break
            lst = by_img[img]
            if rnd < len(lst):
                picked.append(lst[rnd])
                added = True
        if not added:
            break
        rnd += 1
    return picked, len(by_img)


def draw_case(case, img_path, id2gt, pad, scale, dest):
    im = Image.open(img_path).convert("RGB")
    x, y, w, h = case["bbox"]
    x1, y1 = max(0, int(x) - pad), max(0, int(y) - pad)
    x2, y2 = min(im.width, int(x + w) + pad), min(im.height, int(y + h) + pad)
    crop = im.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(crop)
    for gt in id2gt.get(case["image_id"], []):     # 同图 GT：绿框
        gx1, gy1, gx2, gy2 = gt["box"]
        dr.rectangle([(gx1 - x1) * scale, (gy1 - y1) * scale,
                      (gx2 - x1) * scale, (gy2 - y1) * scale], outline=(0, 255, 0), width=2)
    dr.rectangle([(x - x1) * scale, (y - y1) * scale,
                  (x + w - x1) * scale, (y + h - y1) * scale], outline=(255, 0, 0), width=2)
    dr.text((4, 4), "score=%.3f maxIoU=%.3f nGT=%d" % (
        case["score"], case["max_gt_iou"], case["n_gt_in_image"]), fill=(255, 255, 0))
    crop.save(dest)


def main(argv=None):
    a = parse_args(argv)
    guard_test_set(a.gt, a.allow_test_gt)
    prepare_out_dir(a.out_dir)

    preds = ea.load_predictions(a.predictions)
    id2gt, images = ea.load_gt(a.gt)
    fn_by_id = {im["id"]: im["file_name"] for im in images}

    cands = build_candidates(preds, id2gt)
    rows, counts = [], {}
    cid = 0
    for lbl, _, _ in BANDS:
        picked, n_imgs = sample_round_robin(cands[lbl], a.per_band, a.seed)
        counts[lbl] = {"population": len(cands[lbl]), "sampled": len(picked),
                       "images_in_population": n_imgs,
                       "requested": a.per_band,
                       "shortfall": max(0, a.per_band - len(picked))}
        for c in picked:
            cid += 1
            fn = fn_by_id.get(c["image_id"], "")
            crop_rel = os.path.join("crops", "case_%03d.png" % cid)
            src = os.path.join(a.img_prefix, fn) if fn else ""
            if src and os.path.exists(src):
                os.makedirs(os.path.join(a.out_dir, "crops"), exist_ok=True)
                draw_case(c, src, id2gt, a.crop_pad, a.crop_scale,
                          os.path.join(a.out_dir, crop_rel))
            else:
                crop_rel = ""
            rows.append({
                "case_id": cid, "score_band": lbl, "image_id": c["image_id"],
                "file_name": fn or "", "score": "%.6f" % c["score"],
                "x": "%.2f" % c["bbox"][0], "y": "%.2f" % c["bbox"][1],
                "w": "%.2f" % c["bbox"][2], "h": "%.2f" % c["bbox"][3],
                "bbox_wh_area": "%.2f" % c["bbox_wh_area"],
                "max_gt_iou": "%.6f" % c["max_gt_iou"],
                "n_gt_in_image": c["n_gt_in_image"], "crop": crop_rel,
                # 人工判读列，留空待填
                "judgement": "", "notes": ""})

    ea.write_csv(os.path.join(a.out_dir, "cases.csv"), rows, [
        "case_id", "score_band", "image_id", "file_name", "score", "x", "y", "w", "h",
        "bbox_wh_area", "max_gt_iou", "n_gt_in_image", "crop", "judgement", "notes"])
    ea.write_json(os.path.join(a.out_dir, "sample_spec.json"), {
        "purpose": "Stage 0 案例抽查；判读类别 A 真实背景 / B 尺度·定位偏差 / "
                   "C 疑似漏标 / D 无法判断（方案 §6）",
        "population_definition": {
            "kind": "legacy 背景", "score_ge": SCORE_FLOOR,
            "bbox_wh_area_lt": SMALL_WH, "max_gt_iou_lt": MAXGT_IOU_LT},
        "bands": [b[0] for b in BANDS], "per_band": a.per_band, "seed": a.seed,
        "sampling": "按图轮转：第一轮每图至多 1 个，之后逐轮补；不足如实报出，不凑数",
        "inputs": {"predictions": a.predictions, "gt": a.gt,
                   "gt_is_test_set": os.path.basename(a.gt) == "test.json"},
        "counts": counts,
        "no_auto_threshold": "不设任意占比自动判'通过'；疑似漏标（C）逐个核查",
    })
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print("完成。产物目录：%s" % a.out_dir)


if __name__ == "__main__":
    main()
