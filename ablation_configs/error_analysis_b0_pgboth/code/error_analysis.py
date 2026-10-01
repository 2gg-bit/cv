#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三折新 B0 与 PG-both 离线误差分析。

只读六组 predictions.bbox.json + 各自 test.json，纯 numpy + pycocotools；
不导入 ssod / mmdet / mmcv / torch，因此不需要 checkout 钉住。

性质：探索性分析，分析规则冻结（**不是**对未见数据的预注册）。
规则见同目录 analysis_plan_r2_20260929.md；v1 analysis_plan.md 仅存历史，其两套判据已撤除。
本脚本**不指定共同瓶颈、不下机制裁定**。

用法：
    python error_analysis.py run --out-dir <dir>
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ---------------------------------------------------------------- 常量（分析规则冻结，非盲态预注册）
IOU_THR = 0.5
AP75_IOU = 0.75
NEAR_IOU = 0.3
THRESHOLDS = (0.9, 0.8, 0.7, 0.5)
BANDS = [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6)]
MISS_IOU_BANDS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
# 预测侧联合统计用分档（面积 × 分数 × 该预测对同图所有 GT 的最大 IoU）
PRED_SCORE_BANDS = [(0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.0, 0.5)]
MAXGT_IOU_BANDS = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.01)]
PERM_SEED = 0
N_PERM = 10000
FORBIDDEN_IMPORTS = {"ssod", "mmdet", "mmcv", "mmcv_full", "torch"}

# 性质与措辞边界（写死在脚本里，避免报告口径漂移）
ANALYSIS_NATURE = ("探索性分析，分析规则冻结"
                   "（部分结果在撰写规则文件之前已被查看，因此不是对未见数据的预注册）")
MECHANISM_CONCLUSION = "现有证据无法裁定"

FROZEN_TEST_SHA = "19aa601904243be548c1551b247f331f1317dc6a9bf4ac9911e6b07398ca919f"
FROZEN_NUM_IMAGES = 232
FROZEN_NUM_GT = 546

ABLATION_CONFIGS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FROZEN_SCRIPT = "/home/xcc/桌面/snew/sar_opt_d3t_deploy/1资料/09_M2FG评估结果_fold6/score_thr_bkg.py"
PLAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_plan_r2_20260929.md")
PLAN_V1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_plan.md")

RUN_SPECS = [
    {"run_id": "fold6_b0", "fold": 6, "variant": "b0",
     "eval_dir": os.path.join(ABLATION_CONFIGS, "fold6_seed678", "b0_evaluation")},
    {"run_id": "fold6_pg_both", "fold": 6, "variant": "pg_both",
     "eval_dir": os.path.join(ABLATION_CONFIGS, "fold6_seed678", "pg_both_evaluation")},
    {"run_id": "fold7_b0", "fold": 7, "variant": "b0",
     "eval_dir": os.path.join(ABLATION_CONFIGS, "fold7_seed678", "b0_evaluation")},
    {"run_id": "fold7_pg_both", "fold": 7, "variant": "pg_both",
     "eval_dir": os.path.join(ABLATION_CONFIGS, "fold7_seed678", "pg_both_evaluation")},
    {"run_id": "fold8_b0", "fold": 8, "variant": "b0",
     "eval_dir": os.path.join(ABLATION_CONFIGS, "fold8_seed678", "b0_evaluation")},
    {"run_id": "fold8_pg_both", "fold": 8, "variant": "pg_both",
     "eval_dir": os.path.join(ABLATION_CONFIGS, "fold8_seed678", "pg_both_evaluation")},
]

B0_RUN_IDS = ["fold6_b0", "fold7_b0", "fold8_b0"]
MODE_ORDER = ["bkg", "miss", "imprecise", "dupe"]


class AnalysisError(Exception):
    pass


# ---------------------------------------------------------------- 基础工具
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_imports(path):
    """用 AST 遍历 Import / ImportFrom 节点，禁止导入 ssod/mmdet/mmcv/torch。"""
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    found, top_level = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                top_level.add(top)
                if top in FORBIDDEN_IMPORTS:
                    found.append({"lineno": node.lineno, "module": alias.name})
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            top_level.add(top)
            if top in FORBIDDEN_IMPORTS:
                found.append({"lineno": node.lineno, "module": node.module})
    if found:
        raise AnalysisError("禁止的导入: %s" % json.dumps(found, ensure_ascii=False))
    return {"file": path, "ok": True, "forbidden_imports_found": found,
            "top_level_modules": sorted(m for m in top_level if m)}


def iou(a, b):
    """与 score_thr_bkg.py:33-38 逐字等价（无 +1）。"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(gt_path):
    """保持标注顺序（getAnnIds 按 id 升序），跳过 iscrowd，对齐 score_thr_bkg.py:19-30。"""
    coco = COCO(gt_path)
    id2gt, images = {}, []
    for im in coco.dataset["images"]:
        images.append({"id": im["id"], "file_name": im.get("file_name", "")})
        gts = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=im["id"])):
            if a.get("iscrowd", 0):
                continue
            x, y, w, h = a["bbox"]
            gts.append({"id": a["id"], "box": [x, y, x + w, y + h],
                        "ann_area": float(a.get("area", w * h)), "bbox_area": float(w * h)})
        id2gt[im["id"]] = gts
    return id2gt, images


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    for k, p in enumerate(preds):
        if not isinstance(p.get("image_id"), int):
            raise AnalysisError("预测 %d 的 image_id 非整数" % k)
        bbox = p.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise AnalysisError("预测 %d 的 bbox 非四元" % k)
        if not (bbox[2] > 0 and bbox[3] > 0):
            raise AnalysisError("预测 %d 的 w/h 非正: %s" % (k, bbox))
        if not isinstance(p.get("score"), (int, float)):
            raise AnalysisError("预测 %d 的 score 非数" % k)
    return preds


def write_json(path, obj):
    if os.path.exists(path):
        raise AnalysisError("拒绝覆盖已有文件: %s" % path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_text(path, text):
    if os.path.exists(path):
        raise AnalysisError("拒绝覆盖已有文件: %s" % path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv(path, rows, fieldnames):
    if os.path.exists(path):
        raise AnalysisError("拒绝覆盖已有文件: %s" % path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def size_bin(area):
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def band_label(bands, value):
    """左闭右开；落在所有档之外（含 NaN）时返回 None，避免静默归入某一档。"""
    if value is None or not np.isfinite(value):
        return None
    for lo, hi in bands:
        if lo <= value < hi:
            return "[%g,%g)" % (lo, hi)
    return None


def quantiles(values):
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)), "q3": float(np.percentile(arr, 75))}


# ---------------------------------------------------------------- 两种匹配语义
def legacy_match(preds, id2gt):
    """复刻冻结贪心口径。返回 (pred_recs, gt_state)。

    并列打破：预测用稳定排序（同分保持文件原序）；GT argmax 用严格 >，同 IoU 取最小下标。
    """
    by_img = defaultdict(list)
    for idx, p in enumerate(preds):
        x, y, w, h = p["bbox"]
        by_img[p["image_id"]].append((idx, [x, y, x + w, y + h], float(p["score"])))
    pred_recs = []
    gt_state = {}
    for img_id in sorted(by_img):
        items = by_img[img_id]
        gts = id2gt[img_id]
        matched = [False] * len(gts)
        order = sorted(range(len(items)), key=lambda i: -items[i][2])
        for i in order:
            idx, box, score = items[i]
            best_j, best_iou = -1, 0.0
            for j, gt in enumerate(gts):
                v = iou(box, gt["box"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= IOU_THR:
                if not matched[best_j]:
                    matched[best_j] = True
                    rec = {"pred_index": idx, "image_id": img_id, "score": score, "kind": "tp",
                           "best_iou": best_iou, "matched_gt_id": gts[best_j]["id"]}
                    gt_state[(img_id, gts[best_j]["id"])] = {
                        "occupied": True, "matched_pred_iou": best_iou,
                        "matched_pred_score": score, "matched_pred_index": idx}
                else:
                    rec = {"pred_index": idx, "image_id": img_id, "score": score, "kind": "dupe",
                           "best_iou": best_iou, "matched_gt_id": gts[best_j]["id"]}
            else:
                rec = {"pred_index": idx, "image_id": img_id, "score": score, "kind": "bkg",
                       "best_iou": best_iou, "matched_gt_id": None}
            pred_recs.append(rec)
    for img_id in sorted(id2gt):
        for gt in id2gt[img_id]:
            gt_state.setdefault((img_id, gt["id"]), {
                "occupied": False, "matched_pred_iou": None,
                "matched_pred_score": None, "matched_pred_index": None})
    return pred_recs, gt_state


def classify(preds, id2gt):
    """与 score_thr_bkg.py:41-67 同口径的投影：(score, kind) 序列。"""
    recs, _ = legacy_match(preds, id2gt)
    return [(r["score"], r["kind"]) for r in recs]


def summarize_legacy(preds, id2gt):
    """逐字复刻 score_thr_bkg.py:70-94，键名不变。"""
    recs = classify(preds, id2gt)
    n = {"tp": 0, "dupe": 0, "bkg": 0}
    for _, k in recs:
        n[k] += 1
    res = {
        "num_preds": len(preds),
        "background_errors": n["bkg"],
        "duplicates": n["dupe"],
        "true_positives": n["tp"],
    }
    for t in THRESHOLDS:
        key = "ge%g" % t
        res["preds_" + key] = sum(1 for s, _ in recs if s >= t)
        res["bkg_" + key] = sum(1 for s, k in recs if k == "bkg" and s >= t)
        res["tp_" + key] = sum(1 for s, k in recs if k == "tp" and s >= t)
    res["bkg_bins"] = {
        "[%g,%g)" % (lo, hi): sum(1 for s, k in recs if k == "bkg" and lo <= s < hi)
        for lo, hi in BANDS
    }
    res["bkg_below_0.5"] = sum(1 for s, k in recs if k == "bkg" and s < 0.5)
    return res


def coverage_map(preds, id2gt):
    """逐 GT 任意框覆盖视图（含未被 greedy 分配者）。"""
    by_img = defaultdict(list)
    for p in preds:
        x, y, w, h = p["bbox"]
        by_img[p["image_id"]].append(([x, y, x + w, y + h], float(p["score"])))
    cov = {}
    for img_id in sorted(id2gt):
        items = by_img.get(img_id, [])
        for gt in id2gt[img_id]:
            best_iou, best_score, n_ge = 0.0, None, 0
            for box, score in items:
                v = iou(box, gt["box"])
                if v > best_iou:
                    best_iou, best_score = v, score
                if v >= IOU_THR:
                    n_ge += 1
            cov[(img_id, gt["id"])] = {"best_iou_any": best_iou,
                                       "best_pred_score_any": best_score,
                                       "n_preds_iou_ge_0_5": n_ge}
    return cov


# ---------------------------------------------------------------- 逐 GT 特征
def gt_features(id2gt):
    feats = {}
    for img_id in sorted(id2gt):
        gts = id2gt[img_id]
        centers = [((g["box"][0] + g["box"][2]) / 2.0, (g["box"][1] + g["box"][3]) / 2.0) for g in gts]
        for j, g in enumerate(gts):
            nn = None
            for k, c in enumerate(centers):
                if k == j:
                    continue
                d = float(np.hypot(centers[j][0] - c[0], centers[j][1] - c[1]))
                if nn is None or d < nn:
                    nn = d
            feats[(img_id, g["id"])] = {
                "ann_area": g["ann_area"], "bbox_area": g["bbox_area"],
                "size_bin_coco": size_bin(g["ann_area"]), "size_bin_bbox": size_bin(g["bbox_area"]),
                "cx": centers[j][0], "cy": centers[j][1],
                "nn_center_dist": nn, "n_gt_in_image": len(gts),
                "nn_dist_over_sqrt_area": (nn / np.sqrt(g["ann_area"])) if (nn is not None and g["ann_area"] > 0) else None,
            }
    return feats


# ---------------------------------------------------------------- 合成用例
def _mk_gt(spec):
    id2gt = {}
    for img_id, boxes in spec:
        id2gt[img_id] = [{"id": gid, "box": list(b), "ann_area": float((b[2] - b[0]) * (b[3] - b[1])),
                          "bbox_area": float((b[2] - b[0]) * (b[3] - b[1]))} for gid, b in boxes]
    return id2gt


def _mk_pred(items):
    return [{"image_id": im, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s, "category_id": 0}
            for im, b, s in items]


def synthetic_cases():
    """合成用例：两种匹配语义各自符合自身定义，不要求二者输出相同。"""
    out = []

    def run(name, preds, id2gt, expect):
        precs, gstate = legacy_match(preds, id2gt)
        cov = coverage_map(preds, id2gt)
        summ = summarize_legacy(preds, id2gt)
        got = {
            "num_preds": summ["num_preds"],
            "legacy_background_errors": summ["background_errors"],
            "legacy_duplicates": summ["duplicates"],
            "legacy_true_positives": summ["true_positives"],
            "identity_ok": summ["background_errors"] + summ["duplicates"] + summ["true_positives"] == summ["num_preds"],
            "tp_pred_indices": sorted(r["pred_index"] for r in precs if r["kind"] == "tp"),
            "matched_gt_ids": sorted(r["matched_gt_id"] for r in precs if r["kind"] == "tp"),
            "occupied_gt_count": sum(1 for v in gstate.values() if v["occupied"]),
            "coverage_covered_at_50": sum(1 for v in cov.values() if v["best_iou_any"] >= IOU_THR),
        }
        for key, want in expect.items():
            if got[key] != want:
                raise AnalysisError("合成用例 %s 断言失败: %s 期望 %r 实得 %r" % (name, key, want, got[key]))
        out.append({"case": name, "expected": expect, "got": got, "ok": True})

    # A 空预测
    run("empty_predictions", _mk_pred([]), _mk_gt([(1, [(11, [0, 0, 10, 10])])]),
        {"num_preds": 0, "legacy_background_errors": 0, "legacy_true_positives": 0,
         "occupied_gt_count": 0, "coverage_covered_at_50": 0})

    # B 该图无 GT
    run("image_without_gt", _mk_pred([(1, [0, 0, 10, 10], 0.9)]), _mk_gt([(1, [])]),
        {"num_preds": 1, "legacy_background_errors": 1, "legacy_true_positives": 0,
         "occupied_gt_count": 0})

    # C 同分同 IoU：稳定排序 → 先入文件者得 tp，后者为 dupe
    run("equal_score_equal_iou", _mk_pred([(1, [0, 0, 10, 10], 0.5), (1, [0, 0, 10, 10], 0.5)]),
        _mk_gt([(1, [(11, [0, 0, 10, 10])])]),
        {"legacy_true_positives": 1, "legacy_duplicates": 1, "tp_pred_indices": [0]})

    # D GT argmax 同 IoU 取最小下标；另一 GT 因占用而漏检，且被覆盖（stolen）
    run("gt_argmax_tie_lowest_index",
        _mk_pred([(1, [0, 0, 10, 10], 0.9)]),
        _mk_gt([(1, [(100, [0, 0, 10, 10]), (101, [0, 0, 10, 10])])]),
        {"legacy_true_positives": 1, "matched_gt_ids": [100], "occupied_gt_count": 1,
         "coverage_covered_at_50": 2})

    # E1 两条预测争抢同一个 GT，另一个 GT 被几何覆盖却未被占用
    #    GT1=[0,0,10,10] GT2=[0,0,10,20]；X 对 GT1 IoU=1.0 而对 GT2 =0.5 → 取 GT1
    #    Y 对 GT1 IoU=0.833 > 对 GT2 IoU=0.6 → argmax 仍是已被占用的 GT1 → dupe
    run("preds_contesting_one_gt_other_covered",
        _mk_pred([(1, [0, 0, 10, 10], 0.9), (1, [0, 0, 10, 12], 0.8)]),
        _mk_gt([(1, [(1, [0, 0, 10, 10]), (2, [0, 0, 10, 20])])]),
        {"legacy_true_positives": 1, "legacy_duplicates": 1, "matched_gt_ids": [1],
         "occupied_gt_count": 1, "coverage_covered_at_50": 2})

    # E2 两条预测各取一个 GT（互不争抢），两个 GT 都被占用
    run("two_preds_take_two_gts",
        _mk_pred([(1, [0, 0, 10, 10], 0.9), (1, [20, 0, 30, 10], 0.8)]),
        _mk_gt([(1, [(1, [0, 0, 10, 10]), (2, [20, 0, 30, 10])])]),
        {"legacy_true_positives": 2, "legacy_duplicates": 0, "matched_gt_ids": [1, 2],
         "occupied_gt_count": 2, "coverage_covered_at_50": 2})

    # F 一条预测几何上覆盖多个 GT：只占用其 argmax，另一个 GT 未占用但被覆盖
    run("one_pred_covers_two_gts",
        _mk_pred([(1, [0, 0, 10, 10], 0.9)]),
        _mk_gt([(1, [(1, [0, 0, 10, 10]), (2, [2, 0, 12, 10])])]),
        {"legacy_true_positives": 1, "matched_gt_ids": [1], "occupied_gt_count": 1,
         "coverage_covered_at_50": 2})

    # G 反例(a)：某 GT 被占用，但被分配的框不是对它 IoU 最高的框
    #    X=[0,0,10,8] 与 A=[0,0,10,8] 完全重合 → 占用 A；Y=[0,0,6,10] 的 argmax 是 GT=[0,0,10,10]
    #    但 IoU(X, GT)=0.8 > IoU(Y, GT)=0.6
    id2gt_g = _mk_gt([(1, [(1, [0, 0, 10, 8]), (2, [0, 0, 10, 10])])])
    preds_g = _mk_pred([(1, [0, 0, 10, 8], 0.9), (1, [0, 0, 6, 10], 0.8)])
    precs_g, gstate_g = legacy_match(preds_g, id2gt_g)
    cov_g = coverage_map(preds_g, id2gt_g)
    key = (1, 2)
    if not (gstate_g[key]["matched_pred_iou"] < cov_g[key]["best_iou_any"]):
        raise AnalysisError("合成用例 assigned_box_is_not_best_iou_any 断言失败")
    out.append({"case": "assigned_box_is_not_best_iou_any", "ok": True,
                "expected": {"matched_pred_iou < best_iou_any": True},
                "got": {"matched_pred_iou": gstate_g[key]["matched_pred_iou"],
                        "best_iou_any": cov_g[key]["best_iou_any"],
                        "matched_gt_ids": sorted(r["matched_gt_id"] for r in precs_g if r["kind"] == "tp")}})

    # H 反例(b)：GT 被某框覆盖（best_iou_any ≥ 0.5）却未被占用 → miss_stolen（用例 D 的结构）
    return out


# ---------------------------------------------------------------- 影子对照
def _extract_json(stdout):
    """score_thr_bkg.py 可能在 JSON 之前打印 pycocotools 提示：取首个 { 到最后一个 }。"""
    start, end = stdout.find("{"), stdout.rfind("}")
    if start < 0 or end <= start:
        raise AnalysisError("冻结脚本 stdout 中未找到 JSON 片段")
    return json.loads(stdout[start:end + 1])


def shadow_frozen_script(gt_path, runs):
    if not os.path.exists(FROZEN_SCRIPT):
        raise AnalysisError("冻结脚本不存在: %s" % FROZEN_SCRIPT)
    specs = ["%s=%s" % (r["run_id"], os.path.join(r["eval_dir"], "predictions.bbox.json")) for r in runs]
    proc = subprocess.run([sys.executable, FROZEN_SCRIPT, gt_path] + specs,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0:
        raise AnalysisError("冻结脚本退出码 %d\nstderr: %s" % (proc.returncode, proc.stderr[-2000:]))
    ref = _extract_json(proc.stdout)
    per_run, all_equal = {}, True
    for r in runs:
        rid = r["run_id"]
        mine = r["legacy"]
        their = ref["runs"][rid]
        keys = sorted(set(their) & set(mine))
        cmp = {}
        for k in keys:
            equal = their[k] == mine[k]
            all_equal = all_equal and equal
            if not equal:
                cmp[k] = {"ref": their[k], "new": mine[k], "equal": False}
        per_run[rid] = {"compared_keys": keys, "mismatches": cmp,
                        "equal": len(cmp) == 0}
    if not all_equal:
        raise AnalysisError("影子对照不一致: %s" % json.dumps(per_run, ensure_ascii=False))
    return {"frozen_script": FROZEN_SCRIPT, "frozen_script_sha256": sha256_file(FROZEN_SCRIPT),
            "frozen_script_stdout_json_offset": proc.stdout.find("{"),
            "per_run": per_run, "all_equal": all_equal,
            "note": ("影子对照只在 legacy_* 命名空间内进行；既有 09_/11_/12_ 的 JSON 为 186 图 dev 集"
                     " + Soft-NMS，仅作口径规范，不作基线 diff。")}


# ---------------------------------------------------------------- COCO 复算
def coco_recompute(preds, gt_path, images, iou_type="bbox"):
    """复刻 mmdet 2.16 CocoDataset.evaluate 的 COCO 参数。"""
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.params.catIds = sorted(coco_gt.getCatIds())
    ev.params.imgIds = sorted(coco_gt.getImgIds())
    ev.params.maxDets = [100, 300, 1000]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = [None if s is None else float(s) for s in ev.stats]
    mapping = {0: "bbox_mAP", 1: "bbox_mAP_50", 2: "bbox_mAP_75", 3: "bbox_mAP_s", 4: "bbox_mAP_m",
               5: "bbox_mAP_l", 6: "bbox_AR@100", 7: "bbox_AR@300", 8: "bbox_AR@1000",
               9: "bbox_AR_s@1000", 10: "bbox_AR_m@1000", 11: "bbox_AR_l@1000"}
    raw = {mapping[i]: stats[i] for i in mapping}

    # TP/FP/ignore 分解（areaRng='all', IoU 0.5）
    # 本机 pycocotools 的 evaluate() 只按 maxDets[-1] 生成逐图结果，存于 evalImgs，
    # 索引 = a*K + i（无 maxDet 维）；accumulate() 再按 [0:maxDet] 切片。
    # 因每图预测数 <= max_per_img = 100，各 maxDet 档结果一致（由 max_dets_per_image 记录佐证）。
    K = len(ev.params.imgIds)
    a_all = ev.params.areaRngLbl.index("all")
    t_50 = int(np.argmin(np.abs(np.asarray(ev.params.iouThrs) - IOU_THR)))
    tp = fp = ig = 0
    max_dets_per_image = 0
    for i_img in range(K):
        entry = ev.evalImgs[a_all * K + i_img]
        if entry is None:
            continue
        dtm = np.asarray(entry["dtMatches"])
        dtig = np.asarray(entry["dtIgnore"])
        max_dets_per_image = max(max_dets_per_image, int(dtm.shape[1]))
        for d in range(dtm.shape[1]):
            if dtig[t_50, d]:
                ig += 1
            elif dtm[t_50, d] != 0:
                tp += 1
            else:
                fp += 1
    decomp = {"iou_thr": IOU_THR, "areaRng": "all", "maxDet": 100,
              "maxDet_effective_per_image": int(ev.params.maxDets[-1]),
              "max_dets_per_image": max_dets_per_image,
              "coco_true_positives": tp, "coco_false_positives": fp, "coco_ignored": ig,
              "coco_detections_total": tp + fp + ig}

    # 结构断言：逐图结果基于 maxDets[-1]，只在每图预测数不超过该档时才与 maxDet=100 等价
    if max_dets_per_image > 100:
        raise AnalysisError("每图预测数 %d > 100，maxDet=100 档需另行截断，当前口径不可直接使用"
                            % max_dets_per_image)

    # ---- 逐 GT 的 COCO 自身匹配（新/丢失/保持/从未）与 PR 曲线 ----
    # dtMatches / gtMatches 是 evaluateImg 用 maxDet = maxDets[-1] 的结果；
    # 本机 pycocotools 的 dt 在该处已按 -score 稳定排序，故列/行顺序与 ev.ious 一致。
    dt_bbox = {int(d["id"]): d["bbox"] for d in coco_dt.dataset["annotations"]}
    gt_bbox = {int(a["id"]): a["bbox"] for a in coco_gt.dataset["annotations"] if not a.get("iscrowd", 0)}
    iou_thrs = np.asarray(ev.params.iouThrs, dtype=float)
    levels = (("0.50", IOU_THR), ("0.75", AP75_IOU))
    t_idx = {lbl: int(np.argmin(np.abs(iou_thrs - thr))) for lbl, thr in levels}
    per_gt_matches = {lbl: {} for lbl, _ in levels}
    n_pairs = n_impl_diff = 0
    max_impl_diff = 0.0
    for i_img in range(K):
        entry = ev.evalImgs[a_all * K + i_img]
        if entry is None:
            continue
        img_id = int(ev.params.imgIds[i_img])
        cat_id = int(entry["category_id"])
        gt_ids = [int(x) for x in np.asarray(entry["gtIds"]).reshape(-1)]
        dt_ids = [int(x) for x in np.asarray(entry["dtIds"]).reshape(-1)]
        gtm = np.asarray(entry["gtMatches"], dtype=float)
        dts = np.asarray(entry["dtScores"], dtype=float)
        raw_ious = ev.ious.get((img_id, cat_id))
        ious = None
        if raw_ious is not None and len(raw_ious) != 0:
            arr = np.asarray(raw_ious, dtype=float)
            ious = arr if arr.ndim == 2 else None
        for g, gt_id in enumerate(gt_ids):
            for lbl, thr in levels:
                m = float(gtm[t_idx[lbl], g])
                dt_id = int(m) if m > 0 else None
                score = None
                if dt_id is not None and dt_id in dt_ids:
                    score = float(dts[dt_ids.index(dt_id)])
                per_gt_matches[lbl]["%d|%d" % (img_id, gt_id)] = {
                    "matched_dt_id": dt_id, "matched_dt_score": score,
                    "n_dt_ge_thr": int((ious[:, g] >= thr).sum()) if ious is not None else 0,
                    "gt_area": gt_bbox.get(gt_id, [0, 0, 0, 0])[2] * gt_bbox.get(gt_id, [0, 0, 0, 0])[3]}
        # 两种 IoU 实现的实测差异（用同一批框对；不假定一致）
        if ious is not None:
            for d, dt_id in enumerate(dt_ids):
                if dt_id not in dt_bbox:
                    continue
                x, y, w, h = dt_bbox[dt_id]
                for g, gt_id in enumerate(gt_ids):
                    if gt_id not in gt_bbox:
                        continue
                    gx, gy, gw, gh = gt_bbox[gt_id]
                    mine = iou([x, y, x + w, y + h], [gx, gy, gx + gw, gy + gh])
                    diff = abs(mine - float(ious[d, g]))
                    n_pairs += 1
                    if diff > 1e-9:
                        n_impl_diff += 1
                    if diff > max_impl_diff:
                        max_impl_diff = diff

    rec_thrs = [float(x) for x in np.asarray(ev.params.recThrs, dtype=float)]
    rec_arr = np.asarray(ev.eval["recall"], dtype=float)
    m_100 = int(np.argmin(np.abs(np.asarray(ev.params.maxDets, dtype=float) - 100)))
    # 交叉校验：最大可达召回必须等于「该档匹配上的 GT 数 / 非 crowd GT 数」。
    # 两条量来自不同代码路径（ev.eval['recall'] 与 evalImgs 的 gtMatches），不等即说明口径或索引取错
    n_gt_for_recall = sum(1 for a in coco_gt.dataset["annotations"] if not a.get("iscrowd", 0))
    pr_curves = {}
    for lbl, _ in levels:
        prec = np.asarray(ev.eval["precision"], dtype=float)[t_idx[lbl], :, 0, a_all, m_100]
        vals = [None if (v < 0 or not np.isfinite(v)) else float(v) for v in prec]
        finite = [v for v in vals if v is not None]
        # 该 IoU 档真正达到的最大召回：取 COCOeval 自己算的 recall，不靠精度数组的尾部形态推断
        max_rec = float(rec_arr[t_idx[lbl], 0, a_all, m_100])
        n_matched = sum(1 for v in per_gt_matches[lbl].values() if v["matched_dt_id"] is not None)
        if abs(max_rec - n_matched / float(n_gt_for_recall)) > 1e-9:
            raise AnalysisError(
                "最大可达召回交叉校验失败（IoU %s）：ev.eval['recall']=%.6f 但匹配上的 GT 数 %d / %d = %.6f"
                % (lbl, max_rec, n_matched, n_gt_for_recall,
                   n_matched / float(n_gt_for_recall)))
        # 本机 pycocotools 把「超过最大可达召回」的精度存为 0.0，而非 COCO 原版的 -1；
        # 因此不假定 -1，而是实测负值/零值个数，并按召回上限截断求均值
        n_neg = sum(1 for v in vals if v is not None and v < 0)
        n_zero = sum(1 for v in vals if v == 0.0)
        within = [v for v, r in zip(vals, rec_thrs)
                  if v is not None and r <= max_rec + 1e-12]
        pr_curves[lbl] = {
            "recall_grid": rec_thrs, "precision": vals,
            "max_recall_all_dets": max_rec,
            "max_recall_all_dets_source": "ev.eval['recall'][IoU档, cat 0, areaRng='all', maxDet=100]",
            "precision_beyond_max_recall_marker": (
                "-1（COCO 原版）" if n_neg else
                "0.0（本机实测无 -1；超过最大可达召回的召回点精度存为 0.0，不等于「精度为 0」）"),
            "n_precision_negative": n_neg,
            "n_precision_exact_zero": n_zero,
            "mean_precision_within_reached_recall": (float(np.mean(within)) if within else None),
            "mean_precision_full_grid": (float(np.mean(finite)) if finite else None)}

    impl_agreement = {
        "note": "pycocotools 自有 IoU 与冻结脚本手写 iou() 的实测差异；两者并列呈现，不合并、不假定一致",
        "pairs_compared": n_pairs, "pairs_differing_gt_1e-9": n_impl_diff,
        "share_differing": (n_impl_diff / float(n_pairs)) if n_pairs else None,
        "max_abs_diff": max_impl_diff}
    return {"raw": raw, "detection_decomposition": decomp, "per_gt_matches": per_gt_matches,
            "pr_curves": pr_curves, "pr_grid": rec_thrs, "iou_impl_agreement": impl_agreement}


def coco_match_change(matches_b0, matches_pg, label_b0, label_pg, iou_label):
    """按 COCO 自身匹配（非几何覆盖）对照两次运行的逐 GT 匹配状态。

    状态：newly_matched / lost_match / retained / never。
    对状态发生变化的 GT，进一步区分"该侧是否本来就没有 >= 阈值的框"：
      - 丢失且 pg 侧无 >= 阈值的框 → no_qualifying_box_in_pg
      - 丢失但 pg 侧有 >= 阈值的框 → qualifying_box_present_but_assigned_elsewhere（一分一配的排序效应）
    """
    keys = sorted(set(matches_b0) | set(matches_pg))
    counts = {"newly_matched": 0, "lost_match": 0, "retained": 0, "never": 0}
    lost_reason = {"no_qualifying_box_in_pg": 0, "qualifying_box_present_but_assigned_elsewhere": 0}
    new_reason = {"no_qualifying_box_in_b0": 0, "qualifying_box_present_but_assigned_elsewhere": 0}
    rows = []
    for k in keys:
        a, b = matches_b0.get(k, {}), matches_pg.get(k, {})
        da, db = a.get("matched_dt_id"), b.get("matched_dt_id")
        if da is None and db is not None:
            state = "newly_matched"
            new_reason["no_qualifying_box_in_b0" if a.get("n_dt_ge_thr", 0) == 0
                       else "qualifying_box_present_but_assigned_elsewhere"] += 1
        elif da is not None and db is None:
            state = "lost_match"
            lost_reason["no_qualifying_box_in_pg" if b.get("n_dt_ge_thr", 0) == 0
                        else "qualifying_box_present_but_assigned_elsewhere"] += 1
        elif da is not None and db is not None:
            state = "retained"
        else:
            state = "never"
        counts[state] += 1
        # 状态变化才逐条列出（never/retained 共几百条，不逐条铺开）
        if state in ("newly_matched", "lost_match"):
            img_id, gt_id = (int(x) for x in k.split("|"))
            rows.append({
                "iou_level": iou_label, "image_id": img_id, "gt_id": gt_id, "state": state,
                "matched_in_b0": int(da is not None), "matched_in_pg": int(db is not None),
                "matched_dt_score_b0": a.get("matched_dt_score"),
                "matched_dt_score_pg": b.get("matched_dt_score"),
                "n_dt_ge_thr_b0": a.get("n_dt_ge_thr"), "n_dt_ge_thr_pg": b.get("n_dt_ge_thr"),
                "gt_bbox_area": a.get("gt_area", b.get("gt_area"))})
    # 保持匹配者：匹配到的框分数是否变化（框 id 跨 run 不可比，用分数判断是否同一个导出框）
    retained_same = retained_changed = 0
    for k in keys:
        a, b = matches_b0.get(k, {}), matches_pg.get(k, {})
        if a.get("matched_dt_id") is None or b.get("matched_dt_id") is None:
            continue
        if a.get("matched_dt_score") == b.get("matched_dt_score"):
            retained_same += 1
        else:
            retained_changed += 1
    return {"iou_level": iou_label, "labels": [label_b0, label_pg], "counts": counts,
            "lost_reason": lost_reason, "new_reason": new_reason,
            "retained_same_matched_score": retained_same, "retained_changed_matched_score": retained_changed,
            "changed_rows": rows,
            "note": ("COCO 自身匹配（一分一配贪心，maxDet=100，areaRng=all）。"
                     "与'几何覆盖翻转'是两个不同口径，不得互相解释。")}


def compare_metrics(raw, stored):
    """按 metrics.json 的保存精度（三位小数）比对，不一律容忍 0.001。"""
    rows, ok = {}, True
    for k in sorted(stored):
        if k not in raw:
            continue
        r = raw[k]
        same = r is not None and round(float(r), 3) == round(float(stored[k]), 3)
        ok = ok and same
        rows[k] = {"recomputed_raw": r, "recomputed_3dp": None if r is None else round(float(r), 3),
                   "stored": stored[k], "equal_at_stored_precision": same}
    if not ok:
        bad = {k: v for k, v in rows.items() if not v["equal_at_stored_precision"]}
        raise AnalysisError("COCO 复算与 metrics.json 在保存精度下不一致: %s" % json.dumps(bad, ensure_ascii=False))
    return {"per_key": rows, "all_equal_at_stored_precision": ok}


# ---------------------------------------------------------------- 逐 run 汇总
def build_run(run, id2gt, images, feats):
    eval_dir = run["eval_dir"]
    pred_path = os.path.join(eval_dir, "predictions.bbox.json")
    preds = load_predictions(pred_path)
    legacy = summarize_legacy(preds, id2gt)
    pred_recs, gstate = legacy_match(preds, id2gt)
    cov = coverage_map(preds, id2gt)

    # 完整性断言
    if (legacy["background_errors"] + legacy["duplicates"] + legacy["true_positives"]) != legacy["num_preds"]:
        raise AnalysisError("%s: legacy 计数恒等式不成立" % run["run_id"])
    meta = json.load(open(os.path.join(eval_dir, "metadata.json"), "r", encoding="utf-8"))
    if legacy["num_preds"] != meta.get("num_predictions"):
        raise AnalysisError("%s: num_preds %d != metadata.num_predictions %s"
                            % (run["run_id"], legacy["num_preds"], meta.get("num_predictions")))
    stored = json.load(open(os.path.join(eval_dir, "metrics.json"), "r", encoding="utf-8"))
    coco = coco_recompute(preds, os.path.join(eval_dir, "test.json"), images)
    metric_check = compare_metrics(coco["raw"], stored)
    d0 = coco["detection_decomposition"]
    coco_total = d0["coco_true_positives"] + d0["coco_false_positives"] + d0["coco_ignored"]
    if coco_total != legacy["num_preds"]:
        raise AnalysisError("%s: COCO 分解总数 %d != num_preds %d（可能有预测落在 test.json 之外）"
                            % (run["run_id"], coco_total, legacy["num_preds"]))

    # GT 侧
    per_gt, miss_typ = [], {"stolen": 0, "near": 0, "none": 0}
    # 三个尺寸档全部预置为 0，避免"缺键"被误读成"未统计"
    size_strat = {"small": 0, "medium": 0, "large": 0}
    imp_strat = {"small": 0, "medium": 0, "large": 0}
    covered_50 = covered_75 = occupied_n = imprecise_n = 0
    miss_iou_bins = {"[%g,%g)" % (lo, hi): 0 for lo, hi in MISS_IOU_BANDS}
    for key in sorted(gstate):
        img_id, gt_id = key
        st, cv, ft = gstate[key], cov[key], feats[key]
        occ = bool(st["occupied"])
        occupied_n += occ
        covered_50 += cv["best_iou_any"] >= IOU_THR
        covered_75 += cv["best_iou_any"] >= AP75_IOU
        if occ:
            mtype = "na"
            if st["matched_pred_iou"] < AP75_IOU:
                imprecise_n += 1
                imp_strat[ft["size_bin_coco"]] += 1
            cbin = ("ge0.75" if cv["best_iou_any"] >= AP75_IOU
                    else ("0.5to0.75" if cv["best_iou_any"] >= IOU_THR else "lt0.5"))
        else:
            cbin = ("ge0.75" if cv["best_iou_any"] >= AP75_IOU
                    else ("0.5to0.75" if cv["best_iou_any"] >= IOU_THR else "lt0.5"))
            if cv["best_iou_any"] >= IOU_THR:
                mtype = "stolen"
            elif cv["best_iou_any"] >= NEAR_IOU:
                mtype = "near"
            else:
                mtype = "none"
            miss_typ[mtype] += 1
            size_strat[ft["size_bin_coco"]] += 1
            for lo, hi in MISS_IOU_BANDS:
                if lo <= cv["best_iou_any"] < hi:
                    miss_iou_bins["[%g,%g)" % (lo, hi)] += 1
        per_gt.append({
            "run_id": run["run_id"], "image_id": img_id, "gt_id": gt_id,
            "ann_area": ft["ann_area"], "bbox_area": ft["bbox_area"],
            "size_bin_coco": ft["size_bin_coco"], "size_bin_bbox": ft["size_bin_bbox"],
            "cx": ft["cx"], "cy": ft["cy"], "nn_center_dist": ft["nn_center_dist"],
            "nn_dist_over_sqrt_area": ft["nn_dist_over_sqrt_area"],
            "n_gt_in_image": ft["n_gt_in_image"], "occupied": int(occ),
            "matched_pred_iou": st["matched_pred_iou"], "matched_pred_score": st["matched_pred_score"],
            "best_iou_any": cv["best_iou_any"], "best_pred_score_any": cv["best_pred_score_any"],
            "n_preds_iou_ge_0_5": cv["n_preds_iou_ge_0_5"],
            "coverage_bin": cbin, "miss_type": mtype})

    # 预测侧（三轴都是预测框自身的属性：面积档 × 分数档 × 对同图所有 GT 的最大 IoU 档）
    bkg_area = {"small": 0, "medium": 0, "large": 0}
    per_pred = []
    for r in sorted(pred_recs, key=lambda x: (x["image_id"], -x["score"], x["pred_index"])):
        p = preds[r["pred_index"]]
        x, y, w, h = p["bbox"]
        if r["kind"] == "bkg":
            bkg_area[size_bin(w * h)] += 1
        per_pred.append({
            "run_id": run["run_id"], "pred_index": r["pred_index"], "image_id": r["image_id"],
            "score": r["score"], "x1": x, "y1": y, "x2": x + w, "y2": y + h, "w": w, "h": h,
            "kind": r["kind"], "best_iou": r["best_iou"], "matched_gt_id": r["matched_gt_id"],
            "pred_area_bin": size_bin(w * h)})

    n_gt = len(feats)
    joint = pred_side_joint(per_pred)
    ext = {
        "coverage_miss_total": int(n_gt - occupied_n),
        "coverage_miss_stolen": miss_typ["stolen"],
        "coverage_miss_near": miss_typ["near"],
        "coverage_miss_none": miss_typ["none"],
        "coverage_miss_by_size_coco": dict(sorted(size_strat.items())),
        "coverage_miss_best_iou_any_bins": miss_iou_bins,
        "coverage_imprecise_total": int(imprecise_n),
        "coverage_imprecise_by_size_coco": dict(sorted(imp_strat.items())),
        "coverage_occupied": int(occupied_n),
        "coverage_covered_at_50": int(covered_50),
        "coverage_covered_at_75": int(covered_75),
        "bkg_by_pred_bbox_area_bin": dict(sorted(bkg_area.items())),
    }
    counts = {
        "num_preds": legacy["num_preds"],
        "num_gt_noncrowd": int(n_gt),
        "legacy_background_errors": legacy["background_errors"],
        "legacy_duplicates": legacy["duplicates"],
        "legacy_true_positives": legacy["true_positives"],
        "coverage_miss_total": ext["coverage_miss_total"],
        "coverage_imprecise_total": ext["coverage_imprecise_total"],
    }
    summary = {
        "run_id": run["run_id"], "fold": run["fold"], "variant": run["variant"],
        "eval_dir": eval_dir,
        "predictions_path": pred_path, "predictions_sha256": sha256_file(pred_path),
        "test_json_sha256": sha256_file(os.path.join(eval_dir, "test.json")),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "postproc": meta.get("eval_params"),
        "counts": counts,
        "legacy": legacy,
        "identity_check": {
            "lhs": legacy["background_errors"] + legacy["duplicates"] + legacy["true_positives"],
            "rhs": legacy["num_preds"],
            "ok": legacy["background_errors"] + legacy["duplicates"] + legacy["true_positives"] == legacy["num_preds"]},
        "score_bands": {k: legacy[k] for k in sorted(legacy) if k.startswith("bkg_") or k.startswith("tp_") or k.startswith("preds_")},
        "coverage_extensions": ext,
        "pred_side_joint": joint,
        "coco": {"raw": coco["raw"], "detection_decomposition": coco["detection_decomposition"],
                 "iou_impl_agreement": coco["iou_impl_agreement"]},
        "metric_recompute_check": metric_check,
        "tie_breaking": {"pred_order": "稳定排序 by -score，同分保持文件原序",
                         "gt_argmax": "严格 >，同 IoU 取最小 GT 下标"},
        "causes_unknown": {
            "no_good_box_generated": "未知",
            "good_box_filtered_by_score": "未知",
            "good_box_suppressed_by_nms": "未知",
            "note": "本轮只读导出预测，无法区分上述三项成因，不由低重复率或低覆盖率倒推。"},
    }
    return {"run": run, "preds": preds, "pred_recs": pred_recs, "per_gt": per_gt,
            "per_pred": per_pred, "cov": cov, "gstate": gstate, "legacy": legacy,
            "summary": summary, "n_gt": n_gt,
            "coco_matches": coco["per_gt_matches"], "pr_curves": coco["pr_curves"],
            "pr_grid": coco["pr_grid"]}


# ---------------------------------------------------------------- Part A
def _top_quartile_images(vec, image_ids):
    """确定性选取：按 (-count, image_id) 取前 25%。"""
    k = int(np.ceil(len(image_ids) * 0.25))
    ordered = sorted(image_ids, key=lambda i: (-vec.get(i, 0), i))
    return set(ordered[:k])


def per_image_vectors(run_bundle, id2gt):
    image_ids = sorted(id2gt)
    vec = {m: {i: 0 for i in image_ids} for m in ["bkg", "miss", "imprecise"]}
    for r in run_bundle["pred_recs"]:
        if r["kind"] == "bkg":
            vec["bkg"][r["image_id"]] += 1
    for g in run_bundle["per_gt"]:
        if not g["occupied"]:
            vec["miss"][g["image_id"]] += 1
        elif g["matched_pred_iou"] < AP75_IOU:
            vec["imprecise"][g["image_id"]] += 1
    return vec, image_ids


def pred_side_joint(per_pred):
    """预测侧联合统计：面积档 × 分数档 × 最大 GT IoU 档，每格并列 TP / 背景误检 / 重复。

    三个轴**都是预测框自身的属性**（不是 GT 属性），最大 GT IoU 用冻结口径 `iou()`（无 +1）。
    目的：直接给出"高分 × 小框"这一**交集**，而不是由"小框占比高"与"高分占比高"两个边际比例推断。
    空的组合不补零，避免"无此组合"与"有但零"混淆。
    """
    cells, unbanded = {}, 0
    for r in per_pred:
        sb = band_label(PRED_SCORE_BANDS, r["score"])
        ib = band_label(MAXGT_IOU_BANDS, r["best_iou"])
        if sb is None or ib is None:
            unbanded += 1
            continue
        key = (r["pred_area_bin"], sb, ib)
        c = cells.setdefault(key, {"pred_area_bin": r["pred_area_bin"], "score_band": sb,
                                   "max_gt_iou_band": ib, "total": 0, "tp": 0,
                                   "background": 0, "duplicate": 0})
        c["total"] += 1
        if r["kind"] == "tp":
            c["tp"] += 1
        elif r["kind"] == "bkg":
            c["background"] += 1
        else:
            c["duplicate"] += 1
    if unbanded:
        raise AnalysisError("%d 条预测的分数或最大 GT IoU 落在所有分档之外" % unbanded)
    rows = [cells[k] for k in sorted(cells)]

    def _sum(sel):
        out = {"n_cells": 0, "total": 0, "tp": 0, "background": 0, "duplicate": 0}
        for c in rows:
            if sel(c):
                out["n_cells"] += 1
                for k in ("total", "tp", "background", "duplicate"):
                    out[k] += c[k]
        return out

    score_labels = ["[%g,%g)" % (lo, hi) for lo, hi in PRED_SCORE_BANDS]
    iou_labels = ["[%g,%g)" % (lo, hi) for lo, hi in MAXGT_IOU_BANDS]
    # 显式按分数下界挑选，不依赖 PRED_SCORE_BANDS 的排列顺序
    high = ["[%g,%g)" % (lo, hi) for lo, hi in PRED_SCORE_BANDS if lo >= 0.7]
    small_sel = lambda c: c["pred_area_bin"] == "small"
    high_sel = lambda c: c["score_band"] in high
    cross = {
        "definition": "small = bbox 面积 < 32²；high_score = 分数 >= 0.7",
        "small_and_high_score": _sum(lambda c: small_sel(c) and high_sel(c)),
        "small_and_lower_score": _sum(lambda c: small_sel(c) and not high_sel(c)),
        "not_small_and_high_score": _sum(lambda c: (not small_sel(c)) and high_sel(c)),
        "neither": _sum(lambda c: (not small_sel(c)) and (not high_sel(c)))}
    # 结构断言：四个象限与各边际必须都覆盖全部预测，否则说明分档有漏洞或"高分"定义与标注不符
    quad = sum(cross[k]["total"] for k in ("small_and_high_score", "small_and_lower_score",
                                           "not_small_and_high_score", "neither"))
    if quad != len(per_pred):
        raise AnalysisError("面积×分数交叉合计 %d != 预测数 %d" % (quad, len(per_pred)))
    if sum(_sum(lambda c, b=b: c["score_band"] == b)["total"] for b in score_labels) != len(per_pred):
        raise AnalysisError("分数档边际合计 != 预测数")
    if high != ["[%g,%g)" % (lo, hi) for lo, hi in PRED_SCORE_BANDS if lo >= 0.7]:
        raise AnalysisError("高分数档定义异常")
    return {
        "axes": {"pred_area_bin": "bbox w*h（<32² / <96² / 其余）",
                 "score_band": "导出分数（冻结 score_thr=0.05）",
                 "max_gt_iou_band": "该预测对同图所有 GT 的最大 IoU（冻结 iou()，无 +1）"},
        "cells": rows,
        "cell_columns": ["pred_area_bin", "score_band", "max_gt_iou_band", "total", "tp",
                         "background", "duplicate"],
        "marginals": {
            "all": _sum(lambda c: True),
            "by_area_bin": {b: _sum(lambda c, b=b: c["pred_area_bin"] == b)
                            for b in ("small", "medium", "large")},
            "by_score_band": {s: _sum(lambda c, s=s: c["score_band"] == s) for s in score_labels}},
        "cross_small_x_high_score": {
            "definition": "small = bbox 面积 < 32²；high_score = 分数 >= 0.7",
            "small_and_high_score": _sum(lambda c: small_sel(c) and high_sel(c)),
            "small_and_lower_score": _sum(lambda c: small_sel(c) and not high_sel(c)),
            "not_small_and_high_score": _sum(lambda c: (not small_sel(c)) and high_sel(c)),
            "neither": _sum(lambda c: (not small_sel(c)) and (not high_sel(c)))},
        "high_score_small_by_max_gt_iou": {
            ib: _sum(lambda c, ib=ib: small_sel(c) and high_sel(c) and c["max_gt_iou_band"] == ib)
            for ib in iou_labels},
        "note": ("三轴都是预测框属性；本表**不区分成因**（未产生好框 / 被分数过滤 / 被 NMS 抑制一律未知），"
                 "也不对 AP 影响排序。")}


def part_a(bundles, id2gt):
    per_fold_counts, per_fold_shares, rank, vecs = {}, {}, {}, {}
    for rid in B0_RUN_IDS:
        b = bundles[rid]
        c = b["legacy"]
        n_gt = b["n_gt"]
        per_fold_counts[rid] = {
            "bkg": c["background_errors"], "dupe": c["duplicates"],
            "miss": b["summary"]["counts"]["coverage_miss_total"],
            "imprecise": b["summary"]["counts"]["coverage_imprecise_total"]}
        per_fold_shares[rid] = {
            "s_bkg": c["background_errors"] / float(c["num_preds"]),
            "s_dupe": c["duplicates"] / float(c["num_preds"]),
            "s_miss": b["summary"]["counts"]["coverage_miss_total"] / float(n_gt),
            "s_imprecise": b["summary"]["counts"]["coverage_imprecise_total"] / float(n_gt)}
        rank[rid] = sorted(MODE_ORDER, key=lambda m: -per_fold_counts[rid][m])
        vecs[rid] = per_image_vectors(b, id2gt)[0]

    image_ids = sorted(id2gt)
    pearson = {}
    for m in ["bkg", "miss", "imprecise"]:
        pearson[m] = {}
        for i, a in enumerate(B0_RUN_IDS):
            for b_ in B0_RUN_IDS[i + 1:]:
                xa = np.asarray([vecs[a][m][i2] for i2 in image_ids], dtype=float)
                xb = np.asarray([vecs[b_][m][i2] for i2 in image_ids], dtype=float)
                r = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 and xb.std() > 0 else None
                pearson[m]["%s-%s" % (a, b_)] = r

    quart = {}
    for m in ["bkg", "miss", "imprecise"]:
        sets = {rid: _top_quartile_images(vecs[rid][m], image_ids) for rid in B0_RUN_IDS}
        qsize = len(next(iter(sets.values())))
        inter = set.intersection(*[sets[r] for r in B0_RUN_IDS])
        quart[m] = {"quartile_size": qsize, "per_fold_size": {r: len(sets[r]) for r in B0_RUN_IDS},
                    "three_way_intersection": len(inter),
                    "chance_expected": 232 * (qsize / 232.0) ** 3}
    for m in quart:
        quart[m]["chance_expected"] = len(image_ids) * (quart[m]["quartile_size"] / float(len(image_ids))) ** 3

    top10 = {}
    for m in ["bkg", "miss", "imprecise"]:
        top10[m] = {}
        for rid in B0_RUN_IDS:
            vals = sorted(vecs[rid][m].values(), reverse=True)
            tot = sum(vals)
            top10[m][rid] = (sum(vals[:10]) / float(tot)) if tot else 0.0

    strat = {}
    for rid in B0_RUN_IDS:
        lg = bundles[rid]["legacy"]
        bkg_small = bundles[rid]["summary"]["coverage_extensions"]["bkg_by_pred_bbox_area_bin"].get("small", 0)
        strat[rid] = {"bkg_small_share": bkg_small / float(lg["background_errors"]) if lg["background_errors"] else 0.0,
                      "bkg_ge0.9_share": lg["bkg_ge0.9"] / float(lg["background_errors"]) if lg["background_errors"] else 0.0}

    # 跨轴同除得到的比例：仅作描述。单位不同，此项**不作为重要性判据**，也不使用 0.8 之类的切点。
    dom = {rid: per_fold_counts[rid]["bkg"] / float(per_fold_counts[rid]["bkg"]
                                                   + per_fold_counts[rid]["miss"]
                                                   + per_fold_counts[rid]["imprecise"]) for rid in B0_RUN_IDS}
    counting_units = {
        "bkg": "预测框数（legacy 口径）", "dupe": "预测框数（legacy 口径）",
        "miss": "GT 数（coverage 口径）", "imprecise": "GT 数（coverage 口径）",
        "warning": ("bkg / dupe 统计预测框，miss / imprecise 统计 GT，单位不同："
                    "不得按数量排名推断对 AP 的影响，不得跨轴求和或相除，也不得据此写下「瓶颈」")}
    concentration = {m: {"top10_share": top10[m],
                         "three_way_quartile_intersection": quart[m]["three_way_intersection"],
                         "chance_expected": quart[m]["chance_expected"],
                         "per_image_pearson": pearson[m]}
                     for m in ["bkg", "miss", "imprecise"]}
    concentration["threshold_note"] = ("v1 曾用 top10 图占比 < 0.4 作为'非离群驱动'判据，该切点是人为设定；"
                                       "imprecise 的约 0.40 仅擦线越过，**不足以据此排除 imprecise 的重要性**。")
    conclusion = {
        "designation": "不指定共同瓶颈",
        "persistent_common_error_pattern": "bkg（背景误检）",
        "statement": ("不指定共同瓶颈。可以陈述的是：背景误检（legacy bkg）是三折新 B0 导出结果中"
                      "**持续存在的共同错误模式**——逐图计数幅值在三折都最大、三折两两 Pearson ≈ 0.97、"
                      "三折 top-quartile 硬图交集 %d（随机期望 %.2f）。"
                      "这支持'同一类错误在导出结果中反复出现'，**不支持**把它排为第一瓶颈，"
                      "也不支持据此排除 miss / imprecise / dupe。"
                      % (quart["bkg"]["three_way_intersection"], quart["bkg"]["chance_expected"])),
        "why_no_bottleneck": ("bkg / dupe 计预测框、miss / imprecise 计 GT，单位不同，按数量排名不能推断"
                              "对 AP 的影响；imprecise 的集中度仅擦过人为阈值，也不足以排除其重要性。"),
        "large_bin_excluded": "COCO 口径 large 仅 2 个 GT，不作任何稳定性结论",
        "withdrawn_from_v1": ["v1 §5 五条'共同瓶颈'判据及其通过／不通过结论",
                              "v1 以 0.8 切点宣称 bkg'压倒性'",
                              "v1 以 top10 < 0.4 排除 imprecise 的重要性"]}
    return {"per_fold_counts": per_fold_counts, "per_fold_shares": per_fold_shares,
            "rank_by_count": rank,
            "rank_by_count_note": "纯描述性排序（按计数），不代表重要性，也不构成瓶颈判定",
            "per_image_pearson": pearson, "top_quartile_overlap": quart,
            "top10_share": top10, "stratification": strat,
            "counting_units": counting_units,
            "share_among_three_error_axes": {
                "values": dom, "note": "跨轴同除（bkg 为预测框、miss/imprecise 为 GT），仅作描述，不是判据"},
            "concentration": concentration,
            "conclusion": conclusion,
            "namespaces": {
                "bkg": "legacy_background_errors", "dupe": "legacy_duplicates",
                "miss": "coverage_miss_total", "imprecise": "coverage_imprecise_total",
                "warning": "三套命名空间（legacy/coco/coverage）不得相加或互相比较"}}


# ---------------------------------------------------------------- Part B
def mann_whitney_perm(a, b, seed=PERM_SEED, n=N_PERM):
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if len(a) == 0 or len(b) == 0:
        return {"n_a": len(a), "n_b": len(b), "observed_median_diff": None, "p_value": None}
    obs = float(np.median(a) - np.median(b))
    pool = np.asarray(a + b, dtype=float)
    na = len(a)
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(n):
        perm = rng.permutation(pool)
        d = float(np.median(perm[:na]) - np.median(perm[na:]))
        if abs(d) >= abs(obs):
            cnt += 1
    return {"n_a": len(a), "n_b": len(b), "observed_median_diff": obs,
            "p_value": (cnt + 1) / float(n + 1), "n_permutations": n, "seed": seed}


def cliffs_delta(a, b):
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if not a or not b:
        return None
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / float(len(a) * len(b))


def paired_feature_stats(A, B, feats, label_a, label_b):
    """两组的**描述统计**：中位数差、置换检验 p、Cliff's δ、中位数与四分位。

    **不下机制裁定。** 两个特征的 p>0.05、较小的 δ、以及翻转数量比例相近，都不能识别共同机制；
    把结论写成「与…一致」同样不解决问题，故机制结论字段固定为 MECHANISM_CONCLUSION。
    """
    if not A or not B:
        return {"labels": [label_a, label_b], "n_a": len(A), "n_b": len(B),
                "tests": {}, "cliffs_delta": {}, "medians": {},
                "mechanism_conclusion": MECHANISM_CONCLUSION, "reason": "空集合"}
    area_a = [feats[k]["ann_area"] for k in A]
    area_b = [feats[k]["ann_area"] for k in B]
    # 聚集度对"该图只有 1 个 GT"的目标无定义（无最近邻），须剔除后再算中位数与检验
    den_a = [v for v in (feats[k]["nn_dist_over_sqrt_area"] for k in A) if v is not None]
    den_b = [v for v in (feats[k]["nn_dist_over_sqrt_area"] for k in B) if v is not None]
    tests = {"ann_area": mann_whitney_perm(area_a, area_b),
             "nn_dist_over_sqrt_area": mann_whitney_perm(den_a, den_b)}
    deltas = {"ann_area": cliffs_delta(area_a, area_b),
              "nn_dist_over_sqrt_area": cliffs_delta(den_a, den_b)}
    return {"labels": [label_a, label_b], "n_a": len(A), "n_b": len(B),
            "tests": tests, "cliffs_delta": deltas,
            "medians": {"ann_area": {label_a: quantiles(area_a), label_b: quantiles(area_b)},
                        "nn_dist_over_sqrt_area": {label_a: quantiles(den_a), label_b: quantiles(den_b)}},
            "mechanism_conclusion": MECHANISM_CONCLUSION,
            "reason": ("两个特征的 p>0.05、较小的 δ 与翻转数量比例相近，都不能识别共同机制；"
                       "不下机制裁定，上述数值仅作描述统计。")}


def ratio_of(x, y):
    if not y:
        return None
    return {"ratio": x / float(y), "numerator": x, "denominator": y,
            "note": "仅作描述；v1 曾以 [0.67,1.5] 镜像带作裁定依据，该用法已撤除"}


def pr_curve_summary(b0, pg, level):
    """两条 101 点 PR 曲线在给定 IoU 档上的可读出量对照。"""
    cb, cp, grid = b0["pr_curves"][level], pg["pr_curves"][level], b0["pr_grid"]
    ga = np.asarray(grid, dtype=float)
    mb, mp = cb["max_recall_all_dets"], cp["max_recall_all_dets"]
    pts = {}
    for r in (0.5, 0.75, 0.9):
        i = int(np.argmin(np.abs(ga - r)))
        bv, pv = cb["precision"][i], cp["precision"][i]
        pts["recall_%g" % r] = {
            "recall_grid_value": grid[i], "precision_b0": bv, "precision_pg": pv,
            "delta": (pv - bv) if (bv is not None and pv is not None) else None,
            "beyond_max_recall_b0": bool(grid[i] > mb + 1e-12),
            "beyond_max_recall_pg": bool(grid[i] > mp + 1e-12)}
    return {"iou_level": level, "recall_grid": grid, "precision_b0": cb["precision"],
            "precision_pg": cp["precision"], "precision_at_recall": pts,
            "max_recall_all_dets": {"b0": mb, "pg": mp},
            "max_recall_all_dets_source": cb["max_recall_all_dets_source"],
            "precision_beyond_max_recall_marker": cb["precision_beyond_max_recall_marker"],
            "n_precision_negative": {"b0": cb["n_precision_negative"], "pg": cp["n_precision_negative"]},
            "mean_precision_within_reached_recall": {"b0": cb["mean_precision_within_reached_recall"],
                                                     "pg": cp["mean_precision_within_reached_recall"]},
            "mean_precision_full_grid": {"b0": cb["mean_precision_full_grid"],
                                         "pg": cp["mean_precision_full_grid"]},
            "note": ("COCOeval 的 101 点插值精度（areaRng=all, maxDet=100）。"
                     "本机 pycocotools 对「超过最大可达召回」的召回点存 **0.0**（非 COCO 原版的 -1）；"
                     "实测负值个数见 n_precision_negative，故这些 0.0 不等于「精度恰为 0」。"
                     "最大可达召回取自 ev.eval['recall']，不由精度数组尾部形态推断。")}


def part_b(bundles, id2gt, feats):
    fold_blocks = {}
    for fold in (6, 7, 8):
        brid, prid = "fold%d_b0" % fold, "fold%d_pg_both" % fold
        b0, pg = bundles[brid], bundles[prid]
        # 几何覆盖翻转（由逐 GT best_iou_any 定义）——**不是** AP75 的解释，只是覆盖口径的变化
        gains = {t: {k for k in b0["cov"] if b0["cov"][k]["best_iou_any"] < t <= pg["cov"][k]["best_iou_any"]}
                 for t in (0.5, AP75_IOU)}
        losses = {t: {k for k in b0["cov"] if pg["cov"][k]["best_iou_any"] < t <= b0["cov"][k]["best_iou_any"]}
                  for t in (0.5, AP75_IOU)}
        flips = []
        for direction, sets in (("gain", gains), ("loss", losses)):
            for k in sorted(sets[AP75_IOU]):
                ft = feats[k]
                flips.append({
                    "fold": fold, "flip_kind": "geometric_coverage", "direction": direction,
                    "defined_by": "best_iou_any", "image_id": k[0], "gt_id": k[1],
                    "ann_area": ft["ann_area"], "bbox_area": ft["bbox_area"],
                    "size_bin_coco": ft["size_bin_coco"],
                    "nn_dist_over_sqrt_area": ft["nn_dist_over_sqrt_area"],
                    "best_iou_b0": b0["cov"][k]["best_iou_any"], "best_iou_pg": pg["cov"][k]["best_iou_any"],
                    "delta_iou": pg["cov"][k]["best_iou_any"] - b0["cov"][k]["best_iou_any"],
                    "best_score_b0": b0["cov"][k]["best_pred_score_any"],
                    "best_score_pg": pg["cov"][k]["best_pred_score_any"],
                    "n_preds_iou_ge_0_5_b0": b0["cov"][k]["n_preds_iou_ge_0_5"],
                    "n_preds_iou_ge_0_5_pg": pg["cov"][k]["n_preds_iou_ge_0_5"],
                    "contested": (b0["cov"][k]["n_preds_iou_ge_0_5"] >= 2) or (pg["cov"][k]["n_preds_iou_ge_0_5"] >= 2)})
        inner = paired_feature_stats(losses[AP75_IOU], gains[AP75_IOU], feats,
                                     "geometric_coverage_loss75(fold%d)" % fold,
                                     "geometric_coverage_gain75(fold%d)" % fold)
        inner["count_ratio_loss_over_gain"] = ratio_of(len(losses[AP75_IOU]), len(gains[AP75_IOU]))
        # COCO 自身匹配变化（与几何覆盖并列、不混用）
        coco_change = {lbl: coco_match_change(b0["coco_matches"][lbl], pg["coco_matches"][lbl],
                                             "fold%d_b0" % fold, "fold%d_pg_both" % fold, lbl)
                       for lbl in ("0.50", "0.75")}
        # 分档：先给 [0,0.5) 档（冻结口径只给了 bkg_below_0.5 汇总），再给冻结脚本的五档，
        # 使各档之和 == Δbackground_errors，避免读者以为缺档
        fp_delta = {"[0,0.5)": {"b0": b0["legacy"]["bkg_below_0.5"], "pg": pg["legacy"]["bkg_below_0.5"],
                                "delta": pg["legacy"]["bkg_below_0.5"] - b0["legacy"]["bkg_below_0.5"]}}
        for band in sorted(b0["legacy"]["bkg_bins"]):
            fp_delta[band] = {"b0": b0["legacy"]["bkg_bins"][band], "pg": pg["legacy"]["bkg_bins"][band],
                              "delta": pg["legacy"]["bkg_bins"][band] - b0["legacy"]["bkg_bins"][band]}
        fold_blocks[fold] = {
            "delta_metrics": {k: pg["summary"]["coco"]["raw"][k] - b0["summary"]["coco"]["raw"][k]
                              for k in sorted(b0["summary"]["coco"]["raw"])},
            "geometric_coverage_flip_0_75": {
                "definition": "gain_thr = {best_iou_any: b0<thr<=pg}; loss_thr = {best_iou_any: pg<thr<=b0}",
                "gain": sorted("%d|%d" % k for k in gains[AP75_IOU]),
                "loss": sorted("%d|%d" % k for k in losses[AP75_IOU]),
                "n_gain": len(gains[AP75_IOU]), "n_loss": len(losses[AP75_IOU]),
                "net": len(gains[AP75_IOU]) - len(losses[AP75_IOU])},
            "geometric_coverage_flip_0_50": {
                "definition": "gain_thr = {best_iou_any: b0<thr<=pg}; loss_thr = {best_iou_any: pg<thr<=b0}",
                "gain": sorted("%d|%d" % k for k in gains[0.5]),
                "loss": sorted("%d|%d" % k for k in losses[0.5]),
                "n_gain": len(gains[0.5]), "n_loss": len(losses[0.5]),
                "net": len(gains[0.5]) - len(losses[0.5])},
            "paired_features_internal": inner,
            "coco_match_change": coco_change,
            "pr_curve": {lvl: pr_curve_summary(b0, pg, lvl) for lvl in ("0.75", "0.50")},
            "fp_by_band": fp_delta,
            "flips": flips}

    loss7 = {tuple(int(x) for x in s.split("|")) for s in fold_blocks[7]["geometric_coverage_flip_0_75"]["loss"]}
    gain8 = {tuple(int(x) for x in s.split("|")) for s in fold_blocks[8]["geometric_coverage_flip_0_75"]["gain"]}
    primary = paired_feature_stats(loss7, gain8, feats,
                                   "geometric_coverage_loss75(fold7)",
                                   "geometric_coverage_gain75(fold8)")
    primary["count_ratio"] = {
        "loss7_over_gain8": ratio_of(len(loss7), len(gain8)),
        "gain7_over_loss8": ratio_of(fold_blocks[7]["geometric_coverage_flip_0_75"]["n_gain"],
                                     fold_blocks[8]["geometric_coverage_flip_0_75"]["n_loss"])}
    cross = {
        "loss7_intersect_gain8": len(loss7 & gain8),
        "loss7_count": len(loss7), "gain8_count": len(gain8),
        "images_loss7": len({k[0] for k in loss7}), "images_gain8": len({k[0] for k in gain8}),
        "image_overlap": len({k[0] for k in loss7} & {k[0] for k in gain8}),
    }
    # 主比较的 COCO 匹配变化（与几何覆盖翻转并列）
    # 主比较的两组是 loss75(fold7) 与 gain75(fold8)：COCO 侧必须分别读 fold7 的 lost_match
    # 与 fold8 的 newly_matched，**不能只读一折**（曾误把 fold7 的 newly_matched 当作 fold8 的新增）。
    # 另按逐 GT 身份对照两个口径的集合：净增相同不代表逐目标匹配相同。
    primary_coco_change = {}
    for lbl in ("0.50", "0.75"):
        tag = lbl.replace(".", "_")
        c7 = fold_blocks[7]["coco_match_change"][lbl]
        c8 = fold_blocks[8]["coco_match_change"][lbl]
        # 逐 GT 身份对照必须在同一 IoU 档内做：拿 0.75 的覆盖集合去对 0.50 的 COCO 匹配集合是无意义的。
        # 0.75 档的覆盖集合即 loss7 / gain8；0.50 档另有自己的 gain/loss 列表。
        cov_loss = {tuple(int(x) for x in s.split("|"))
                    for s in fold_blocks[7]["geometric_coverage_flip_%s" % tag]["loss"]}
        cov_gain = {tuple(int(x) for x in s.split("|"))
                    for s in fold_blocks[8]["geometric_coverage_flip_%s" % tag]["gain"]}
        k7_lost = {(r["image_id"], r["gt_id"]) for r in c7["changed_rows"]
                   if r["state"] == "lost_match"}
        k8_new = {(r["image_id"], r["gt_id"]) for r in c8["changed_rows"]
                  if r["state"] == "newly_matched"}
        primary_coco_change[lbl] = {
            "fold7_loss_side": {
                "lost_match": c7["counts"]["lost_match"], "lost_reason": c7["lost_reason"],
                "newly_matched": c7["counts"]["newly_matched"],
                "net": c7["counts"]["newly_matched"] - c7["counts"]["lost_match"]},
            "fold8_gain_side": {
                "newly_matched": c8["counts"]["newly_matched"], "new_reason": c8["new_reason"],
                "lost_match": c8["counts"]["lost_match"],
                "net": c8["counts"]["newly_matched"] - c8["counts"]["lost_match"]},
            "identity_overlap": {
                "iou_level": lbl,
                "definition": ("几何覆盖用逐 GT best_iou_any 跨 %s；COCO 匹配用 COCOeval 自身的一分一配；"
                               "集合身份按 (image_id, gt_id) 对照，两侧的 '覆盖' 定义不同。" % lbl),
                "coverage_loss_fold7": len(cov_loss), "coco_lost_fold7": len(k7_lost),
                "coverage_loss_fold7_and_coco_lost": len(cov_loss & k7_lost),
                "coverage_loss_fold7_only": len(cov_loss - k7_lost),
                "coco_lost_fold7_only": len(k7_lost - cov_loss),
                "coverage_gain_fold8": len(cov_gain), "coco_new_fold8": len(k8_new),
                "coverage_gain_fold8_and_coco_new": len(cov_gain & k8_new),
                "coverage_gain_fold8_only": len(cov_gain - k8_new),
                "coco_new_fold8_only": len(k8_new - cov_gain),
                "coverage_net_fold7": fold_blocks[7]["geometric_coverage_flip_%s" % tag]["net"],
                "coco_net_fold7": c7["counts"]["newly_matched"] - c7["counts"]["lost_match"],
                "coverage_net_fold8": fold_blocks[8]["geometric_coverage_flip_%s" % tag]["net"],
                "coco_net_fold8": c8["counts"]["newly_matched"] - c8["counts"]["lost_match"]}}
    primary_coco_change["note"] = (
        "fold7_loss_side 取 fold7 的 lost_match，fold8_gain_side 取 fold8 的 newly_matched —— "
        "主比较的两组分别是 loss75(fold7) 与 gain75(fold8)，两侧必须各读自己的折。"
        "净增（net）相同 **不代表逐目标匹配相同**：两个口径的集合身份逐档见各自 identity_overlap，"
        "不得互推，也不得用其一解释另一。")
    # 高分背景框（含"高分小框且对任何 GT 的 IoU < 0.1"这一交集）vs 精度变化
    def _clutter_small_high(run_bundle):
        return sum(1 for r in run_bundle["per_pred"]
                   if r["pred_area_bin"] == "small" and r["score"] >= 0.7
                   and r["kind"] == "bkg" and r["best_iou"] < 0.1)

    bkg_hi = {}
    for fold in (6, 7, 8):
        fb = fold_blocks[fold]["fp_by_band"]
        b0c, pgc = _clutter_small_high(bundles["fold%d_b0" % fold]), \
            _clutter_small_high(bundles["fold%d_pg_both" % fold])
        bkg_hi[fold] = {
            "delta_bkg_ge_0.9": fb["[0.9,1.01)"]["delta"],
            "delta_bkg_ge_0.7": sum(fb[b]["delta"] for b in ("[0.9,1.01)", "[0.8,0.9)", "[0.7,0.8)")),
            "delta_background_errors": sum(v["delta"] for v in fb.values()),
            "b0_high_score_small_bkg_iou_lt_0.1": b0c,
            "pg_high_score_small_bkg_iou_lt_0.1": pgc,
            "delta_high_score_small_bkg_iou_lt_0.1": pgc - b0c,
            "b0_mAP": bundles["fold%d_b0" % fold]["summary"]["coco"]["raw"]["bbox_mAP"],
            "pg_mAP": bundles["fold%d_pg_both" % fold]["summary"]["coco"]["raw"]["bbox_mAP"],
            "delta_mAP": fold_blocks[fold]["delta_metrics"]["bbox_mAP"],
            "delta_mAP_75": fold_blocks[fold]["delta_metrics"]["bbox_mAP_75"]}
    high_score_bkg_vs_metric = {
        "per_fold": bkg_hi,
        "intersection_definition": ("`high_score_small_bkg_iou_lt_0.1` = 预测框 bbox 面积 < 32²、分数 >= 0.7、"
                                   "legacy 判为背景误检、且对同图所有 GT 的最大 IoU（冻结 iou()）< 0.1 的条数。"
                                   "**该列只能读作「与已标注 GT 重叠很低」，不能读作「空间上远离 GT」**"
                                   "——一个很小的框整体落在一个大 GT 框内部时，IoU 也可以非常低。"
                                   "因此它既可能是真实背景，也可能是严重尺度偏差，还可能是**标注遗漏**；"
                                   "三者的区分需要抽查代表案例，属下一步工作，本轮不作判断。"),
        "statement": ("fold7 与 fold8 的高分背景框都减少（`bkg_ge_0.9` %+d / %+d，分数 >= 0.7 档 %+d / %+d，"
                      "高分小框且与已标注 GT 最大 IoU < 0.1 的背景框 %+d / %+d），"
                      "但两折 mAP 一降一升（%+.4f / %+.4f）；"
                      "fold6 的高分小框背景框反而增加（%+d）而 mAP 上升（%+.4f）。"
                      % (bkg_hi[7]["delta_bkg_ge_0.9"], bkg_hi[8]["delta_bkg_ge_0.9"],
                         bkg_hi[7]["delta_bkg_ge_0.7"], bkg_hi[8]["delta_bkg_ge_0.7"],
                         bkg_hi[7]["delta_high_score_small_bkg_iou_lt_0.1"],
                         bkg_hi[8]["delta_high_score_small_bkg_iou_lt_0.1"],
                         bkg_hi[7]["delta_mAP"], bkg_hi[8]["delta_mAP"],
                         bkg_hi[6]["delta_high_score_small_bkg_iou_lt_0.1"], bkg_hi[6]["delta_mAP"])),
        "implication": ("高分背景框数量（含「高分小框」这一交集）的变化与 mAP 变化方向不一致；"
                        "该数量变化本身不足以解释精度变化，不得据此推出精度变化方向，"
                        "也不得把它当作因果证据。")}
    return {"folds": fold_blocks,
            "primary_cross_fold_features": primary,
            "primary_cross_fold_coco_match_change": primary_coco_change,
            "cross_fold": cross,
            "mechanism_conclusion": MECHANISM_CONCLUSION,
            "terminology": {
                "geometric_coverage_flip": ("由逐 GT best_iou_any 定义、跨 0.5 / 0.75 的覆盖变化；"
                                            "**不得直接当作 AP75 的解释**"),
                "coco_match_change": ("COCOeval 自身的一分一配匹配变化（maxDet=100, areaRng=all）；"
                                      "与几何覆盖并列列出，两者不混用")},
            "flip_definition": ("几何覆盖翻转：gain_thr = {b0<thr<=pg}; loss_thr = {pg<thr<=b0}; "
                                "使用逐 GT best_iou_any"),
            "feature_note": "面积与聚集度均仅由 GT 算出，B0/PG-both 共用",
            "high_score_bkg_vs_metric": high_score_bkg_vs_metric,
            "power_caveat": ("翻转集为数十个目标（fold7 几何覆盖失去 40 / fold8 得到 34），"
                             "p 值与 δ 只作描述；样本量下不足以支持机制结论"),
            "fold6_role": "参考；fold6 已被用于选方案，其统计不构成对 PG-both 的验证证据"}


# ---------------------------------------------------------------- Part C
def part_c(bundles, pa, pb):
    ext = {rid: bundles[rid]["summary"]["coverage_extensions"] for rid in B0_RUN_IDS}
    lg = {rid: bundles[rid]["legacy"] for rid in B0_RUN_IDS}
    miss_near = {rid: ext[rid]["coverage_miss_near"] for rid in B0_RUN_IDS}
    miss_total = {rid: ext[rid]["coverage_miss_total"] for rid in B0_RUN_IDS}
    imp = {rid: ext[rid]["coverage_imprecise_total"] for rid in B0_RUN_IDS}
    bkg_hi = {rid: lg[rid]["bkg_ge0.9"] / float(lg[rid]["background_errors"]) for rid in B0_RUN_IDS}
    bkg_small = {rid: ext[rid]["bkg_by_pred_bbox_area_bin"].get("small", 0) / float(lg[rid]["background_errors"])
                 for rid in B0_RUN_IDS}
    joint = {rid: bundles[rid]["summary"]["pred_side_joint"]["cross_small_x_high_score"] for rid in B0_RUN_IDS}
    joint_iou = {rid: bundles[rid]["summary"]["pred_side_joint"]["high_score_small_by_max_gt_iou"]
                 for rid in B0_RUN_IDS}
    hs = pb["high_score_bkg_vs_metric"]

    def t3(d):
        """按 fold6/7/8_b0 固定顺序渲染三元组。"""
        return " / ".join(str(d[rid]) for rid in B0_RUN_IDS)

    # 报告正文只印这些短行；原始数值仍完整写入 candidates.json
    s_hi = t3({rid: "%.3f" % bkg_hi[rid] for rid in B0_RUN_IDS})
    s_sm = t3({rid: "%.3f" % bkg_small[rid] for rid in B0_RUN_IDS})
    s_cross = t3({rid: "%d = TP %d + 背景 %d + 重复 %d" % (
        joint[rid]["small_and_high_score"]["total"], joint[rid]["small_and_high_score"]["tp"],
        joint[rid]["small_and_high_score"]["background"], joint[rid]["small_and_high_score"]["duplicate"])
        for rid in B0_RUN_IDS})
    s_lt01 = t3({rid: joint_iou[rid]["[0,0.1)"]["background"] for rid in B0_RUN_IDS})
    s_lt01_share = t3({rid: "%.1f%%" % (100.0 * joint_iou[rid]["[0,0.1)"]["background"]
                                       / joint[rid]["small_and_high_score"]["background"])
                       for rid in B0_RUN_IDS})
    s_mid = t3({rid: joint_iou[rid]["[0.1,0.5)"]["background"] for rid in B0_RUN_IDS})
    s_pear = " / ".join("%.3f" % v for v in pa["per_image_pearson"]["bkg"].values())
    s_miss_sz = " | ".join("small %d / medium %d / large %d" % (
        ext[rid]["coverage_miss_by_size_coco"].get("small", 0),
        ext[rid]["coverage_miss_by_size_coco"].get("medium", 0),
        ext[rid]["coverage_miss_by_size_coco"].get("large", 0)) for rid in B0_RUN_IDS)
    s_imp_sz = " | ".join("small %d / medium %d / large %d" % (
        ext[rid]["coverage_imprecise_by_size_coco"].get("small", 0),
        ext[rid]["coverage_imprecise_by_size_coco"].get("medium", 0),
        ext[rid]["coverage_imprecise_by_size_coco"].get("large", 0)) for rid in B0_RUN_IDS)

    items = [
        {"mode": "bkg", "direction": "高分小目标背景抑制（模型/训练侧）",
         "status": "候选方向，**尚未确定为修改方向**（已另行展开为待评审方案，尚未实施、未经训练验证）",
         "support": {"bkg_ge0.9_share": bkg_hi, "bkg_small_bbox_share": bkg_small,
                     "small_x_high_score_交集计数": joint,
                     "small_x_high_score_且最大GT_IoU_lt_0.1": {rid: joint_iou[rid]["[0,0.1)"] for rid in B0_RUN_IDS},
                     "per_image_pearson": pa["per_image_pearson"]["bkg"],
                     "top_quartile_three_way": pa["top_quartile_overlap"]["bkg"]["three_way_intersection"],
                     "chance": pa["top_quartile_overlap"]["bkg"]["chance_expected"]},
         "support_summary": [
             "高分背景框占全部背景误检：%s（fold6/7/8_b0）" % s_hi,
             "背景误检中的小框（bbox 面积 < 32²）占比：%s" % s_sm,
             "高分（≥0.7）∧ 小框（bbox<32²）的构成：%s" % s_cross,
             "该交集中与已标注 GT 最大 IoU < 0.1 的背景框：%s（占该交集背景 %s）" % (s_lt01, s_lt01_share),
             "同一交集中最大 GT IoU 落在 [0.1,0.5) 的背景框：%s（数量级远小，非近邻偏移为主）" % s_mid,
             "逐图 Pearson：%s；三折 top-quartile 交集 %d（随机期望 %.2f）" % (
                 s_pear, pa["top_quartile_overlap"]["bkg"]["three_way_intersection"],
                 pa["top_quartile_overlap"]["bkg"]["chance_expected"])],
         "counter": ("高分背景框数量不足以解释精度变化：" + hs["statement"] + hs["implication"] +
                     " 另外，本方向只在三折 B0 上描述过错误分布，尚未证明抑制它们能提高 mAP。"),
         "falsifiable": ("若在高分小框上做抑制后，独立 SAR 留出集上的 mAP 不升，或高分背景框减少而 mAP 同时下降，"
                         "则该方向不成立")},
        {"mode": "miss", "direction": "小目标召回 / 近邻漏检找回（模型/训练侧）",
         "support": {"miss_total": miss_total, "miss_near_at_0.3": miss_near,
                     "miss_best_iou_any_bins": {rid: ext[rid]["coverage_miss_best_iou_any_bins"] for rid in B0_RUN_IDS},
                     "miss_by_size_coco": {rid: ext[rid]["coverage_miss_by_size_coco"] for rid in B0_RUN_IDS}},
         "support_summary": [
             "漏检 GT 数：%s" % t3(miss_total),
             "其中 best_iou_any ≥ 0.3 的近邻漏检：%s" % t3(miss_near),
             "漏检 GT 按 COCO 尺寸（ann.area）：%s" % s_miss_sz],
         "counter": "漏检总量与背景误检不在同一计数单位上，不能据此比较量级或重要性",
         "falsifiable": "若漏检 GT 的 best_iou_any 直方图集中在低档，则不存在可找回的近邻"},
        {"mode": "imprecise", "direction": "定位精修（框回归侧）",
         "support": {"imprecise_total": imp,
                     "imprecise_by_size_coco": {rid: ext[rid]["coverage_imprecise_by_size_coco"] for rid in B0_RUN_IDS},
                     "coverage_covered_at_50": {rid: ext[rid]["coverage_covered_at_50"] for rid in B0_RUN_IDS},
                     "coverage_covered_at_75": {rid: ext[rid]["coverage_covered_at_75"] for rid in B0_RUN_IDS}},
         "support_summary": [
             "被占用但被分配框的 IoU < 0.75 的 GT：%s" % t3(imp),
             "几何覆盖 ≥0.5 / ≥0.75 的 GT 数：%s；%s" % (
                 t3({rid: ext[rid]["coverage_covered_at_50"] for rid in B0_RUN_IDS}),
                 t3({rid: ext[rid]["coverage_covered_at_75"] for rid in B0_RUN_IDS})),
             "imprecise 群体按 COCO 尺寸（ann.area）：%s" % s_imp_sz],
         "counter": ("v1 曾以 top10 图占比 < 0.4 判其'非离群驱动'并据此降权，该切点是人为设定，"
                     "接近阈值的观测不足以排除本方向"),
         "falsifiable": "若各折 imprecise 群体的 size 分布方向不一致，则难以跨折推广"},
        {"mode": "dupe", "direction": "重复 / 争抢框处理",
         "support": {"legacy_duplicates": {rid: lg[rid]["duplicates"] for rid in B0_RUN_IDS}},
         "support_summary": ["legacy 重复框数：%s" % t3({rid: lg[rid]["duplicates"] for rid in B0_RUN_IDS})],
         "counter": "硬 NMS 下重复数仅个位数，本轮不构成可行动方向；仅在将来做 soft-NMS / 候选级回放时才有意义",
         "falsifiable": "若换用 soft-NMS 后重复数显著上升，则该方向重新进入候选"}]
    return {"ordered_by": "固定模式顺序 bkg → miss → imprecise → dupe；不作排序、不选模块、不强排方向",
            "note": "报告正文只印 support_summary 的短行；support 中的原始数值完整保存在本文件（candidates.json）",
            "selection_status": ("本轮**不选定修改方向**。bkg 方向保留为**候选**，已另行展开为待评审的单一方案文档"
                                 "（`proposal_small_bkg_roi_reweight_20260929.md`，首版仅 `sup2`）："
                                 "尚未实施、尚无任何训练或验证结果；该文件不使用「高分」命名，"
                                 "因为其公式中没有分数条件"),
            "items": items,
            "boundary": "不得据本测试集调整 score/NMS 阈值；不得叠加 M2/FG；成因字段本轮记“未知”"}



# ---------------------------------------------------------------- 报告
def build_report(pa, pb, pc, bundles, runs_meta, self_test, shadow, plan_sha, plan_v1_sha):
    L = []
    A = L.append

    def _p(v, fmt="%.4f"):
        return "n/a" if v is None else (fmt % v)

    A("# 三折新 B0 与 PG-both 离线误差分析报告（探索性分析）")
    A("")
    A("**性质**：%s" % ANALYSIS_NATURE)
    A("")
    A("分析规则：`analysis_plan_r2_20260929.md`（sha256 `%s`）；v1 规则文件 `analysis_plan.md`"
      "（sha256 `%s`）仅存历史。" % (plan_sha, plan_v1_sha))
    A("")
    A("> 本文**不指定共同瓶颈，也不下机制裁定**。v1 的两套判据（§5 共同瓶颈五条、§6 同一现象反向）"
      "已按 2026-09-29 评审意见撤除，理由见规则文件 §0；撤除清单见 "
      "`cross_fold/descriptive_stats.json → conclusion.withdrawn_from_v1`。")
    A("")
    A("只读六组导出预测，不做训练、不重跑推理、不调后处理阈值、不叠加 M2/FG。")
    A("")
    A("## 1. 输入与校验")
    A("")
    A("| run | num_preds | predictions sha256 (前12) | test.json sha256 (前12) | 身份恒等式 | COCO 复算(保存精度) |")
    A("|---|---|---|---|---|---|")
    for rid in [r["run_id"] for r in runs_meta]:
        s = bundles[rid]["summary"]
        A("| %s | %d | `%s` | `%s` | %s | %s |" % (
            rid, s["counts"]["num_preds"], s["predictions_sha256"][:12], s["test_json_sha256"][:12],
            "成立" if s["identity_check"]["ok"] else "**不成立**",
            "一致" if s["metric_recompute_check"]["all_equal_at_stored_precision"] else "**不一致**"))
    A("")
    A("- 冻结 test.json sha256：`%s`；232 图 / 546 非 crowd 标注。" % FROZEN_TEST_SHA)
    A("- 影子对照（与冻结 `score_thr_bkg.py` 逐项比对 `legacy_*`）：%s；被对照脚本 sha256 `%s`。"
      % ("全部一致" if shadow["all_equal"] else "**不一致**", shadow["frozen_script_sha256"]))
    A("- 合成用例：%d 项全部通过。" % len(self_test["synthetic_cases"]))
    A("")
    A("## 2. 口径与并列打破规则")
    A("")
    A("三套计数**禁止相加、禁止互相比较**：")
    A("")
    A("| 命名空间 | 语义 | 来源 |")
    A("|---|---|---|")
    A("| `legacy_*` | 组内按分数降序、每条预测取单一最大 IoU GT、一对一占用 | 复刻 `score_thr_bkg.py` |")
    A("| `coco_*` | COCOeval 的 TP/FP/ignore 与 AP/AR | pycocotools 自有 IoU 与 `areaRng` |")
    A("| `coverage_*` | 逐 GT 覆盖视图（`best_iou_any` / `miss_*` / `imprecise_*`） | legacy 占用 + 逐 GT 任意框最大 IoU |")
    A("")
    A("- 并列打破：预测用稳定排序（同分保持文件原序）；GT argmax 用严格 `>`，同 IoU 取最小 GT 下标。")
    A("- `imprecise` 用**被分配**框的 IoU；`miss_*` 细分用 `best_iou_any`。两处不同 IoU 不得互换。")
    A("- 尺寸分箱：`size_bin_coco` 用 `ann.area`（COCOeval `areaRng` 口径，large 仅 2 个 GT）；")
    A("  `size_bin_bbox` 用 bbox 面积（large 19）。**不对 APl / `AR_l@1000` 下任何结论。**")
    A("- 计数单位：%s" % pa["counting_units"]["warning"])
    A("")
    A("两种 IoU 实现的实测差异（同一批框对，逐 run 比较 pycocotools 自有 IoU 与冻结脚本手写 `iou()`）：")
    A("")
    A("| run | 比较框对数 | 差异 > 1e-9 的对数 | 占比 | 最大绝对差 |")
    A("|---|---|---|---|---|")
    for rid in [r["run_id"] for r in runs_meta]:
        ag = bundles[rid]["summary"]["coco"]["iou_impl_agreement"]
        A("| %s | %d | %d | %s | %.3e |" % (rid, ag["pairs_compared"], ag["pairs_differing_gt_1e-9"],
                                           _p(ag["share_differing"], "%.4f"), ag["max_abs_diff"]))
    A("")
    A("> 两个口径（冻结 `iou()` 与 pycocotools）**并列呈现、不合并**。实测结果：本数据上六组全部为零差异，"
      "因此两种实现的分歧在本轮不构成混淆来源；该结论只对本次这批框对成立，不外推。")
    A("> 本表中几何覆盖与 `legacy` 一律用冻结口径，COCO 匹配与 PR 曲线用 pycocotools 口径。")
    A("")
    A("**PR 曲线的召回上限与「精度为 0」的含义（本机实测，不按 COCO 原版假定）：**")
    A("")
    A("- 最大可达召回取自 COCOeval 自身的 `ev.eval['recall']`（areaRng=all, maxDet=100），"
      "并与「该档匹配上的 GT 数 / 非 crowd GT 数」交叉校验，六组两个 IoU 档全部相等，否则脚本报错退出。")
    A("- 本机 pycocotools 把**超过最大可达召回**的召回点的精度存为 **0.0**，而非 COCO 原版的 -1；"
      "六组实测 `n_precision_negative` 全为 0。因此 101 点曲线尾部的 0.0 **不是**「精度恰为 0」的观测，"
      "读出时必须按各侧自己的召回上限截断（见 §5.3）。")
    A("")
    A("## 3. Part A：三折新 B0 误差分解（描述统计，不指定瓶颈）")
    A("")
    A("计数（**注意计数对象不同，不得跨轴比较**）：")
    A("")
    A("| 模式 | 计数对象 | fold6_b0 | fold7_b0 | fold8_b0 |")
    A("|---|---|---|---|---|")
    for m in MODE_ORDER:
        A("| %s | %s | %d | %d | %d |" % (
            m, pa["counting_units"][m],
            pa["per_fold_counts"]["fold6_b0"][m], pa["per_fold_counts"]["fold7_b0"][m],
            pa["per_fold_counts"]["fold8_b0"][m]))
    A("")
    A("占比（分母不同：预测框占比用 num_preds，GT 占比用 546）：")
    A("")
    A("| run | s_bkg=bkg/preds | s_dupe=dupe/preds | s_miss=miss/GT | s_imprecise=imp/GT |")
    A("|---|---|---|---|---|")
    for rid in B0_RUN_IDS:
        s = pa["per_fold_shares"][rid]
        A("| %s | %.4f | %.4f | %.4f | %.4f |" % (rid, s["s_bkg"], s["s_dupe"], s["s_miss"], s["s_imprecise"]))
    A("")
    A("按计数排序（**纯描述性，不代表重要性**）：%s"
      % "；".join("%s: %s" % (rid, " > ".join(pa["rank_by_count"][rid])) for rid in B0_RUN_IDS))
    A("")
    A("逐图相关（Pearson，232 图）：")
    A("")
    A("| 模式 | fold6-7 | fold7-8 | fold6-8 |")
    A("|---|---|---|---|")
    for m in ["bkg", "miss", "imprecise"]:
        p = pa["per_image_pearson"][m]
        fmt = lambda v: "n/a" if v is None else "%.3f" % v
        A("| %s | %s | %s | %s |" % (m, fmt(p["fold6_b0-fold7_b0"]), fmt(p["fold7_b0-fold8_b0"]),
                                     fmt(p["fold6_b0-fold8_b0"])))
    A("")
    A("三折 top-quartile 硬图交集：")
    A("")
    A("| 模式 | 每折 quartile 大小 | 三折交集 | 随机期望 |")
    A("|---|---|---|---|")
    for m in ["bkg", "miss", "imprecise"]:
        q = pa["top_quartile_overlap"][m]
        A("| %s | %d | %d | %.2f |" % (m, q["quartile_size"], q["three_way_intersection"], q["chance_expected"]))
    A("")
    A("集中度（描述统计，**不用于通过／不通过判定**）：")
    A("")
    A("| 模式 | top10 图占比 fold6 / fold7 / fold8 |")
    A("|---|---|")
    for m in ["bkg", "miss", "imprecise"]:
        A("| %s | %.3f / %.3f / %.3f |" % (
            m, pa["top10_share"][m]["fold6_b0"], pa["top10_share"][m]["fold7_b0"],
            pa["top10_share"][m]["fold8_b0"]))
    A("")
    A("> %s" % pa["concentration"]["threshold_note"])
    A("")
    A("分层（bkg，由预测框属性算）：")
    A("")
    A("| run | 小框占比(bbox 面积) | bkg_ge0.9 占比 |")
    A("|---|---|---|")
    for rid in B0_RUN_IDS:
        st = pa["stratification"][rid]
        A("| %s | %.3f | %.3f |" % (rid, st["bkg_small_share"], st["bkg_ge0.9_share"]))
    A("")
    A("跨轴同除（bkg 为预测框、miss/imprecise 为 GT，**单位不同，仅作描述**）：%s"
      % ", ".join("%s=%.4f" % (rid, v) for rid, v in sorted(pa["share_among_three_error_axes"]["values"].items())))
    A("")
    A("**结论（Part A）**：%s" % pa["conclusion"]["statement"])
    A("")
    A("%s" % pa["conclusion"]["why_no_bottleneck"])
    A("")
    A("从 v1 撤除：%s" % "；".join(pa["conclusion"]["withdrawn_from_v1"]))
    A("")
    A("> %s" % pa["conclusion"]["large_bin_excluded"])
    A("")
    A("## 4. 预测侧联合统计：面积 × 分数 × 最大 GT IoU")
    A("")
    A("三个轴都是**预测框自身**的属性（不是 GT 属性）。目的：直接给出「高分 × 小框」这一**交集**的"
      "TP / 背景误检 / 重复构成，而不是由「小框占比高」与「高分占比高」两个边际比例推断。")
    A("")
    A("### 4.1 2×2 交叉（small = bbox 面积 < 32²；high_score = 分数 ≥ 0.7）")
    A("")
    A("| run | 组合 | total | tp | background | duplicate |")
    A("|---|---|---|---|---|---|")
    for rid in B0_RUN_IDS:
        cs = bundles[rid]["summary"]["pred_side_joint"]["cross_small_x_high_score"]
        for name, key in (("small 且 高分", "small_and_high_score"),
                          ("small 且 非高分", "small_and_lower_score"),
                          ("非 small 且 高分", "not_small_and_high_score"),
                          ("两者都不是", "neither")):
            c = cs[key]
            A("| %s | %s | %d | %d | %d | %d |" % (rid, name, c["total"], c["tp"], c["background"],
                                                  c["duplicate"]))
    A("")
    A("### 4.2 高分小框按最大 GT IoU 档分解")
    A("")
    A("| run | 最大 GT IoU 档 | total | tp | background | duplicate |")
    A("|---|---|---|---|---|---|")
    for rid in B0_RUN_IDS:
        d = bundles[rid]["summary"]["pred_side_joint"]["high_score_small_by_max_gt_iou"]
        for k in sorted(d):
            c = d[k]
            A("| %s | %s | %d | %d | %d | %d |" % (rid, k, c["total"], c["tp"], c["background"],
                                                  c["duplicate"]))
    A("")
    A("### 4.3 背景误检最多的三轴格（按 background 降序，仅列前三）")
    A("")
    A("| run | 面积档 | 分数档 | 最大 GT IoU 档 | total | tp | background | duplicate |")
    A("|---|---|---|---|---|---|---|---|")
    for rid in B0_RUN_IDS:
        cells = sorted(bundles[rid]["summary"]["pred_side_joint"]["cells"],
                       key=lambda c: (-c["background"], c["pred_area_bin"], c["score_band"],
                                      c["max_gt_iou_band"]))
        for c in cells[:3]:
            A("| %s | %s | %s | %s | %d | %d | %d | %d |" % (
                rid, c["pred_area_bin"], c["score_band"], c["max_gt_iou_band"],
                c["total"], c["tp"], c["background"], c["duplicate"]))
    A("")
    A("> 完整三轴格见 `per_run/<run_id>/summary.json → pred_side_joint.cells` 与 `partA/joint_cells.csv`；"
      "本表**不区分成因**，也不对 AP 影响排序。")
    A("")
    A("> **读表的两个限制：**① `最大 GT IoU` 低只说明**与已标注 GT 重叠很低**，不说明空间上远离 GT"
      "（小框整体落在大 GT 框内时 IoU 也可以很低），因此不能把 `[0,0.1)` 档一概视为「纯背景」；"
      "它可能是真实背景、严重尺度偏差，或**标注遗漏**，三者区分需抽查代表案例。"
      "② `total` 是预测框计数、`tp/background/duplicate` 是 legacy 一对一分配的结果，"
      "与 §3 的 `coverage_*`（逐 GT 口径）不是同一命名空间，不得相减或相除。")
    A("")
    A("## 5. Part B：B0 ↔ PG-both 逐目标配对（描述统计，不下机制裁定）")
    A("")
    A("两个口径**并列列出、不混用**：")
    A("")
    A("- **几何覆盖翻转**：由逐 GT `best_iou_any` 定义、跨越 0.5 / 0.75 的变化。"
      "**不得直接当作 AP75 的解释。**")
    A("- **COCO 匹配变化**：COCOeval 自身一分一配匹配的新增 / 丢失 / 保持 / 从未（`maxDet=100`, `areaRng=all`）。")
    A("")
    A("| fold | 指标 | B0 | PG-both | Δ |")
    A("|---|---|---|---|---|")
    for fold in (6, 7, 8):
        for k in ["bbox_mAP", "bbox_mAP_75", "bbox_mAP_s", "bbox_AR@100"]:
            b0 = bundles["fold%d_b0" % fold]["summary"]["coco"]["raw"][k]
            pg = bundles["fold%d_pg_both" % fold]["summary"]["coco"]["raw"][k]
            A("| %d | %s | %.4f | %.4f | %+.4f |" % (fold, k, b0, pg, pg - b0))
    A("")
    A("> 上行 B0 / PG 用**复算原始值**（未舍入），Δ 由原始值相减。若改用 `metrics.json` 的三位小数值相减"
      "（台账口径），ΔmAP 为 fold6 %+.4f / fold7 %+.4f / fold8 %+.4f；两者可能相差 1 个末位，"
      "**以台账口径为准**。"
      % tuple(bundles["fold%d_pg_both" % f]["summary"]["metric_recompute_check"]["per_key"]["bbox_mAP"]["stored"]
              - bundles["fold%d_b0" % f]["summary"]["metric_recompute_check"]["per_key"]["bbox_mAP"]["stored"]
              for f in (6, 7, 8)))
    A("")
    A("### 5.1 几何覆盖翻转计数（`best_iou_any` 口径）")
    A("")
    A("| fold | 0.75 gain | 0.75 loss | 0.75 net | 0.50 gain | 0.50 loss | 0.50 net |")
    A("|---|---|---|---|---|---|---|")
    for fold in (6, 7, 8):
        f = pb["folds"][fold]
        A("| %d | %d | %d | %+d | %d | %d | %+d |" % (
            fold, f["geometric_coverage_flip_0_75"]["n_gain"], f["geometric_coverage_flip_0_75"]["n_loss"],
            f["geometric_coverage_flip_0_75"]["net"],
            f["geometric_coverage_flip_0_50"]["n_gain"], f["geometric_coverage_flip_0_50"]["n_loss"],
            f["geometric_coverage_flip_0_50"]["net"]))
    A("")
    A("### 5.2 COCO 匹配变化（COCOeval 自身匹配，与 5.1 不同口径）")
    A("")
    A("| fold | IoU | 新匹配 | 丢失匹配 | 保持 | 从未 | 丢失:该侧无合格框 | 丢失:有合格框但被分走 | 新匹配:原侧无合格框 | 新匹配:有合格框但被分走 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for fold in (6, 7, 8):
        for lvl in ("0.75", "0.50"):
            cm = pb["folds"][fold]["coco_match_change"][lvl]
            c, lr, nr = cm["counts"], cm["lost_reason"], cm["new_reason"]
            A("| %d | %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (
                fold, lvl, c["newly_matched"], c["lost_match"], c["retained"], c["never"],
                lr["no_qualifying_box_in_pg"], lr["qualifying_box_present_but_assigned_elsewhere"],
                nr["no_qualifying_box_in_b0"], nr["qualifying_box_present_but_assigned_elsewhere"]))
    A("")
    A("> 「有合格框但被分走」= 该 GT 在那一侧**存在** IoU ≥ 阈值的预测框，但被同一图里别的 GT 先占走；"
      "这是分数排序与一分一配的直接体现，与「该侧根本没有合格框」是两种不同情形。")
    A("")
    A("### 5.3 PR 曲线（COCOeval 101 点插值，IoU 0.75 / 0.50）")
    A("")
    A("| fold | IoU | precision@R=0.5 (B0 → PG) | @R=0.75 | @R=0.9 | 最大可达召回 B0 / PG |")
    A("|---|---|---|---|---|---|")
    for fold in (6, 7, 8):
        for lvl in ("0.75", "0.50"):
            pr = pb["folds"][fold]["pr_curve"][lvl]
            pa_r = pr["precision_at_recall"]
            mr = pr["max_recall_all_dets"]

            def _cell(pt, side, maxrec):
                # 超过最大可达召回的召回点，本机存为 0.0，不能当作「精度为 0」的观测
                return "超上限(0.0)" if pt["recall_grid_value"] > maxrec + 1e-12 \
                    else _p(pt["precision_%s" % side])

            A("| %d | %s | %s → %s | %s → %s | %s → %s | %s / %s |" % (
                fold, lvl,
                _cell(pa_r["recall_0.5"], "b0", mr["b0"]), _cell(pa_r["recall_0.5"], "pg", mr["pg"]),
                _cell(pa_r["recall_0.75"], "b0", mr["b0"]), _cell(pa_r["recall_0.75"], "pg", mr["pg"]),
                _cell(pa_r["recall_0.9"], "b0", mr["b0"]), _cell(pa_r["recall_0.9"], "pg", mr["pg"]),
                _p(mr["b0"]), _p(mr["pg"])))
    A("")
    A("> 「超上限(0.0)」= 该召回点已高于该侧实际达到的最大召回。本机 pycocotools 把这类点的精度存为 **0.0**，"
      "而非 COCO 原版的 -1（实测负值个数见 `partB/paired.json → ...n_precision_negative`，六组全为 0），"
      "因此它**不是**「精度恰为 0」的观测，读出时应忽略该格。")
    A("> 最大可达召回取自 COCOeval 自身的 `ev.eval['recall']`（areaRng=all, maxDet=100），"
      "不由精度数组的尾部形态推断。完整 101 点曲线见 `partB/paired.json → folds.<f>.pr_curve` 或 `partB/pr_curves.csv`。")
    A("")
    A("### 5.4 主比较：几何覆盖损失 `loss75(fold7)` vs 几何覆盖增益 `gain75(fold8)`")
    A("")
    pr = pb["primary_cross_fold_features"]
    A("- 机制结论：**%s**（%s）" % (pr["mechanism_conclusion"], pr["reason"]))
    A("- 样本量：n_a=%d（fold7 覆盖损失）、n_b=%d（fold8 覆盖增益）" % (pr["n_a"], pr["n_b"]))
    for f in ["ann_area", "nn_dist_over_sqrt_area"]:
        t, d, m = pr["tests"][f], pr["cliffs_delta"][f], pr["medians"][f]
        A("- %s：中位差 %s，p=%s，Cliff's δ=%s，中位数 %s vs %s" % (
            f, _p(t["observed_median_diff"], "%.4g"), _p(t["p_value"]),
            _p(d, "%.3f"),
            _p(m["geometric_coverage_loss75(fold7)"]["median"], "%.4g"),
            _p(m["geometric_coverage_gain75(fold8)"]["median"], "%.4g")))
    cr = pr["count_ratio"]
    A("- 计数比：|loss7|/|gain8| = %s；|gain7|/|loss8| = %s（仅描述；v1 的 [0.67,1.5] 镜像带裁定已撤除）" % (
        _p(cr["loss7_over_gain8"]["ratio"] if cr["loss7_over_gain8"] else None, "%.3f"),
        _p(cr["gain7_over_loss8"]["ratio"] if cr["gain7_over_loss8"] else None, "%.3f")))
    A("- 反向证据：`|loss75(f7) ∩ gain75(f8)| = %d`（同一目标未同时在两组出现）；涉及图 %d vs %d，图重叠 %d。"
      % (pb["cross_fold"]["loss7_intersect_gain8"], pb["cross_fold"]["images_loss7"],
         pb["cross_fold"]["images_gain8"], pb["cross_fold"]["image_overlap"]))
    cm = pb["primary_cross_fold_coco_match_change"]
    cm7, cm8 = cm["0.75"]["fold7_loss_side"], cm["0.75"]["fold8_gain_side"]
    A("- COCO 匹配变化（主比较，IoU 0.75）：**fold7** 丢失 %d 个匹配（该侧无合格框 %d、有合格框但被分走 %d），"
      "同折新增 %d、净 %+d；**fold8** 新增 %d 个匹配（原侧无合格框 %d、有合格框但被分走 %d），"
      "同折丢失 %d、净 %+d。"
      % (cm7["lost_match"], cm7["lost_reason"]["no_qualifying_box_in_pg"],
         cm7["lost_reason"]["qualifying_box_present_but_assigned_elsewhere"],
         cm7["newly_matched"], cm7["net"],
         cm8["newly_matched"], cm8["new_reason"]["no_qualifying_box_in_b0"],
         cm8["new_reason"]["qualifying_box_present_but_assigned_elsewhere"],
         cm8["lost_match"], cm8["net"]))
    for lvl in ("0.75", "0.50"):
        io = cm[lvl]["identity_overlap"]
        A("- **两套计数的逐目标对照（IoU %s）**：几何覆盖 loss 侧(fold7) %d 个，其中与 COCO 丢失匹配重合 %d、"
          "仅几何覆盖有 %d、仅 COCO 有 %d；几何覆盖 gain 侧(fold8) %d 个，其中与 COCO 新增匹配重合 %d、"
          "仅几何覆盖有 %d、仅 COCO 有 %d。该档净增：fold7 几何覆盖 %+d / COCO %+d；"
          "fold8 几何覆盖 %+d / COCO %+d。"
          % (lvl, io["coverage_loss_fold7"], io["coverage_loss_fold7_and_coco_lost"],
             io["coverage_loss_fold7_only"], io["coco_lost_fold7_only"],
             io["coverage_gain_fold8"], io["coverage_gain_fold8_and_coco_new"],
             io["coverage_gain_fold8_only"], io["coco_new_fold8_only"],
             io["coverage_net_fold7"], io["coco_net_fold7"],
             io["coverage_net_fold8"], io["coco_net_fold8"]))
    io75, io50 = cm["0.75"]["identity_overlap"], cm["0.50"]["identity_overlap"]
    A("- **净增相同不代表逐目标匹配相同**：几何覆盖（逐 GT `best_iou_any`）与 COCO 匹配"
      "（COCOeval 一分一配）是两个口径，上两行的集合身份即为二者的差异；两者不得互推，"
      "也不得用其一解释另一。0.75 档两侧集合逐目标完全相同（错配 0 / 0）：fold7 失去 %d = COCO 丢失 %d，"
      "fold8 得到 %d = COCO 新增 %d。0.50 档则已分叉：fold8 的覆盖侧得到 %d 个、COCO 侧只新增 %d 个"
      "（有 %d 个目标几何上已跨过 0.50 却未在 COCO 的 0.50 档匹配上），两折净增也因此不同号"
      "（fold7 %+d vs %+d，fold8 %+d vs %+d）——同一批预测在不同 IoU 档下并不对应同一组目标。"
      % (io75["coverage_loss_fold7"], io75["coco_lost_fold7"],
         io75["coverage_gain_fold8"], io75["coco_new_fold8"],
         io50["coverage_gain_fold8"], io50["coco_new_fold8"],
         io50["coverage_gain_fold8_only"],
         io50["coverage_net_fold7"], io50["coco_net_fold7"],
         io50["coverage_net_fold8"], io50["coco_net_fold8"]))
    A("- **措辞边界**：上述 p 值与 δ 只是描述统计，**不构成机制证据**；"
      "p>0.05 不表示「证明相同」，较小的 δ 也不表示「同一机制」。%s" % pb["power_caveat"])
    A("")
    A("### 5.5 三折各自内部对（几何覆盖损失 vs 几何覆盖增益）描述统计")
    A("")
    A("| fold | n_loss | n_gain | ann_area p / δ | 聚集度 p / δ | abs(loss)/abs(gain) | 角色 |")
    A("|---|---|---|---|---|---|---|")
    for fold in (6, 7, 8):
        v = pb["folds"][fold]["paired_features_internal"]
        cr2 = v["count_ratio_loss_over_gain"]
        ta, td = v["tests"]["ann_area"], v["cliffs_delta"]["ann_area"]
        tn, tnd = v["tests"]["nn_dist_over_sqrt_area"], v["cliffs_delta"]["nn_dist_over_sqrt_area"]
        A("| %d | %d | %d | %s / %s | %s / %s | %s | %s |" % (
            fold, v["n_a"], v["n_b"], _p(ta["p_value"]), _p(td, "%.3f"),
            _p(tn["p_value"]), _p(tnd, "%.3f"),
            _p(cr2["ratio"] if cr2 else None, "%.3f"),
            "参考（fold6 已被用于选方案）" if fold == 6 else "新增折"))
    A("")
    A("> %s" % pb["fold6_role"])
    A("")
    A("### 5.6 高分背景框数量 vs 精度变化")
    A("")
    A("| fold | Δbkg ≥0.9 | Δbkg ≥0.7 | Δbackground_errors | Δ高分小框背景(与已标注 GT 最大 IoU<0.1) | ΔmAP | ΔmAP75 |")
    A("|---|---|---|---|---|---|---|")
    hs = pb["high_score_bkg_vs_metric"]["per_fold"]
    for fold in (6, 7, 8):
        h = hs[fold]
        A("| %d | %+d | %+d | %+d | %+d | %+.4f | %+.4f |" % (
            fold, h["delta_bkg_ge_0.9"], h["delta_bkg_ge_0.7"], h["delta_background_errors"],
            h["delta_high_score_small_bkg_iou_lt_0.1"], h["delta_mAP"], h["delta_mAP_75"]))
    A("")
    A("> %s" % pb["high_score_bkg_vs_metric"]["intersection_definition"])
    A("")
    A("> %s" % pb["high_score_bkg_vs_metric"]["statement"])
    A("")
    A("> %s" % pb["high_score_bkg_vs_metric"]["implication"])
    A("")
    A("### 5.7 背景误检分档 B0→PG 变化（legacy 口径，各档之和 = Δbackground_errors）")
    A("")
    bands = sorted(pb["folds"][6]["fp_by_band"])
    A("| fold | Δbackground_errors | " + " | ".join(bands) + " |")
    A("|" + "---|" * (len(bands) + 2))
    for fold in (6, 7, 8):
        fb = pb["folds"][fold]["fp_by_band"]
        A("| %d | %+d | %s |" % (
            fold, sum(fb[b]["delta"] for b in bands),
            " | ".join("%+d" % fb[b]["delta"] for b in bands)))
    A("")
    A("### 5.8 术语与口径")
    A("")
    A("- 几何覆盖翻转：%s" % pb["terminology"]["geometric_coverage_flip"])
    A("- COCO 匹配变化：%s" % pb["terminology"]["coco_match_change"])
    A("")
    A("## 6. Part C：候选方向（证据表，脚本不选模块）")
    A("")
    A("%s" % pc["ordered_by"])
    A("")
    A("%s" % pc["selection_status"])
    A("")
    A("> %s" % pc["note"])
    A("")
    for it in pc["items"]:
        A("### %s：%s" % (it["mode"], it["direction"]))
        A("")
        if it.get("status"):
            A("- 状态：%s" % it["status"])
        A("- 支持证据（fold6_b0 / fold7_b0 / fold8_b0）：")
        for line in it["support_summary"]:
            A("  - %s" % line)
        A("- 反证/不利证据：%s" % it["counter"])
        A("- 可证伪陈述：%s" % it["falsifiable"])
        A("")
    A("> %s" % pc["boundary"])
    A("")
    A("## 7. 边界与非目标")
    A("")
    A("- 不训练、不重跑推理；只读六组导出预测与各自 test.json。")
    A("- 不据本测试集调整任何 score/NMS 阈值；不叠加 M2/FG。")
    A("- 成因字段（未产生好框 / 被分数过滤 / 被 NMS 抑制）本轮**一律记“未知”**，不由低重复率或低覆盖率倒推。")
    A("- 不对 APl / `AR_l@1000` 下结论（COCO 口径 large 仅 2 个 GT）。")
    A("- 本轮**不产生新的验证证据**：232 图测试集此前已用于 AP75 讨论，本轮只得诊断线索。")
    A("  后续独立留出必须是**此前未参与选择的 SAR 图像/场景**；仅更换 3-shot 标注折或随机种子、")
    A("  但仍评价同一 232 图，**不构成新的独立测试集**。")
    A("- `ShipRSImageNet` 实为光学遥感（见 `audit_softnms_cr4o9epx/CORRECTION.md`），其“独立 SAR”表述已撤回；")
    A("  ShipRSImageNet test 保持封存，不得当作独立 SAR 验证复用。")
    A("- 候选级（RPN/ROI）归因属阶段二，本轮不执行。")
    A("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 编排
def run_all(out_dir):
    started = time.time()
    started_str = time.strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise AnalysisError("out-dir 已存在且非空，拒绝覆盖：%s\n冲突文件（前 20）：%s"
                            % (out_dir, sorted(os.listdir(out_dir))[:20]))
    os.makedirs(out_dir, exist_ok=True)

    # 步骤 1 依赖自审
    import_audit = audit_imports(os.path.abspath(__file__))

    # 步骤 2 输入核对
    for r in RUN_SPECS:
        for fn in ("predictions.bbox.json", "test.json", "metrics.json", "metadata.json"):
            p = os.path.join(r["eval_dir"], fn)
            if not os.path.exists(p):
                raise AnalysisError("缺少输入: %s" % p)
    shas = {r["run_id"]: sha256_file(os.path.join(r["eval_dir"], "test.json")) for r in RUN_SPECS}
    if len(set(shas.values())) != 1 or list(shas.values())[0] != FROZEN_TEST_SHA:
        raise AnalysisError("六个 test.json 哈希不一致或与冻结值不符: %s" % json.dumps(shas, indent=2))

    gt_path = os.path.join(RUN_SPECS[0]["eval_dir"], "test.json")
    id2gt, images = load_gt(gt_path)
    if len(images) != FROZEN_NUM_IMAGES:
        raise AnalysisError("图像数 %d != %d" % (len(images), FROZEN_NUM_IMAGES))
    n_gt = sum(len(v) for v in id2gt.values())
    if n_gt != FROZEN_NUM_GT:
        raise AnalysisError("非 crowd GT 数 %d != %d" % (n_gt, FROZEN_NUM_GT))
    feats = gt_features(id2gt)

    # 步骤 3 合成用例
    synth = synthetic_cases()

    # 逐 run 汇总
    bundles, runs_meta = {}, []
    for r in RUN_SPECS:
        b = build_run(r, id2gt, images, feats)
        bundles[r["run_id"]] = b
        runs_meta.append(r)

    # 步骤 4 影子对照
    shadow_specs = [{"run_id": r["run_id"], "eval_dir": r["eval_dir"],
                     "legacy": bundles[r["run_id"]]["legacy"]} for r in RUN_SPECS]
    shadow = shadow_frozen_script(gt_path, shadow_specs)

    self_test = {"dependency_audit": import_audit, "synthetic_cases": synth,
                 "shadow_frozen_script": shadow,
                 "all_passed": bool(shadow["all_equal"]) and all(c["ok"] for c in synth)}

    # 步骤 5 分析
    pa = part_a(bundles, id2gt)
    pb = part_b(bundles, id2gt, feats)
    pc = part_c(bundles, pa, pb)

    plan_sha = sha256_file(PLAN_PATH)
    plan_v1_sha = sha256_file(PLAN_V1_PATH)

    # 步骤 6 写产物
    os.makedirs(os.path.join(out_dir, "per_run"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cross_fold"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "partA"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "partB"), exist_ok=True)

    gt_fields = ["run_id", "image_id", "gt_id", "ann_area", "bbox_area", "size_bin_coco", "size_bin_bbox",
                 "cx", "cy", "nn_center_dist", "nn_dist_over_sqrt_area", "n_gt_in_image", "occupied",
                 "matched_pred_iou", "matched_pred_score", "best_iou_any", "best_pred_score_any",
                 "n_preds_iou_ge_0_5", "coverage_bin", "miss_type"]
    pred_fields = ["run_id", "pred_index", "image_id", "score", "x1", "y1", "x2", "y2", "w", "h",
                   "kind", "best_iou", "matched_gt_id", "pred_area_bin"]
    flip_fields = ["fold", "flip_kind", "direction", "defined_by", "image_id", "gt_id", "ann_area",
                   "bbox_area", "size_bin_coco", "nn_dist_over_sqrt_area", "best_iou_b0", "best_iou_pg",
                   "delta_iou", "best_score_b0", "best_score_pg", "n_preds_iou_ge_0_5_b0",
                   "n_preds_iou_ge_0_5_pg", "contested"]
    joint_fields = ["run_id", "pred_area_bin", "score_band", "max_gt_iou_band", "total", "tp",
                    "background", "duplicate"]
    coco_match_fields = ["fold", "iou_level", "image_id", "gt_id", "state", "matched_in_b0",
                         "matched_in_pg", "matched_dt_score_b0", "matched_dt_score_pg",
                         "n_dt_ge_thr_b0", "n_dt_ge_thr_pg", "gt_bbox_area"]
    pr_fields = ["fold", "iou_level", "recall_grid_value", "precision_b0", "precision_pg", "delta",
                 "beyond_max_recall_b0", "beyond_max_recall_pg"]
    img_fields = ["image_id", "file_name", "fold", "n_gt", "bkg", "miss", "imprecise", "is_top_quartile_bkg"]

    for r in RUN_SPECS:
        rid = r["run_id"]
        b = bundles[rid]
        d = os.path.join(out_dir, "per_run", rid)
        os.makedirs(d, exist_ok=True)
        write_json(os.path.join(d, "summary.json"), b["summary"])
        write_csv(os.path.join(d, "per_gt.csv"), sorted(b["per_gt"], key=lambda x: (x["image_id"], x["gt_id"])), gt_fields)
        write_csv(os.path.join(d, "per_pred.csv"),
                  sorted(b["per_pred"], key=lambda x: (x["image_id"], -x["score"], x["pred_index"])), pred_fields)

    write_json(os.path.join(out_dir, "cross_fold", "descriptive_stats.json"), pa)
    # 预测侧联合统计（面积 × 分数 × 最大 GT IoU），六组都给，便于逐组复核
    joint_rows = []
    for rid in [r["run_id"] for r in RUN_SPECS]:
        for c in bundles[rid]["summary"]["pred_side_joint"]["cells"]:
            joint_rows.append(dict(c, run_id=rid))
    write_csv(os.path.join(out_dir, "partA", "joint_cells.csv"),
              sorted(joint_rows, key=lambda x: (x["run_id"], x["pred_area_bin"], x["score_band"],
                                                x["max_gt_iou_band"])), joint_fields)
    per_image_rows = []
    for rid in B0_RUN_IDS:
        fold = bundles[rid]["run"]["fold"]
        v, imgs = per_image_vectors(bundles[rid], id2gt)
        q = _top_quartile_images(v["bkg"], imgs)
        name = {im["id"]: im["file_name"] for im in images}
        for i in imgs:
            per_image_rows.append({"image_id": i, "file_name": name.get(i, ""), "fold": fold,
                                   "n_gt": len(id2gt[i]), "bkg": v["bkg"][i], "miss": v["miss"][i],
                                   "imprecise": v["imprecise"][i],
                                   "is_top_quartile_bkg": int(i in q)})
    write_csv(os.path.join(out_dir, "cross_fold", "per_image.csv"),
              sorted(per_image_rows, key=lambda x: (x["fold"], x["image_id"])), img_fields)
    write_json(os.path.join(out_dir, "partB", "paired.json"), pb)
    all_flips = sorted([f for fold in (6, 7, 8) for f in pb["folds"][fold]["flips"]],
                       key=lambda x: (x["fold"], x["direction"], x["image_id"], x["gt_id"]))
    write_csv(os.path.join(out_dir, "partB", "geometric_coverage_flips.csv"), all_flips, flip_fields)
    coco_rows = []
    for fold in (6, 7, 8):
        for lvl in ("0.75", "0.50"):
            for r_ in pb["folds"][fold]["coco_match_change"][lvl]["changed_rows"]:
                coco_rows.append(dict(r_, fold=fold))
    write_csv(os.path.join(out_dir, "partB", "coco_match_changes.csv"),
              sorted(coco_rows, key=lambda x: (x["fold"], x["iou_level"], x["state"], x["image_id"], x["gt_id"])),
              coco_match_fields)
    pr_rows = []
    for fold in (6, 7, 8):
        for lvl in ("0.75", "0.50"):
            pr = pb["folds"][fold]["pr_curve"][lvl]
            mb, mp = pr["max_recall_all_dets"]["b0"], pr["max_recall_all_dets"]["pg"]
            for i, rv in enumerate(pr["recall_grid"]):
                bv, pv = pr["precision_b0"][i], pr["precision_pg"][i]
                pr_rows.append({"fold": fold, "iou_level": lvl, "recall_grid_value": rv,
                                "precision_b0": bv, "precision_pg": pv,
                                "delta": (pv - bv) if (bv is not None and pv is not None) else None,
                                "beyond_max_recall_b0": bool(rv > mb + 1e-12),
                                "beyond_max_recall_pg": bool(rv > mp + 1e-12)})
    write_csv(os.path.join(out_dir, "partB", "pr_curves.csv"), pr_rows, pr_fields)
    write_json(os.path.join(out_dir, "candidates.json"), pc)
    write_json(os.path.join(out_dir, "self_test.json"), self_test)
    write_text(os.path.join(out_dir, "误差分析报告.md"),
               build_report(pa, pb, pc, bundles, runs_meta, self_test, shadow, plan_sha, plan_v1_sha))

    ended = time.time()
    manifest = {
        "analysis_nature": ANALYSIS_NATURE,
        "mechanism_conclusion": MECHANISM_CONCLUSION,
        "analysis_plan": {"path": PLAN_PATH, "sha256": plan_sha},
        "analysis_plan_v1": {"path": PLAN_V1_PATH, "sha256": plan_v1_sha,
                             "role": "历史材料；其 §5 共同瓶颈判据与 §6 机制裁定已于 2026-09-29 撤除"},
        "script": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pycocotools_file": os.path.abspath(sys.modules["pycocotools"].__file__)},
        "frozen_test_sha256": FROZEN_TEST_SHA, "num_images": len(images), "num_gt_noncrowd": n_gt,
        "runs": [{"run_id": r["run_id"], "fold": r["fold"], "variant": r["variant"],
                  "eval_dir": r["eval_dir"],
                  "predictions_sha256": bundles[r["run_id"]]["summary"]["predictions_sha256"],
                  "metrics_sha256": sha256_file(os.path.join(r["eval_dir"], "metrics.json")),
                  "metadata_sha256": sha256_file(os.path.join(r["eval_dir"], "metadata.json")),
                  "postproc": bundles[r["run_id"]]["summary"]["postproc"]} for r in RUN_SPECS],
        "constants": {"IOU_THR": IOU_THR, "AP75_IOU": AP75_IOU, "NEAR_IOU": NEAR_IOU,
                      "THRESHOLDS": list(THRESHOLDS), "BANDS": BANDS, "PERM_SEED": PERM_SEED,
                      "N_PERM": N_PERM,
                      "PRED_SCORE_BANDS": PRED_SCORE_BANDS, "MAXGT_IOU_BANDS": MAXGT_IOU_BANDS,
                      "PR_CURVE_RECALL_CEILING_SOURCE": "ev.eval['recall'][IoU档, cat 0, areaRng='all', maxDet=100]",
                      "PR_CURVE_BEYOND_CEILING_MARKER": (
                          "本机 pycocotools 对超过最大可达召回的召回点存 0.0（非 COCO 原版 -1）；"
                          "实测 n_precision_negative 六组全为 0，读曲线须按各侧召回上限截断")},
        "outputs": ["summary.json", "manifest.json", "self_test.json", "run_metadata.json",
                    "candidates.json", "误差分析报告.md",
                    "per_run/<run_id>/{summary.json,per_gt.csv,per_pred.csv}",
                    "cross_fold/{descriptive_stats.json,per_image.csv}",
                    "partA/joint_cells.csv",
                    "partB/{paired.json,geometric_coverage_flips.csv,coco_match_changes.csv,pr_curves.csv}"],
        "determinism_note": ("统计产物按 indent=2, ensure_ascii=False, sort_keys=True 序列化；"
                             "CSV 行序固定。run_metadata.json 含时间/耗时/输出绝对路径，不参与逐字节一致性。")}
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    run_metadata = {
        "started_at": started_str, "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": ended - started, "out_dir_abs": os.path.abspath(out_dir),
        "pycocotools_import_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "note": "本文件含时间与绝对路径，按既定规则不计入确定性逐字节比较。"}
    write_json(os.path.join(out_dir, "run_metadata.json"), run_metadata)

    summary = {
        "status": "complete",
        "runs": {r["run_id"]: bundles[r["run_id"]]["summary"]["counts"] for r in RUN_SPECS},
        "identity_all_ok": all(bundles[r["run_id"]]["summary"]["identity_check"]["ok"] for r in RUN_SPECS),
        "metric_recompute_all_ok": all(bundles[r["run_id"]]["summary"]["metric_recompute_check"]
                                       ["all_equal_at_stored_precision"] for r in RUN_SPECS),
        "self_test_all_passed": self_test["all_passed"],
        "analysis_nature": ANALYSIS_NATURE,
        "part_a_conclusion": {"designation": pa["conclusion"]["designation"],
                             "persistent_common_error_pattern": pa["conclusion"]["persistent_common_error_pattern"]},
        "part_b_mechanism_conclusion": pb["mechanism_conclusion"],
        "part_b_primary_descriptive": {
            "n_a": pb["primary_cross_fold_features"]["n_a"],
            "n_b": pb["primary_cross_fold_features"]["n_b"],
            "ann_area": {"p": pb["primary_cross_fold_features"]["tests"]["ann_area"]["p_value"],
                         "cliffs_delta": pb["primary_cross_fold_features"]["cliffs_delta"]["ann_area"]},
            "nn_dist_over_sqrt_area": {
                "p": pb["primary_cross_fold_features"]["tests"]["nn_dist_over_sqrt_area"]["p_value"],
                "cliffs_delta": pb["primary_cross_fold_features"]["cliffs_delta"]["nn_dist_over_sqrt_area"]}},
        "high_score_bkg_vs_metric": {
            "statement": pb["high_score_bkg_vs_metric"]["statement"],
            "implication": pb["high_score_bkg_vs_metric"]["implication"]},
        "withdrawn_from_v1": pa["conclusion"]["withdrawn_from_v1"],
        "namespace_warning": "legacy / coco / coverage 三套计数禁止相加或互相比较",
        "causes_unknown": {"no_good_box_generated": "未知", "good_box_filtered_by_score": "未知",
                           "good_box_suppressed_by_nms": "未知"},
        "candidate_direction_selected": None,
        "boundary": ("不训练、不调阈值、不叠加 M2/FG；成因字段记“未知”；"
                     "APl/AR_l 不作结论；本轮不产生新的独立验证证据；不指定共同瓶颈、不下机制裁定。")}
    write_json(os.path.join(out_dir, "summary.json"), summary)

    print("[输入核对] 身份恒等式：六组全部成立")
    print("[输入核对] num_preds 与 metadata 一致：六组")
    print("[输入核对] 非 crowd GT = %d，图像 = %d" % (n_gt, len(images)))
    print("[依赖自审] 禁止导入：无（%s）" % ", ".join(import_audit["top_level_modules"]))
    print("[合成用例] %d 项全部通过" % len(synth))
    print("[影子对照] legacy_* 逐项一致（被对照脚本 sha256 %s）" % shadow["frozen_script_sha256"][:12])
    print("[COCO 复算] 六组在保存精度下一致")
    print("[Part A] %s（持续共同错误模式：%s）"
          % (pa["conclusion"]["designation"], pa["conclusion"]["persistent_common_error_pattern"]))
    print("[Part B] 机制结论：%s" % pb["mechanism_conclusion"])
    print("[Part B] %s" % pb["high_score_bkg_vs_metric"]["statement"])
    print("产物目录：%s" % os.path.abspath(out_dir))
    return 0


def main():
    ap = argparse.ArgumentParser(description="三折新 B0 与 PG-both 离线误差分析")
    ap.add_argument("mode", choices=["run"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    try:
        return run_all(os.path.abspath(args.out_dir))
    except AnalysisError as e:
        sys.stderr.write("[分析失败] %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
