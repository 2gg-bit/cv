"""Effect table for the reweight_l1 round, computed from the existing predictions only.

Reads the six already-produced `predictions.bbox.json` (three folds x B0 / reweight_l1)
and reports how the high-score small-box background detections and the matching true
positives moved.  No training, no re-inference, no GPU.

The counting protocol is NOT reimplemented here: it is imported from the frozen
error-analysis implementation (`code/error_analysis.py`), whose `legacy_*` namespace was
shadow-verified against the frozen `score_thr_bkg.py`.  That script is executed again
here as part of the run, so a mismatch in the shared keys aborts instead of producing a
table.

What the table is and is not: it is a check on whether the outcome matches the original
hypothesis (suppressing high-scoring small background boxes).  It is not a causal
demonstration -- the reweight changes optimisation, and this table only counts exported
detections.  In particular, a background count that does not fall does not remove the
measured mAP gain recorded in `paired_coco_delta.json`.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime

ROUND_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_ROOT = os.path.dirname(os.path.dirname(ROUND_DIR))
ERROR_ANALYSIS = os.path.join(
    ABLATION_ROOT, "ablation_configs", "error_analysis_b0_pgboth", "code",
    "error_analysis.py")

HIGH_SCORE_EXTENSION = 0.9  # 扩展切点；冻结口径里的"高分"是 score >= 0.7
SMALL_AREA_DEF = "bbox w*h < 32^2"


def load_error_analysis():
    spec = importlib.util.spec_from_file_location("_frozen_error_analysis",
                                                  ERROR_ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_specs():
    specs = []
    for fold in (6, 7, 8):
        for variant in ("b0", "reweight_l1"):
            d = os.path.join(ABLATION_ROOT, "ablation_configs",
                             "fold%d_seed678" % fold,
                             "%s_evaluation" % variant)
            specs.append({"run_id": "fold%d_%s" % (fold, variant), "fold": fold,
                          "variant": variant, "eval_dir": d})
    return specs


def small_high_counts(per_pred, threshold):
    """小框(<32^2) 且分数 >= threshold 的背景/TP 计数（冻结 size_bin 的扩展切片）。"""
    out = {"bkg": 0, "tp": 0, "dupe": 0, "total": 0}
    for r in per_pred:
        if r["pred_area_bin"] != "small" or r["score"] < threshold:
            continue
        out["total"] += 1
        out[r["kind"]] += 1
    return out


def metrics_for(bundle, ea):
    summary = bundle["summary"]
    legacy = summary["legacy"]
    joint = summary["pred_side_joint"]
    small_high = joint["cross_small_x_high_score"]["small_and_high_score"]
    ext = summary["coverage_extensions"]
    return {
        "num_preds": summary["counts"]["num_preds"],
        "tp": summary["counts"]["legacy_true_positives"],
        "bkg": summary["counts"]["legacy_background_errors"],
        "dupe": summary["counts"]["legacy_duplicates"],
        "bkg_ge0.9": legacy["bkg_ge0.9"],
        "tp_ge0.9": legacy["tp_ge0.9"],
        "bkg_small": ext["bkg_by_pred_bbox_area_bin"]["small"],
        "bkg_small_high0.7": small_high["background"],
        "tp_small_high0.7": small_high["tp"],
        "small_high0.7_total": small_high["total"],
        "bkg_small_ge0.9_extension": small_high_counts(bundle["per_pred"],
                                                       HIGH_SCORE_EXTENSION)["bkg"],
        "tp_small_ge0.9_extension": small_high_counts(bundle["per_pred"],
                                                     HIGH_SCORE_EXTENSION)["tp"],
    }


ROW_LABELS = [
    ("num_preds", "预测总数"),
    ("bkg", "背景误检总数"),
    ("bkg_ge0.9", "　其中分数 >= 0.9"),
    ("bkg_small", "　其中小框 < 32^2"),
    ("bkg_small_high0.7", "　其中小框且分数 >= 0.7"),
    ("bkg_small_ge0.9_extension", "　其中小框且分数 >= 0.9（扩展切点）"),
    ("tp", "TP 总数"),
    ("tp_ge0.9", "　其中分数 >= 0.9"),
    ("tp_small_high0.7", "　其中小框且分数 >= 0.7"),
    ("tp_small_ge0.9_extension", "　其中小框且分数 >= 0.9（扩展切点）"),
    ("small_high0.7_total", "小框且分数 >= 0.7 的预测数（含背景/TP/重复）"),
]

TABLE_METRICS = [k for k, _ in ROW_LABELS]


def build_table(per_run):
    folds = (6, 7, 8)
    table = {}
    for key in TABLE_METRICS:
        row = {"baseline": {}, "variant": {}, "delta": {}}
        for fold in folds:
            b = per_run["fold%d_b0" % fold][key]
            v = per_run["fold%d_reweight_l1" % fold][key]
            row["baseline"][str(fold)] = b
            row["variant"][str(fold)] = v
            row["delta"][str(fold)] = v - b
        deltas = [row["delta"][str(f)] for f in folds]
        row["mean_delta"] = sum(deltas) / float(len(deltas))
        row["all_folds_same_sign"] = (all(d > 0 for d in deltas)
                                      or all(d < 0 for d in deltas))
        table[key] = row
    return table


def render_markdown(table):
    lines = ["| 指标 | " + " | ".join(
        "f%d B0 | f%d 变体 | f%d Δ" % (f, f, f) for f in (6, 7, 8)) +
        " | 平均 Δ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, label in ROW_LABELS:
        row = table[key]
        cells = []
        for fold in (6, 7, 8):
            cells += ["%d" % row["baseline"][str(fold)],
                      "%d" % row["variant"][str(fold)],
                      "%+d" % row["delta"][str(fold)]]
        cells.append("%+.2f" % row["mean_delta"])
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True,
                        help="output JSON path; refuses to overwrite")
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    if os.path.exists(out_path):
        raise SystemExit("refusing to overwrite %s" % out_path)
    if not os.path.isdir(os.path.dirname(out_path)):
        raise SystemExit("output directory missing: %s" % os.path.dirname(out_path))

    ea = load_error_analysis()
    specs = run_specs()
    for spec in specs:
        for fn in ("predictions.bbox.json", "test.json", "metrics.json",
                   "metadata.json"):
            p = os.path.join(spec["eval_dir"], fn)
            if not os.path.exists(p):
                raise SystemExit("missing input: %s" % p)

    gt_path = os.path.join(specs[0]["eval_dir"], "test.json")
    id2gt, images = ea.load_gt(gt_path)
    if len(images) != ea.FROZEN_NUM_IMAGES:
        raise SystemExit("images %d != %d" % (len(images), ea.FROZEN_NUM_IMAGES))
    n_gt = sum(len(v) for v in id2gt.values())
    if n_gt != ea.FROZEN_NUM_GT:
        raise SystemExit("non-crowd GT %d != %d" % (n_gt, ea.FROZEN_NUM_GT))
    feats = ea.gt_features(id2gt)

    shas = {s["run_id"]: sha256_file(os.path.join(s["eval_dir"], "test.json"))
            for s in specs}
    if len(set(shas.values())) != 1 or list(shas.values())[0] != ea.FROZEN_TEST_SHA:
        raise SystemExit("test.json hashes differ or do not match the frozen value")

    bundles, per_run = {}, {}
    for spec in specs:
        bundle = ea.build_run(spec, id2gt, images, feats)
        bundles[spec["run_id"]] = bundle
        per_run[spec["run_id"]] = metrics_for(bundle, ea)

    shadow_specs = [{"run_id": s["run_id"], "eval_dir": s["eval_dir"],
                     "legacy": bundles[s["run_id"]]["legacy"]} for s in specs]
    shadow = ea.shadow_frozen_script(gt_path, shadow_specs)

    table = build_table(per_run)
    folds = (6, 7, 8)
    identity_ok = all(
        per_run["fold%d_%s" % (f, v)]["bkg"] + per_run["fold%d_%s" % (f, v)]["dupe"]
        + per_run["fold%d_%s" % (f, v)]["tp"]
        == per_run["fold%d_%s" % (f, v)]["num_preds"]
        for f in folds for v in ("b0", "reweight_l1"))

    mAP_s = {}
    paired_path = os.path.join(ROUND_DIR, "paired_coco_delta.json")
    if os.path.exists(paired_path):
        paired = json.load(open(paired_path, "r", encoding="utf-8"))
        for row in paired["pairs"]:
            mAP_s[row["label"]] = {
                "baseline": row["baseline"]["raw"]["mAP_s"],
                "variant": row["variant"]["raw"]["mAP_s"],
                "delta": row["delta_raw"]["mAP_s"]}

    payload = {
        "step": "效果对照表：高分小框背景误检与对应 TP 的变化",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "computed_from": "已有 predictions.bbox.json（未重跑推理、未训练、未用 GPU）",
        "protocol": {
            "source_of_counts": "冻结的 code/error_analysis.py（legacy_* 贪心口径，"
                                "与 score_thr_bkg.py 影子对照）",
            "error_analysis_path": ERROR_ANALYSIS,
            "error_analysis_sha256": sha256_file(ERROR_ANALYSIS),
            "iou_thr": ea.IOU_THR,
            "high_score_frozen": "score >= 0.7（冻结代码里的 PRED_SCORE_BANDS 高分段）",
            "high_score_extension": HIGH_SCORE_EXTENSION,
            "small_definition": SMALL_AREA_DEF,
            "shadow_frozen_script": {
                "frozen_script": shadow["frozen_script"],
                "frozen_script_sha256": shadow["frozen_script_sha256"],
                "all_equal": shadow["all_equal"],
                "compared_keys_per_run": {k: len(v["compared_keys"])
                                          for k, v in shadow["per_run"].items()},
            },
            "identity_check_bkg_dupe_tp_eq_num_preds": identity_ok,
        },
        "runs": {s["run_id"]: {"eval_dir": s["eval_dir"],
                               "predictions_sha256": sha256_file(
                                   os.path.join(s["eval_dir"], "predictions.bbox.json")),
                               "checkpoint_sha256": json.load(open(
                                   os.path.join(s["eval_dir"], "metadata.json"),
                                   "r", encoding="utf-8")).get("checkpoint_sha256")}
                 for s in specs},
        "per_run_counts": per_run,
        "table": table,
        "table_markdown": render_markdown(table),
        "coco_mAP_s_from_paired_coco_delta": mAP_s,
        "reading": {
            "purpose": "判断结果是否与最初假设一致（加权抑制高分小框背景）",
            "not_a_causal_proof": ("该表只计数导出预测，不区分成因（未产生好框 / 被分数过滤 / "
                                   "被 NMS 抑制一律未知），不能作为因果证明"),
            "negative_bkg_result_does_not_erase_gain": ("即使背景数没有下降，也不影响 "
                                                        "paired_coco_delta.json 里已记录的 mAP 增益"),
            "exploratory": "232 图测试集此前已参与方案选择，本轮仍是探索性配对实验，非独立验证",
        },
        "status": "complete" if identity_ok and shadow["all_equal"] else "inconsistent",
    }
    if payload["status"] != "complete":
        raise SystemExit("consistency checks failed; nothing written")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")

    print(payload["table_markdown"])
    print()
    print("shadow vs frozen score_thr_bkg.py: all_equal=%s (keys compared per run: %s)"
          % (shadow["all_equal"], ", ".join(
              "%s=%d" % (k, len(v["compared_keys"]))
              for k, v in sorted(shadow["per_run"].items()))))
    print("wrote %s" % out_path)


if __name__ == "__main__":
    sys.exit(main())
