"""Independent M0 localization audit — offline stage-wise IoU tracking.

Consumes the per-fold diagnosis JSON produced by
`tools/diagnose_teacher_stages.py` (frozen-baseline inference path) on the new
M0 (seed=678, `work_dirs/m0_seed678/training/{fold}/iter_32000.pth`).

For each teacher and each IoU threshold in {0.5, 0.75, 0.85, 0.9}, it answers:

  Q1. After RPN and after ROI regression, is an accurate box already present?
      -> rpn_cov / pre_cov (GT covered by any proposal / any regressed box).

  Q2. Is an accurate box lost because its classification score is too low?
      -> score_filter_loss = pre_cov - scr_cov (scr = regressed box with
         p_ship > score_thr).

  Q3. Does NMS keep a worse-localized box for the *same* target?
      -> replay greedy NMS with suppression recording; count pairs where the
         suppressed box (same GT) has IoU >= thr while the kept suppressor has
         IoU < thr (a good box was dropped in favour of a worse one).

Failure-mode decomposition (the core distinction the audit must make):
      box_inaccurate  = GT not covered by any regressed box at thr
                        ("框本身不准" — no accurate box exists to keep)
      good_not_retained = GT covered pre-NMS but absent from the final
                        detections ("好框没有被保留" — an accurate box existed
                        but was lost in score filtering / NMS / truncation).

This script only reads artifacts; it never modifies weights, configs,
predictions, logs or the original experiment directories. Outputs go to a
fresh per-fold analysis directory.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

IOU_THRESHOLDS = [0.5, 0.75, 0.85, 0.9]
SCORE_THR = 0.05
NMS_IOU = 0.5
TEACHERS = ["teacher1", "teacher2"]


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rescale(box, sf):
    return [box[0] / sf[0], box[1] / sf[1], box[2] / sf[2], box[3] / sf[3]]


def nms_with_suppression(boxes, scores, thr=NMS_IOU):
    """Greedy NMS replay (matches mmcv batched_nms ordering).

    Returns (kept_indices, suppression_pairs) where each pair is
    (suppressed_idx, suppressor_idx).
    """
    if not boxes:
        return [], []
    order = np.argsort(np.asarray(scores, dtype=float))[::-1]
    keep = []
    pairs = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ovr = np.array([iou(boxes[i], boxes[j]) for j in order[1:]], dtype=float)
        sup = order[1:][ovr > thr]
        for j in sup:
            pairs.append((int(j), i))
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep, pairs


def build_gt_map(gt_path):
    from pycocotools.coco import COCO
    coco = COCO(gt_path)
    fn2gt = {}
    for im in coco.dataset['images']:
        gts = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=im['id'])):
            if a.get('iscrowd', 0):
                continue
            x, y, w, h = a['bbox']
            gts.append((a['id'], [x, y, x + w, y + h]))
        fn2gt[im['file_name']] = gts
    return coco, fn2gt


def verify(diag, coco, recorded_ap):
    """Recompute teacher2 AP from teacher2_formal and compare to recorded."""
    from pycocotools.cocoeval import COCOeval
    preds = []
    fn2id = {im['file_name']: im['id'] for im in coco.dataset['images']}
    cat_id = coco.getCatIds()[0]
    for r in diag['records']:
        img_id = fn2id[r['filename']]
        for b in r['teacher2_formal']['boxes']:
            x1, y1, x2, y2, s = b
            preds.append({'image_id': img_id, 'category_id': cat_id,
                          'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(s)})
    dt = coco.loadRes(preds)
    ce = COCOeval(coco, dt, 'bbox')
    ce.evaluate(); ce.accumulate(); ce.summarize()
    ap = round(float(ce.stats[0]), 4)
    ap50 = round(float(ce.stats[1]), 4)
    ap75 = round(float(ce.stats[2]), 4)
    return {'recomputed_ap': ap, 'ap50': ap50, 'ap75': ap75,
            'recorded_ap': recorded_ap,
            'ap_match': abs(ap - recorded_ap) < 0.005,
            'num_preds': len(preds)}


def stage_coverage(diag, fn2gt, out_dir):
    rows = []
    records = diag['records']
    for t in TEACHERS:
        for thr in IOU_THRESHOLDS:
            rpn_cov = pre_cov = scr_cov = nms_cov = final_cov = final_1to1 = 0
            total = 0
            for r in records:
                sf = r['scale_factor']
                rpn = [rescale(b[:4], sf) for b in r[t]['rpn_proposals']['boxes']]
                pre = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
                sc = [float(s[0]) for s in r[t]['roi_pre_nms']['scores']]
                fin = [rescale(b[:4], sf) for b in r[t]['final_det']['boxes']]
                scr_idx = [i for i, s in enumerate(sc) if s > SCORE_THR]
                scr_boxes = [pre[i] for i in scr_idx]
                scr_scores = [sc[i] for i in scr_idx]
                nms_keep, _ = nms_with_suppression(scr_boxes, scr_scores)
                nms_boxes = [scr_boxes[i] for i in nms_keep]
                for gt_id, gt in fn2gt[r['filename']]:
                    total += 1
                    if any(iou(gt, p) >= thr for p in rpn): rpn_cov += 1
                    if any(iou(gt, p) >= thr for p in pre): pre_cov += 1
                    if any(iou(gt, p) >= thr for p in scr_boxes): scr_cov += 1
                    if any(iou(gt, p) >= thr for p in nms_boxes): nms_cov += 1
                    if any(iou(gt, p) >= thr for p in fin): final_cov += 1
                    if fin and max(iou(gt, p) for p in fin) >= thr: final_1to1 += 1
            rows.append({
                'teacher': t, 'iou_thr': thr, 'total_gt': total,
                'rpn_cov': rpn_cov, 'pre_cov': pre_cov,
                'scr_cov': scr_cov, 'nms_cov': nms_cov, 'final_cov': final_cov,
                'final_1to1': final_1to1,
                # loss decomposition (cumulative funnel)
                'score_filter_loss': pre_cov - scr_cov,
                'nms_loss': scr_cov - nms_cov,
                'maximg_loss': nms_cov - final_cov,
            })
    with open(os.path.join(out_dir, 'stage_coverage.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def loss_attribution(diag, fn2gt, out_dir):
    """Per-GT, per-threshold failure-mode classification (teacher2 focus)."""
    rows = []
    records = diag['records']
    for thr in IOU_THRESHOLDS:
        for t in TEACHERS:
            for r in records:
                sf = r['scale_factor']
                pre = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
                sc = [float(s[0]) for s in r[t]['roi_pre_nms']['scores']]
                fin = [rescale(b[:4], sf) for b in r[t]['final_det']['boxes']]
                scr_idx = [i for i, s in enumerate(sc) if s > SCORE_THR]
                scr_boxes = [pre[i] for i in scr_idx]
                scr_scores = [sc[i] for i in scr_idx]
                nms_keep, _ = nms_with_suppression(scr_boxes, scr_scores)
                nms_boxes = [scr_boxes[i] for i in nms_keep]
                for gt_id, gt in fn2gt[r['filename']]:
                    pre_c = any(iou(gt, p) >= thr for p in pre)
                    scr_c = any(iou(gt, p) >= thr for p in scr_boxes)
                    nms_c = any(iou(gt, p) >= thr for p in nms_boxes)
                    fin_c = any(iou(gt, p) >= thr for p in fin)
                    if not pre_c:
                        mode = 'box_inaccurate'
                    elif not scr_c:
                        mode = 'score_filter_loss'
                    elif not nms_c:
                        mode = 'nms_suppression'
                    elif not fin_c:
                        mode = 'maximg_truncation'
                    else:
                        mode = 'retained'
                    rows.append({'iou_thr': thr, 'teacher': t,
                                 'filename': r['filename'], 'gt_id': gt_id,
                                 'mode': mode})
    with open(os.path.join(out_dir, 'loss_attribution.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['iou_thr', 'teacher', 'filename', 'gt_id', 'mode'])
        w.writeheader(); w.writerows(rows)
    # aggregate
    agg = {}
    for thr in IOU_THRESHOLDS:
        for t in TEACHERS:
            sub = [x for x in rows if x['iou_thr'] == thr and x['teacher'] == t]
            cnt = Counter(x['mode'] for x in sub)
            agg[f'{thr}_{t}'] = dict(cnt)
    return agg, rows


def nms_quality(diag, fn2gt, out_dir):
    """Q3: does NMS keep a worse-localized box for the same target?

    Replays NMS and records, for each suppression pair sharing the same best
    GT, whether the suppressed (dropped) box had IoU >= thr while the kept
    suppressor had IoU < thr — i.e. a good box was dropped in favour of a worse
    one for the *same* target. Also records cross-target suppression volume.
    """
    rows = []
    records = diag['records']
    for thr in IOU_THRESHOLDS:
        for t in TEACHERS:
            stats = Counter()
            for r in records:
                sf = r['scale_factor']
                pre = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
                sc = [float(s[0]) for s in r[t]['roi_pre_nms']['scores']]
                scr_idx = [i for i, s in enumerate(sc) if s > SCORE_THR]
                scr_boxes = [pre[i] for i in scr_idx]
                scr_scores = [sc[i] for i in scr_idx]
                gts = fn2gt[r['filename']]
                def best_gt(box):
                    bgt, bi = None, 0.0
                    for gt_id, gt in gts:
                        v = iou(gt, box)
                        if v > bi: bi, bgt = v, gt_id
                    return bgt, bi
                _, pairs = nms_with_suppression(scr_boxes, scr_scores)
                for j, i in pairs:
                    bgt_j, bij_j = best_gt(scr_boxes[j])
                    bgt_i, bij_i = best_gt(scr_boxes[i])
                    same = bgt_j is not None and bgt_j == bgt_i
                    if same and bij_j >= thr and bij_i < thr:
                        stats['same_target_good_dropped_for_worse'] += 1
                    elif same:
                        stats['same_target_other'] += 1
                    else:
                        stats['cross_target_suppression'] += 1
            rows.append({'iou_thr': thr, 'teacher': t, **dict(stats)})
    with open(os.path.join(out_dir, 'nms_quality.csv'), 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', required=True)
    ap.add_argument('--test-ann', default='data/ssdd/annotations/test.json')
    ap.add_argument('--recorded-ap', type=float, default=None)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    diag = json.load(open(args.diag))
    coco, fn2gt = build_gt_map(args.test_ann)

    ver = verify(diag, coco, args.recorded_ap)
    with open(os.path.join(args.out_dir, 'verification.json'), 'w') as f:
        json.dump(ver, f, indent=2, ensure_ascii=False)
    print('=== 核验 (teacher2 AP 复算 vs 记录) ===')
    print(json.dumps(ver, indent=2, ensure_ascii=False))

    sc = stage_coverage(diag, fn2gt, args.out_dir)
    agg, _ = loss_attribution(diag, fn2gt, args.out_dir)
    nq = nms_quality(diag, fn2gt, args.out_dir)

    print('\n=== 阶段覆盖 (RPN -> pre -> score_thr -> NMS -> final) ===')
    for r in sc:
        if r['iou_thr'] in (0.75, 0.85):
            print(f"{r['teacher']} IoU={r['iou_thr']}: "
                  f"RPN={r['rpn_cov']} pre={r['pre_cov']} scr={r['scr_cov']} "
                  f"NMS={r['nms_cov']} final={r['final_cov']} 1to1={r['final_1to1']}/{r['total_gt']}")

    print('\n=== 失败归因 (teacher2, IoU=0.75/0.85) ===')
    for thr in (0.75, 0.85):
        print(f'  IoU={thr} teacher2: {agg.get(str(thr)+"_teacher2", {})}')

    print('\n=== NMS 同目标质量 (Q3) ===')
    for r in nq:
        print(f"  IoU={r['iou_thr']} {r['teacher']}: {r}")

    summary = {'verification': ver, 'stage_coverage': sc,
               'loss_attribution': agg, 'nms_quality': nq}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nwritten to {args.out_dir}')


if __name__ == '__main__':
    main()
