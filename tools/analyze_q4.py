"""Q4 redefinition + reconciliation + candidate/NMS evidence.

Per teacher, per GT, per IoU threshold (>=0.50, 0.75):
  - qualified candidate = ROI-regressed, pre-score-filter box with IoU >= thr
  - q = max ship foreground score among qualified candidates
  - grouping:
      no qualified candidate          -> 候选定位不足
      qualified but all <= score_thr  -> 分数过滤损失
      pass score_thr but q < 0.3      -> 低置信
      exists qualified with score>=0.3 -> 高分
  - record BOTH: best-IoU candidate (and its score), and best-score qualified
    candidate (and its IoU).

Reconciliation: per-teacher per-IoU, 分数过滤损失数 must equal
    (ROI 原始框覆盖 GT 数) - (分数筛选后覆盖 GT 数).
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
from pycocotools.coco import COCO


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rescale(box, sf):
    return [box[0] / sf[0], box[1] / sf[1], box[2] / sf[2], box[3] / sf[3]]


def nms(boxes, scores, thr=0.5):
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ovr = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def build_gt_map(coco):
    fn2gt = {}
    for im in coco.dataset['images']:
        gts = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=im['id'])):
            if a.get('iscrowd', 0):
                continue
            x, y, w, h = a['bbox']
            gts.append((a['id'], [x, y, x + w, y + h]))
        fn2gt[im['file_name']] = gts
    return fn2gt


def q4_per_gt(diag, coco, out_dir, score_thr=0.05):
    fn2gt = build_gt_map(coco)
    records = diag['records']
    rows = []
    for thr in [0.5, 0.75]:
        for t in ['teacher1', 'teacher2']:
            for r in records:
                sf = r['scale_factor']
                pre_boxes = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
                pre_scores = [float(s[0]) for s in r[t]['roi_pre_nms']['scores']]
                fin = [rescale(b[:4], sf) for b in r[t]['final_det']['boxes']]
                for gt_id, gt in fn2gt[r['filename']]:
                    qual = [i for i, b in enumerate(pre_boxes) if iou(gt, b) >= thr]
                    if not qual:
                        group, q, best_iou_i, best_score_i = '候选定位不足', None, -1, -1
                    else:
                        q = max(pre_scores[i] for i in qual)
                        best_iou_i = max(qual, key=lambda i: iou(gt, pre_boxes[i]))
                        best_score_i = max(qual, key=lambda i: pre_scores[i])
                        qual_pass = [i for i in qual if pre_scores[i] > score_thr]
                        if not qual_pass:
                            group = '分数过滤损失'
                        elif q < 0.3:
                            group = '低置信'
                        else:
                            group = '高分'
                    fin_iou = max((iou(gt, b) for b in fin), default=0.0)
                    rows.append({
                        'iou_thr': thr, 'teacher': t, 'filename': r['filename'],
                        'gt_id': gt_id, 'group': group,
                        'q_score': (round(q, 4) if q is not None else 'null'),
                        'best_iou': round(iou(gt, pre_boxes[best_iou_i]), 3) if best_iou_i >= 0 else 'null',
                        'best_iou_score': round(pre_scores[best_iou_i], 4) if best_iou_i >= 0 else 'null',
                        'best_score_iou': round(iou(gt, pre_boxes[best_score_i]), 3) if best_score_i >= 0 else 'null',
                        'best_score_score': round(pre_scores[best_score_i], 4) if best_score_i >= 0 else 'null',
                        'final_iou': round(fin_iou, 3),
                    })
    with open(os.path.join(out_dir, 'q4_per_gt.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def reconcile(diag, coco, out_dir, score_thr=0.05):
    """per-teacher per-IoU: 分数过滤损失数 = pre覆盖 - score_thr覆盖."""
    fn2gt = build_gt_map(coco)
    records = diag['records']
    rows = []
    for thr in [0.5, 0.75]:
        for t in ['teacher1', 'teacher2']:
            pre_cov = scr_cov = 0
            filt_loss = 0
            total = 0
            for r in records:
                sf = r['scale_factor']
                pre = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
                sc = [float(s[0]) for s in r[t]['roi_pre_nms']['scores']]
                scr = [pre[i] for i, s in enumerate(sc) if s > score_thr]
                for gt_id, gt in fn2gt[r['filename']]:
                    total += 1
                    cov_pre = any(iou(gt, b) >= thr for b in pre)
                    cov_scr = any(iou(gt, b) >= thr for b in scr)
                    if cov_pre: pre_cov += 1
                    if cov_scr: scr_cov += 1
                    if cov_pre and not cov_scr:
                        filt_loss += 1
            rows.append({
                'iou_thr': thr, 'teacher': t, 'total_gt': total,
                'pre_covered': pre_cov, 'score_thr_covered': scr_cov,
                'filt_loss_by_diff': pre_cov - scr_cov,
                'filt_loss_by_count': filt_loss,
                'match': (pre_cov - scr_cov) == filt_loss,
            })
    with open(os.path.join(out_dir, 'q4_reconciliation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', required=True)
    ap.add_argument('--test-ann', default='data/ssdd/annotations/test.json')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    diag = json.load(open(args.diag))
    coco = COCO(args.test_ann)

    rows = q4_per_gt(diag, coco, args.out_dir)
    print('=== Q4 分组统计（每教师每IoU）===')
    for thr in [0.5, 0.75]:
        for t in ['teacher1', 'teacher2']:
            sub = [r for r in rows if r['iou_thr'] == thr and r['teacher'] == t]
            cnt = Counter(r['group'] for r in sub)
            print(f'IoU={thr} {t}: {dict(cnt)}')

    print('=== 对账 ===')
    rec = reconcile(diag, coco, args.out_dir)
    for r in rec:
        print(f"IoU={r['iou_thr']} {r['teacher']}: pre={r['pre_covered']} scr={r['score_thr_covered']} "
              f"diff={r['filt_loss_by_diff']} count={r['filt_loss_by_count']} 匹配={r['match']}")


if __name__ == '__main__':
    main()
