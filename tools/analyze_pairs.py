"""Step 3: candidate provenance + real NMS suppression evidence.

candidate_pairs.csv: for each GT's qualified candidates, save the SAME proposal
  before/after regression: GT IoU, width/height/center shift, full-precision score.
nms_pairs.csv: real suppression relation from greedy NMS replay.
"""
import argparse
import csv
import json
import os

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


def candidate_pairs(diag, coco, out_dir, thr=0.5):
    fn2gt = build_gt_map(coco)
    records = diag['records']
    rows = []
    for t in ['teacher1', 'teacher2']:
        for r in records:
            sf = r['scale_factor']
            rpn = [rescale(b[:4], sf) for b in r[t]['rpn_proposals']['boxes']]
            pre = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
            sc = [float(s[0]) for s in r[t]['roi_pre_nms']['scores']]
            for gt_id, gt in fn2gt[r['filename']]:
                for i in range(len(pre)):
                    if iou(gt, pre[i]) < thr:
                        continue
                    # same proposal before (rpn[i]) and after (pre[i]) regression
                    w_r = rpn[i][2] - rpn[i][0]; h_r = rpn[i][3] - rpn[i][1]
                    w_p = pre[i][2] - pre[i][0]; h_p = pre[i][3] - pre[i][1]
                    cx_r = (rpn[i][0] + rpn[i][2]) / 2; cy_r = (rpn[i][1] + rpn[i][3]) / 2
                    cx_p = (pre[i][0] + pre[i][2]) / 2; cy_p = (pre[i][1] + pre[i][3]) / 2
                    rows.append({
                        'teacher': t, 'filename': r['filename'], 'gt_id': gt_id,
                        'proposal_id': i,
                        'iou_rpn': round(iou(gt, rpn[i]), 4),
                        'iou_pre': round(iou(gt, pre[i]), 4),
                        'score_full': repr(sc[i]),
                        'w_rpn': round(w_r, 2), 'w_pre': round(w_p, 2),
                        'h_rpn': round(h_r, 2), 'h_pre': round(h_p, 2),
                        'center_shift': round(((cx_p-cx_r)**2 + (cy_p-cy_r)**2)**0.5, 2),
                    })
    with open(os.path.join(out_dir, 'candidate_pairs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def nms_pairs(diag, coco, out_dir, thr=0.5):
    fn2gt = build_gt_map(coco)
    records = diag['records']
    rows = []
    for t in ['teacher1', 'teacher2']:
        for r in records:
            sf = r['scale_factor']
            pre = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
            sc = [float(s[0]) for s in r[t]['roi_pre_nms']['scores']]
            scr_idx = [i for i, s in enumerate(sc) if s > 0.05]
            boxes = [pre[i] for i in scr_idx]
            scores = [sc[i] for i in scr_idx]
            # greedy NMS replay with suppression recording
            order = np.argsort(scores)[::-1]
            suppressed = {}  # suppressed_idx -> suppressor_idx
            while order.size > 0:
                i = order[0]
                if order.size == 1:
                    break
                ovr = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
                sup = order[1:][ovr > 0.5]
                for j in sup:
                    suppressed.setdefault(int(scr_idx[j]), int(scr_idx[i]))
                order = order[np.where(ovr <= 0.5)[0] + 1]
            for j, i in suppressed.items():
                # GT membership: each box's best GT
                def best_gt(box):
                    bgt, bi = None, 0.0
                    for gt_id, gt in fn2gt[r['filename']]:
                        v = iou(gt, box)
                        if v > bi: bi, bgt = v, gt_id
                    return bgt, bi
                bgt_j, bij_j = best_gt(pre[j])
                bgt_i, bij_i = best_gt(pre[i])
                rows.append({
                    'teacher': t, 'filename': r['filename'],
                    'suppressed_id': j, 'suppressor_id': i,
                    'suppressed_score': repr(sc[j]), 'suppressor_score': repr(sc[i]),
                    'suppressed_gt': bgt_j, 'suppressed_gt_iou': round(bij_j, 3),
                    'suppressor_gt': bgt_i, 'suppressor_gt_iou': round(bij_i, 3),
                    'pair_iou': round(iou(pre[i], pre[j]), 3),
                    'same_gt': bgt_j == bgt_i,
                })
    with open(os.path.join(out_dir, 'nms_pairs.csv'), 'w', newline='') as f:
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

    cp = candidate_pairs(diag, coco, args.out_dir)
    print(f'candidate_pairs: {len(cp)} 条')
    np_rows = nms_pairs(diag, coco, args.out_dir)
    print(f'nms_pairs: {len(np_rows)} 条')
    same = sum(1 for r in np_rows if r['same_gt'])
    print(f'  NMS 抑制中同目标(可能去重): {same}, 跨目标(可能误抑制): {len(np_rows)-same}')


if __name__ == '__main__':
    main()
