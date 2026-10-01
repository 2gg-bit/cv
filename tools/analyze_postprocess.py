"""Steps 4-5: split post-process loss into categories + regression shrink check."""
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
        order = order[order.size and np.where(ovr <= thr)[0] + 1]
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


def postprocess_losses(diag, coco, out_dir, thr=0.5):
    fn2gt = build_gt_map(coco)
    records = diag['records']
    rows = []
    for t in ['teacher1', 'teacher2']:
        for r in records:
            sf = r['scale_factor']
            pre_boxes = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
            pre_scores = [s[0] for s in r[t]['roi_pre_nms']['scores']]
            fin = [rescale(b[:4], sf) for b in r[t]['final_det']['boxes']]
            scr_idx = [i for i, s in enumerate(pre_scores) if s > 0.05]
            scr_boxes = [pre_boxes[i] for i in scr_idx]
            scr_sc = [pre_scores[i] for i in scr_idx]
            nms_idx = nms(scr_boxes, scr_sc, 0.5)
            nms_boxes = [scr_boxes[i] for i in nms_idx]
            for gt_id, gt in fn2gt[r['filename']]:
                best_pre = max((iou(gt, b) for b in pre_boxes), default=0.0)
                if best_pre < thr:
                    continue  # not a post-process loss
                best_scr = max((iou(gt, b) for b in scr_boxes), default=0.0)
                best_nms = max((iou(gt, b) for b in nms_boxes), default=0.0)
                best_fin = max((iou(gt, b) for b in fin), default=0.0)
                if best_scr < thr:
                    stage = 'score_filter'
                elif best_nms < thr:
                    stage = 'nms_suppression'
                elif best_fin < thr:
                    stage = 'max_per_img_truncation'
                else:
                    continue
                # find the best pre candidate for context
                pre_best_i = max(range(len(pre_boxes)), key=lambda i: iou(gt, pre_boxes[i]))
                pre_best_iou = iou(gt, pre_boxes[pre_best_i])
                rows.append({
                    'teacher': t, 'filename': r['filename'], 'gt_id': gt_id,
                    'gt_box': gt, 'stage': stage,
                    'best_pre_iou': round(pre_best_iou, 3),
                    'best_pre_score': round(pre_scores[pre_best_i], 3),
                    'best_scr_iou': round(best_scr, 3),
                    'best_nms_iou': round(best_nms, 3),
                    'best_fin_iou': round(best_fin, 3),
                })
    with open(os.path.join(out_dir, 'postprocess_losses.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['teacher'])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    return Counter(r['stage'] for r in rows), rows


def regression_check(diag, coco, out_dir):
    fn2gt = build_gt_map(coco)
    records = diag['records']
    rows = []
    for t in ['teacher1', 'teacher2']:
        for r in records:
            sf = r['scale_factor']
            rpn = [rescale(b[:4], sf) for b in r[t]['rpn_proposals']['boxes']]
            pre = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
            # rpn[i] -> pre[i] is the same proposal before/after regression
            assert len(rpn) == len(pre)
            for gt_id, gt in fn2gt[r['filename']]:
                # best matching proposal (by rpn iou)
                best_i = max(range(len(rpn)), key=lambda i: iou(gt, rpn[i]))
                iou_rpn = iou(gt, rpn[best_i])
                iou_pre = iou(gt, pre[best_i])
                if iou_rpn >= 0.5 or iou_pre >= 0.5:
                    # width/height before & after
                    w_r = rpn[best_i][2] - rpn[best_i][0]; h_r = rpn[best_i][3] - rpn[best_i][1]
                    w_p = pre[best_i][2] - pre[best_i][0]; h_p = pre[best_i][3] - pre[best_i][1]
                    cx_r = (rpn[best_i][0] + rpn[best_i][2]) / 2; cy_r = (rpn[best_i][1] + rpn[best_i][3]) / 2
                    cx_p = (pre[best_i][0] + pre[best_i][2]) / 2; cy_p = (pre[best_i][1] + pre[best_i][3]) / 2
                    cx_gt = (gt[0] + gt[2]) / 2; cy_gt = (gt[1] + gt[3]) / 2
                    rows.append({
                        'teacher': t, 'filename': r['filename'], 'gt_id': gt_id,
                        'iou_rpn': round(iou_rpn, 3), 'iou_pre': round(iou_pre, 3),
                        'iou_delta': round(iou_pre - iou_rpn, 3),
                        'w_rpn': round(w_r, 1), 'w_pre': round(w_p, 1),
                        'h_rpn': round(h_r, 1), 'h_pre': round(h_p, 1),
                        'center_shift': round(((cx_p - cx_r)**2 + (cy_p - cy_r)**2)**0.5, 1),
                        'degrade_75_to_50': iou_rpn >= 0.75 and 0.5 <= iou_pre < 0.75,
                    })
    with open(os.path.join(out_dir, 'regression_pairs.csv'), 'w', newline='') as f:
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

    print('=== 步骤4: 后处理损失分类 (IoU=0.5) ===')
    cnt, _ = postprocess_losses(diag, coco, args.out_dir, thr=0.5)
    print(dict(cnt))

    print('=== 步骤5: 回归检查 ===')
    rows = regression_check(diag, coco, args.out_dir)
    degrade = sum(1 for r in rows if r['degrade_75_to_50'])
    iou_deltas = [r['iou_delta'] for r in rows]
    print(f'回归对总数: {len(rows)}')
    print(f'  IoU 退化(75->50): {degrade} 个')
    print(f'  IoU delta 均值: {np.mean(iou_deltas):.3f} (正=改善, 负=退化)')
    print(f'  IoU delta < -0.1: {sum(1 for d in iou_deltas if d < -0.1)} 个')
    print(f'  IoU delta > 0.1: {sum(1 for d in iou_deltas if d > 0.1)} 个')


if __name__ == '__main__':
    main()
