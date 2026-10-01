"""Background false-positive review for M0(seed=678) vs M1-A.

Extracts TIDE BackgroundError detections (IoU<0.1 vs all GT), records their
image / box / score / nearest-GT IoU, and dumps a per-fold manifest. Also
prepares fixed sample images (original + GT + pred boxes).
"""
import argparse
import csv
import json
import os

import numpy as np
from pycocotools.coco import COCO
from tidecv import TIDE, Data
from tidecv.errors.main_errors import BackgroundError


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_data(coco, pred_path):
    cat_id = coco.getCatIds()[0]
    gt = Data('gt')
    gt_boxes_by_img = {}
    for img in coco.dataset['images']:
        gts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img['id'])):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            gt.add_ground_truth(img['id'], cat_id, box=[x, y, x + w, y + h])
            gts.append([x, y, x + w, y + h])
        gt_boxes_by_img[img['id']] = gts
    pred = Data('pred')
    for p in json.load(open(pred_path)):
        x, y, w, h = p['bbox']
        pred.add_detection(p['image_id'], p['category_id'], p['score'],
                           box=[x, y, x + w, y + h])
    return gt, pred, gt_boxes_by_img


def extract_bkg(gt, pred, gt_boxes_by_img):
    tide = TIDE()
    run = tide.evaluate(gt, pred, mode=TIDE.BOX, name='model')
    rows = []
    for e in run.error_dict[BackgroundError]:
        pd = e.pred
        img_id = pd['image']  # TIDE internal id == COCO image id we passed
        box = list(pd['bbox'])
        score = float(pd['score'])
        # nearest GT IoU
        gts = gt_boxes_by_img.get(img_id, [])
        best_iou = max((iou(box, g) for g in gts), default=0.0)
        rows.append({
            'image_id': img_id, 'score': score,
            'box': [round(v, 1) for v in box], 'nearest_gt_iou': round(best_iou, 3),
        })
    rows.sort(key=lambda r: -r['score'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/ssdd/annotations/test.json')
    ap.add_argument('--pred', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    coco = COCO(args.gt)
    gt, pred, gt_boxes = build_data(coco, args.pred)
    rows = extract_bkg(gt, pred, gt_boxes)

    # 高分 top20 + 分层抽样 20
    top20 = rows[:20]
    # 分层：按分数分 4 档各取 5
    if len(rows) > 20:
        scores = np.array([r['score'] for r in rows[20:]])
        qs = np.quantile(scores, [0.25, 0.5, 0.75])
        stratified = []
        for q in qs:
            near = [r for r in rows[20:] if r['score'] <= q][:5]
            stratified.extend(near)
        stratified = stratified[:20]
    else:
        stratified = rows

    print(f'背景虚警总数: {len(rows)}')
    print(f'  score>0.999 且 IoU<0.1: {sum(1 for r in rows if r["score"]>0.999)}')
    print(f'  score>0.9 且 IoU<0.1: {sum(1 for r in rows if r["score"]>0.9)}')
    print(f'  分数分布: min={rows[-1]["score"]:.3f} max={rows[0]["score"]:.3f}')

    # 输出明细
    with open(os.path.join(args.out, 'bkg_manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({'total': len(rows), 'top20': top20, 'stratified': stratified},
              open(os.path.join(args.out, 'bkg_samples.json'), 'w'), indent=2)
    print(f'written {args.out}/bkg_manifest.csv + bkg_samples.json')


if __name__ == '__main__':
    main()
