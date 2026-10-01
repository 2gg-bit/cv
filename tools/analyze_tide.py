"""TIDE error decomposition for M0(seed=678) vs M1-A.

Per fold, per model, runs TIDE and reports the six error types:
Cls / Loc / Both / Dupe / Bkg / Miss. Also does COCO one-to-one GT pairing
between M0 and M1-A to attribute detection gains/losses per GT.
"""
import argparse
import json
import os

import numpy as np
from pycocotools.coco import COCO
from tidecv import TIDE


def tide_decompose(gt_path, pred_path, name):
    from tidecv import Data
    coco = COCO(gt_path)
    gt_data = Data('gt')
    cat_id = coco.getCatIds()[0]
    for img in coco.dataset['images']:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img['id'])):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            gt_data.add_ground_truth(img['id'], cat_id, box=[x, y, w, h])
    pred_data = Data('pred')
    for p in json.load(open(pred_path)):
        x, y, w, h = p['bbox']
        pred_data.add_detection(p['image_id'], p['category_id'], p['score'],
                                box=[x, y, w, h])
    tide = TIDE()
    tide.evaluate(gt_data, pred_data, mode=TIDE.BOX, name=name)
    main_errors = tide.get_main_errors()  # {name: {err_type: float 比例}}
    counts = main_errors.get(name, {})
    return counts, None


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def one_to_one_match(gt_boxes, pred_boxes, thr=0.5):
    """greedy one-to-one: each pred matches at most one GT."""
    matched = set()
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        best_i, best_iou = -1, 0.0
        for gi, gb in enumerate(gt_boxes):
            if gi in matched:
                continue
            v = iou(gb, pb)
            if v > best_iou:
                best_iou, best_i = v, gi
        if best_i >= 0 and best_iou >= thr:
            matched.add(best_i)
            pairs.append((best_i, pi))
    return matched, pairs


def gt_pairing(gt_path, pred_a_path, pred_b_path, thr=0.5):
    coco = COCO(gt_path)
    pred_a = json.load(open(pred_a_path))
    pred_b = json.load(open(pred_b_path))
    # group predictions by image_id
    from collections import defaultdict
    a_by_img = defaultdict(list)
    b_by_img = defaultdict(list)
    for p in pred_a:
        a_by_img[p['image_id']].append((p['bbox'], p['score']))
    for p in pred_b:
        b_by_img[p['image_id']].append((p['bbox'], p['score']))

    stats = {'both': 0, 'only_a': 0, 'only_b': 0, 'neither': 0}
    per_gt = []
    for img in coco.dataset['images']:
        img_id = img['id']
        gts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            gts.append((ann['id'], [x, y, x + w, y + h]))
        if not gts:
            continue
        # boxes to [x1,y1,x2,y2]
        a_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b, _ in a_by_img[img_id]]
        b_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b, _ in b_by_img[img_id]]
        gt_boxes = [g[1] for g in gts]
        a_matched, _ = one_to_one_match(gt_boxes, a_boxes, thr)
        b_matched, _ = one_to_one_match(gt_boxes, b_boxes, thr)
        for gi, (gt_id, gb) in enumerate(gts):
            in_a = gi in a_matched
            in_b = gi in b_matched
            if in_a and in_b:
                stats['both'] += 1; label = 'both'
            elif in_a and not in_b:
                stats['only_a'] += 1; label = 'only_a'
            elif not in_a and in_b:
                stats['only_b'] += 1; label = 'only_b'
            else:
                stats['neither'] += 1; label = 'neither'
            per_gt.append({'image_id': img_id, 'gt_id': gt_id, 'label': label,
                           'file': img['file_name']})
    return stats, per_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/ssdd/annotations/test.json')
    ap.add_argument('--m0-dir', required=True)
    ap.add_argument('--m1-dir', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=[6, 7, 8])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_tide = {}
    all_pair = {}
    for f in args.folds:
        m0_pred = f'{args.m0_dir}/{f}/predictions.bbox.json'
        m1_pred = f'{args.m1_dir}/{f}/m1_a/predictions.bbox.json'
        print(f'=== fold {f} TIDE ===')
        for name, pred in [('M0', m0_pred), ('M1-A', m1_pred)]:
            counts, ap_ = tide_decompose(args.gt, pred, name)
            all_tide[f'{f}_{name}'] = counts
            total = sum(counts.values())
            print(f'  {name}: Cls={counts["Cls"]} Loc={counts["Loc"]} Both={counts["Both"]} '
                  f'Dupe={counts["Dupe"]} Bkg={counts["Bkg"]} Miss={counts["Miss"]} (总错误={total})')
        stats, per_gt = gt_pairing(args.gt, m0_pred, m1_pred)
        all_pair[f] = stats
        print(f'  GT配对(M0 vs M1-A): both={stats["both"]} only_M0={stats["only_a"]} '
              f'only_M1={stats["only_b"]} neither={stats["neither"]}')

    json.dump({'tide': all_tide, 'pairing': all_pair},
              open(os.path.join(args.out, 'tide_pairing.json'), 'w'), indent=2)
    print(f'written {args.out}/tide_pairing.json')


if __name__ == '__main__':
    main()
