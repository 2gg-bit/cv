"""Replay four score/NMS combinations on M1 pre-NMS candidates.

    A = M1-A (NMS by p, score p)
    B = M1-B (NMS by p*q, score p*q)
    plus two counterfactuals:
    NMS-by-p  + score p*q
    NMS-by-p*q + score p

This separates "ranking changed the kept box set" from "final score changed".
"""
import argparse
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_keep(boxes, scores, thr=0.5, max_per_img=100):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_per_img:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ovr = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def rescale_box(box, sf):
    return [box[0] / sf[0], box[1] / sf[1], box[2] / sf[2], box[3] / sf[3]]


def eval_ap(coco, preds):
    cocoDt = coco.loadRes(preds)
    ce = COCOeval(coco, cocoDt, 'bbox')
    ce.evaluate(); ce.accumulate(); ce.summarize()
    return {'AP': round(float(ce.stats[0]), 3), 'AP50': round(float(ce.stats[1]), 3),
            'AP75': round(float(ce.stats[2]), 3)}


def run(cands_json, test_ann, score_thr=0.05, nms_thr=0.5, max_per_img=100):
    data = json.load(open(cands_json))
    coco = COCO(test_ann)
    cat_id = coco.getCatIds()[0]
    fn2id = {im['file_name']: im['id'] for im in coco.dataset['images']}

    combos = {
        'NMS_p_score_p':   (lambda p, q: p, lambda p, q: p),
        'NMS_p_score_pq':  (lambda p, q: p, lambda p, q: p * q),
        'NMS_pq_score_p':  (lambda p, q: p * q, lambda p, q: p),
        'NMS_pq_score_pq': (lambda p, q: p * q, lambda p, q: p * q),
    }
    results = {}
    for name, (rank_fn, score_fn) in combos.items():
        preds = []
        for r in data['records']:
            sf = r['scale_factor']
            # candidate filter p > score_thr
            valid = [(c, rescale_box(c['box'][:4], sf)) for c in r['candidates'] if c['p'] > score_thr]
            if not valid:
                continue
            cs = [c for c, _ in valid]
            boxes = [b for _, b in valid]
            p = np.array([c['p'] for c in cs])
            q = np.array([c['q'] for c in cs])
            rank_scores = rank_fn(p, q)
            final_scores = score_fn(p, q)
            keep = nms_keep(boxes, rank_scores, nms_thr, max_per_img)
            img_id = fn2id[r['filename']]
            for k in keep:
                b = boxes[k]
                preds.append({'image_id': img_id, 'category_id': cat_id,
                              'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                              'score': float(final_scores[k])})
        results[name] = eval_ap(coco, preds)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', required=True)
    ap.add_argument('--test-ann', default='data/ssdd/annotations/test.json')
    args = ap.parse_args()
    res = run(args.cands, args.test_ann)
    print('=== 四种评分/NMS 组合 ===')
    for k, v in res.items():
        print(f'{k}: AP={v["AP"]} AP50={v["AP50"]} AP75={v["AP75"]}')


if __name__ == '__main__':
    main()
