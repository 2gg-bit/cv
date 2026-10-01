"""Bidirectional background-candidate comparison between M0 and M1-A.

Uses ONE unified rule (IoU < 0.1 vs all GT) on the raw predictions.bbox.json
(no extra score filtering), records the JSON row index so every background
candidate traces back to its source prediction, spatially matches the two
models, and emits side-by-side images.
"""
import argparse
import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_bg_candidates(pred_path, gt_boxes_by_img, bg_thr=0.1):
    preds = json.load(open(pred_path))
    rows = []
    for idx, p in enumerate(preds):
        x, y, w, h = p['bbox']
        box = [x, y, x + w, y + h]
        gts = gt_boxes_by_img.get(p['image_id'], [])
        best = max((iou(box, g) for g in gts), default=0.0)
        if best < bg_thr:
            rows.append({
                'json_idx': idx, 'image_id': p['image_id'],
                'score': float(p['score']), 'box': box,
                'nearest_gt_iou': round(best, 3),
            })
    return rows


def match_spatial(rows_a, rows_b, thr=0.5):
    """greedy one-to-one spatial match: each pred matched at most once, same image."""
    matched_a = set(); matched_b = set()
    pairs = []
    for ia, ra in enumerate(rows_a):
        if ia in matched_a:
            continue
        best_ib, best_iou = -1, 0.0
        for ib, rb in enumerate(rows_b):
            if ib in matched_b or ra['image_id'] != rb['image_id']:
                continue
            v = iou(ra['box'], rb['box'])
            if v >= thr and v > best_iou:
                best_iou, best_ib = v, ib
        if best_ib >= 0:
            matched_a.add(ia); matched_b.add(best_ib)
            pairs.append({'a_json_idx': rows_a[ia]['json_idx'],
                          'b_json_idx': rows_b[best_ib]['json_idx'],
                          'image_id': ra['image_id'],
                          'iou': round(best_iou, 3),
                          'a_box': rows_a[ia]['box'],
                          'b_box': rows_b[best_ib]['box'],
                          'a_score': rows_a[ia]['score'],
                          'b_score': rows_b[best_ib]['score']})
    return pairs, matched_a, matched_b


def draw_side_by_side(img_path, gt_boxes, preds_a, preds_b, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = img
    canvas[:, w:] = img
    for g in gt_boxes:
        cv2.rectangle(canvas, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                      (0, 0, 255), 2)
        cv2.rectangle(canvas, (w + int(g[0]), int(g[1])), (w + int(g[2]), int(g[3])),
                      (0, 0, 255), 2)
    for p in preds_a:
        b = p['box']
        cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (255, 0, 0), 2)
    for p in preds_b:
        b = p['box']
        cv2.rectangle(canvas, (w + int(b[0]), int(b[1])), (w + int(b[2]), int(b[3])),
                      (255, 0, 0), 2)
    cv2.putText(canvas, 'M0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(canvas, 'M1-A', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(out_path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/ssdd/annotations/test.json')
    ap.add_argument('--img-dir', default='data/ssdd/test_images')
    ap.add_argument('--m0', required=True)
    ap.add_argument('--m1', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--top-n', type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    coco = COCO(args.gt)
    id2file = {im['id']: im['file_name'] for im in coco.dataset['images']}
    gt_boxes = {}
    for im in coco.dataset['images']:
        gts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=im['id'])):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            gts.append([x, y, x + w, y + h])
        gt_boxes[im['id']] = gts

    bg_a = load_bg_candidates(args.m0, gt_boxes)
    bg_b = load_bg_candidates(args.m1, gt_boxes)
    print(f'M0 背景候选: {len(bg_a)} | M1-A 背景候选: {len(bg_b)}')

    pairs, matched_a, matched_b = match_spatial(bg_a, bg_b)
    only_a = [i for i in range(len(bg_a)) if i not in matched_a]
    only_b = [i for i in range(len(bg_b)) if i not in matched_b]
    print(f'空间匹配(同图 IoU>=0.5): 共同={len(pairs)} | 仅M0={len(only_a)} | 仅M1={len(only_b)}')

    # 并排图：取 M0 分数最高的 top_n 背景候选（含匹配的 M1 框），以及 M1 新增的 top 若干
    top_a = sorted(bg_a, key=lambda r: -r['score'])[:args.top_n]
    os.makedirs(os.path.join(args.out, 'side_by_side'), exist_ok=True)
    for r in top_a:
        img_id = r['image_id']
        fname = id2file.get(img_id)
        if fname is None:
            continue
        # find matched M1 box at same image (any IoU>=0.5)
        matched_b_boxes = [bg_b[j] for j in matched_b
                           if bg_b[j]['image_id'] == img_id and iou(r['box'], bg_b[j]['box']) >= 0.5]
        preds_b = matched_b_boxes if matched_b_boxes else []
        out = os.path.join(args.out, 'side_by_side', f'{fname}_s{r["score"]:.2f}.jpg')
        draw_side_by_side(os.path.join(args.img_dir, fname), gt_boxes.get(img_id, []),
                          [r], preds_b, out)

    # 汇总 + 完整明细
    summary = {
        'm0_bg_count': len(bg_a), 'm1_bg_count': len(bg_b),
        'matched': len(pairs), 'only_m0': len(only_a), 'only_m1': len(only_b),
    }
    json.dump(summary, open(os.path.join(args.out, 'compare_summary.json'), 'w'), indent=2)
    detail = {
        'matched': pairs,
        'only_m0': [bg_a[i] for i in only_a],
        'only_m1': [bg_b[i] for i in only_b],
    }
    json.dump(detail, open(os.path.join(args.out, 'compare_detail.json'), 'w'), indent=2)
    print(f'written {args.out}/compare_summary.json + compare_detail.json + side_by_side/')


if __name__ == '__main__':
    main()
