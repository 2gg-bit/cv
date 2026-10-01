"""Fixed-stratified audit sample images: matched / only_m0 / only_m1.

matched: side-by-side M0 (left) and M1-A (right), each box labeled with its
json index and true score. only_m0 / only_m1: single model box labeled.
"""
import argparse
import json
import os
import random

import cv2
import numpy as np
from pycocotools.coco import COCO


def draw(img, box, label, color, offset_x=0):
    cv2.rectangle(img, (offset_x + int(box[0]), int(box[1])),
                  (offset_x + int(box[2]), int(box[3])), color, 2)
    cv2.putText(img, label, (offset_x + int(box[0]), max(15, int(box[1]) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/ssdd/annotations/test.json')
    ap.add_argument('--img-dir', default='data/ssdd/test_images')
    ap.add_argument('--detail', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    coco = COCO(args.gt)
    id2file = {im['id']: im['file_name'] for im in coco.dataset['images']}
    id2gt = {}
    for im in coco.dataset['images']:
        gts = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=im['id'])):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            gts.append([x, y, x + w, y + h])
        id2gt[im['id']] = gts

    detail = json.load(open(args.detail))
    for cls, items in detail.items():
        random.shuffle(items)
        for k, item in enumerate(items[:args.per_class]):
            img_id = item['image_id']
            fname = id2file.get(img_id)
            if fname is None:
                continue
            img = cv2.imread(os.path.join(args.img_dir, fname))
            if img is None:
                continue
            if cls == 'matched':
                h, w = img.shape[:2]
                canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
                canvas[:, :w] = img; canvas[:, w:] = img
                for g in id2gt.get(img_id, []):
                    draw(canvas, g, '', (0, 0, 255), 0)
                    draw(canvas, g, '', (0, 0, 255), w)
                draw(canvas, item['a_box'], f'id={item["a_json_idx"]} s={item["a_score"]:.3f}', (255, 0, 0), 0)
                draw(canvas, item['b_box'], f'id={item["b_json_idx"]} s={item["b_score"]:.3f}', (255, 0, 0), w)
                cv2.putText(canvas, 'M0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(canvas, 'M1-A', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(args.out_dir, f'matched_{k:02d}_{fname}'), canvas)
            else:
                for g in id2gt.get(img_id, []):
                    draw(img, g, '', (0, 0, 255))
                draw(img, item['box'], f'id={item["json_idx"]} s={item["score"]:.3f}', (255, 0, 0))
                cv2.imwrite(os.path.join(args.out_dir, f'{cls}_{k:02d}_{fname}'), img)
    print(f'样例图已保存到 {args.out_dir}/')


if __name__ == '__main__':
    main()
