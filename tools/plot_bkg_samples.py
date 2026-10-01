"""Plot background false-positive samples: original + GT (red) + pred (blue)."""
import argparse
import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', default='data/ssdd/annotations/test.json')
    ap.add_argument('--img-dir', default='data/ssdd/test_images')
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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

    samples = json.load(open(args.samples))
    for key in ['top20', 'stratified']:
        for i, s in enumerate(samples.get(key, [])):
            img_id = s['image_id']
            fname = id2file.get(img_id)
            if fname is None:
                continue
            path = os.path.join(args.img_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            # GT red
            for g in id2gt.get(img_id, []):
                cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                              (0, 0, 255), 2)
            # pred blue (thick)
            b = s['box']
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (255, 0, 0), 3)
            cv2.putText(img, f's={s["score"]:.3f} iou={s["nearest_gt_iou"]}',
                        (int(b[0]), max(15, int(b[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            out = os.path.join(args.out_dir, f'{key}_{i:02d}_{fname}')
            cv2.imwrite(out, img)
    print(f'样例图已保存到 {args.out_dir}/')


if __name__ == '__main__':
    main()
