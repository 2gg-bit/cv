"""NMS replay endpoint check: reproduce M1-A/B with original mmcv batched_nms.

Replays the final NMS + scoring on the exported pre-NMS candidates using the
ORIGINAL mmcv.ops.batched_nms (not a numpy reimplementation), then compares the
resulting per-box predictions against the eval_teacher2_export.py output.

Run on CPU (candidates are small); does not compete with the training GPU.
"""
import argparse
import json
import os

import numpy as np
import torch
from mmcv.ops import batched_nms


def load_preds(json_path):
    data = json.load(open(json_path))
    # predictions.bbox.json is a flat list
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', required=True)
    ap.add_argument('--pred-a', required=True)
    ap.add_argument('--pred-b', required=True)
    args = ap.parse_args()

    cands = json.load(open(args.cands))
    pred_a = load_preds(args.pred_a)
    pred_b = load_preds(args.pred_b)

    # build per-image candidate lookup (image_id -> list of (box, p, q))
    # image_id mapping from filename -> coco image id is done via the pred files
    # We rebuild predictions from candidates for A and B, then compare.
    from pycocotools.coco import COCO
    coco = COCO('data/ssdd/annotations/test.json')
    fn2id = {im['file_name']: im['id'] for im in coco.dataset['images']}
    cat_id = coco.getCatIds()[0]

    replay_a, replay_b = [], []
    for r in cands['records']:
        sf = r['scale_factor']
        img_id = fn2id[r['filename']]
        valid = [c for c in r['candidates'] if c['p'] > 0.05]
        if not valid:
            continue
        boxes = torch.tensor([[c['box'][0]/sf[0], c['box'][1]/sf[1],
                               c['box'][2]/sf[2], c['box'][3]/sf[3]]
                              for c in valid], dtype=torch.float32)
        p = torch.tensor([c['p'] for c in valid], dtype=torch.float32)
        q = torch.tensor([c['q'] for c in valid], dtype=torch.float32)
        labels = torch.zeros(len(valid), dtype=torch.long)
        nms_cfg = dict(type='nms', iou_threshold=0.5)
        # A: NMS by p, score p (dets score column == p)
        dets_a, _ = batched_nms(boxes, p, labels, nms_cfg)
        for b in dets_a[:100]:
            replay_a.append({'image_id': img_id, 'category_id': cat_id,
                             'bbox': [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                             'score': float(b[4])})
        # B: NMS by p*q, score p*q (dets score column == joint)
        joint = p * q
        dets_b, _ = batched_nms(boxes, joint, labels, nms_cfg)
        for b in dets_b[:100]:
            replay_b.append({'image_id': img_id, 'category_id': cat_id,
                             'bbox': [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                             'score': float(b[4])})

    def normalize(preds):
        return sorted((p['image_id'], round(p['bbox'][0], 3), round(p['bbox'][1], 3),
                       round(p['bbox'][2], 3), round(p['bbox'][3], 3), round(p['score'], 4))
                      for p in preds)

    na, nb = normalize(replay_a), normalize(replay_b)
    ea, eb = normalize(pred_a), normalize(pred_b)

    print(f'候选图数: {len(cands["records"])}')
    print(f'A 回放框数: {len(na)} vs 端点导出: {len(ea)}')
    print(f'B 回放框数: {len(nb)} vs 端点导出: {len(eb)}')
    print(f'A 逐框一致: {na == ea}')
    print(f'B 逐框一致: {nb == eb}')
    if na != ea:
        diff_a = [x for x in ea if x not in set(na)]
        print(f'  A 差异框(端点有回放无): {len(diff_a)}')
    if nb != eb:
        diff_b = [x for x in eb if x not in set(nb)]
        print(f'  B 差异框(端点有回放无): {len(diff_b)}')


if __name__ == '__main__':
    main()
