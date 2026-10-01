"""Offline Hard-NMS replay acceptance for the M0 candidate cache.

Reads the candidate cache produced by export_m0_candidates.py (one forward pass
per image; candidates + reference Hard-NMS output from the SAME _bbox_forward)
and re-applies the entry score filter + the *real* mmcv.ops.batched_nms +
max_per_img on the candidates.  Compares:

  1. replay  vs  reference_hard_nms (same forward pass)  -> must be per-box,
     bit-exact (this proves the candidate cache is complete and the replay
     logic is correct).
  2. replay  vs  formal predictions.bbox.json (a *separate* process) -> expected
     to match per-box up to FP16 non-determinism (a few boxes may differ by
     <1 px); this documents how faithfully the cache mirrors the official eval.

Acceptance is per-box exact on (1); (2) is reported for transparency, not used
to relax the (1) criterion.
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from mmcv.ops import batched_nms

SCORE_THR = 0.05
NMS_IOU = 0.5
MAX_PER_IMG = 100


def replay_hard_nms(cands_json):
    """Replay Hard NMS on the candidate cache -> per-image list of dets.

    The batched_nms operator must run on **CUDA**, matching the reference
    (get_bboxes -> multiclass_nms runs on GPU).  PyTorch's CPU sort and CUDA sort
    break exact-score ties differently, so a CPU replay would keep a different
    duplicate on the ~3 exact-score ties in the test set (verified).
    """
    preds = []  # list of {image_id, box:[x1,y1,x2,y2], score, label}
    for r in cands_json['records']:
        cands = r['candidates']
        if not cands:
            continue
        sf = r['scale_factor']
        boxes = torch.tensor([c['box'] for c in cands], dtype=torch.float32)
        p = torch.tensor([c['p'] for c in cands], dtype=torch.float32)
        sf_t = torch.tensor(sf, dtype=torch.float32)
        # rescale to original image space (matches get_bboxes rescale=True)
        boxes = (boxes.view(-1, 1, 4) / sf_t).view(-1, 4)
        valid = p > SCORE_THR
        if not valid.any():
            continue
        boxes = boxes[valid].cuda()
        p = p[valid].cuda()
        labels = torch.zeros(boxes.size(0), dtype=torch.long, device=boxes.device)
        dets, _ = batched_nms(boxes, p, labels,
                              dict(type='nms', iou_threshold=NMS_IOU))
        dets = dets[:MAX_PER_IMG].cpu()
        for d in dets:
            preds.append({'image_id': r['image_id'],
                          'box': [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                          'score': float(d[4]), 'label': 0})
    return preds


def _ref_key(box, score):
    return (round(box[0], 6), round(box[1], 6), round(box[2], 6), round(box[3], 6),
            round(score, 6))


def compare_replay_vs_reference(replay, cands_json):
    ref_by_img = defaultdict(list)
    for r in cands_json['records']:
        for d in r['reference_hard_nms']:
            ref_by_img[r['image_id']].append(d)
    rep_by_img = defaultdict(list)
    for d in replay:
        rep_by_img[d['image_id']].append(d)

    all_imgs = sorted(set(ref_by_img) | set(rep_by_img))
    report = {'num_replay': len(replay), 'num_reference': sum(len(v) for v in ref_by_img.values()),
              'image_coverage_match': set(ref_by_img) == set(rep_by_img),
              'count_mismatch': [], 'box_mismatch': [], 'exact': True}
    for im in all_imgs:
        rk = sorted(_ref_key(d['box'], d['score']) for d in rep_by_img[im])
        fk = sorted(_ref_key(d['box'], d['score']) for d in ref_by_img[im])
        if len(rk) != len(fk):
            report['count_mismatch'].append({'image_id': im, 'replay': len(rk), 'ref': len(fk)})
            report['exact'] = False
            continue
        for a, b in zip(rk, fk):
            if a != b:
                report['box_mismatch'].append({'image_id': im, 'replay': a, 'ref': b})
                report['exact'] = False
    return report


def compare_replay_vs_official(replay, pred_json_path):
    official = json.load(open(pred_json_path))
    off_by_img = defaultdict(list)
    for p in official:
        x, y, w, h = p['bbox']
        off_by_img[p['image_id']].append((round(x, 6), round(y, 6), round(x + w, 6), round(y + h, 6), round(p['score'], 6)))
    rep_by_img = defaultdict(list)
    for d in replay:
        rep_by_img[d['image_id']].append(_ref_key(d['box'], d['score']))
    all_imgs = sorted(set(off_by_img) | set(rep_by_img))
    report = {'num_replay': len(replay), 'num_official': len(official),
              'image_coverage_match': set(off_by_img) == set(rep_by_img),
              'count_mismatch': [], 'box_mismatch': [], 'exact': True}
    for im in all_imgs:
        rk = sorted(rep_by_img[im])
        fk = sorted(off_by_img[im])
        if len(rk) != len(fk):
            report['count_mismatch'].append({'image_id': im, 'replay': len(rk), 'official': len(fk)})
            report['exact'] = False
            continue
        for a, b in zip(rk, fk):
            if a != b:
                report['box_mismatch'].append({'image_id': im, 'replay': a, 'official': b})
                report['exact'] = False
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', required=True, help='export_m0_candidates.py output')
    ap.add_argument('--pred', default=None, help='optional official predictions.bbox.json')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cands_json = json.load(open(args.cands))
    replay = replay_hard_nms(cands_json)

    rep_vs_ref = compare_replay_vs_reference(replay, cands_json)
    rep_vs_ref.update({'score_thr': SCORE_THR, 'nms_iou': NMS_IOU, 'max_per_img': MAX_PER_IMG})
    with open(os.path.join(args.out_dir, 'hard_nms_replay_acceptance.json'), 'w') as f:
        json.dump(rep_vs_ref, f, indent=2, ensure_ascii=False)

    print('=== Hard NMS 回放验收 (replay vs 同前向参考) ===')
    print(f"replay={rep_vs_ref['num_replay']} reference={rep_vs_ref['num_reference']} "
          f"coverage_match={rep_vs_ref['image_coverage_match']} "
          f"count_mismatch={len(rep_vs_ref['count_mismatch'])} "
          f"box_mismatch={len(rep_vs_ref['box_mismatch'])} "
          f"EXACT={rep_vs_ref['exact']}")
    if rep_vs_ref['box_mismatch']:
        print('  mismatch examples:', rep_vs_ref['box_mismatch'][:5])

    if args.pred:
        rep_vs_off = compare_replay_vs_official(replay, args.pred)
        with open(os.path.join(args.out_dir, 'hard_nms_replay_vs_official.json'), 'w') as f:
            json.dump(rep_vs_off, f, indent=2, ensure_ascii=False)
        print('=== Hard NMS 回放 vs 官方 predictions.bbox.json (跨进程) ===')
        print(f"replay={rep_vs_off['num_replay']} official={rep_vs_off['num_official']} "
              f"coverage_match={rep_vs_off['image_coverage_match']} "
              f"count_mismatch={len(rep_vs_off['count_mismatch'])} "
              f"box_mismatch={len(rep_vs_off['box_mismatch'])} "
              f"EXACT={rep_vs_off['exact']}")
        if rep_vs_off['box_mismatch']:
            print('  mismatch examples:', rep_vs_off['box_mismatch'][:5])


if __name__ == '__main__':
    main()
