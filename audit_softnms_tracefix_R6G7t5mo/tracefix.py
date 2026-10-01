"""Trace-fix: rebuild A/B candidate->output tracking records and verify.

Fixes the A-group index-mapping bug: the previous replay discarded the `keep`
indices returned by `batched_nms` and used the output *rank* `k` directly to
index the filtered->original mapping (`vidx[k]`), which is wrong because
`dets` is sorted by score descending (rank != filtered-array index).  The
correct chain is:

    original candidates + ids
      -> score filter (valid)                  [filtered index]
      -> batched_nms returns keep               [keep[k] = filtered index]
      -> original index = vidx[keep[k]]
      -> sort/truncate box/id/score together
      -> tracking record

This script re-runs only the cache post-processing (no model forward), rebuilds
the tracking records for both A (Hard NMS, CUDA) and B (Linear Soft-NMS, CPU),
and produces machine-readable verification:
  * {A,B}_trace.json              tracking records
  * prediction_equivalence.json   replay vs the already-verified predictions
  * trace_acceptance.json         per-record cache traceability + score rules
  * geometric_stats.json          max_iou_lt_01 / max_iou_01_to_05 / max_iou_lt_05
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from mmcv.ops import batched_nms, soft_nms

SCORE_THR = 0.05
NMS_IOU = 0.5
MAX_PER_IMG = 100


def rescale(boxes, sf):
    b = torch.tensor(boxes, dtype=torch.float32)
    sf_t = torch.tensor(sf, dtype=torch.float32)
    return (b.view(-1, 1, 4) / sf_t).view(-1, 4)


def _make_trace(r, cands, orig_idx, k, box, score):
    return {
        'image_id': r['image_id'],
        'proposal_id': cands[orig_idx]['proposal_id'],
        'cache_row_idx': orig_idx,
        'rank': k,
        'original_p': cands[orig_idx]['p'],
        'output_score': float(score),
        'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
    }


def replay_hard_nms(cands_json):
    """A: Hard NMS (CUDA). FIXED index mapping via batched_nms `keep`."""
    preds, trace = [], []
    for r in cands_json['records']:
        cands = r['candidates']
        if not cands:
            continue
        p = torch.tensor([c['p'] for c in cands], dtype=torch.float32)
        boxes = rescale([c['box'] for c in cands], r['scale_factor'])
        valid = p > SCORE_THR
        if not valid.any():
            continue
        boxes_c = boxes[valid].cuda()
        p_c = p[valid].cuda()
        labels = torch.zeros(boxes_c.size(0), dtype=torch.long, device=boxes_c.device)
        dets, keep = batched_nms(boxes_c, p_c, labels, dict(type='nms', iou_threshold=NMS_IOU))
        dets = dets[:MAX_PER_IMG].cpu()
        keep = keep[:MAX_PER_IMG].cpu()
        vidx = valid.nonzero(as_tuple=False).squeeze(1)  # filtered -> original index
        for k in range(dets.size(0)):
            d = dets[k]
            orig_idx = int(vidx[int(keep[k].item())].item())  # FIXED: keep[k], not k
            preds.append({'image_id': r['image_id'],
                          'box': [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                          'score': float(d[4]), 'label': 0})
            trace.append(_make_trace(r, cands, orig_idx, k, d[:4], d[4]))
    return preds, trace


def replay_soft_nms(cands_json):
    """B: Linear Soft-NMS (CPU). Uses soft_nms's returned `inds` (correct)."""
    preds, trace = [], []
    for r in cands_json['records']:
        cands = r['candidates']
        if not cands:
            continue
        p = torch.tensor([c['p'] for c in cands], dtype=torch.float32)
        boxes = rescale([c['box'] for c in cands], r['scale_factor'])
        valid = p > SCORE_THR
        if not valid.any():
            continue
        boxes_f = boxes[valid]
        p_f = p[valid]
        vidx = valid.nonzero(as_tuple=False).squeeze(1)
        dets, inds = soft_nms(boxes_f, p_f, iou_threshold=NMS_IOU, sigma=0.5,
                              min_score=0.05, method='linear', offset=0)
        dets = dets[:MAX_PER_IMG]
        for k in range(dets.size(0)):
            d = dets[k]
            orig_idx = int(vidx[int(inds[k].item())].item())  # inds[k] = filtered index
            preds.append({'image_id': r['image_id'],
                          'box': [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                          'score': float(d[4]), 'label': 0})
            trace.append(_make_trace(r, cands, orig_idx, k, d[:4], d[4]))
    return preds, trace


def load_verified_preds(path):
    # predictions.bbox.json is a flat list of {image_id, category_id, bbox:[x,y,w,h], score}
    preds = []
    for p in json.load(open(path)):
        x, y, w, h = p['bbox']
        preds.append({'image_id': p['image_id'],
                      'box': [x, y, x + w, y + h], 'score': p['score']})
    return preds


def pred_multiset_key(p):
    return (p['image_id'], round(p['box'][0], 6), round(p['box'][1], 6),
            round(p['box'][2], 6), round(p['box'][3], 6), round(p['score'], 6))


def verify_prediction_equivalence(replay, verified):
    rk = defaultdict(list); vk = defaultdict(list)
    for p in replay:
        rk[p['image_id']].append(pred_multiset_key(p))
    for p in verified:
        vk[p['image_id']].append(pred_multiset_key(p))
    all_imgs = sorted(set(rk) | set(vk))
    mismatch = []
    for im in all_imgs:
        a, b = sorted(rk[im]), sorted(vk[im])
        if a != b:
            mismatch.append({'image_id': im, 'replay': len(a), 'verified': len(b)})
    return {'replay_boxes': len(replay), 'verified_boxes': len(verified),
            'image_coverage_match': set(rk) == set(vk),
            'per_image_mismatch': len(mismatch),
            'per_box_exact': len(mismatch) == 0,
            'examples': mismatch[:10]}


def verify_trace_acceptance(trace, preds, cands_json, method):
    # build cache lookup: image_id -> candidates list (for box trace-back)
    cands_by_img = {r['image_id']: r['candidates'] for r in cands_json['records']}
    sf_by_img = {r['image_id']: r['scale_factor'] for r in cands_json['records']}
    n_trace_ok = n_output_ok = n_score_ok = 0
    bad = []
    assert len(trace) == len(preds), f'trace({len(trace)}) != preds({len(preds)})'
    for tr, pr in zip(trace, preds):
        iid = tr['image_id']
        cands = cands_by_img[iid]
        sf = sf_by_img[iid]
        # 1. trace back to cache
        row_ok = (0 <= tr['cache_row_idx'] < len(cands)
                  and cands[tr['cache_row_idx']]['proposal_id'] == tr['proposal_id']
                  and cands[tr['cache_row_idx']]['p'] == tr['original_p'])
        # original box (rescaled) == output box (NMS/soft_nms keep boxes unchanged)
        ob = rescale([cands[tr['cache_row_idx']]['box']], sf)[0] if row_ok else None
        box_ok = (ob is not None and
                  round(float(ob[0]), 6) == round(tr['box'][0], 6) and
                  round(float(ob[1]), 6) == round(tr['box'][1], 6) and
                  round(float(ob[2]), 6) == round(tr['box'][2], 6) and
                  round(float(ob[3]), 6) == round(tr['box'][3], 6))
        # 2. score rule
        if method == 'A':
            score_ok = tr['output_score'] == tr['original_p']
        else:
            score_ok = tr['output_score'] <= tr['original_p'] + 1e-12
        # 3. output correspondence (trace vs pred, same rank/order)
        out_ok = (tr['image_id'] == pr['image_id']
                  and round(tr['box'][0], 6) == round(pr['box'][0], 6)
                  and round(tr['box'][1], 6) == round(pr['box'][1], 6)
                  and round(tr['box'][2], 6) == round(pr['box'][2], 6)
                  and round(tr['box'][3], 6) == round(pr['box'][3], 6)
                  and round(tr['output_score'], 6) == round(pr['score'], 6))
        if row_ok and box_ok:
            n_trace_ok += 1
        if score_ok:
            n_score_ok += 1
        if out_ok:
            n_output_ok += 1
        if not (row_ok and box_ok and score_ok and out_ok):
            bad.append({'image_id': iid, 'rank': tr['rank'],
                        'row_ok': row_ok, 'box_ok': box_ok,
                        'score_ok': score_ok, 'out_ok': out_ok,
                        'proposal_id': tr['proposal_id']})
    return {
        'method': method,
        'n_records': len(trace),
        'trace_to_cache_ok': n_trace_ok,
        'score_rule_ok': n_score_ok,
        'output_correspondence_ok': n_output_ok,
        'count_match': len(trace) == len(preds),
        'all_pass': (n_trace_ok == len(trace) and n_score_ok == len(trace)
                     and n_output_ok == len(trace)),
        'bad_examples': bad[:10],
        'bad_count': len(bad),
    }


def geometric_stats(preds, gt_path):
    from pycocotools.coco import COCO
    coco = COCO(gt_path)
    id2gt = defaultdict(list)
    for im in coco.dataset['images']:
        for a in coco.loadAnns(coco.getAnnIds(imgIds=im['id'])):
            if a.get('iscrowd', 0):
                continue
            x, y, w, h = a['bbox']
            id2gt[im['id']].append([x, y, x + w, y + h])

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    lt01 = to05 = 0
    for p in preds:
        gts = id2gt[p['image_id']]
        mx = max((iou(p['box'], g) for g in gts), default=0.0)
        if mx < 0.1:
            lt01 += 1
        elif mx < 0.5:
            to05 += 1
    lt05 = lt01 + to05
    return {'num_preds': len(preds), 'max_iou_lt_01': lt01,
            'max_iou_01_to_05': to05, 'max_iou_lt_05': lt05,
            'reconcile_ok': lt05 == lt01 + to05}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred-a', required=True)
    ap.add_argument('--pred-b', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--fold', type=int, required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cands_json = json.load(open(args.cands))

    out = {'fold': args.fold, 'checkpoint': cands_json.get('checkpoint')}

    pa, ta = replay_hard_nms(cands_json)
    pb, tb = replay_soft_nms(cands_json)
    json.dump(ta, open(os.path.join(args.out_dir, 'A_trace.json'), 'w'))
    json.dump(tb, open(os.path.join(args.out_dir, 'B_trace.json'), 'w'))

    va = load_verified_preds(args.pred_a)
    vb = load_verified_preds(args.pred_b)

    out['prediction_equivalence'] = {
        'A': verify_prediction_equivalence(pa, va),
        'B': verify_prediction_equivalence(pb, vb),
    }
    out['trace_acceptance'] = {
        'A': verify_trace_acceptance(ta, pa, cands_json, 'A'),
        'B': verify_trace_acceptance(tb, pb, cands_json, 'B'),
    }
    out['geometric_stats'] = {
        'A': geometric_stats(pa, args.gt),
        'B': geometric_stats(pb, args.gt),
    }
    json.dump(out, open(os.path.join(args.out_dir, 'tracefix_result.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
