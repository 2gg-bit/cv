"""Offline A/B post-processing evaluation on a candidate cache.

A = original Hard NMS (CUDA mmcv batched_nms, matches the baseline).
B = fixed-parameter Linear Soft-NMS (CPU mmcv soft_nms), per protocol:
      method=linear, iou_threshold=0.5, min_score=0.05, offset=0.

Both methods reuse the SAME pre-NMS candidate cache (one forward pass per image,
see export_m0_candidates.py) and apply the SAME entry filter (p_ship > 0.05).

Soft-NMS rules (step 5):
  1. output score = Soft-NMS *returned* new (decayed) score, never the original
     score re-fetched via returned indices;
  2. sort by the new score, then take at most 100 per image;
  3. no additional 0.3 / 0.5 threshold after Soft-NMS;
  4. empty / single / tied candidates handled;
  5. the cache's original scores are never mutated.

For each fold it writes predictions.bbox.json + metrics (AP/AP50/AP75/AP85/
APs/APm/APl/AR@100) + background/duplicate/box counts + candidate->output
index & score change records + post-processing timing (incl. CPU/GPU transfer).
"""
import argparse
import hashlib
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from mmcv.ops import batched_nms, soft_nms

SCORE_THR = 0.05
NMS_IOU = 0.5
MAX_PER_IMG = 100


def _rescale(boxes, sf):
    b = torch.tensor(boxes, dtype=torch.float32)
    sf_t = torch.tensor(sf, dtype=torch.float32)
    return (b.view(-1, 1, 4) / sf_t).view(-1, 4)


def replay_hard_nms(cands_json):
    """A: Hard NMS on CUDA (matches baseline). Returns (preds, changes, timing)."""
    preds, changes = [], []
    t_xfer = t_nms = 0.0
    for r in cands_json['records']:
        cands = r['candidates']
        if not cands:
            continue
        t0 = time.perf_counter()
        boxes = torch.tensor([c['box'] for c in cands], dtype=torch.float32)
        p = torch.tensor([c['p'] for c in cands], dtype=torch.float32)
        boxes = _rescale([c['box'] for c in cands], r['scale_factor'])
        valid = p > SCORE_THR
        if not valid.any():
            continue
        boxes_c = boxes[valid].cuda()
        p_c = p[valid].cuda()
        torch.cuda.synchronize()
        t_xfer += time.perf_counter() - t0
        t1 = time.perf_counter()
        labels = torch.zeros(boxes_c.size(0), dtype=torch.long, device=boxes_c.device)
        dets, _ = batched_nms(boxes_c, p_c, labels, dict(type='nms', iou_threshold=NMS_IOU))
        dets = dets[:MAX_PER_IMG].cpu()
        torch.cuda.synchronize()
        t_nms += time.perf_counter() - t1
        # map kept boxes back to original candidate indices (filtered order)
        vidx = valid.nonzero(as_tuple=False).squeeze(1)
        for k, d in enumerate(dets):
            orig_idx = int(vidx[k].item())
            preds.append({'image_id': r['image_id'],
                          'box': [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                          'score': float(d[4]), 'label': 0})
            changes.append({'image_id': r['image_id'],
                            'proposal_id': cands[orig_idx]['proposal_id'],
                            'original_p': cands[orig_idx]['p'],
                            'output_score': float(d[4]), 'rank': k})
    return preds, changes, {'transfer_s': t_xfer, 'nms_s': t_nms, 'device': 'cuda',
                            'dtype': 'float32'}


def replay_soft_nms(cands_json):
    """B: Linear Soft-NMS on CPU. Returns (preds, changes, timing)."""
    preds, changes = [], []
    t_soft = 0.0
    for r in cands_json['records']:
        cands = r['candidates']
        if not cands:
            continue
        boxes = torch.tensor([c['box'] for c in cands], dtype=torch.float32)
        p = torch.tensor([c['p'] for c in cands], dtype=torch.float32)
        boxes = _rescale([c['box'] for c in cands], r['scale_factor'])
        valid = p > SCORE_THR
        if not valid.any():
            continue
        boxes_f = boxes[valid]
        p_f = p[valid]
        vidx = valid.nonzero(as_tuple=False).squeeze(1)
        t0 = time.perf_counter()
        # CPU soft_nms returns (dets [x1,y1,x2,y2,new_score], inds), sorted by new score desc
        dets, inds = soft_nms(boxes_f, p_f, iou_threshold=NMS_IOU, sigma=0.5,
                              min_score=0.05, method='linear', offset=0)
        dets = dets[:MAX_PER_IMG]
        t_soft += time.perf_counter() - t0
        for k in range(dets.size(0)):
            d = dets[k]
            orig_idx = int(vidx[int(inds[k].item())].item())
            preds.append({'image_id': r['image_id'],
                          'box': [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                          'score': float(d[4]), 'label': 0})
            changes.append({'image_id': r['image_id'],
                            'proposal_id': cands[orig_idx]['proposal_id'],
                            'original_p': cands[orig_idx]['p'],
                            'output_score': float(d[4]), 'rank': k})
    return preds, changes, {'soft_nms_s': t_soft, 'device': 'cpu', 'dtype': 'float32'}


def write_predictions(preds, out_path):
    coco = []
    for p in preds:
        x1, y1, x2, y2 = p['box']
        coco.append({'image_id': p['image_id'], 'category_id': p['label'],
                     'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': p['score']})
    with open(out_path, 'w') as f:
        json.dump(coco, f)
    return coco


def compute_metrics(pred_json, gt_path):
    import io
    import contextlib
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_json)
    ce = COCOeval(coco_gt, coco_dt, 'bbox')
    ce.evaluate(); ce.accumulate()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ce.summarize()
    stats = ce.stats
    prec = ce.eval['precision']  # [T=10, R=101, K, A=4, M=3]
    ap85 = float(prec[7, :, 0, 0, 2].mean())  # IoU=0.85 (idx 7), all area, maxDet=100
    metrics = {
        'AP': float(stats[0]), 'AP50': float(stats[1]), 'AP75': float(stats[2]),
        'AP85': ap85,
        'APs': float(stats[3]), 'APm': float(stats[4]), 'APl': float(stats[5]),
        'AR@100': float(stats[8]),
    }
    return metrics, buf.getvalue()


def compute_bkg_dupe(preds, gt_path):
    from pycocotools.coco import COCO
    coco = COCO(gt_path)
    fn2id = {im['file_name']: im['id'] for im in coco.dataset['images']}
    id2gt = {}
    for im in coco.dataset['images']:
        gts = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=im['id'])):
            if a.get('iscrowd', 0):
                continue
            x, y, w, h = a['bbox']
            gts.append([x, y, x + w, y + h])
        id2gt[im['id']] = gts

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    by_img = defaultdict(list)
    for p in preds:
        by_img[p['image_id']].append((p['box'], p['score']))
    n_bkg = n_dupe = n_tp = 0
    for img_id, items in by_img.items():
        gts = id2gt[img_id]
        matched = [False] * len(gts)
        # sort preds by score desc for greedy one-to-one matching
        order = sorted(range(len(items)), key=lambda i: -items[i][1])
        for i in order:
            b = items[i][0]
            best_j, best_iou = -1, 0.0
            for j, gt in enumerate(gts):
                v = iou(b, gt)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= 0.5:
                if not matched[best_j]:
                    matched[best_j] = True
                    n_tp += 1
                else:
                    n_dupe += 1
            else:
                n_bkg += 1
    return {'num_preds': len(preds), 'background_errors': n_bkg,
            'duplicates': n_dupe, 'true_positives': n_tp}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def write_metadata(out_dir, method, fold, cands_json, gt_path, n_preds, eval_params):
    ckpt = os.path.abspath(cands_json.get('checkpoint', ''))
    meta = {
        'method': method,
        'fold': fold,
        'checkpoint': ckpt,
        'checkpoint_sha256': sha256_file(ckpt) if os.path.exists(ckpt) else None,
        'config': cands_json.get('config'),
        'test_ann_file': os.path.abspath(gt_path),
        'test_ann_sha256': sha256_file(gt_path),
        'num_images': cands_json.get('num_images'),
        'num_predictions': n_preds,
        'eval_params': eval_params,
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    }
    json.dump(meta, open(os.path.join(out_dir, f'{method}_metadata.json'), 'w'),
              indent=2, ensure_ascii=False)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--method', choices=['A', 'B', 'AB'], default='AB')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cands_json = json.load(open(args.cands))
    fold = cands_json.get('fold')

    result = {'fold': fold, 'checkpoint': cands_json.get('checkpoint')}
    if args.method in ('A', 'AB'):
        pa, ca, ta = replay_hard_nms(cands_json)
        write_predictions(pa, os.path.join(args.out_dir, 'A_predictions.bbox.json'))
        json.dump(ca, open(os.path.join(args.out_dir, 'A_changes.json'), 'w'))
        m, evallog = compute_metrics(
            os.path.join(args.out_dir, 'A_predictions.bbox.json'), args.gt)
        json.dump(m, open(os.path.join(args.out_dir, 'A_metrics.json'), 'w'), indent=2)
        open(os.path.join(args.out_dir, 'A_eval.log'), 'w').write(evallog)
        write_metadata(args.out_dir, 'A', fold, cands_json, args.gt, len(pa),
                       {'algorithm': 'Hard NMS', 'score_thr': SCORE_THR,
                        'nms_iou': NMS_IOU, 'max_per_img': MAX_PER_IMG,
                        'device': 'cuda', 'dtype': 'float32'})
        result['A'] = {'metrics': m, 'bkg_dupe': compute_bkg_dupe(pa, args.gt), 'timing': ta}
    if args.method in ('B', 'AB'):
        pb, cb, tb = replay_soft_nms(cands_json)
        write_predictions(pb, os.path.join(args.out_dir, 'B_predictions.bbox.json'))
        json.dump(cb, open(os.path.join(args.out_dir, 'B_changes.json'), 'w'))
        m, evallog = compute_metrics(
            os.path.join(args.out_dir, 'B_predictions.bbox.json'), args.gt)
        json.dump(m, open(os.path.join(args.out_dir, 'B_metrics.json'), 'w'), indent=2)
        open(os.path.join(args.out_dir, 'B_eval.log'), 'w').write(evallog)
        write_metadata(args.out_dir, 'B', fold, cands_json, args.gt, len(pb),
                       {'algorithm': 'Linear Soft-NMS', 'method': 'linear',
                        'iou_threshold': NMS_IOU, 'min_score': 0.05, 'offset': 0,
                        'entry_score_thr': SCORE_THR, 'max_per_img': MAX_PER_IMG,
                        'device': 'cpu', 'dtype': 'float32'})
        result['B'] = {'metrics': m, 'bkg_dupe': compute_bkg_dupe(pb, args.gt), 'timing': tb}

    if 'A' in result and 'B' in result:
        diff = {k: round(result['B']['metrics'][k] - result['A']['metrics'][k], 4)
                for k in result['A']['metrics']}
        result['B_minus_A'] = diff

    json.dump(result, open(os.path.join(args.out_dir, 'summary.json'), 'w'),
              indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
