"""Analyze diagnosis v2 JSON for stage-wise target loss.

Steps 2-3: verify baseline consistency + build multi-IoU stage coverage table.
Outputs into a fresh analysis directory (never overwrites predictions/anns/logs).
"""
import argparse
import csv
import json
import os

import numpy as np
from pycocotools.coco import COCO


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def rescale(box, sf):
    return [box[0] / sf[0], box[1] / sf[1], box[2] / sf[2], box[3] / sf[3]]


def nms(boxes, scores, thr=0.5):
    """greedy NMS, returns kept indices (no max_per_img truncation)."""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ovr = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]
    return keep


def build_gt_map(coco):
    fn2gt = {}
    for im in coco.dataset['images']:
        ann_ids = coco.getAnnIds(imgIds=im['id'])
        gts = []
        for a in coco.loadAnns(ann_ids):
            if a.get('iscrowd', 0):
                continue
            x, y, w, h = a['bbox']
            gts.append((a['id'], [x, y, x + w, y + h]))
        fn2gt[im['file_name']] = gts
    return fn2gt


def load_diag(path):
    d = json.load(open(path))
    return d


def verify(diag, coco, out_dir):
    records = diag['records']
    n_imgs = len(records)
    coco_imgs = coco.dataset['images']
    # image id coverage
    fn_in_diag = {r['filename'] for r in records}
    fn_in_coco = {im['file_name'] for im in coco_imgs}
    n_gt = sum(len(coco.getAnnIds(imgIds=im['id'])) for im in coco_imgs)
    # teacher2 formal predictions
    preds = []
    for r in records:
        for im in coco_imgs:
            if im['file_name'] == r['filename']:
                img_id = im['id']; break
        cat_id = coco.getCatIds()[0]
        for b in r['teacher2_formal']['boxes']:
            x1, y1, x2, y2, s = b
            preds.append({'image_id': img_id, 'category_id': cat_id,
                          'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(s)})
    report = {
        'num_images_diag': n_imgs,
        'num_images_coco': len(coco_imgs),
        'image_coverage_match': fn_in_diag == fn_in_coco,
        'num_gt': n_gt,
        'num_preds': len(preds),
    }
    # AP via coco
    from pycocotools.cocoeval import COCOeval
    cocoDt = coco.loadRes(preds)
    ce = COCOeval(coco, cocoDt, 'bbox')
    ce.evaluate(); ce.accumulate(); ce.summarize()
    report['ap'] = round(float(ce.stats[0]), 3)
    report['ap50'] = round(float(ce.stats[1]), 3)
    report['ap75'] = round(float(ce.stats[2]), 3)
    with open(os.path.join(out_dir, 'verification.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def stage_coverage(diag, coco, out_dir):
    fn2gt = build_gt_map(coco)
    records = diag['records']
    ious = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
    rows = []
    for t in ['teacher1', 'teacher2']:
        for thr in ious:
            # stage stats
            covered_rpn = covered_pre = covered_scr = covered_nms = covered_final = 0
            final_matched = 0  # one-to-one recall on final
            for r in records:
                sf = r['scale_factor']
                gts = fn2gt[r['filename']]
                # stage boxes (rescaled to orig)
                rpn = [rescale(b[:4], sf) for b in r[t]['rpn_proposals']['boxes']]
                pre_boxes = [rescale(b[:4], sf) for b in r[t]['roi_pre_nms']['boxes']]
                pre_scores = [s[0] for s in r[t]['roi_pre_nms']['scores']]
                fin = [rescale(b[:4], sf) for b in r[t]['final_det']['boxes']]
                # score_thr filtered (0.05) and NMS (iou=0.5) simulated from pre
                scr_idx = [i for i, s in enumerate(pre_scores) if s > 0.05]
                scr_boxes = [pre_boxes[i] for i in scr_idx]
                scr_sc = [pre_scores[i] for i in scr_idx]
                nms_idx = nms(scr_boxes, scr_sc, thr=0.5)
                nms_boxes = [scr_boxes[i] for i in nms_idx]
                for gt_id, gt in gts:
                    if any(iou(gt, p) >= thr for p in rpn):
                        covered_rpn += 1
                    if any(iou(gt, p) >= thr for p in pre_boxes):
                        covered_pre += 1
                    if any(iou(gt, p) >= thr for p in scr_boxes):
                        covered_scr += 1
                    if any(iou(gt, p) >= thr for p in nms_boxes):
                        covered_nms += 1
                    if any(iou(gt, p) >= thr for p in fin):
                        covered_final += 1
                    # one-to-one recall on final (greedy best IoU)
                    if fin:
                        best = max(iou(gt, p) for p in fin)
                        if best >= thr:
                            final_matched += 1
            total = sum(len(fn2gt[r['filename']]) for r in records)
            rows.append({
                'teacher': t, 'iou_thr': thr, 'total_gt': total,
                'rpn_covered': covered_rpn, 'pre_covered': covered_pre,
                'score_thr_covered': covered_scr, 'nms_covered': covered_nms,
                'final_covered': covered_final, 'final_1to1_recall': final_matched,
            })
    with open(os.path.join(out_dir, 'stage_coverage.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', required=True)
    ap.add_argument('--test-ann', default='data/ssdd/annotations/test.json')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag = load_diag(args.diag)
    coco = COCO(args.test_ann)

    print('=== 步骤2: 核验 ===')
    rep = verify(diag, coco, args.out_dir)
    print(json.dumps(rep, indent=2, ensure_ascii=False))

    print('=== 步骤3: 阶段覆盖表 ===')
    rows = stage_coverage(diag, coco, args.out_dir)
    # 打印关键行（iou 0.5 和 0.75）
    for r in rows:
        if r['iou_thr'] in (0.5, 0.75):
            print(f"{r['teacher']} IoU={r['iou_thr']}: RPN={r['rpn_covered']} "
                  f"pre={r['pre_covered']} score_thr={r['score_thr_covered']} "
                  f"NMS={r['nms_covered']} final={r['final_covered']} "
                  f"1to1={r['final_1to1_recall']}/{r['total_gt']}")
    print(f"written to {args.out_dir}")


if __name__ == '__main__':
    main()
