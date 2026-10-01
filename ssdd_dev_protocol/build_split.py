"""Build the SSDD dev split (dev set carved from the original training pool).

Pre-agreed rule (fixed BEFORE seeing any AP, not tunable by results):
  * source      = SSDD train.json (928 images, 2041 ship boxes)
  * dev target  ~20% of the original pool (~186 images)
  * excluded    the 9 supervised few-shot images of folds 6/7/8 (stay in training)
  * duplicate   exact content-duplicate groups are treated as atomic units
  * seed        numpy RandomState(678)
  * scene       no explicit scene metadata in the annotations; exact-duplicate
                grouping is applied, image-number-based scene isolation is NOT
                claimed (documented limitation).

Outputs (all COCO-format, ship category id=0):
  * data/dev.json                                  dev annotation
  * data/train_pool.json                           training pool (train minus dev)
  * data/dev_image_ids.json                        dev image id list
  * data/instances_train2017.{fold}@3-unlabeled-dev.json   per-fold new unlabeled
  * data/split_manifest.json                       rule + seed + counts + hashes
  * data/leakage_check.json                        overlap / retention / leakage
"""
import hashlib
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SEED = 678
TARGET = 186  # ~20% of 928

TRAIN = f'{ROOT}/../data/ssdd/annotations/train.json'
JPEG = f'{ROOT}/../data/ssdd/JPEGImages'
SEMI = f'{ROOT}/../data/ssdd/annotations/semi_supervised'
OUT = f'{ROOT}/data'

FOLD_SUP = {
    6: ['000020.jpg', '000613.jpg', '000893.jpg'],
    7: ['000255.jpg', '000312.jpg', '000678.jpg'],
    8: ['000050.jpg', '000664.jpg', '001014.jpg'],
}


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    train = json.load(open(TRAIN))
    imgs = train['images']
    anns = train['annotations']
    id2anns = {}
    for a in anns:
        id2anns.setdefault(a['image_id'], []).append(a)

    # duplicate groups (exact content)
    h2f = {}
    for im in imgs:
        h2f.setdefault(md5(os.path.join(JPEG, im['file_name'])), []).append(im['file_name'])
    dup_groups = [sorted(v) for v in h2f.values() if len(v) > 1]
    dup_set = set(sum(dup_groups, []))

    sup_set = set(sum(FOLD_SUP.values(), []))
    # 保护：包含监督图的重复组必须整组留在训练池（防止监督图与其重复图被拆到开发/训练两侧）
    sup_protected = set()
    protected_groups = []
    for g in dup_groups:
        if set(g) & sup_set:
            sup_protected |= set(g)
            protected_groups.append(g)
    non_sup = [im for im in imgs if im['file_name'] not in sup_set
               and im['file_name'] not in sup_protected]

    # split units (non-supervised): duplicate groups + unique images
    unit_of = {}
    for g in dup_groups:
        for fn in g:
            unit_of[fn] = tuple(g)
    units = []
    seen = set()
    for im in non_sup:
        fn = im['file_name']
        if fn in seen:
            continue
        u = unit_of.get(fn, (fn,))
        for x in u:
            seen.add(x)
        units.append(u)

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(units))
    dev_fns, cnt = [], 0
    for i in order:
        u = units[i]
        if cnt + len(u) > TARGET + 3:  # allow slight overshoot from group granularity
            continue
        dev_fns.extend(u)
        cnt += len(u)
        if cnt >= TARGET:
            break
    dev_fns = sorted(dev_fns)
    dev_set = set(dev_fns)

    def coco_subset(imgs_sel, keep_ids=True):
        sel = set(imgs_sel)
        sub_imgs = [im for im in imgs if im['file_name'] in sel]
        idmap = {im['id']: im['id'] for im in sub_imgs} if keep_ids else {
            im['id']: j + 1 for j, im in enumerate(sub_imgs)}
        sub_anns = [dict(a, image_id=idmap[a['image_id']]) for a in anns
                    if a['image_id'] in idmap and idmap[a['image_id']] is not None]
        return {
            'images': [dict(im, id=idmap[im['id']]) for im in sub_imgs],
            'annotations': sub_anns,
            'categories': [{'supercategory': 'none', 'id': 0, 'name': 'ship'}],
        }

    # dev annotation (keep ORIGINAL image_id from train.json)
    json.dump(coco_subset(dev_set, keep_ids=True), open(f'{OUT}/dev.json', 'w'))
    # training pool (train minus dev, keep original ids)
    train_pool_fns = [im['file_name'] for im in imgs if im['file_name'] not in dev_set]
    json.dump(coco_subset(train_pool_fns, keep_ids=True), open(f'{OUT}/train_pool.json', 'w'))
    json.dump(sorted(int(fn.replace('.jpg', '')) for fn in dev_fns),
              open(f'{OUT}/dev_image_ids.json', 'w'))

    # per-fold new unlabeled (original unlabeled minus dev)
    for f in (6, 7, 8):
        ul = json.load(open(f'{SEMI}/instances_train2017.{f}@3-unlabeled.json'))
        new_imgs = [im for im in ul['images'] if im['file_name'] not in dev_set]
        keep = {im['id'] for im in new_imgs}
        new_anns = [a for a in ul['annotations'] if a['image_id'] in keep]
        json.dump({'images': new_imgs, 'annotations': new_anns,
                   'categories': ul['categories']},
                  open(f'{OUT}/instances_train2017.{f}@3-unlabeled-dev.json', 'w'))

    # ---- manifest ----
    def sha256(p):
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for c in iter(lambda: fh.read(8 * 1024 * 1024), b''):
                h.update(c)
        return h.hexdigest()

    manifest = {
        'split_rule': '~20% of SSDD train.json (928) as dev; exclude 9 supervised (fold 6/7/8); '
                      'exact-duplicate groups atomic; numpy RandomState(678) permutation, greedy fill to >=186',
        'seed': SEED,
        'target_images': TARGET,
        'num_train_pool': len(imgs),
        'num_dev': len(dev_fns),
        'num_supervised_excluded': len(sup_set),
        'num_duplicate_groups': len(dup_groups),
        'num_images_in_duplicate_groups': len(dup_set),
        'dev_image_filenames': dev_fns,
        'dev_image_ids': [im['id'] for im in imgs if im['file_name'] in dev_set],
        'supervised_filenames': sorted(sup_set),
        'scene_grouping': 'no explicit scene metadata; exact-duplicate grouping applied; '
                          'image-number scene isolation NOT claimed (limitation)',
        'hashes': {
            'train.json': sha256(TRAIN),
            'dev.json': sha256(f'{OUT}/dev.json'),
            'train_pool.json': sha256(f'{OUT}/train_pool.json'),
        },
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    }
    json.dump(manifest, open(f'{OUT}/split_manifest.json', 'w'), indent=2, ensure_ascii=False)

    # ---- leakage check ----
    assert not (dev_set & sup_set), 'supervised images must stay in training (not dev)'
    assert not (dev_set & sup_protected), 'protected duplicate groups must stay in training (not dev)'
    test = json.load(open(f'{ROOT}/../data/ssdd/annotations/test.json'))
    test_fns = {im['file_name'] for im in test['images']}
    leakage = {
        'dev_vs_supervised_overlap': sorted(dev_set & sup_set),
        'dev_vs_supervised_overlap_count': len(dev_set & sup_set),
        'protected_duplicate_groups_with_supervised': protected_groups,
        'dev_vs_sup_protected_overlap': sorted(dev_set & sup_protected),
        'dev_vs_test_overlap': sorted(dev_set & test_fns),
        'dev_vs_test_overlap_count': len(dev_set & test_fns),
        'dev_all_from_train_pool': all(fn in {im['file_name'] for im in imgs} for fn in dev_fns),
        'dev_count': len(dev_fns),
        'supervised_retained_in_training_pool': sorted(sup_set) == sorted(
            set(sup_set) & set(train_pool_fns)),
        'dev_not_in_new_unlabeled': all(
            all(fn not in dev_set for fn in [im['file_name'] for im in
                json.load(open(f'{OUT}/instances_train2017.{f}@3-unlabeled-dev.json'))['images']])
            for f in (6, 7, 8)),
    }
    json.dump(leakage, open(f'{OUT}/leakage_check.json', 'w'), indent=2, ensure_ascii=False)

    print('dev images:', len(dev_fns))
    print('dev filenames:', dev_fns)
    print('train pool (minus dev):', len(train_pool_fns))
    print('duplicate groups:', len(dup_groups), 'images:', len(dup_set))
    print('leakage:', json.dumps(leakage, ensure_ascii=False))
    print('manifest written to', f'{OUT}/split_manifest.json')


if __name__ == '__main__':
    main()
