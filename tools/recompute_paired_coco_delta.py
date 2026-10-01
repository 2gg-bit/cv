"""Recompute COCO bbox metrics at full precision for paired baseline-vs-variant runs.

Why this exists: every frozen evaluation artifact (`metrics.json`, `eval.log`, the
pycocotools summary) stores the COCO stats rounded to three decimals, so a delta read
off those files would be a delta of rounded numbers.  The standing rule for this
project is that deltas come from unrounded values, so this script replays the *same*
COCOeval on *both* sides of each pair and reports the raw values together with their
difference.  The stored three-decimal values are used only as a consistency check.

Fidelity to the frozen evaluation path
(`tools/eval_teacher2_export.py` -> `CocoDataset.evaluate`) is enforced two ways:

  * the parameter block below is a transcription of the vendored source, and
  * `check_provenance()` re-reads that source and the resolved configs and asserts the
    transcribed lines are still present, the config's `evaluation` dict carries no
    COCO-parameter overrides, and the dataset uses the expected single class -- all
    before any number is computed.

Both sides of a pair are evaluated with the identical code path and identical
parameters; the reported delta is `variant - baseline`.

CPU only, pure numpy/pycocotools: it never imports torch/mmdet/ssod, never touches a
checkpoint for inference, and never writes into an input directory.
"""

import argparse
import contextlib
import hashlib
import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------------------------------------------------------------------
# Frozen COCOeval parameters, transcribed from the vendored mmdet source.
#
# mmdet/datasets/coco.py, CocoDataset.evaluate():
#     proposal_nums=(100, 300, 1000)                       -> params.maxDets
#     iou_thrs = np.linspace(.5, 0.95,
#         int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)  -> params.iouThrs
#     cocoEval.params.catIds = self.cat_ids
#     cocoEval.params.imgIds = self.img_ids
#     cocoEval.params.useCats     (left at the pycocotools default True)
#     areaRng / areaRngLbl / recThrs left at the pycocotools defaults
#     classwise=False
#
# tools/eval_teacher2_export.py, get_eval_kwargs() keeps only `metric='bbox'` out of
# cfg.evaluation (everything else it strips), so no COCO parameter is overridden.
# ---------------------------------------------------------------------------
PROPOSAL_NUMS = (100, 300, 1000)
IOU_THRS = np.linspace(
    .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
IOU_TYPE = 'bbox'
CLASSWISE = False

# mmdet/datasets/coco.py, the `coco_metric_names` mapping.
STATS_INDEX = {
    'mAP': 0,
    'mAP_50': 1,
    'mAP_75': 2,
    'mAP_s': 3,
    'mAP_m': 4,
    'mAP_l': 5,
    'AR@100': 6,
    'AR@300': 7,
    'AR@1000': 8,
    'AR_s@1000': 9,
    'AR_m@1000': 10,
    'AR_l@1000': 11,
}
METRIC_ITEMS = tuple(STATS_INDEX)
PRIMARY_ITEMS = ('mAP', 'mAP_50', 'mAP_75', 'AR@100')

EXPECTED_EVALUATION_DICT = (
    "dict(interval=4000, metric='bbox', type='SubModulesDistEvalHook')")
FORBIDDEN_EVAL_KEYS = ('proposal_nums', 'iou_thrs', 'classwise', 'metric_items')
EXPECTED_CLASS_DECL = "classes=('ship', )"

VENDORED_COCO_PY = Path(
    '/home/xcc/dual_teacher_project/thirdparty/mmdetection/mmdet/datasets/coco.py')
EVAL_TOOL = Path(__file__).resolve().parent / 'eval_teacher2_export.py'

# Lines that must still be present in the vendored source for the transcription above
# to describe it.  Each entry is (needle, what it pins down).
VENDORED_NEEDLES = (
    ('proposal_nums=(100, 300, 1000),', 'bbox evaluate default proposal_nums'),
    ('cocoEval.params.catIds = self.cat_ids', 'catIds assignment'),
    ('cocoEval.params.imgIds = self.img_ids', 'imgIds assignment'),
    ('cocoEval.params.maxDets = list(proposal_nums)', 'maxDets assignment'),
    ('cocoEval.params.iouThrs = iou_thrs', 'iouThrs assignment'),
    ("'AR@100': 6,", 'stats index mapping'),
)
EVAL_TOOL_NEEDLES = (
    ('def get_eval_kwargs(cfg):', 'eval kwargs helper'),
    ('eval_kwargs = cfg.get("evaluation", {}).copy()', 'source of eval kwargs'),
)

FROZEN_TEST_ANN_SHA256 = (
    '19aa601904243be548c1551b247f331f1317dc6a9bf4ac9911e6b07398ca919f')
FROZEN_TEST_ANN_IMAGES = 232


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def mmdet_round(value):
    """Reproduce mmdet's storage of a stat: float(f'{stat:.3f}')."""
    return float('{:.3f}'.format(value))


def extract_call(text, name):
    """Return the exact `name = <call(...)>` text, paren-balanced, else None."""
    match = re.search(r'^%s\s*=\s*' % re.escape(name), text, flags=re.MULTILINE)
    if match is None:
        return None
    start = match.end()
    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def check_provenance(resolved_config_paths):
    """Assert the frozen parameters still describe the frozen evaluation path."""
    checks = []

    vendored_text = VENDORED_COCO_PY.read_text(encoding='utf-8')
    for needle, label in VENDORED_NEEDLES:
        found = needle in vendored_text
        checks.append({'check': 'vendored_coco_py__%s' % label,
                       'needle': needle, 'ok': found})
        if not found:
            raise SystemExit(
                'vendored source no longer matches the transcribed parameters: '
                'missing %r in %s' % (needle, VENDORED_COCO_PY))

    tool_text = EVAL_TOOL.read_text(encoding='utf-8')
    for needle, label in EVAL_TOOL_NEEDLES:
        found = needle in tool_text
        checks.append({'check': 'eval_tool__%s' % label,
                       'needle': needle, 'ok': found})
        if not found:
            raise SystemExit(
                'eval tool no longer matches: missing %r in %s'
                % (needle, EVAL_TOOL))

    for cfg_path in resolved_config_paths:
        text = Path(cfg_path).read_text(encoding='utf-8')
        call = extract_call(text, 'evaluation')
        if call is None:
            raise SystemExit('no `evaluation = ` assignment in %s' % cfg_path)
        if call != EXPECTED_EVALUATION_DICT:
            raise SystemExit(
                'unexpected evaluation dict in %s: %s' % (cfg_path, call))
        checks.append({'check': 'evaluation_dict__%s' % Path(cfg_path).parent.name,
                       'needle': call, 'ok': True})
        for key in FORBIDDEN_EVAL_KEYS:
            if key in call:
                raise SystemExit(
                    'evaluation dict in %s carries COCO parameter %r'
                    % (cfg_path, key))
        if EXPECTED_CLASS_DECL not in text:
            raise SystemExit(
                'expected single-class declaration %r not found in %s'
                % (EXPECTED_CLASS_DECL, cfg_path))
        checks.append({'check': 'class_decl__%s' % Path(cfg_path).parent.name,
                       'needle': EXPECTED_CLASS_DECL, 'ok': True})

    return {
        'vendored_coco_py': {'path': str(VENDORED_COCO_PY),
                             'sha256': sha256_file(VENDORED_COCO_PY)},
        'eval_tool': {'path': str(EVAL_TOOL), 'sha256': sha256_file(EVAL_TOOL)},
        'checks': checks,
    }


def eval_params_for(cat_ids, img_ids):
    return {
        'iouType': IOU_TYPE,
        'catIds': list(cat_ids),
        'imgIds': list(img_ids),
        'maxDets': list(PROPOSAL_NUMS),
        'iouThrs': [float(v) for v in IOU_THRS],
        'classwise': CLASSWISE,
    }


def compute_stats(predictions_path, gt_path, resolved_config_path):
    """Replay the frozen CocoDataset.evaluate on one predictions file."""
    gt = COCO(str(gt_path))
    preds = load_json(predictions_path)

    # mmdet's CocoDataset uses the snake_case pycocotools wrappers from
    # mmdet/datasets/api_wrappers/coco_api.py; the official camelCase calls below are
    # the same functions.  COCOeval.evaluate() applies np.unique to both lists, so the
    # order returned here cannot influence the result.
    categories = gt.dataset.get('categories', [])
    cat_ids = [c['id'] for c in categories if c['name'] == 'ship']
    if len(cat_ids) != 1:
        raise SystemExit('expected exactly one ship category, got %r' % (cat_ids,))
    img_ids = sorted(gt.getImgIds())

    params = eval_params_for(cat_ids, img_ids)

    coco_dt = gt.loadRes(preds)
    coco_eval = COCOeval(gt, coco_dt, IOU_TYPE)
    coco_eval.params.catIds = params['catIds']
    coco_eval.params.imgIds = params['imgIds']
    coco_eval.params.maxDets = params['maxDets']
    coco_eval.params.iouThrs = np.array(params['iouThrs'])
    assert coco_eval.params.useCats == 1, 'pycocotools default useCats changed'

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats
    raw = {item: float(stats[STATS_INDEX[item]]) for item in METRIC_ITEMS}
    return {
        'raw': raw,
        'cat_ids': cat_ids,
        'img_ids': img_ids,
        'params': params,
        'num_predictions': len(preds),
        'summarize_text': buffer.getvalue().strip(),
    }


def collect_side(label, run_dir):
    """Read one run's artifacts and recompute its raw COCO stats."""
    run_dir = Path(run_dir)
    paths = {
        'predictions': run_dir / 'predictions.bbox.json',
        'test_ann': run_dir / 'test.json',
        'metrics': run_dir / 'metrics.json',
        'metadata': run_dir / 'metadata.json',
        'resolved_config': run_dir / 'resolved_config.py',
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise SystemExit('%s (%s) missing files: %s'
                         % (label, run_dir, ', '.join(missing)))

    metadata = load_json(paths['metadata'])
    stored = load_json(paths['metrics'])

    side = {
        'label': label,
        'dir': str(run_dir),
        'sha256': {name: sha256_file(path) for name, path in paths.items()},
        'metadata': {
            'checkpoint': metadata.get('checkpoint'),
            'checkpoint_sha256': metadata.get('checkpoint_sha256'),
            'resolved_config_sha256': metadata.get('resolved_config_sha256'),
            'test_ann_sha256': metadata.get('test_ann_sha256'),
            'num_test_images': metadata.get('num_test_images'),
            'num_predictions': metadata.get('num_predictions'),
            'version': metadata.get('version'),
        },
    }

    checkpoint = metadata.get('checkpoint')
    if checkpoint and Path(checkpoint).exists():
        side['checkpoint_rehash'] = {
            'path': checkpoint,
            'sha256_now': sha256_file(checkpoint),
            'matches_metadata': sha256_file(checkpoint)
            == metadata.get('checkpoint_sha256'),
        }
    else:
        side['checkpoint_rehash'] = {
            'path': checkpoint,
            'sha256_now': None,
            'matches_metadata': None,
            'note': 'checkpoint file not present on disk; metadata value not '
                    're-verified for this side',
        }

    computed = compute_stats(paths['predictions'], paths['test_ann'],
                             paths['resolved_config'])
    side['raw'] = computed['raw']
    side['cat_ids'] = computed['cat_ids']
    side['img_ids'] = computed['img_ids']
    side['params'] = computed['params']
    side['num_predictions'] = computed['num_predictions']
    side['summarize_text'] = computed['summarize_text']

    rounded_ok = {}
    for item in METRIC_ITEMS:
        key = 'bbox_%s' % item
        if key not in stored:
            rounded_ok[key] = None
            continue
        rounded_ok[key] = {
            'stored': stored[key],
            'recomputed_rounded': mmdet_round(computed['raw'][item]),
            'match': stored[key] == mmdet_round(computed['raw'][item]),
        }
    side['rounding_vs_metrics_json'] = rounded_ok
    side['rounding_all_match'] = all(
        v is not None and v['match'] for v in rounded_ok.values())
    side['missing_metric_keys'] = sorted(
        k for k, v in rounded_ok.items() if v is None)

    side['checks'] = {
        'predictions_count_matches_metadata':
            computed['num_predictions'] == metadata.get('num_predictions'),
        'test_ann_sha256_matches_metadata':
            side['sha256']['test_ann'] == metadata.get('test_ann_sha256'),
        'test_ann_is_frozen_snapshot':
            side['sha256']['test_ann'] == FROZEN_TEST_ANN_SHA256,
        'num_test_images_is_frozen':
            metadata.get('num_test_images') == FROZEN_TEST_ANN_IMAGES,
        'num_images_seen_by_cocoeval':
            len(computed['img_ids']) == metadata.get('num_test_images'),
        'resolved_config_sha256_matches_metadata':
            side['sha256']['resolved_config']
            == metadata.get('resolved_config_sha256'),
        'rounding_all_match': side['rounding_all_match'],
    }
    return side


def compare_pair(pair):
    baseline = collect_side('%s:baseline' % pair['label'], pair['baseline_dir'])
    variant = collect_side('%s:variant' % pair['label'], pair['variant_dir'])

    shared = {
        'test_ann_identical_across_sides':
            baseline['sha256']['test_ann'] == variant['sha256']['test_ann'],
        'image_ids_identical_across_sides':
            baseline['img_ids'] == variant['img_ids'],
        'params_identical_across_sides':
            baseline['params'] == variant['params'],
        'checkpoints_differ':
            baseline['metadata']['checkpoint']
            != variant['metadata']['checkpoint'],
        'baseline_rounding_all_match': baseline['rounding_all_match'],
        'variant_rounding_all_match': variant['rounding_all_match'],
    }

    delta = {}
    for item in METRIC_ITEMS:
        delta[item] = variant['raw'][item] - baseline['raw'][item]
    delta_from_stored = {}
    for item in METRIC_ITEMS:
        key = 'bbox_%s' % item
        b = baseline['rounding_vs_metrics_json'].get(key)
        v = variant['rounding_vs_metrics_json'].get(key)
        if b and v and b['stored'] is not None and v['stored'] is not None:
            delta_from_stored[item] = v['stored'] - b['stored']
        else:
            delta_from_stored[item] = None

    return {
        'label': pair['label'],
        'baseline': baseline,
        'variant': variant,
        'shared_checks': shared,
        'consistent': all(shared.values()),
        'delta_raw': delta,
        'delta_from_stored_3dp': delta_from_stored,
        'primary': {item: {
            'baseline': baseline['raw'][item],
            'variant': variant['raw'][item],
            'delta': delta[item],
        } for item in PRIMARY_ITEMS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pair', nargs=3, action='append', required=True,
        metavar=('LABEL', 'BASELINE_DIR', 'VARIANT_DIR'),
        help='one paired comparison; repeat once per fold')
    parser.add_argument('--out', required=True,
                        help='output JSON path (parent dir must exist; refuses to '
                             'overwrite an existing file)')
    parser.add_argument('--expect-pairs', type=int, default=0,
                        help='if set, the run is only "complete" when exactly this '
                             'many pairs are supplied; with fewer pairs the per-fold '
                             'deltas are still reported but no equal-weight mean is '
                             'produced and the exit code is non-zero')
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise SystemExit('refusing to overwrite %s' % out_path)
    if not out_path.parent.exists():
        raise SystemExit('output parent directory missing: %s' % out_path.parent)

    pairs = [{'label': label, 'baseline_dir': b, 'variant_dir': v}
             for label, b, v in args.pair]

    if args.expect_pairs and len(pairs) > args.expect_pairs:
        raise SystemExit('got %d pairs but expected at most %d'
                         % (len(pairs), args.expect_pairs))
    partial = bool(args.expect_pairs) and len(pairs) < args.expect_pairs

    resolved_configs = []
    for pair in pairs:
        for key in ('baseline_dir', 'variant_dir'):
            resolved_configs.append(
                str(Path(pair[key]) / 'resolved_config.py'))

    provenance = check_provenance(resolved_configs)

    results = [compare_pair(pair) for pair in pairs]

    labels = [row['label'] for row in results]
    all_consistent = all(row['consistent'] for row in results)

    aggregate = None
    if all_consistent and not partial:
        aggregate = {
            'n_folds': len(results),
            'folds': labels,
            'weighting': 'equal weight over folds',
            'per_fold': {row['label']: {
                'mAP': row['delta_raw']['mAP'],
                'mAP_50': row['delta_raw']['mAP_50'],
                'mAP_75': row['delta_raw']['mAP_75'],
                'AR@100': row['delta_raw']['AR@100'],
            } for row in results},
        }
        for item in PRIMARY_ITEMS:
            key = 'mean_delta_%s' % item
            aggregate[key] = float(
                np.mean([row['delta_raw'][item] for row in results]))

    if not all_consistent:
        status = 'inconsistent'
    elif partial:
        status = 'incomplete'
    else:
        status = 'complete'

    payload = {
        'script': {'path': str(Path(__file__).resolve()),
                   'sha256': sha256_file(Path(__file__).resolve())},
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'unrounded paired COCO delta: both sides recomputed through the '
                   'same frozen COCOeval path; stored 3-decimal values are used only '
                   'as a consistency assertion',
        'delta_convention': 'delta = variant - baseline',
        'parameters': {
            'iou_type': IOU_TYPE,
            'proposal_nums': list(PROPOSAL_NUMS),
            'iou_thrs': [float(v) for v in IOU_THRS],
            'classwise': CLASSWISE,
            'use_cats': 'pycocotools default (True)',
            'area_rng': 'pycocotools default',
            'rec_thrs': 'pycocotools default',
            'stats_index': STATS_INDEX,
        },
        'environment': {
            'python': __import__('sys').version,
            'numpy': np.__version__,
            'pycocotools': _pycocotools_version(),
            'cuda_used': False,
        },
        'provenance': provenance,
        'pairs': results,
        'aggregate': aggregate,
        'expected_pairs': args.expect_pairs,
        'reporting_scope': ('per-fold deltas only; no equal-weight mean was produced '
                            'because fewer folds than expected were supplied'
                            if partial else 'all supplied folds'),
        'status': status,
    }

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write('\n')

    for row in results:
        print('[%s] consistent=%s' % (row['label'], row['consistent']))
        for item in PRIMARY_ITEMS:
            entry = row['primary'][item]
            print('   %-6s baseline=%.6f variant=%.6f delta=%+.6f  (3dp: %+.3f)'
                  % (item, entry['baseline'], entry['variant'], entry['delta'],
                     row['delta_from_stored_3dp'][item]
                     if row['delta_from_stored_3dp'][item] is not None
                     else float('nan')))
        if not row['consistent']:
            print('   failed checks: %s'
                  % [k for k, v in row['shared_checks'].items() if not v])
            print('   baseline checks: %s'
                  % [k for k, v in row['baseline']['checks'].items() if not v])
            print('   variant checks: %s'
                  % [k for k, v in row['variant']['checks'].items() if not v])
    if aggregate is not None:
        print('equal-weight mean delta over %d folds: mAP %+.6f  AP50 %+.6f  '
              'AP75 %+.6f  AR@100 %+.6f'
              % (aggregate['n_folds'], aggregate['mean_delta_mAP'],
                 aggregate['mean_delta_mAP_50'], aggregate['mean_delta_mAP_75'],
                 aggregate['mean_delta_AR@100']))
    else:
        print('no equal-weight mean produced (status=%s): only the per-fold deltas '
              'above are reported' % status)
    print('wrote %s (status=%s)' % (out_path, status))

    if status != 'complete':
        raise SystemExit(1)


def _pycocotools_version():
    try:
        import importlib.metadata as md
        return md.version('pycocotools')
    except Exception:
        try:
            import pycocotools
            return getattr(pycocotools, '__version__', 'unknown')
        except Exception:
            return 'unknown'


if __name__ == '__main__':
    main()
