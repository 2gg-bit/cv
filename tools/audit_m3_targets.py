"""Read-only M3 feasibility audit on the fixed 186-image SSDD dev split.

The GPU ``collect`` command saves every model output before reading ground truth
for analysis. ``analyze`` only needs NumPy. Neither command trains, evaluates AP,
selects a checkpoint, or authorizes a later training run. Run from the cv root.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEV_PATH = ROOT / "ssdd_dev_protocol/data/dev.json"
IMAGE_COUNT = 186
NOTICE = ("OFFLINE FEASIBILITY ONLY: this is not AP evaluation, not a training "
          "result, and does not authorize training. Test-scale weak views do not "
          "reproduce the full training augmentation or assignment process.")
BOX_KEYS = ("anchor_bbox", "teacher1_bbox", "teacher2_bbox", "selected_bbox")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def write_json(path, value):
    # JSON null marks nonfinite model diagnostics; never silently round floats.
    with open(path, "x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare_output_dir(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("output directory must be NEW (even an empty directory is refused): {}".format(path))
    path.mkdir(parents=True, exist_ok=False)
    return path


def validate_manifest(images, image_ids, expected_count=IMAGE_COUNT):
    expected = [item["id"] for item in images]
    actual = list(image_ids)
    if any(type(i) is not int for i in expected + actual):
        raise ValueError("image IDs must be integers")
    if len(expected) != expected_count or len(set(expected)) != expected_count:
        raise ValueError("dev annotation must contain {} unique images".format(expected_count))
    if len(actual) != expected_count or len(set(actual)) != expected_count:
        raise ValueError("candidate manifest must contain {} unique images".format(expected_count))
    if set(expected) != set(actual):
        raise ValueError("candidate image IDs differ from the dev split")


def validate_dev_path(path):
    if Path(path).resolve() != DEV_PATH.resolve():
        raise ValueError("only ssdd_dev_protocol/data/dev.json is allowed; official test is forbidden")


def validate_m3_config(model_cfg, min_anchor_iou=0.5):
    """Validate the predeclared experiment; never silently change its routing."""
    if model_cfg.get("type") != "DualTeacher" or model_cfg.get("train_cfg") is None:
        raise ValueError("audit requires a DualTeacher config with training thresholds retained")
    train_cfg = model_cfg["train_cfg"]
    if train_cfg.get("m3_enabled") is not True:
        raise ValueError("audit config must explicitly enable M3")
    if train_cfg.get("m3_target_mode") != "lower_uncertainty":
        raise ValueError("audit config must declare m3_target_mode='lower_uncertainty'")
    configured_iou = train_cfg.get("m3_min_anchor_iou")
    if (not isinstance(configured_iou, (int, float)) or isinstance(configured_iou, bool)
            or not math.isfinite(configured_iou) or configured_iou != 0.5):
        raise ValueError("audit config m3_min_anchor_iou is fixed to 0.5")
    if not math.isfinite(min_anchor_iou) or min_anchor_iou != 0.5:
        raise ValueError("audit --min-anchor-iou is fixed to 0.5; threshold sweeps are forbidden")


def validate_pipeline(pipeline):
    allowed = {"LoadImageFromFile", "MultiScaleFlipAug", "Resize", "RandomFlip",
               "Normalize", "Pad", "ImageToTensor", "DefaultFormatBundle", "Collect"}
    for step in pipeline:
        kind = step.get("type")
        if kind not in allowed:
            raise ValueError("unsupported test pipeline transform: {}".format(kind))
        if kind == "Collect" and step.get("keys") != ["img"]:
            raise ValueError("test pipeline must collect img only, never GT")
        if kind == "MultiScaleFlipAug":
            if step.get("flip", False):
                raise ValueError("test flip/TTA is forbidden")
            scale = step.get("img_scale")
            if isinstance(scale, list) and scale and isinstance(scale[0], (tuple, list)):
                if len(scale) != 1:
                    raise ValueError("multi-scale test/TTA is forbidden")
            if isinstance(step.get("scale_factor"), list) and len(step["scale_factor"]) != 1:
                raise ValueError("multi-scale test/TTA is forbidden")
            validate_pipeline(step["transforms"])


def ensure_transform(meta):
    """Return original->input transform; fixed test pipeline must never flip."""
    if meta.get("flip", False):
        raise ValueError("unexpected flipped test view")
    if "transform_matrix" in meta:
        matrix = np.asarray(meta["transform_matrix"], dtype=np.float64)
    else:
        scale = np.asarray(meta.get("scale_factor"), dtype=np.float64).reshape(-1)
        if scale.size == 1:
            sx = sy = scale[0]
        elif scale.size == 4 and scale[0] == scale[2] and scale[1] == scale[3]:
            sx, sy = scale[:2]
        else:
            raise ValueError("missing/unsupported scale_factor")
        matrix = np.diag([sx, sy, 1.0])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("invalid transform_matrix")
    # The accepted pipeline only resizes. Refuse an unreported crop/rotation.
    if not np.allclose(matrix, np.diag(np.diag(matrix)), rtol=0, atol=1e-10):
        raise ValueError("audit test transform must be pure scale")
    if np.any(np.diag(matrix) <= 0) or matrix[2, 2] != 1:
        raise ValueError("audit test transform must be positive scale")
    return matrix


def original_boxes(boxes, matrix):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    inv = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    if not len(boxes):
        return boxes.copy()
    # Transform all corners uniformly without clipping or geometry repair.
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]],
                        boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
    points = np.concatenate([corners, np.ones(corners.shape[:2] + (1,))], axis=2)
    mapped = points @ inv.T
    mapped = mapped[..., :2] / mapped[..., 2:]
    result = np.concatenate([mapped.min(axis=1), mapped.max(axis=1)], axis=1)
    # Preserve invalid/reversed input geometry so analysis cannot make it valid.
    invalid = (boxes[:, 2:] <= boxes[:, :2]).any(axis=1) | ~np.isfinite(boxes).all(axis=1)
    result[invalid] = boxes[invalid] / np.array([matrix[0, 0], matrix[1, 1]] * 2)
    return result


def pairwise_iou(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    valid_a = np.isfinite(first).all(axis=1) & (first[:, 2:] > first[:, :2]).all(axis=1)
    valid_b = np.isfinite(second).all(axis=1) & (second[:, 2:] > second[:, :2]).all(axis=1)
    a, b = np.nan_to_num(first), np.nan_to_num(second)
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    result = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    result[~valid_a, :] = 0
    result[:, ~valid_b] = 0
    return result


def random_source(image_id, anchor_index, seed, valid1, valid2):
    """No Python/NumPy/Torch global RNG consumption; stable across image order."""
    options = [source for source, valid in ((1, valid1), (2, valid2)) if valid]
    if not options:
        return 0
    digest = hashlib.sha256("m3:{}:{}:{}".format(seed, image_id, anchor_index).encode()).digest()
    return options[int.from_bytes(digest[:8], "big") % len(options)]


def safe_list(array):
    """Nonfinite model values become explicit null diagnostics, not invalid JSON."""
    array = np.asarray(array)
    return [safe_list(row) for row in array] if array.ndim else (
        float(array) if np.isfinite(array) else None)


def _box(row, name):
    value = np.asarray(row[name], dtype=np.float64)
    if value.shape != (4,):
        raise ValueError("{} must contain four coordinates".format(name))
    return value


def validate_candidates(payload, images, expected_count=IMAGE_COUNT):
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported candidates schema")
    rows = payload["images"]
    validate_manifest(images, [row["image_id"] for row in rows], expected_count)
    names = {item["id"]: item.get("file_name") for item in images}
    for image in rows:
        if image.get("file_name") != names[image["image_id"]]:
            raise ValueError("image filename differs from dev manifest")
        for index, row in enumerate(image["candidates"]):
            if row["anchor_index"] != index or type(row["source_id"]) is not int or row["source_id"] not in (0, 1, 2):
                raise ValueError("invalid anchor order or source ID")
            for name in BOX_KEYS:
                _box(row, name)
            for name in ("baseline_uncertainty", "teacher1_uncertainty", "teacher2_uncertainty"):
                if np.asarray(row[name]).shape != (4,):
                    raise ValueError("uncertainty must contain four coordinates")
            anchor, selected = _box(row, "anchor_bbox"), _box(row, "selected_bbox")
            if not np.isfinite(anchor).all() or not np.isfinite(selected).all():
                raise ValueError("anchor and selected boxes must be finite")
            source = row["source_id"]
            expected = anchor if source == 0 else _box(row, "teacher{}_bbox".format(source))
            if not np.array_equal(expected, selected):
                raise ValueError("selected box does not match its declared source")
            if source and not row["teacher{}_valid".format(source)]:
                raise ValueError("selected source is invalid")
            for key in ("teacher1_valid", "teacher2_valid", "branch1_original_reg_eligible", "branch2_original_reg_eligible"):
                if type(row[key]) is not bool:
                    raise ValueError("eligibility and validity flags must be boolean")
            if row["branch1_original_reg_eligible"] != row["branch2_original_reg_eligible"]:
                raise ValueError("original regression branches must share fused uncertainty eligibility")
            if not math.isfinite(row["pseudo_score"]):
                raise ValueError("pseudo score must be finite")


def _eligible(uncertainty, threshold):
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    return bool(np.isfinite(uncertainty).all() and uncertainty.mean() < threshold)


def _method_stats(records, method, total_gt):
    values = np.array([r["ious"][method] for r in records], dtype=np.float64)
    baseline = np.array([r["ious"]["baseline"] for r in records], dtype=np.float64)
    delta = values - baseline
    thresholds = {}
    for threshold in (0.5, 0.75, 0.85):
        covered = {(r["image_id"], r["gt_id"]) for r in records if r["ious"][method] >= threshold}
        thresholds[str(threshold)] = dict(
            candidates_at_or_above=int((values >= threshold).sum()),
            candidate_fraction=float((values >= threshold).mean()) if len(values) else None,
            unique_gt_covered=len(covered), total_dev_gt=total_gt,
            gt_coverage_fraction=len(covered) / total_gt if total_gt else None)
    return dict(count=len(values), mean_iou=float(values.mean()) if len(values) else None,
                mean_iou_delta=float(delta.mean()) if len(delta) else None,
                median_iou_delta=float(np.median(delta)) if len(delta) else None,
                improved=int((delta > 1e-12).sum()), degraded=int((delta < -1e-12).sum()),
                unchanged=int((np.abs(delta) <= 1e-12).sum()), thresholds=thresholds)


def analyze_candidates(payload, annotation, seed=678, reg_threshold=0.02,
                       expected_count=IMAGE_COUNT):
    """GT is used only here, after immutable candidate collection has completed."""
    validate_candidates(payload, annotation["images"], expected_count)
    gt = {image["id"]: [] for image in annotation["images"]}
    excluded = 0
    for item in annotation.get("annotations", []):
        if item["image_id"] not in gt:
            raise ValueError("GT refers to an image outside the fixed dev split")
        x, y, width, height = item["bbox"]
        if item.get("iscrowd", 0) or item.get("ignore", 0) or width <= 0 or height <= 0:
            excluded += 1
            continue
        if not np.isfinite([x, y, width, height]).all():
            raise ValueError("nonfinite GT")
        gt[item["image_id"]].append((item["id"], [x, y, x + width, y + height]))
    total_gt = sum(len(rows) for rows in gt.values())
    diagnostics, source_counts = [], {"0": 0, "1": 0, "2": 0}
    total_candidates = 0
    for image in payload["images"]:
        ground_truth = gt[image["image_id"]]
        gt_boxes = [row[1] for row in ground_truth]
        for row in image["candidates"]:
            total_candidates += 1
            source_counts[str(row["source_id"])] += 1
            if not ground_truth:
                continue
            # Associate once using the anchor; never rematch an alternative box.
            overlaps = pairwise_iou([row["anchor_bbox"]], gt_boxes)[0]
            associated = int(np.argmax(overlaps))
            associated_box = [gt_boxes[associated]]
            boxes = {"baseline": row["anchor_bbox"], "teacher1": row["teacher1_bbox"],
                     "teacher2": row["teacher2_bbox"], "selected": row["selected_bbox"]}
            sampled = random_source(image["image_id"], row["anchor_index"], seed,
                                    row["teacher1_valid"], row["teacher2_valid"])
            boxes["random_valid_teacher"] = row["anchor_bbox"] if sampled == 0 else row["teacher{}_bbox".format(sampled)]
            ious = {key: float(pairwise_iou([box], associated_box)[0, 0]) for key, box in boxes.items()}
            # This oracle is diagnostic only, constrained to runtime-valid boxes.
            oracle_options = [("baseline", 0)] + [("teacher{}".format(k), k) for k in (1, 2) if row["teacher{}_valid".format(k)]]
            oracle_method, oracle_source = max(oracle_options, key=lambda entry: ious[entry[0]])
            ious["oracle_diagnostic_only"] = ious[oracle_method]
            # Preserve the exact runtime comparison/dtype at the threshold.
            baseline_eligible = row["branch1_original_reg_eligible"]
            diagnostics.append(dict(image_id=image["image_id"], anchor_index=row["anchor_index"],
                                    gt_id=ground_truth[associated][0], anchor_gt_iou=float(overlaps[associated]),
                                    source_id=row["source_id"], random_source_id=sampled,
                                    oracle_diagnostic_source_id=oracle_source, ious=ious,
                                    branch1_original_reg_eligible=baseline_eligible,
                                    branch2_original_reg_eligible=baseline_eligible,
                                    teacher1_uncertainty_eligible=_eligible(row["teacher1_uncertainty"], reg_threshold),
                                    teacher2_uncertainty_eligible=_eligible(row["teacher2_uncertainty"], reg_threshold)))
    methods = ("baseline", "teacher1", "teacher2", "selected", "random_valid_teacher", "oracle_diagnostic_only")
    cohorts = {"all_anchors_with_gt": diagnostics,
               "anchor_gt_iou_ge_0.5": [r for r in diagnostics if r["anchor_gt_iou"] >= .5]}
    for branch in (1, 2):
        cohorts["branch{}_original_reg_eligible".format(branch)] = [
            r for r in diagnostics if r["branch{}_original_reg_eligible".format(branch)]]
        cohorts["branch{}_original_reg_eligible_anchor_gt_iou_ge_0.5".format(branch)] = [
            r for r in diagnostics if r["branch{}_original_reg_eligible".format(branch)] and r["anchor_gt_iou"] >= .5]
    report = dict(notice=NOTICE, schema_version=1, num_images=len(payload["images"]),
                  image_ids=[image["image_id"] for image in payload["images"]],
                  num_empty_images=sum(not image["candidates"] for image in payload["images"]),
                  total_candidates=total_candidates, candidates_without_gt=total_candidates - len(diagnostics),
                  num_gt=total_gt, excluded_crowd_ignore_or_degenerate_gt=excluded,
                  source_counts=source_counts, seed=seed, reg_pseudo_threshold=reg_threshold,
                  association_rule="max IoU(anchor, GT), fixed for all alternatives; GT ties use annotation order",
                  original_eligibility_rule="mean((teacher1_unc + teacher2_unc) / 2) < reg_pseudo_threshold; same for both original branches",
                  coverage_rule="unique anchor-associated GT with candidate IoU >= threshold / all usable dev GT; not COCO recall/AP",
                  teacher_candidate_rule="raw teacher means retained even if invalid; random and oracle only use runtime-valid teachers",
                  oracle_rule="diagnostic upper bound over baseline and valid teachers; never a selector",
                  random_rule="SHA256(m3:seed:image_id:anchor_index), uniform choice among runtime-valid teachers; baseline if none",
                  cohorts={name: {method: _method_stats(rows, method, total_gt) for method in methods}
                           for name, rows in cohorts.items()})
    return report, diagnostics


def load_collection(directory):
    directory = Path(directory)
    metadata = read_json(directory / "metadata.json")
    if metadata.get("collection_complete") is not True:
        raise ValueError("candidate collection is incomplete")
    for name, field in (("candidates.json", "candidates_sha256"), ("dev.json", "dev_sha256"),
                        ("resolved_config.py", "resolved_config_sha256")):
        if sha256_file(directory / name) != metadata[field]:
            raise ValueError("{} SHA256 mismatch".format(name))
    if metadata.get("fixed_dev_relative_path") != "ssdd_dev_protocol/data/dev.json":
        raise ValueError("collection was not made on the fixed dev split")
    payload, annotation = read_json(directory / "candidates.json"), read_json(directory / "dev.json")
    validate_candidates(payload, annotation["images"])
    if metadata["image_ids"] != [row["image_id"] for row in payload["images"]]:
        raise ValueError("metadata/candidate image order mismatch")
    return payload, annotation, metadata


def run_analysis(input_dir, output_dir):
    payload, annotation, metadata = load_collection(input_dir)
    report, diagnostics = analyze_candidates(payload, annotation, metadata["seed"], metadata["reg_pseudo_threshold"])
    report["collection_candidates_sha256"] = metadata["candidates_sha256"]
    report["collection_metadata_sha256"] = sha256_file(Path(input_dir) / "metadata.json")
    write_json(Path(output_dir) / "analysis.json", report)
    write_json(Path(output_dir) / "gt_diagnostics.json", diagnostics)
    return report


def _source_hashes():
    paths = sorted((ROOT / "ssod").rglob("*.py")) + [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def _rng_fingerprints(torch):
    return dict(torch_cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                torch_cuda_sha256=[hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                                   for state in torch.cuda.get_rng_state_all()])


def collect(args):
    # Lazy imports keep --help and CPU analysis independent of the CUDA stack.
    import mmcv
    import mmdet
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from ssod.utils import patch_config
    from ssod.models.m3_routing import select_regression_targets

    if not torch.cuda.is_available():
        raise ValueError("collect requires the existing CUDA/MMCV training environment")
    config_path, checkpoint_path = Path(args.config).resolve(), Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise ValueError("checkpoint must be a local full DualTeacher checkpoint; downloads are forbidden")
    cfg = Config.fromfile(str(config_path))
    cfg.merge_from_dict(dict(fold=args.fold, percent=3))
    cfg = patch_config(cfg)
    validate_m3_config(cfg.model, args.min_anchor_iou)
    if isinstance(cfg.data.test, list):
        raise ValueError("concatenated test datasets are forbidden")
    validate_dev_path(cfg.data.test.ann_file)
    validate_pipeline(cfg.data.test.pipeline)
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", None)
    # Disable initialization URLs; strict full-state loading supplies every tensor.
    def no_pretrain(value):
        if isinstance(value, dict):
            if "pretrained" in value:
                value["pretrained"] = None
            if isinstance(value.get("init_cfg"), dict) and value["init_cfg"].get("type") == "Pretrained":
                value["init_cfg"] = None
            for child in value.values():
                no_pretrain(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                no_pretrain(child)
    no_pretrain(cfg.model)
    cfg.model.train_cfg.pop("load1_from", None)
    cfg.model.train_cfg.pop("load2_from", None)
    cfg.auto_resume, cfg.resume_from, cfg.load_from = False, None, None
    out = prepare_output_dir(args.out_dir)
    cfg.dump(str(out / "resolved_config.py"))
    checkpoint_hash = sha256_file(checkpoint_path)
    dev_hash = sha256_file(DEV_PATH)
    # Only the image manifest is consumed before inference. GT is analyzed later.
    manifest = read_json(DEV_PATH)["images"]
    dataset = build_dataset(cfg.data.test)
    image_ids = [int(item) for item in dataset.img_ids]
    validate_manifest(manifest, image_ids)
    if len(dataset) != IMAGE_COUNT:
        raise ValueError("dataset length differs from the fixed 186-image dev split")
    files = {item["id"]: item["file_name"] for item in manifest}
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                                   dist=False, shuffle=False, seed=args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=True)
    model.CLASSES = dataset.CLASSES
    model.eval()
    modelx = MMDataParallel(model.cuda(0), device_ids=[0])
    modelx.eval()
    if any(module.training for module in model.modules()):
        raise ValueError("all model modules must remain in eval mode")
    parameter_versions = {name: tensor._version for name, tensor in model.named_parameters()}
    buffer_hash = lambda: {name: hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
                           for name, tensor in model.named_buffers()}
    before_buffers = buffer_hash()
    dtype_counts = {}
    for parameter in model.parameters():
        dtype_counts[str(parameter.dtype)] = dtype_counts.get(str(parameter.dtype), 0) + parameter.numel()
    before_rng = _rng_fingerprints(torch)
    images = []
    reg_threshold = float(cfg.model.train_cfg.reg_pseudo_threshold)
    def array(tensor):
        return tensor.detach().cpu().numpy()
    with torch.no_grad():
        for position, data in enumerate(data_loader):
            if set(data) != {"img", "img_metas"}:
                raise ValueError("model input must contain only img and img_metas")
            # Use old MMCV DataContainer scatter, exactly as MMDataParallel does.
            _, scattered = modelx.scatter((), data, modelx.device_ids)
            batch = scattered[0]
            img, metas = batch["img"], batch["img_metas"]
            if isinstance(img, list):
                if len(img) != 1 or len(metas) != 1:
                    raise ValueError("one test view is required")
                img, metas = img[0], metas[0]
            if img.shape[0] != 1 or len(metas) != 1:
                raise ValueError("audit requires a one-image batch")
            meta = metas[0]
            if any(key.startswith("gt_") for key in meta):
                raise ValueError("GT must not occur in model metadata")
            matrix = ensure_transform(meta)
            meta["transform_matrix"] = matrix.astype(np.float32)
            # extract_teacher_info bypasses forward's auto_fp16 decorator.
            if cfg.get("fp16") is not None:
                img = img.half()
            first, second = modelx.module.extract_teacher_info(img, metas)
            anchors = first["det_bboxes"][0]
            box1, box2 = first["teacher1_reg_boxes"][0], first["teacher2_reg_boxes"][0]
            unc1, unc2 = first["teacher1_reg_unc"][0], first["teacher2_reg_unc"][0]
            selected, sources = select_regression_targets(anchors[:, :4], box1, box2, unc1, unc2,
                                                         min_anchor_iou=args.min_anchor_iou, mode="lower_uncertainty")
            if not torch.equal(selected, first["reg_target_bboxes"][0]) or not torch.equal(sources, first["m3_source"][0]):
                raise ValueError("extractor M3 outputs disagree with the shared runtime selector")
            _, valid1 = select_regression_targets(anchors[:, :4], box1, box2, unc1, unc2,
                                                  min_anchor_iou=args.min_anchor_iou, mode="teacher1")
            _, valid2 = select_regression_targets(anchors[:, :4], box1, box2, unc1, unc2,
                                                  min_anchor_iou=args.min_anchor_iou, mode="teacher2")
            pixel_boxes = {key: original_boxes(array(value), matrix) for key, value in
                           (("anchor_bbox", anchors[:, :4]), ("teacher1_bbox", box1),
                            ("teacher2_bbox", box2), ("selected_bbox", selected))}
            source_values, valid1_values, valid2_values = array(sources), array(valid1), array(valid2)
            anchor_values, unc1_values, unc2_values = array(anchors), array(unc1), array(unc2)
            original_eligible = array(-anchors[:, 5:9].mean(dim=-1) > -reg_threshold)
            candidates = []
            for index in range(len(anchors)):
                row = {key: safe_list(value[index]) for key, value in pixel_boxes.items()}
                row.update(anchor_index=index, pseudo_score=float(anchor_values[index, 4]),
                           baseline_uncertainty=safe_list(anchor_values[index, 5:9]),
                           teacher1_uncertainty=safe_list(unc1_values[index]),
                           teacher2_uncertainty=safe_list(unc2_values[index]),
                           source_id=int(source_values[index]), teacher1_valid=bool(valid1_values[index] == 1),
                           teacher2_valid=bool(valid2_values[index] == 2),
                           branch1_original_reg_eligible=bool(original_eligible[index]),
                           branch2_original_reg_eligible=bool(original_eligible[index]))
                candidates.append(row)
            raw = {}
            for teacher, info in (("teacher1", first), ("teacher2", second)):
                raw_boxes = array(info["raw_det_bboxes"][0])
                raw[teacher] = dict(boxes_xyxy=safe_list(original_boxes(raw_boxes[:, :4], matrix)),
                                    scores=safe_list(raw_boxes[:, 4]), labels=array(info["raw_det_labels"][0]).tolist())
            image_id = image_ids[position]
            actual_name = meta.get("ori_filename", meta.get("filename", ""))
            if Path(actual_name).name != Path(files[image_id]).name:
                raise ValueError("loader image order/filename differs from the manifest")
            images.append(dict(image_id=image_id, file_name=files[image_id], candidates=candidates,
                               raw_teacher_detections=raw, original_to_input_matrix=matrix.tolist(),
                               img_shape=list(meta["img_shape"]), ori_shape=list(meta["ori_shape"]),
                               input_dtype=str(img.dtype)))
            print("collected {}/{} image_id={} candidates={}".format(position + 1, IMAGE_COUNT, image_id, len(candidates)), flush=True)
    after_rng = _rng_fingerprints(torch)
    if before_buffers != buffer_hash() or parameter_versions != {name: tensor._version for name, tensor in model.named_parameters()}:
        raise ValueError("model state changed during read-only inference")
    payload = dict(schema_version=1, coordinate_system="original image pixels xyxy; no clipping or rounding",
                   raw_teacher_scores_note="teacher detection scores belong to their separate raw detection sets, not to aligned jitter means",
                   uncertainty_note="normalized jitter uncertainty is dimensionless and is not coordinate-transformed; nonfinite diagnostics use null",
                   images=images)
    validate_candidates(payload, manifest)
    write_json(out / "candidates.json", payload)
    # All model candidates are sealed before the annotation snapshot or GT analysis.
    if sha256_file(DEV_PATH) != dev_hash or sha256_file(checkpoint_path) != checkpoint_hash:
        raise ValueError("dev file or checkpoint changed during collection")
    shutil.copyfile(DEV_PATH, out / "dev.json")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, universal_newlines=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, universal_newlines=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = dict(collection_complete=True, notice=NOTICE, fold=args.fold, percent=3, seed=args.seed,
                    image_ids=image_ids, num_images=len(image_ids), total_candidates=sum(len(i["candidates"]) for i in images),
                    fixed_dev_relative_path="ssdd_dev_protocol/data/dev.json", dev_sha256=dev_hash,
                    checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                    config=str(config_path), config_sha256=sha256_file(config_path),
                    resolved_config_sha256=sha256_file(out / "resolved_config.py"),
                    candidates_sha256=sha256_file(out / "candidates.json"), source_sha256=_source_hashes(),
                    git_revision=revision, git_dirty=dirty, model_parameter_dtype_counts=dtype_counts,
                    software=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                  mmcv=mmcv.__version__, mmdet=mmdet.__version__, cuda=torch.version.cuda),
                    gpu=torch.cuda.get_device_name(0), fp16=cfg.get("fp16"), no_grad=True, all_modules_eval=True,
                    strict_full_checkpoint_load=True, parameters_version_unchanged=True, buffers_sha256_unchanged=True,
                    rng_before_teacher_extraction=before_rng, rng_after_teacher_extraction=after_rng,
                    rng_note="model construction consumes seeded RNG before extraction; jitter consumes Torch RNG; random analysis does not",
                    cudnn_benchmark=False, cudnn_deterministic=True, bitwise_cross_device_reproducibility_guaranteed=False,
                    reg_pseudo_threshold=reg_threshold, min_anchor_iou=args.min_anchor_iou,
                    jitter_times=cfg.model.train_cfg.jitter_times, jitter_scale=cfg.model.train_cfg.jitter_scale,
                    pipeline=cfg.data.test.pipeline, source_id_map={"0": "original anchor", "1": "teacher1 jitter mean", "2": "teacher2 jitter mean"})
    write_json(out / "metadata.json", metadata)
    return run_analysis(out, out)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=NOTICE)
    # add_subparsers(required=...) was introduced in Python 3.7.
    commands = parser.add_subparsers(dest="command")
    collector = commands.add_parser("collect", help="collect GPU teacher candidates, then separately analyze fixed dev GT")
    collector.add_argument("config", help="DualTeacher dev config; original train_cfg is retained")
    collector.add_argument("checkpoint", help="local full-model checkpoint, loaded strict=True")
    collector.add_argument("--fold", type=int, required=True, choices=(6, 7, 8))
    collector.add_argument("--out-dir", required=True, help="new directory; any existing path is refused")
    collector.add_argument("--seed", type=int, default=678)
    collector.add_argument("--min-anchor-iou", type=float, default=.5, choices=(.5,),
                           help="fixed candidate guard: only 0.5 is permitted; no threshold sweep")
    analyzer = commands.add_parser("analyze", help="CPU/NumPy reanalysis of an intact completed collection")
    analyzer.add_argument("collection_dir")
    analyzer.add_argument("--out-dir", required=True, help="new analysis directory; collection is read-only")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required: collect or analyze")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(NOTICE, flush=True)
    if args.command == "collect":
        report = collect(args)
    else:
        out = prepare_output_dir(args.out_dir)
        report = run_analysis(args.collection_dir, out)
    print("Completed {} images, {} candidates. {}".format(report["num_images"], report["total_candidates"], NOTICE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
