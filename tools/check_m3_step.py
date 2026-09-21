"""Bounded M3 GPU acceptance on one real training-pipeline batch, no updates.

Run from the repository root with PYTHONPATH=.; requires the pinned MMDetection
stack, CUDA, the existing training split, and trusted local checkpoints. Default
execution includes backward; --forward-only is an explicitly weaker check.
Exit codes: 0 passed, 1 failed, 2 inconclusive (no sampled target changed).
"""

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import traceback
import types

import numpy as np
import torch
from mmcv import Config, DictAction


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--fold", type=int, default=6)
    parser.add_argument("--seed", type=int, default=678)
    parser.add_argument("--out-dir", required=True, help="New directory; never reused")
    parser.add_argument("--checkpoint", help="Full four-branch M2 checkpoint, strict load")
    parser.add_argument("--fp32", action="store_true", help="Diagnostic override only")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--batch-index", type=int, default=0,
                        help="Zero-based fixed training-loader batch; no AP/GT selection")
    parser.add_argument("--loss-atol", type=float, default=1e-6)
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--grad-atol", type=float, default=1e-6)
    parser.add_argument("--grad-rtol", type=float, default=1e-4)
    parser.add_argument("--loss-scale", type=float, default=1.0,
                        help="Fixed backward scale; gradients are divided by it")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    if args.batch_index < 0 or args.fold not in (6, 7, 8):
        parser.error("Use a nonnegative batch index and an existing dev fold (6/7/8)")
    for name in ("loss_atol", "loss_rtol", "grad_atol", "grad_rtol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error("Tolerances must be finite and nonnegative")
    if not math.isfinite(args.loss_scale) or args.loss_scale <= 0:
        parser.error("--loss-scale must be finite and positive")
    return args


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def clear_grads(model):
    # Explicit None works on the pinned old PyTorch stack and distinguishes an
    # absent gradient from an existing zero tensor (especially for teachers).
    for parameter in model.parameters():
        parameter.grad = None


def strict_full_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("Full checkpoint has no state_dict")
    if all(key.startswith("module.") for key in state):
        state = OrderedDict((key[7:], value) for key, value in state.items())
    if not all(any(key.startswith(branch + ".") for key in state)
               for branch in ("teacher1", "teacher2", "student1", "student2")):
        raise ValueError("--checkpoint requires all four branches")
    if not all(torch.is_tensor(value) and bool(torch.isfinite(value).all())
               for value in state.values()):
        raise ValueError("Checkpoint contains non-tensor or nonfinite state")
    model.load_state_dict(state, strict=True)
    for key, actual in model.state_dict().items():
        expected = state[key].to(dtype=actual.dtype)
        if not bool(torch.isfinite(actual).all()) or not torch.equal(actual.cpu(), expected):
            raise RuntimeError("Strict checkpoint verification failed: " + key)


def flatten_tensors(value, prefix=""):
    if torch.is_tensor(value):
        return {prefix: value.detach().float().cpu().clone()}
    if isinstance(value, dict):
        output = {}
        for key in sorted(value):
            output.update(flatten_tensors(value[key], prefix + key))
        return output
    if isinstance(value, (list, tuple)):
        output = {}
        for index, item in enumerate(value):
            output.update(flatten_tensors(item, "{}[{}]".format(prefix, index)))
        return output
    raise TypeError("Unexpected non-tensor loss entry: " + prefix)


def compare_tensors(first, second, atol, rtol, exclude=()):
    keys = sorted(set(first) | set(second))
    details = {}
    for key in keys:
        if key.split("[")[0] in exclude:
            continue
        if key not in first or key not in second or first[key].shape != second[key].shape:
            details[key] = dict(close=False, reason="missing key or shape mismatch")
            continue
        left, right = first[key], second[key]
        difference = (left.double() - right.double()).abs()
        denominator = torch.max(left.double().abs(), right.double().abs()).clamp(min=1e-12)
        details[key] = dict(
            close=bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
            max_abs=float(difference.max()) if difference.numel() else 0.0,
            max_relative=float((difference / denominator).max()) if difference.numel() else 0.0)
    return dict(passed=all(item["close"] for item in details.values()),
                atol=atol, rtol=rtol, tensors=details)


def install_sampling_observers(model, events):
    """Observe real sampler output before the actual M3 implementation uses it."""
    for branch in ("student1", "student2"):
        head = getattr(model, branch).roi_head
        original = head._bbox_forward_train

        def observe(self, x, samples, gt_bboxes, gt_labels, img_metas,
                    _original=original, _branch=branch):
            targets = self._m3_targets
            event = dict(branch=_branch, routed=targets is not None, images=[])
            for index, sample in enumerate(samples):
                changed = 0
                if targets is not None:
                    if targets[index].requires_grad:
                        raise AssertionError("M3 targets must be detached")
                    chosen = targets[index][sample.pos_assigned_gt_inds]
                    changed = int((chosen != sample.pos_gt_bboxes).any(dim=1).sum())
                event["images"].append(dict(
                    positives=len(sample.pos_bboxes), negatives=len(sample.neg_bboxes),
                    changed_positive_targets=changed,
                    original_gt_sha256=tensor_hash(gt_bboxes[index]),
                    labels_sha256=tensor_hash(gt_labels[index]),
                    positives_sha256=tensor_hash(sample.pos_bboxes),
                    negatives_sha256=tensor_hash(sample.neg_bboxes),
                    assigned_gt_sha256=tensor_hash(sample.pos_assigned_gt_inds)))
            events.append(event)
            bbox_head = self.bbox_head
            get_targets, loss = bbox_head.get_targets, bbox_head.loss
            original_targets = []

            def record_targets(*args, **kwargs):
                result = get_targets(*args, **kwargs)
                original_targets[:] = [value.detach().clone() for value in result]
                return result

            def record_loss(cls_score, bbox_pred, rois, *target_args, **kwargs):
                if len(target_args) != 4 or len(original_targets) != 4:
                    raise AssertionError("Unexpected bbox target call signature")
                changed = (target_args[2] != original_targets[2]).any(dim=1)
                negatives = ~original_targets[3].bool().any(dim=1)
                event["actual_bbox_target_rows_changed"] = int(changed.sum())
                event["labels_and_weights_unchanged"] = all(
                    torch.equal(target_args[index], original_targets[index]) for index in (0, 1, 3))
                event["negative_targets_unchanged"] = not bool((changed & negatives).any())
                if not event["labels_and_weights_unchanged"] or not event["negative_targets_unchanged"]:
                    raise AssertionError("M3 changed classification/weights/negative targets")
                return loss(cls_score, bbox_pred, rois, *target_args, **kwargs)

            bbox_head.get_targets, bbox_head.loss = record_targets, record_loss
            try:
                return _original(x, samples, gt_bboxes, gt_labels, img_metas)
            finally:
                bbox_head.get_targets, bbox_head.loss = get_targets, loss

        head._bbox_forward_train = types.MethodType(observe, head)


def sampling_signature(events):
    return [dict(branch=event["branch"], images=[
        {key: value for key, value in image.items() if key != "changed_positive_targets"}
        for image in event["images"]]) for event in events]


def run_condition(model, batch, mode, args, use_fp16, state, rng, events):
    from mmcv.parallel import scatter

    model.load_state_dict(state, strict=True)
    clear_grads(model)
    model.train()
    model.freeze("teacher1")
    model.freeze("teacher2")
    model.m3_enabled = mode != "off"
    model.m3_target_mode = "original" if mode == "original" else "lower_uncertainty"
    restore_rng(rng)
    events[:] = []
    inputs = scatter(copy.deepcopy(batch), [torch.cuda.current_device()])[0]
    # Use real MMDetection auto_fp16 wrappers and the same autocast facility as
    # modern MMCV's FP16 hook. No optimizer/scaler step, clipping, or EMA runs.
    with torch.set_grad_enabled(not args.forward_only):
        with torch.cuda.amp.autocast(enabled=use_fp16):
            losses = model(return_loss=True, **inputs)
            total_loss, _ = model._parse_losses(losses)
        loss_tensors = flatten_tensors(losses)
        if not all(bool(torch.isfinite(value).all()) for value in loss_tensors.values()):
            raise AssertionError("Nonfinite losses in " + mode)
        if not args.forward_only:
            (total_loss * args.loss_scale).backward()
    gradients, norms, missing = {}, {}, []
    teacher_gradients = []
    for name, parameter in model.named_parameters():
        if name.startswith(("teacher1.", "teacher2.")):
            if parameter.requires_grad or parameter.grad is not None:
                teacher_gradients.append(name)
            continue
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float().cpu() / args.loss_scale
            if not bool(torch.isfinite(gradient).all()):
                raise AssertionError("Nonfinite gradient in {}: {}".format(mode, name))
            gradients[name] = gradient
            norms[name] = float(gradient.double().norm())
        elif parameter.requires_grad:
            missing.append(name)
    if teacher_gradients:
        raise AssertionError("Teacher gradient/freeze failure: " + repr(teacher_gradients))
    if not args.forward_only:
        for branch in ("student1.", "student2."):
            if not any(value > 0 for name, value in norms.items() if name.startswith(branch)):
                raise AssertionError("Backward produced no nonzero gradients for " + branch)
    condition = dict(
        losses={key: value.tolist() for key, value in loss_tensors.items()},
        total_loss=float(total_loss.detach()), student_gradient_norms=norms,
        student_parameters_without_gradient=missing, teacher_gradients=teacher_gradients,
        sampling=copy.deepcopy(events),
        changed_positive_targets=sum(image["changed_positive_targets"]
                                     for event in events for image in event["images"]),
        actual_bbox_target_rows_changed=sum(event["actual_bbox_target_rows_changed"]
                                            for event in events))
    del losses, total_loss, inputs
    clear_grads(model)
    torch.cuda.empty_cache()
    return condition, loss_tensors, gradients


def execute(args, out_dir, report):
    import mmcv
    import mmdet
    from mmcv.runner import wrap_fp16_model
    from mmcv.utils import import_modules_from_strings
    from mmdet.models import build_detector
    from ssod.apis import set_random_seed
    from ssod.datasets import build_dataloader, build_dataset
    from ssod.utils import get_root_logger, patch_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fixtures are not GPU acceptance")
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    # The inherited dev config contains ${percent}/${fold} but intentionally
    # leaves their values to the entry point. Resolve the fixed 3-shot protocol
    # only after injecting every required variable (also overriding cfg-options).
    cfg.fold, cfg.percent, cfg.seed = args.fold, 3, args.seed
    cfg.work_dir = str(out_dir)
    cfg = patch_config(cfg)
    if cfg.model.type != "DualTeacher" or not cfg.model.train_cfg.get("m2_enabled", False):
        raise ValueError("Use the M3 dev config with M2 enabled")
    if cfg.model.model.roi_head.type != "M3RoIHead" or not cfg.model.train_cfg.m3_enabled:
        raise ValueError("Use the M3 dev config with M3RoIHead and m3_enabled=True")
    train_cfg = cfg.model.train_cfg
    if (train_cfg.get("m2_force_weight_one", False)
            or train_cfg.get("m3_target_mode") != "lower_uncertainty"
            or train_cfg.get("m3_min_anchor_iou") != 0.5):
        raise ValueError("Fixed protocol requires M2 weights, lower_uncertainty and anchor IoU 0.5")
    if cfg.get("load_from") or cfg.get("resume_from"):
        raise ValueError("Use explicit --checkpoint; config load_from/resume_from must be None")
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    # Single-process data preparation makes the batch reproducible without
    # spawning workers. All original training transforms and sampler remain.
    cfg.data.workers_per_gpu = 0
    cfg.gpu_ids = [torch.cuda.current_device()]
    use_fp16 = cfg.get("fp16") is not None and not args.fp32
    if args.fp32:
        cfg.fp16 = None
    cfg.dump(str(out_dir / "resolved_config.py"))
    report["resolved_config_sha256"] = file_hash(out_dir / "resolved_config.py")
    report["training_annotations"] = {
        stream: dict(path=str(Path(cfg.data.train[stream].ann_file).resolve()),
                     sha256=file_hash(cfg.data.train[stream].ann_file))
        for stream in ("sup1", "sup2", "unsup")}
    get_root_logger(log_file=str(out_dir / "acceptance.log"), log_level="INFO")
    set_random_seed(args.seed, deterministic=True)
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
                           test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    if args.checkpoint:
        strict_full_checkpoint(model, args.checkpoint)
        checkpoint_paths = [args.checkpoint]
    else:
        model.init_from_pretrained()
        checkpoint_paths = [model.load1_from, model.load2_from]
    report["initialization"] = dict(
        kind="full_strict" if args.checkpoint else "phase1_phase2_strict",
        checkpoints=[dict(path=str(Path(path).resolve()), sha256=file_hash(path))
                     for path in checkpoint_paths])
    dataset = build_dataset(cfg.data.train)
    model.CLASSES = dataset.CLASSES
    # This repository registers only DistributedGroupSemiCrossBalanceSampler.
    # dist=True selects that real sampler; get_dist_info returns rank 0/world 1
    # without initializing a process group, so forward/backward stays local.
    loader = build_dataloader(dataset, cfg.data.samples_per_gpu, 0, num_gpus=1,
                              dist=True, seed=args.seed,
                              sampler_cfg=copy.deepcopy(cfg.data.get("sampler", {}).get("train", {})))
    iterator = iter(loader)
    for _ in range(args.batch_index + 1):
        batch = next(iterator)
    from mmcv.parallel import scatter
    preview = scatter(copy.deepcopy(batch), [torch.cuda.current_device()])[0]
    metas = preview["img_metas"]
    tags = [meta["tag"] for meta in metas]
    if set(tags) != {"sup1", "sup2", "unsup_teacher", "unsup_student"}:
        raise ValueError("Expected all three streams plus teacher/student unsup views: " + repr(tags))
    report["batch"] = dict(index=args.batch_index, tags=tags,
                           filenames=[meta.get("filename") for meta in metas],
                           image_sha256=tensor_hash(preview["img"]))
    del preview
    model.cuda()
    if use_fp16:
        wrap_fp16_model(model)
    state = OrderedDict((key, value.detach().cpu().clone())
                        for key, value in model.state_dict().items())
    rng = rng_snapshot()
    events = []
    install_sampling_observers(model, events)
    report["runtime"] = dict(torch=torch.__version__, mmcv=mmcv.__version__,
                             mmdet=mmdet.__version__, numpy=np.__version__,
                             cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                             cudnn_benchmark=torch.backends.cudnn.benchmark,
                             fp16=use_fp16, backward=not args.forward_only,
                             backward_loss_scale=args.loss_scale,
                             optimizer_updates=0, ema_updates=0)
    report["conditions"], report["comparisons"] = {}, {}
    try:
        baseline, baseline_losses, baseline_grads = run_condition(
            model, batch, "off", args, use_fp16, state, rng, events)
        report["conditions"]["off"] = baseline
        original, losses, grads = run_condition(
            model, batch, "original", args, use_fp16, state, rng, events)
        report["conditions"]["original"] = original
        report["comparisons"]["off_vs_original_losses"] = compare_tensors(
            baseline_losses, losses, args.loss_atol, args.loss_rtol)
        if not args.forward_only:
            report["comparisons"]["off_vs_original_gradients"] = compare_tensors(
                baseline_grads, grads, args.grad_atol, args.grad_rtol)
        del baseline_grads, grads, losses
        routed, losses, grads = run_condition(
            model, batch, "lower_uncertainty", args, use_fp16, state, rng, events)
        report["conditions"]["lower_uncertainty"] = routed
        report["comparisons"]["off_vs_routed_nonregression"] = compare_tensors(
            baseline_losses, losses, args.loss_atol, args.loss_rtol,
            exclude=("unsup1_loss_bbox", "unsup2_loss_bbox"))
        report["comparisons"]["original_sampling_preserved"] = dict(passed=(
            sampling_signature(baseline["sampling"]) == sampling_signature(original["sampling"])
            == sampling_signature(routed["sampling"])))
        report["comparisons"]["original_targets_preserved"] = dict(passed=(
            original["actual_bbox_target_rows_changed"] == 0))
        report["comparisons"]["both_student_routing_paths_executed"] = dict(passed=all(
            any(event["branch"] == branch and event["routed"] for event in routed["sampling"])
            for branch in ("student1", "student2")))
        passed = all(result["passed"] for result in report["comparisons"].values())
        report["status"] = ("failed" if not passed else "passed" if
                            routed["actual_bbox_target_rows_changed"] > 0 else "inconclusive")
        if report["status"] == "inconclusive":
            report["reason"] = ("No sampled positive regression target changed. Repeat a "
                                "predeclared next --batch-index in a NEW output directory; "
                                "do not select batches using AP or held-out GT.")
    finally:
        clear_grads(model)
        model.load_state_dict(state, strict=True)
        restore_rng(rng)
        report["state_restored_exactly"] = all(
            torch.equal(value.detach().cpu(), state[key]) for key, value in model.state_dict().items())
        if not report["state_restored_exactly"]:
            report["status"] = "failed"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), Path(args.config).resolve(),
                    ROOT / "ssod/models/dual_teacher.py",
                    ROOT / "ssod/models/m3_routing.py",
                    ROOT / "ssod/models/roi_heads/m3_roi_head.py"]
    report = dict(status="failed", arguments=vars(args),
                  limits=("One training batch; explicit elementwise tolerances bound this "
                          "check, not a proof against GPU nondeterminism or an AP result. "
                          "No optimizer, gradient clipping, dynamic-scaler update or EMA."),
                  source_sha256={str(path): file_hash(path) for path in source_paths})
    try:
        report["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
        report["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=str(ROOT)).decode()
        execute(args, out_dir, report)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    finally:
        with open(str(out_dir / "report.json"), "w") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print("M3 acceptance {}: {}".format(report["status"], out_dir / "report.json"))
    return {"passed": 0, "failed": 1, "inconclusive": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
