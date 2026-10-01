"""A1-A6 acceptance for the sup2 small-background ROI reweighting. No training.

Runs one frozen training batch through the same weights several times and only
inspects what came out. Nothing is stepped: no optimizer update, no EMA update,
no checkpoint written, no runner or hook is built.

Variants, all built from the same config and the same checkpoint:

  b0        the unmodified config (StandardRoIHead / Shared2FCBBoxHead)
  b0_repeat the unmodified config again, as the determinism control: whatever
            b0-vs-b0_repeat differs by is the noise floor of this setup, not the
            effect of the change
  lambda0   SmallBkgReweight* with reweight.enable=True and lambda_=0.0
  lambda1   SmallBkgReweight* with reweight.enable=True and lambda_=1.0

Checks (proposal v3 section 5.1):

  A1  existence    reweighted ROIs > 0, and every one of them is background --
                   checked twice: by the head's own `label`s and independently by
                   IoU against the image's GT boxes
  A2  formula      weight values subset of {1.0, 1+lambda}; the 1+lambda positions
                   correspond one-to-one to "background and original-scale area
                   < 32^2", recomputed from the raw ROI boxes and `scale_factor`
  A3  isolation    against lambda0, every loss term except `sup2_loss_cls` is
                   bitwise identical
  A4  branch gate  sup1 / unsup_* never reweight a single ROI
  A5  lambda0      bitwise identical losses to b0 (within the b0-vs-b0_repeat
                   noise floor)
  A6  parameters   0 new trainable parameters, state_dict keys identical to b0,
                   b0 weights loadable with strict=True

Usage (run from the checkout root, on the training machine):
    python tools/verify_small_bkg_reweight.py \
        --config ablation_configs/fold6_seed678/b0.py \
        --checkpoint work_dirs/ablation_v1/fold6_seed678/b0/iter_32000.pth \
        --out-dir ablation_configs/reweight_small_bkg_20260929
"""

import argparse
import collections
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# Import the training entry point the same way tools/fg_replay_diagnostic.py
# does: the environment's egg-link points at another checkout, so this one is
# pinned before anything under ssod is imported.
_spec = importlib.util.spec_from_file_location(
    "_ablation_entry", ROOT / "tools" / "train_ablation.py")
_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry)

NEW_MODULE = "ssod.models.roi_heads.small_bkg_reweight"
NEW_MODULE_PATH = "ssod/models/roi_heads/small_bkg_reweight.py"
VARIANT_CONFIG_NAME = "variant_config.py"

REQUIRED_TAGS = ("sup1", "sup2", "unsup_teacher", "unsup_student")
BASELINE_TAG = "sup2"
MAX_AREA = 32.0 ** 2
COUNTER_KEYS = ("optimizer_step", "optimizer_zero_grad", "ema_momentum_update",
                "ema_before_train_iter", "ema_before_run", "ema_after_train_iter")
EXPECTED_CONFIG_DIFFS = {
    "custom_imports",
    "model.model.roi_head.type",
    "model.model.roi_head.bbox_head.type",
    "model.model.roi_head.bbox_head.reweight",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def state_fingerprint(model):
    return (
        {name: digest_tensor(p) for name, p in model.named_parameters()},
        {name: digest_tensor(b) for name, b in model.named_buffers()},
    )


def rng_fingerprint():
    parts = [
        repr(random.getstate()),
        repr(np.random.get_state()),
        torch.get_rng_state().numpy().tobytes().hex(),
        repr([t.numpy().tobytes().hex() for t in torch.cuda.get_rng_state_all()]),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def reseed_all(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def iou_matrix(boxes_a, boxes_b):
    """IoU of every a-box against every b-box (plain intersection/union)."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]))
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def install_instrumentation():
    """Count every optimizer / EMA write, then prove the counters can move."""
    from ssod.utils.hooks.mean_teacher import MeanTeacher

    counters = collections.Counter()
    # `zero_grad` is inherited from the base class, but concrete optimizers
    # (SGD, Adam, ...) override `step` with their own implementation, so
    # patching torch.optim.Optimizer.step alone silently misses the real call.
    # Patch the base *and* every subclass defining its own `step`; the positive
    # control below drives SGD, so a silent miss cannot pass.
    step_targets = [torch.optim.Optimizer]
    for name in dir(torch.optim):
        candidate = getattr(torch.optim, name)
        if (isinstance(candidate, type)
                and issubclass(candidate, torch.optim.Optimizer)
                and "step" in candidate.__dict__
                and candidate not in step_targets):
            step_targets.append(candidate)
    original = dict(
        step={cls: cls.step for cls in step_targets},
        zero_grad=torch.optim.Optimizer.zero_grad,
        momentum_update=MeanTeacher.momentum_update,
        before_train_iter=MeanTeacher.before_train_iter,
        before_run=MeanTeacher.before_run,
        after_train_iter=MeanTeacher.after_train_iter,
    )

    def make_optimizer_step(base_step):
        def optimizer_step(self, *args, **kwargs):
            counters["optimizer_step"] += 1
            return base_step(self, *args, **kwargs)
        return optimizer_step

    def optimizer_zero_grad(self, *args, **kwargs):
        counters["optimizer_zero_grad"] += 1
        return original["zero_grad"](self, *args, **kwargs)

    def momentum_update(self, model, momentum):
        counters["ema_momentum_update"] += 1
        return original["momentum_update"](self, model, momentum)

    def before_train_iter(self, runner):
        counters["ema_before_train_iter"] += 1
        return original["before_train_iter"](self, runner)

    def before_run(self, runner):
        counters["ema_before_run"] += 1
        return original["before_run"](self, runner)

    def after_train_iter(self, runner):
        counters["ema_after_train_iter"] += 1
        return original["after_train_iter"](self, runner)

    for cls, base_step in original["step"].items():
        cls.step = make_optimizer_step(base_step)
    torch.optim.Optimizer.zero_grad = optimizer_zero_grad
    MeanTeacher.momentum_update = momentum_update
    MeanTeacher.before_train_iter = before_train_iter
    MeanTeacher.before_run = before_run
    MeanTeacher.after_train_iter = after_train_iter

    # Positive controls: the counters must move when the code paths really run,
    # otherwise "0" below would prove nothing.
    param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    param.grad = torch.zeros(1)
    optimizer = torch.optim.SGD([param], lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    other = torch.optim.Adam([param], lr=0.1)
    other.step()
    stub = torch.nn.Module()
    for name in ("student1", "student2", "teacher1", "teacher2"):
        child = torch.nn.Module()
        child.p = torch.nn.Parameter(torch.ones(1))
        setattr(stub, name, child)
    MeanTeacher().momentum_update(stub, 0.5)
    proof = dict(
        optimizer_step_seen=counters["optimizer_step"],
        optimizer_zero_grad_seen=counters["optimizer_zero_grad"],
        ema_momentum_update_seen=counters["ema_momentum_update"],
        step_patched=all(cls.step is not base_step
                         for cls, base_step in original["step"].items()),
        step_patched_classes=[cls.__name__ for cls in original["step"]],
        momentum_update_patched=MeanTeacher.momentum_update is not original["momentum_update"],
    )
    proof["passed"] = (proof["optimizer_step_seen"] == 2   # SGD + Adam
                       and proof["optimizer_zero_grad_seen"] == 1
                       and proof["ema_momentum_update_seen"] == 1
                       and proof["step_patched"] and proof["momentum_update_patched"])
    counters.clear()
    return counters, proof


def extended_receipt():
    receipt = _entry.source_manifest()
    module = importlib.import_module(NEW_MODULE)
    loaded = Path(module.__file__).resolve()
    expected = (ROOT / NEW_MODULE_PATH).resolve()
    if loaded != expected:
        raise RuntimeError(
            "{} imported from {}, expected {}".format(NEW_MODULE, loaded, expected))
    receipt["sources"][NEW_MODULE] = dict(path=str(loaded), sha256=sha256_file(loaded))
    receipt["extensions"] = dict(
        new_module=NEW_MODULE,
        in_train_ablation_source_modules=(NEW_MODULE in getattr(_entry, "SOURCE_MODULES", {})),
        harness=dict(path=str(Path(__file__).resolve()), sha256=sha256_file(__file__)),
        train_ablation=dict(path=str(ROOT / "tools" / "train_ablation.py"),
                            sha256=sha256_file(ROOT / "tools" / "train_ablation.py")),
    )
    return receipt


def run_variant(pass_name, variant, cfg, dataset, batch, checkpoint, device,
                counters, capture=None):
    """One forward. Returns losses, the reweight log and the state evidence."""
    from mmcv.parallel import scatter
    from mmcv.runner import build_optimizer, load_checkpoint, wrap_fp16_model
    from mmdet.models import build_detector

    # cfg.model is the DualTeacher dict and carries its own train_cfg/test_cfg;
    # this mirrors tools/check_ablation_step.py exactly
    model = build_detector(cfg.model)
    from ssod.models.roi_heads.small_bkg_reweight import enable_reweight_diagnostics
    enable_reweight_diagnostics(model)
    if pass_name == "train":
        wrap_fp16_model(model)
    # strict=True: a key mismatch anywhere is a hard failure, not a warning
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model._pretrained_initialized = True  # never re-read Phase 1/2 files here
    model.CLASSES = dataset.CLASSES  # DualTeacher's box-logging path reads self.CLASSES
    model.cuda(device)
    model.train()
    model.freeze("teacher1")
    model.freeze("teacher2")

    optimizer = build_optimizer(model, cfg.optimizer)
    param_groups = [len(group["params"]) for group in optimizer.param_groups]

    keys = sorted(model.state_dict().keys())
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    heads = {}
    for branch in ("teacher1", "teacher2", "student1", "student2"):
        head = getattr(getattr(model, branch), "roi_head", None)
        head = getattr(head, "bbox_head", None)
        heads[branch] = head
        if head is not None and hasattr(head, "reset_reweight_log"):
            head.reset_reweight_log()

    params_before, buffers_before = state_fingerprint(model)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all())
    reseed_all(state)
    rng_before = rng_fingerprint()
    if capture is not None:
        capture["target"] = getattr(model.student2, "roi_head", None)
    inputs = scatter(copy.deepcopy(batch), [device])[0]
    before_forwards = dict(counters)
    started = time.time()
    with torch.cuda.amp.autocast(enabled=(pass_name == "train")):
        losses = model(return_loss=True, **inputs)
    elapsed = time.time() - started
    deltas = {key: counters[key] - before_forwards.get(key, 0) for key in COUNTER_KEYS}
    # keep the batch tensors for the independent GT check before dropping them
    gt_boxes = [box.detach().float().cpu().numpy() for box in inputs["gt_bboxes"]]
    metas = [dict(tag=str(meta["tag"]), filename=meta.get("filename"),
                  scale_factor=(None if meta.get("scale_factor") is None
                                else [float(v) for v in meta["scale_factor"]]))
             for meta in inputs["img_metas"]]
    # Each branch's forward_train returns *lists* of tensors (one entry per
    # image), so the raw dict cannot be compared key-wise. `_parse_losses` is
    # the framework's own reduction (tensor -> mean, list -> sum of means) and
    # is what a real training step logs, so compare on exactly that. Its
    # aggregate `loss` entry is a derived sum of the others, so it is kept out
    # of the per-key table and reported separately as `total_loss`.
    total_loss, parsed = model._parse_losses(losses)
    loss_values = {key: as_float(value)
                   for key, value in parsed.items() if key != "loss"}
    total_loss = as_float(total_loss)
    # keep the sup2 branch's frozen tensors (only that head; the others are
    # teachers / the pseudo-label streams and are not under test)
    sup2_capture = None
    if capture is not None:
        sup2_capture = capture["hits"][0] if capture["hits"] else None
        capture["hits"] = []
        capture["target"] = None

    params_after, buffers_after = state_fingerprint(model)
    grads = sorted(name for name, p in model.named_parameters() if p.grad is not None)
    logs = {}
    for branch, head in heads.items():
        if head is not None and hasattr(head, "reweight_log"):
            logs[branch] = copy.deepcopy(head.reweight_log)
    changed_params = sorted(k for k in params_before if params_before[k] != params_after.get(k))
    changed_buffers = sorted(k for k in buffers_before if buffers_before[k] != buffers_after.get(k))

    del losses, inputs, optimizer, model
    torch.cuda.empty_cache()
    return dict(
        variant=variant, pass_name=pass_name, losses=loss_values,
        total_loss=total_loss, sup2_capture=sup2_capture,
        checkpoint_loaded_strict=True,  # reached only because strict=True passed
        elapsed_seconds=elapsed, counter_deltas=dict(deltas),
        rng_before=rng_before, n_params=n_params, n_trainable=n_trainable,
        state_dict_keys=keys, param_groups=param_groups,
        changed_params=changed_params, changed_buffers=changed_buffers,
        params_unchanged=(not changed_params), n_changed_buffers=len(changed_buffers),
        grads_present=grads, logs=logs, gt_boxes=gt_boxes, img_metas=metas,
    )


def as_float(value):
    """`_parse_losses` already returns python floats, raw losses are tensors."""
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    if isinstance(value, (list, tuple)):
        return float(sum(as_float(item) for item in value))
    return float(value)


# --------------------------------------------------------------- head level
# The end-to-end loss comparison cannot resolve the lambda effect. Measured on
# this batch: two runs of the *same* variant differ in every cls-type loss
# (b0 vs b0_repeat: max |delta| 0.586 fp32 / 0.195 amp), because the
# semi-supervised branches threshold the teacher's scores into pseudo-labels --
# a bit-level difference in one score flips a whole label set. bbox-type losses
# are bitwise stable, since only positives carry regression weight. The effect
# is therefore also measured where it lives: on frozen tensors, with the sampled
# set, the scores and the head weights held fixed across variants.

SAMPLING_FIELDS = ("pos_bboxes", "neg_bboxes", "pos_gt_bboxes", "pos_gt_labels")


def clone_sampling_results(sampling_results):
    """Detach the per-image sampling result fields the head actually reads."""
    return [dict((field, getattr(res, field).detach().float().cpu())
                 for field in SAMPLING_FIELDS) for res in sampling_results]


def rebuild_sampling_results(clones):
    from mmdet.core.bbox.samplers.sampling_result import SamplingResult
    rebuilt = []
    for clone in clones:
        res = SamplingResult.__new__(SamplingResult)
        for field, value in clone.items():
            setattr(res, field, value)
        rebuilt.append(res)
    return rebuilt


def install_capture():
    """Stash the sup2 branch's frozen tensors for the head-level stage.

    Only the branch marked via ``store['target']`` is captured -- copying every
    roi head's output costs GPU memory for nothing -- and no module reference is
    kept, so a finished variant's model can still be collected.
    """
    from mmdet.core import bbox2roi
    from mmdet.models.roi_heads import StandardRoIHead
    from ssod.models.roi_heads.small_bkg_reweight import SmallBkgReweightRoIHead

    store = dict(target=None, hits=[], originals={})

    def wrap(cls):
        store["originals"][cls] = cls._bbox_forward_train

        def patched(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
            results = store["originals"][cls](
                self, x, sampling_results, gt_bboxes, gt_labels, img_metas)
            if self is store["target"]:
                # `rois` is a local of _bbox_forward_train, not part of its
                # return value; rebuild it exactly as that method does
                rois = bbox2roi([res.bboxes for res in sampling_results])
                store["hits"].append(dict(
                    rois=rois.detach().cpu(),
                    cls_score=results["cls_score"].detach().cpu(),
                    bbox_pred=results["bbox_pred"].detach().cpu(),
                    sampling_results=clone_sampling_results(sampling_results),
                    gt_bboxes=[box.detach().cpu() for box in gt_bboxes],
                    gt_labels=[lbl.detach().cpu() for lbl in gt_labels],
                    img_metas=[dict(
                        tag=str(meta.get("tag")),
                        filename=meta.get("filename"),
                        scale_factor=(None if meta.get("scale_factor") is None
                                      else [float(v) for v in meta["scale_factor"]]))
                        for meta in img_metas],
                ))
            return results

        cls._bbox_forward_train = patched

    wrap(StandardRoIHead)
    wrap(SmallBkgReweightRoIHead)  # the subclass overrides the parent's method
    return store


def head_level_stage(cfg_b0, cfg_lambda0, cfg_lambda1, checkpoint, capture,
                     max_area, lambda_value):
    """lambda=0 regression and lambda=1 effect on frozen tensors, deterministic.

    All three heads receive the same ``cls_score`` / ``bbox_pred`` / ``rois``
    and the same sampled set, so the only variable is the reweighting itself.
    """
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    if capture is None:
        raise RuntimeError("no sup2 capture was recorded")

    heads, rcnn_cfg = {}, None
    for name, cfg in (("b0", cfg_b0), ("lambda0", cfg_lambda0),
                      ("lambda1", cfg_lambda1)):
        model = build_detector(cfg.model)  # CPU: only get_targets/loss are used
        from ssod.models.roi_heads.small_bkg_reweight import enable_reweight_diagnostics
        enable_reweight_diagnostics(model)
        load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
        heads[name] = model.student2.roi_head.bbox_head
        rcnn_cfg = model.student2.roi_head.train_cfg
    del model

    # the amp pass stores fp16 tensors; bring everything to fp32 so all three
    # heads are compared on identical inputs
    cls_score = capture["cls_score"].float()
    bbox_pred = capture["bbox_pred"].float()
    rois = capture["rois"].float()
    sampling_results = rebuild_sampling_results(capture["sampling_results"])
    gt_bboxes, gt_labels = capture["gt_bboxes"], capture["gt_labels"]
    metas = capture["img_metas"]
    if [meta["tag"] for meta in metas] != [BASELINE_TAG] * len(metas):
        raise RuntimeError("the captured branch is not the %s stream" % BASELINE_TAG)
    reweight_ctx = dict(scale_factors=[meta["scale_factor"] for meta in metas],
                        tags=[meta["tag"] for meta in metas])

    targets, losses, avg_factors, factors_seen = {}, {}, {}, {}
    for name, head in heads.items():
        extra = {} if name == "b0" else dict(reweight_ctx=reweight_ctx)
        targets[name] = head.get_targets(sampling_results, gt_bboxes, gt_labels,
                                         rcnn_cfg, **extra)
        avg_factors[name] = max(torch.sum(targets[name][1] > 0).float().item(), 1.)
        losses[name] = {key: as_float(value) for key, value
                        in head.loss(cls_score, bbox_pred, rois,
                                     *targets[name]).items()}
        if name != "b0":
            factors_seen[name] = sorted({value for record in head.reweight_log
                                         for value in record.get("factors", [])})

    def tensor_diff(left, right):
        return max(float((a - b).abs().max().item()) for a, b in zip(left, right))

    diff_l0 = tensor_diff(targets["lambda0"], targets["b0"])
    diff_l1 = tensor_diff(targets["lambda1"], targets["b0"])
    changed_l0 = sorted(key for key in losses["b0"]
                        if losses["lambda0"].get(key) != losses["b0"][key])
    changed_l1 = sorted(key for key in losses["b0"]
                        if losses["lambda1"].get(key) != losses["b0"][key])
    denominators_equal = (avg_factors["lambda0"] == avg_factors["b0"]
                          == avg_factors["lambda1"])
    return dict(
        capture_from=capture.get("source", "unknown"),
        n_images=len(metas), tags=[meta["tag"] for meta in metas],
        scale_factors=[meta["scale_factor"] for meta in metas],
        target_max_abs_diff_lambda0_vs_b0=diff_l0,
        target_max_abs_diff_lambda1_vs_b0=diff_l1,
        targets_bitwise_equal_lambda0=(diff_l0 == 0.0),
        factors_seen=factors_seen,
        losses=losses, avg_factor=avg_factors, denominators_equal=denominators_equal,
        changed_keys_lambda0_vs_b0=changed_l0,
        changed_keys_lambda1_vs_b0=changed_l1,
        loss_bitwise_equal_lambda0=(not changed_l0),
        only_cls_changed_lambda1=(changed_l1 == ["loss_cls"]),
        cls_increased_lambda1=losses["lambda1"]["loss_cls"] > losses["b0"]["loss_cls"],
        cls_ratio_lambda1_over_b0=(losses["lambda1"]["loss_cls"]
                                   / losses["b0"]["loss_cls"]),
        acc_equal_lambda1=losses["lambda1"].get("acc") == losses["b0"].get("acc"),
        passed_lambda0=(diff_l0 == 0.0 and not changed_l0 and denominators_equal),
        passed_lambda1=(diff_l1 > 0.0 and changed_l1 == ["loss_cls"]
                        and losses["lambda1"]["loss_cls"] > losses["b0"]["loss_cls"]
                        and denominators_equal),
    )


def compare_losses(left, right):
    keys = sorted(set(left) | set(right))
    rows = {}
    for key in keys:
        a, b = left.get(key), right.get(key)
        rows[key] = dict(
            left=a, right=b,
            abs_diff=None if (a is None or b is None) else abs(a - b),
            bitwise_equal=(a is not None and b is not None and a == b),
        )
    max_diff = max([row["abs_diff"] for row in rows.values() if row["abs_diff"] is not None] or [0.0])
    return dict(keys_equal=sorted(left) == sorted(right), max_abs_diff=max_diff, rows=rows)


def probe_records(run, lambda_value, max_area, gt_index):
    """Per-record formula/background evidence plus the independent GT check."""
    probe = dict(by_tag={}, records=[], issues=[])
    for branch, records in sorted(run["logs"].items()):
        for record in records:
            entry = dict(branch=branch, tag=record["tag"],
                         eligible=record["eligible"],
                         n_neg=record["num_neg"], num_pos=record["num_pos"],
                         n_reweighted=record["n_reweighted"],
                         skipped=record.get("skipped"))
            tag_key = "{}|{}".format(branch, record["tag"])
            summary = probe["by_tag"].setdefault(
                tag_key, dict(branch=branch, tag=record["tag"], images=0,
                              n_neg=0, n_reweighted=0))
            summary["images"] += 1
            summary["n_neg"] += record["num_neg"]
            summary["n_reweighted"] += record["n_reweighted"]
            if not record["eligible"] or record["num_neg"] == 0:
                probe["records"].append(entry)
                continue

            boxes = np.asarray(record["neg_boxes"], dtype=float)
            areas = np.asarray(record["areas"], dtype=float)
            mask = np.asarray(record["small_mask"], dtype=bool)
            factors = np.asarray(record["factors"], dtype=float)
            before = np.asarray(record["weight_before"], dtype=float)
            after = np.asarray(record["weight_after"], dtype=float)
            scale = record["scale_factor"]
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            if scale is not None:
                widths = widths / float(scale[0])
                heights = heights / float(scale[1])
            expected_areas = widths * heights
            expected_mask = expected_areas < max_area
            expected_factors = np.where(expected_mask, 1.0 + lambda_value, 1.0)

            issues = []
            if not np.allclose(areas, expected_areas, rtol=1e-5, atol=1e-6):
                issues.append("area restoration does not reproduce the logged areas")
            if not np.array_equal(mask, expected_mask):
                issues.append("reweight mask is not 'area < %.1f' at original scale" % max_area)
            if not np.array_equal(factors, expected_factors):
                issues.append("factors are not 1+lambda exactly on the mask")
            if not np.array_equal(after, before * factors):
                issues.append("weight_after != weight_before * factor")
            if not record["neg_labels_all_bg"]:
                issues.append("a reweighted ROI is not labelled background")
            values = sorted(set(factors.tolist()))
            if not set(values) <= {1.0, 1.0 + lambda_value}:
                issues.append("weight values outside {1.0, 1+lambda}")

            gt = np.asarray(run["gt_boxes"][gt_index], dtype=float)
            iou_max = iou_matrix(boxes, gt).max(axis=1) if gt.size else np.zeros(boxes.shape[0])
            hit = iou_max[mask]
            if hit.size and float(hit.max()) >= 0.5 + 1e-6:
                issues.append("a reweighted ROI overlaps a GT above the IoU 0.5 threshold")
            entry.update(
                factor_values=values,
                area_min=float(expected_areas.min()), area_max=float(expected_areas.max()),
                reweighted_area_min=float(expected_areas[mask].min()) if mask.any() else None,
                reweighted_area_max=float(expected_areas[mask].max()) if mask.any() else None,
                neg_labels_all_bg=record["neg_labels_all_bg"],
                max_iou_vs_gt_among_reweighted=(None if not hit.size else float(hit.max())),
                min_iou_margin_to_0_5=(None if not hit.size else float(0.5 - hit.max())),
                issues=issues,
            )
            probe["records"].append(entry)
            probe["issues"].extend("{}/{}: {}".format(branch, record["tag"], issue)
                                   for issue in issues)
    return probe


def evaluate_checks(per_pass, per_pass_probe, per_pass_comparisons,
                    head_level=None):
    checks = {}

    def collect(name, predicate, detail):
        checks[name] = dict(passed=bool(predicate), detail=detail)

    # --- A1 / A2 / A4 come from the lambda1 probe of every precision pass ---
    a1, a2, a4 = {}, {}, {}
    for pass_name, probe in sorted(per_pass_probe.items()):
        reweighted = [r for r in probe["records"] if r.get("n_reweighted")]
        n_total = sum(r["n_reweighted"] for r in probe["records"] if r["eligible"])
        backgrounds = [r for r in reweighted if r.get("neg_labels_all_bg") is True
                       and (r.get("max_iou_vs_gt_among_reweighted") or 0.0) < 0.5]
        a1[pass_name] = dict(
            n_reweighted_total=n_total,
            images_with_reweighted=len(reweighted),
            all_background_by_label=all(r.get("neg_labels_all_bg") for r in reweighted) if reweighted else None,
            all_background_by_gt_iou=len(backgrounds) == len(reweighted),
            max_iou_among_reweighted=max(
                [r["max_iou_vs_gt_among_reweighted"] or 0.0 for r in reweighted] or [0.0]),
            passed=bool(reweighted) and n_total > 0
            and len(backgrounds) == len(reweighted),
        )
        a2[pass_name] = dict(
            issues=probe["issues"],
            factor_value_sets=sorted({tuple(r["factor_values"]) for r in reweighted}),
            n_checked=len(reweighted),
            passed=bool(reweighted) and not probe["issues"],
        )
        gated = [r for r in probe["records"] if r["tag"] != BASELINE_TAG]
        a4[pass_name] = dict(
            gated_images=len(gated),
            gated_reweighted=sum(r["n_reweighted"] for r in gated),
            by_tag={key: val for key, val in sorted(probe["by_tag"].items())},
            passed=all(r["n_reweighted"] == 0 for r in gated),
        )
    collect("A1", all(v["passed"] for v in a1.values()), a1)
    collect("A2", all(v["passed"] for v in a2.values()), a2)
    collect("A4", all(v["passed"] for v in a4.values()), a4)

    # --- A3: lambda1 vs lambda0 ---
    a3, a5, a6 = {}, {}, {}
    for pass_name in sorted(per_pass):
        cmp_l1_l0 = per_pass_comparisons[pass_name]["lambda1_vs_lambda0"]
        cmp_l0_b0 = per_pass_comparisons[pass_name]["lambda0_vs_b0"]
        cmp_ctrl = per_pass_comparisons[pass_name]["b0_repeat_vs_b0"]
        noise = cmp_ctrl["max_abs_diff"]
        changed = sorted(key for key, row in cmp_l1_l0["rows"].items()
                         if row["abs_diff"] is not None and row["abs_diff"] > max(noise, 0.0))
        cls_row = cmp_l1_l0["rows"].get("sup2_loss_cls", {})
        total_l1_l0 = per_pass_comparisons[pass_name]["total_loss_lambda1_vs_lambda0"]["rows"]["total_loss"]
        # The effect (~2e-4) is far below this batch's same-variant noise floor,
        # so end to end only the negative claim is available: no key *other*
        # than the agreed classification loss moves beyond two identical runs
        # of one variant. The positive claim ("only loss_cls changed, with the
        # denominator held") is measured at head level in evaluate_checks().
        others_changed = [key for key in changed if key != "sup2_loss_cls"]
        a3[pass_name] = dict(
            noise_floor=noise,
            changed_keys_above_noise=changed,
            changed_keys_other_than_cls_above_noise=others_changed,
            effect_resolvable_end_to_end=bool(changed),
            sup2_loss_cls_lambda0=cls_row.get("right"),
            sup2_loss_cls_lambda1=cls_row.get("left"),
            sup2_loss_cls_increased=bool(cls_row.get("left", 0) > cls_row.get("right", 0)),
            total_loss_lambda0=total_l1_l0["right"],
            total_loss_lambda1=total_l1_l0["left"],
            total_loss_delta=total_l1_l0["abs_diff"],
            passed=bool(cmp_l1_l0["keys_equal"] and not others_changed),
        )
        total_l0_b0 = per_pass_comparisons[pass_name]["total_loss_lambda0_vs_b0"]["rows"]["total_loss"]
        a5[pass_name] = dict(
            noise_floor=noise,
            max_abs_diff=cmp_l0_b0["max_abs_diff"],
            bitwise_equal=bool(cmp_l0_b0["max_abs_diff"] == 0.0),
            total_loss_delta=total_l0_b0["abs_diff"],
            total_loss_bitwise_equal=bool(total_l0_b0["bitwise_equal"]),
            passed=cmp_l0_b0["keys_equal"] and cmp_l0_b0["max_abs_diff"] <= max(noise, 0.0)
            and total_l0_b0["abs_diff"] <= max(noise, 0.0),
        )
        keys_b0 = set(per_pass[pass_name]["b0"]["state_dict_keys"])
        keys_l1 = set(per_pass[pass_name]["lambda1"]["state_dict_keys"])
        a6[pass_name] = dict(
            state_dict_keys_equal=sorted(keys_b0) == sorted(keys_l1),
            n_keys_b0=len(keys_b0), n_keys_lambda1=len(keys_l1),
            n_params_b0=per_pass[pass_name]["b0"]["n_params"],
            n_params_lambda1=per_pass[pass_name]["lambda1"]["n_params"],
            n_trainable_b0=per_pass[pass_name]["b0"]["n_trainable"],
            n_trainable_lambda1=per_pass[pass_name]["lambda1"]["n_trainable"],
            b0_checkpoint_loaded_strict=per_pass[pass_name]["lambda1"]["checkpoint_loaded_strict"],
            passed=(sorted(keys_b0) == sorted(keys_l1)
                    and per_pass[pass_name]["b0"]["n_trainable"]
                    == per_pass[pass_name]["lambda1"]["n_trainable"]),
        )
    collect("A3", all(v["passed"] for v in a3.values()), a3)
    collect("A5", all(v["passed"] for v in a5.values()), a5)
    collect("A6", all(v["passed"] for v in a6.values()), a6)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ablation_configs/fold6_seed678/b0.py")
    parser.add_argument("--checkpoint",
                        default="work_dirs/ablation_v1/fold6_seed678/b0/iter_32000.pth")
    parser.add_argument("--out-dir", default="ablation_configs/reweight_small_bkg_20260929")
    parser.add_argument("--passes", default="train,fp32")
    parser.add_argument("--scan-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None,
                        help="default: the config's own seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing to write into a non-empty directory: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    passes = [p for p in args.passes.split(",") if p]
    for name in passes:
        if name not in ("train", "fp32"):
            raise ValueError("unknown precision pass: %s" % name)

    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from this checkout root to preserve relative data paths")
    _entry.pin_repository()
    if not torch.cuda.is_available():
        raise SystemExit("run this check on the training machine with CUDA")

    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector  # noqa: F401  (registers pipelines)
    from mmdet.models.builder import HEADS
    from ssod.apis import set_random_seed
    from ssod.datasets import build_dataloader, build_dataset
    from ssod.utils import get_root_logger, patch_config
    from ssod.utils.ablation import differences

    importlib.import_module(NEW_MODULE)  # register the experimental classes

    started = time.time()
    command = " ".join([sys.executable, "tools/verify_small_bkg_reweight.py"] +
                       sys.argv[1:])
    logger = get_root_logger(log_file=str(out_dir / "harness.log"), log_level="INFO")
    logger.info("[acceptance] command: %s", command)

    receipt = extended_receipt()
    write_json(out_dir / "source_receipt.json", receipt)
    logger.info("[acceptance] source receipt written (%d modules, new module registered: %s)",
                len(receipt["sources"]), receipt["extensions"]["in_train_ablation_source_modules"])

    counters, instrumentation_proof = install_instrumentation()
    logger.info("[acceptance] instrumentation self-test: %s",
                json.dumps(instrumentation_proof, sort_keys=True))

    # NB: copy.deepcopy() of a mmcv Config yields a ConfigDict, and patch_config
    # requires a real Config, so every cfg is re-read from its file instead.
    seed = args.seed if args.seed is not None else Config.fromfile(args.config).get("seed")
    cfg_b0 = patch_config(Config.fromfile(args.config))
    if cfg_b0.model.model.roi_head.type != "StandardRoIHead" or \
            cfg_b0.model.model.roi_head.bbox_head.type != "Shared2FCBBoxHead":
        raise SystemExit("the reference config is not an unmodified B0 roi_head")
    # the config's own work_dir belongs to the training runs; redirect the
    # box-visualisation writer (ssod.utils.logger reads $WORK_DIR) into out_dir
    os.environ["WORK_DIR"] = str(out_dir / "work_dir_override")
    (out_dir / "work_dir_override").mkdir(exist_ok=True)

    cfg_lambda1 = patch_config(Config.fromfile(args.config))
    cfg_lambda1.custom_imports = dict(imports=[NEW_MODULE], allow_failed_imports=False)
    cfg_lambda1.model.model.roi_head.type = "SmallBkgReweightRoIHead"
    cfg_lambda1.model.model.roi_head.bbox_head.type = "SmallBkgReweightBBoxHead"
    cfg_lambda1.model.model.roi_head.bbox_head.reweight = dict(
        enable=True, lambda_=1.0, max_area=MAX_AREA, tag=BASELINE_TAG)
    variant_path = out_dir / VARIANT_CONFIG_NAME
    cfg_lambda1.dump(str(variant_path))
    config_diffs = differences(dict(cfg_b0), dict(cfg_lambda1))
    diff_paths = sorted(row["path"] for row in config_diffs)
    if set(diff_paths) != EXPECTED_CONFIG_DIFFS:
        raise SystemExit("unexpected config difference set: %s" % diff_paths)
    # a real training run would load the emitted config, so use that round trip
    os.environ["WORK_DIR"] = str(out_dir / "work_dir_override")
    cfg_lambda1 = patch_config(Config.fromfile(str(variant_path)))
    if cfg_lambda1.model.model.roi_head.type != "SmallBkgReweightRoIHead":
        raise SystemExit("the emitted variant config does not resolve to the new roi head")
    if "SmallBkgReweightRoIHead" not in HEADS or "SmallBkgReweightBBoxHead" not in HEADS:
        raise SystemExit("custom_imports did not register the new classes")
    cfg_lambda0 = patch_config(Config.fromfile(str(variant_path)))
    cfg_lambda0.model.model.roi_head.bbox_head.reweight = dict(
        enable=True, lambda_=0.0, max_area=MAX_AREA, tag=BASELINE_TAG)

    set_random_seed(seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    dataset = build_dataset(cfg_b0.data.train)
    loader = build_dataloader(
        dataset, cfg_b0.data.samples_per_gpu, 0, num_gpus=1, dist=True, seed=seed,
        sampler_cfg=copy.deepcopy(cfg_b0.data.get("sampler", {}).get("train", {})))
    device = torch.cuda.current_device()
    iterator = iter(loader)
    batch, batch_index, batch_tags = None, None, None
    for index in range(args.scan_limit):
        candidate = next(iterator)
        probe_inputs = scatter(copy.deepcopy(candidate), [device])[0]
        tags = [str(meta["tag"]) for meta in probe_inputs["img_metas"]]
        del probe_inputs
        if sorted(tags) == sorted(REQUIRED_TAGS):
            batch, batch_index, batch_tags = candidate, index, tags
            break
    if batch is None:
        raise SystemExit("no scanned batch covered all four supervision streams")
    logger.info("[acceptance] batch %d tags=%s", batch_index, batch_tags)
    sup2_index = batch_tags.index(BASELINE_TAG)

    variants = (
        ("b0", cfg_b0),
        ("b0_repeat", cfg_b0),
        ("lambda0", cfg_lambda0),
        ("lambda1", cfg_lambda1),
    )
    capture = install_capture()
    per_pass, per_pass_probe = {}, {}
    for pass_name in passes:
        per_pass[pass_name] = {}
        for variant, cfg in variants:
            logger.info("[acceptance] pass=%s variant=%s ...", pass_name, variant)
            result = run_variant(pass_name, variant, cfg, dataset, batch,
                                 args.checkpoint, device, counters, capture=capture)
            per_pass[pass_name][variant] = result
            logger.info("[acceptance] pass=%s variant=%s done in %.1fs counters=%s",
                        pass_name, variant, result["elapsed_seconds"],
                        json.dumps(result["counter_deltas"], sort_keys=True))
        per_pass_probe[pass_name] = probe_records(
            per_pass[pass_name]["lambda1"], 1.0, MAX_AREA, sup2_index)

    per_pass_comparisons = {}
    for pass_name in passes:
        runs = per_pass[pass_name]
        per_pass_comparisons[pass_name] = dict(
            lambda1_vs_lambda0=compare_losses(runs["lambda1"]["losses"], runs["lambda0"]["losses"]),
            lambda0_vs_b0=compare_losses(runs["lambda0"]["losses"], runs["b0"]["losses"]),
            b0_repeat_vs_b0=compare_losses(runs["b0_repeat"]["losses"], runs["b0"]["losses"]),
            lambda1_vs_b0=compare_losses(runs["lambda1"]["losses"], runs["b0"]["losses"]),
            total_loss_lambda1_vs_lambda0=compare_losses(
                {"total_loss": runs["lambda1"]["total_loss"]},
                {"total_loss": runs["lambda0"]["total_loss"]}),
            total_loss_lambda0_vs_b0=compare_losses(
                {"total_loss": runs["lambda0"]["total_loss"]},
                {"total_loss": runs["b0"]["total_loss"]}),
            total_loss_b0_repeat_vs_b0=compare_losses(
                {"total_loss": runs["b0_repeat"]["total_loss"]},
                {"total_loss": runs["b0"]["total_loss"]}),
        )

    # deterministic head-level stage, on the baseline run's own frozen tensors
    # (prefer the fp32 pass: its tensors are fp32, so no upcast is involved)
    baseline_pass = "fp32" if "fp32" in passes else passes[0]
    baseline_capture = per_pass[baseline_pass]["b0"]["sup2_capture"]
    if baseline_capture is not None:
        baseline_capture = dict(baseline_capture, source="%s|b0" % baseline_pass)
    head_level = head_level_stage(cfg_b0, cfg_lambda0, cfg_lambda1,
                                  args.checkpoint, baseline_capture,
                                  MAX_AREA, 1.0)
    logger.info("[acceptance] head level (lambda0 vs b0 bitwise: %s, "
                "lambda1 changed=%s, cls %.6g -> %.6g, avg_factor %s)",
                head_level["loss_bitwise_equal_lambda0"],
                head_level["changed_keys_lambda1_vs_b0"],
                head_level["losses"]["b0"]["loss_cls"],
                head_level["losses"]["lambda1"]["loss_cls"],
                head_level["avg_factor"])

    checks = evaluate_checks(per_pass, per_pass_probe, per_pass_comparisons,
                             head_level)
    checks["A3"]["head_level"] = head_level
    checks["A3"]["end_to_end_passed"] = checks["A3"]["passed"]
    checks["A3"]["end_to_end_note"] = (
        "the end-to-end loss table cannot resolve this effect on this batch: two "
        "runs of the same variant already differ in every cls-type loss, so the "
        "positive form of A3 ('only sup2_loss_cls changed, denominator held') is "
        "established at head level instead; end to end only the negative form "
        "holds (no other key moves beyond the same-variant noise floor)")
    checks["A3"]["passed"] = bool(checks["A3"]["passed"]
                                  and head_level["passed_lambda1"])
    checks["A5"]["head_level"] = head_level
    checks["A5"]["end_to_end_passed"] = checks["A5"]["passed"]
    checks["A5"]["passed"] = bool(checks["A5"]["passed"]
                                  and head_level["passed_lambda0"])

    guardrails = dict(
        instrumentation=instrumentation_proof,
        optimizer_step_calls=counters["optimizer_step"],
        optimizer_zero_grad_calls=counters["optimizer_zero_grad"],
        ema_momentum_update_calls=counters["ema_momentum_update"],
        ema_before_train_iter_calls=counters["ema_before_train_iter"],
        ema_before_run_calls=counters["ema_before_run"],
        ema_after_train_iter_calls=counters["ema_after_train_iter"],
        runs_with_parameters_unchanged=[
            "{}|{}".format(pass_name, variant)
            for pass_name in passes for variant in per_pass[pass_name]
            if per_pass[pass_name][variant]["params_unchanged"]],
        runs_with_gradients=[
            "{}|{}".format(pass_name, variant)
            for pass_name in passes for variant in per_pass[pass_name]
            if per_pass[pass_name][variant]["grads_present"]],
        buffers_changed=[
            dict(pass_name=pass_name, variant=variant,
                 n_changed=per_pass[pass_name][variant]["n_changed_buffers"])
            for pass_name in passes for variant in per_pass[pass_name]],
    )
    guardrails["no_optimizer_or_ema_update"] = (guardrails["optimizer_step_calls"] == 0
                                                and guardrails["optimizer_zero_grad_calls"] == 0
                                                and guardrails["ema_momentum_update_calls"] == 0
                                                and guardrails["ema_before_train_iter_calls"] == 0
                                                and guardrails["ema_before_run_calls"] == 0
                                                and guardrails["ema_after_train_iter_calls"] == 0)
    guardrails["no_backward"] = not guardrails["runs_with_gradients"]
    n_runs = sum(len(per_pass[pass_name]) for pass_name in passes)
    guardrails["all_parameters_bitwise_unchanged"] = (
        len(guardrails["runs_with_parameters_unchanged"]) == n_runs)

    acceptance = dict(
        status=("passed" if all(check["passed"] for check in checks.values())
                and guardrails["no_optimizer_or_ema_update"]
                and guardrails["no_backward"]
                and guardrails["all_parameters_bitwise_unchanged"]
                else "failed"),
        checks=checks,
        guardrails=guardrails,
        precision_passes=passes,
        batch_index=batch_index,
        batch_tags=batch_tags,
        sup2_image_index=sup2_index,
        config=dict(path=str(Path(args.config).resolve()), sha256=sha256_file(args.config)),
        emitted_variant_config=dict(path=str(variant_path),
                                    sha256=sha256_file(variant_path),
                                    config_diff_paths=diff_paths),
        checkpoint=dict(path=str(Path(args.checkpoint).resolve()),
                        sha256=sha256_file(args.checkpoint)),
        seed=seed, command=command, cwd=str(Path.cwd()),
        python=sys.version.split()[0],
        torch=torch.__version__, cuda=torch.version.cuda,
        elapsed_seconds=time.time() - started,
    )
    write_json(out_dir / "A_acceptance.json", acceptance)
    write_json(out_dir / "loss_tables.json",
               {pass_name: {variant: per_pass[pass_name][variant]["losses"]
                            for variant in per_pass[pass_name]} for pass_name in passes})
    write_json(out_dir / "loss_comparisons.json", per_pass_comparisons)
    write_json(out_dir / "head_level_evidence.json", head_level)
    write_json(out_dir / "reweight_probe.json", per_pass_probe)
    write_json(out_dir / "run_evidence.json", {
        "{}|{}".format(pass_name, variant): dict(
            elapsed_seconds=per_pass[pass_name][variant]["elapsed_seconds"],
            total_loss=per_pass[pass_name][variant]["total_loss"],
            checkpoint_loaded_strict=per_pass[pass_name][variant]["checkpoint_loaded_strict"],
            rng_before=per_pass[pass_name][variant]["rng_before"],
            counter_deltas=per_pass[pass_name][variant]["counter_deltas"],
            n_params=per_pass[pass_name][variant]["n_params"],
            n_trainable=per_pass[pass_name][variant]["n_trainable"],
            param_groups=per_pass[pass_name][variant]["param_groups"],
            params_unchanged=per_pass[pass_name][variant]["params_unchanged"],
            n_changed_buffers=per_pass[pass_name][variant]["n_changed_buffers"],
            changed_buffers=per_pass[pass_name][variant]["changed_buffers"],
            grads_present=per_pass[pass_name][variant]["grads_present"],
            reweight_log_counts={key: len(value)
                                 for key, value in per_pass[pass_name][variant]["logs"].items()},
        )
        for pass_name in passes for variant in per_pass[pass_name]})

    logger.info("[acceptance] status=%s", acceptance["status"])
    for name in sorted(checks):
        logger.info("[acceptance] %s: %s", name,
                    "PASS" if checks[name]["passed"] else "FAIL")
    print(json.dumps({name: checks[name]["passed"] for name in sorted(checks)},
                     sort_keys=True))
    print(json.dumps({key: value for key, value in guardrails.items()
                      if isinstance(value, (bool, int))}, sort_keys=True))
    print("[acceptance] status=%s -> %s" % (acceptance["status"], out_dir / "A_acceptance.json"))
    if acceptance["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
