"""Supplementary verification for the sup2 small-background ROI reweighting.

Three scoped checks, in this order:

1.  Make B0 and B0-repeat comparable.  One batch is fixed; before *every*
    replay the model is reloaded, the buffers are restored, the Python / NumPy /
    Torch-CPU / CUDA RNG states are restored from a single shared snapshot and
    the scattered inputs are rebuilt.  Every tensor that crosses a
    randomness-relevant boundary is recorded in execution order (rpn output ->
    proposals -> the `torch.randperm` choice -> sampled indices/labels ->
    head inputs/targets -> loss) so the first divergence can be located, not
    just observed.  Three policies are traced:

      fixed      one snapshot, restored before every replay
      entry      the retained acceptance harness's behaviour: take a fresh
                 snapshot at the start of each replay and restore *that*, so
                 replay N continues replay N-1's stream
      cuda_only  restore the CUDA stream only, leaving the CPU streams running

    The retained run's `rng_before` fingerprints differ across all 8 replays;
    that, not the model, is what produced the 0.586 / 0.195 spread.

2.  Only once `fixed` replays agree, attribute lambda=0 / lambda=1 against B0
    end to end, with the pre-registered tolerance 0.0 (bitwise).  lambda=0 must
    reproduce B0 exactly; lambda=1 must change `sup2_loss_cls` and nothing else,
    by the amount the frozen head-level basis predicts.

3.  One backward with no weight update.  On frozen head-level inputs, the
    reweighted rows' classification-logit gradients must equal the unweighted
    ones times the weight factor; on the full training-precision path the
    gradients must be finite and the teachers must carry none.  No optimizer
    step and no EMA update is executed anywhere in this file.

This writes into a fresh --out-dir; the retained acceptance artifacts under
`ablation_configs/reweight_small_bkg_20260929/` are read-only inputs.
"""

import argparse
import collections
import copy
import gc
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

# Reuse the retained harness' validated helpers (source receipt, optimizer/EMA
# counters with their positive self-test, state fingerprints, json writer). Its
# `main` is guarded, so importing it only defines names.
_HARNESS_PATH = ROOT / "tools" / "verify_small_bkg_reweight.py"
_hspec = importlib.util.spec_from_file_location("_reweight_harness", _HARNESS_PATH)
_harness = importlib.util.module_from_spec(_hspec)
_hspec.loader.exec_module(_harness)

_entry = _harness._entry
sha256_file = _harness.sha256_file
digest_tensor = _harness.digest_tensor
state_fingerprint = _harness.state_fingerprint
rng_fingerprint = _harness.rng_fingerprint
write_json = _harness.write_json
as_float = _harness.as_float
clone_sampling_results = _harness.clone_sampling_results
rebuild_sampling_results = _harness.rebuild_sampling_results
install_instrumentation = _harness.install_instrumentation
install_capture = _harness.install_capture
REQUIRED_TAGS = _harness.REQUIRED_TAGS
BASELINE_TAG = _harness.BASELINE_TAG
MAX_AREA = _harness.MAX_AREA
COUNTER_KEYS = _harness.COUNTER_KEYS
VARIANT_CONFIG_NAME = _harness.VARIANT_CONFIG_NAME
EXPECTED_CONFIG_DIFFS = _harness.EXPECTED_CONFIG_DIFFS
RETAINED_OUT_DIR = ROOT / "ablation_configs" / "reweight_small_bkg_20260929"

# ------------------------------------------------------------------ constants
# pre-registered, not to be relaxed because a run happened to be noisy
TOLERANCE = 0.0            # "unchanged" claims are required to be bitwise
GRAD_RATIO_TOL = 1e-6      # algebraic identity in fp32, not a sampled quantity
EXPECTED_SUP2_WEIGHT = 0.2
SUP2_WEIGHT_TOL = 1e-6
POLICIES = ("fixed", "entry", "cuda_only")


# ------------------------------------------------------------------- rng state

def snapshot_rng():
    """A detached copy of every RNG stream the sampler and the model read."""
    return (random.getstate(),
            copy.deepcopy(np.random.get_state()),
            torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()])


def restore_rng(snapshot, include=("python", "numpy", "torch_cpu", "torch_cuda")):
    if "python" in include:
        random.setstate(snapshot[0])
    if "numpy" in include:
        # set_state keeps a reference to the array it is handed, so pass a copy
        np.random.set_state(copy.deepcopy(snapshot[1]))
    if "torch_cpu" in include:
        torch.set_rng_state(snapshot[2].clone())
    if "torch_cuda" in include:
        torch.cuda.set_rng_state_all([state.clone() for state in snapshot[3]])


def cpu_rng_sha256():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------- trace
# One ordered event list per forward. Every event keeps the tensors that crossed
# the boundary, so a divergence can be reported with its magnitude and not just
# its position.
TRACE = dict(installed=False, originals={}, events=None, names={}, choice_calls=0)

SAMPLE_FIELDS = ("pos_inds", "neg_inds", "pos_assigned_gt_inds", "pos_bboxes",
                 "neg_bboxes", "pos_gt_bboxes", "pos_gt_labels")


def _flatten(obj, out):
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten(item, out)
    elif isinstance(obj, dict):
        for key in sorted(obj):
            _flatten(obj[key], out)
    return out


def _label(module):
    return TRACE["names"].get(id(module))


def emit(kind, fields, tensors, branch=None, extra=None):
    events = TRACE["events"]
    if events is None:
        return
    kept = [(name, tensor) for name, tensor in zip(fields, tensors)
            if torch.is_tensor(tensor)]
    event = dict(
        seq=len(events), kind=kind, branch=branch,
        fields=[name for name, _ in kept],
        shapes=[list(tensor.shape) for _, tensor in kept],
        dtypes=[str(tensor.dtype).replace("torch.", "") for _, tensor in kept],
        digests=[digest_tensor(tensor) for _, tensor in kept],
    )
    if extra:
        event.update(extra)
    # `payload` keeps the original dtype: int64 indices must stay exact, and a
    # half tensor must be compared as half.
    events.append(dict(event=event,
                       payload=[tensor.detach().cpu() for _, tensor in kept]))


def install_trace():
    """Wrap each boundary that decides, or consumes, the sampled set."""
    if TRACE["installed"]:
        return
    from mmdet.core.bbox.samplers.base_sampler import BaseSampler
    from mmdet.core.bbox.samplers.random_sampler import RandomSampler
    from mmdet.models.dense_heads.rpn_head import RPNHead
    from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead

    def keep(cls, name):
        TRACE["originals"][(cls, name)] = getattr(cls, name)

    # this vendored mmdet has no RegionProposalNetwork; `self.rpn` is an RPNHead
    # whose `forward` is inherited from AnchorHead, so bind it on RPNHead to
    # keep the wrapper narrow
    keep(RPNHead, "forward")

    def rpn_forward(self, x):
        out = TRACE["originals"][(RPNHead, "forward")](self, x)
        tensors = _flatten(out, [])
        emit("rpn_raw", ["out_%d" % i for i in range(len(tensors))], tensors,
             _label(self), dict(n_tensors=len(tensors)))
        return out

    RPNHead.forward = rpn_forward

    keep(RPNHead, "get_bboxes")

    def rpn_get_bboxes(self, *args, **kwargs):
        out = TRACE["originals"][(RPNHead, "get_bboxes")](self, *args, **kwargs)
        tensors = _flatten(out, [])
        emit("rpn_proposals", ["image_%d" % i for i in range(len(tensors))],
             tensors, _label(self))
        return out

    RPNHead.get_bboxes = rpn_get_bboxes

    # the actual random consumer: `torch.randperm` on the CPU torch stream
    keep(RandomSampler, "random_choice")

    def random_choice(self, gallery, num):
        TRACE["choice_calls"] += 1
        before = cpu_rng_sha256()
        out = TRACE["originals"][(RandomSampler, "random_choice")](
            self, gallery, num)
        after = cpu_rng_sha256()
        emit("sampler_choice", ["gallery", "chosen"], [gallery, out],
             _label(self),
             dict(call_index=TRACE["choice_calls"] - 1, num=int(num),
                  gallery_numel=int(gallery.numel()),
                  rng_cpu_before_sha256=before, rng_cpu_after_sha256=after))
        return out

    RandomSampler.random_choice = random_choice

    keep(BaseSampler, "sample")

    def sample(self, *args, **kwargs):
        res = TRACE["originals"][(BaseSampler, "sample")](self, *args, **kwargs)
        emit("sampler_result", SAMPLE_FIELDS,
             [getattr(res, name, None) for name in SAMPLE_FIELDS], _label(self),
             dict(num_pos=int(res.pos_inds.numel()),
                  num_neg=int(res.neg_inds.numel()),
                  num_proposals=(int(args[1].size(0)) if len(args) > 1 else None),
                  sampler=type(self).__name__))
        return res

    BaseSampler.sample = sample

    # single emit point for both the sampled targets and the head inputs
    keep(BBoxHead, "loss")

    def bbox_loss(self, cls_score, bbox_pred, rois, labels, label_weights,
                  bbox_targets, bbox_weights, reduction_override=None):
        emit("head_loss_inputs",
             ["cls_score", "bbox_pred", "rois", "labels", "label_weights",
              "bbox_targets", "bbox_weights"],
             [cls_score, bbox_pred, rois, labels, label_weights, bbox_targets,
              bbox_weights],
             _label(self))
        return TRACE["originals"][(BBoxHead, "loss")](
            self, cls_score, bbox_pred, rois, labels, label_weights,
            bbox_targets, bbox_weights, reduction_override)

    BBoxHead.loss = bbox_loss
    TRACE["installed"] = True


def build_name_map(model):
    names = {}
    for branch in ("teacher1", "teacher2", "student1", "student2"):
        sub = getattr(model, branch, None)
        if sub is None:
            continue
        rpn = getattr(sub, "rpn", None)
        if rpn is not None:
            names[id(rpn)] = branch + "|rpn"
            # the RPN has its own assigner/sampler (AnchorHead builds them)
            for attribute in ("sampler", "assigner", "bbox_sampler"):
                obj = getattr(rpn, attribute, None)
                if obj is not None:
                    names[id(obj)] = "%s|rpn_%s" % (branch, attribute)
        roi_head = getattr(sub, "roi_head", None)
        if roi_head is None:
            continue
        head = getattr(roi_head, "bbox_head", None)
        if head is not None:
            names[id(head)] = branch + "|bbox_head"
        for owner in (roi_head, head):
            for attribute in ("bbox_sampler", "sampler", "bbox_assigner"):
                obj = getattr(owner, attribute, None)
                if obj is not None:
                    names[id(obj)] = "%s|%s" % (branch, attribute)
    return names


def compare_traces(left, right, tolerance):
    """Ordered comparison; returns the first divergence plus every change."""
    if len(left) != len(right):
        return dict(structural_match=False, n_events_left=len(left),
                    n_events_right=len(right), first_divergence=None,
                    n_changed_events=None, changed_events=None)
    changed, first = [], None
    for (left_item, right_item) in zip(left, right):
        left_event = left_item["event"]
        right_event = right_item["event"]
        structural = (left_event["kind"] == right_event["kind"]
                      and left_event["branch"] == right_event["branch"]
                      and left_event["fields"] == right_event["fields"]
                      and left_event["shapes"] == right_event["shapes"]
                      and left_event["dtypes"] == right_event["dtypes"])
        if not structural:
            if first is None:
                first = dict(seq=left_event["seq"], kind=left_event["kind"],
                             branch=left_event["branch"], reason="structure",
                             left_fields=left_event["fields"],
                             right_fields=right_event["fields"])
            changed.append(dict(seq=left_event["seq"], kind=left_event["kind"],
                                branch=left_event["branch"],
                                reason="structure"))
            continue
        if left_event["digests"] == right_event["digests"]:
            continue
        details, changed_fields = [], []
        for index, field in enumerate(left_event["fields"]):
            left_tensor = left_item["payload"][index]
            right_tensor = right_item["payload"][index]
            if left_event["digests"][index] == right_event["digests"][index]:
                continue
            changed_fields.append(field)
            row = dict(field=field, shape=list(left_tensor.shape),
                       dtype=left_event["dtypes"][index])
            if left_tensor.shape == right_tensor.shape:
                both = left_tensor.double(), right_tensor.double()
                diff = (both[0] - both[1]).abs()
                row.update(max_abs_diff=float(diff.max()) if diff.numel() else 0.0,
                           n_differing=int((left_tensor != right_tensor).sum().item()),
                           n_elements=int(left_tensor.numel()))
            else:
                row.update(reason="shape_mismatch",
                           right_shape=list(right_tensor.shape))
            details.append(row)
        changed.append(dict(seq=left_event["seq"], kind=left_event["kind"],
                            branch=left_event["branch"], fields=changed_fields,
                            detail=details))
        if first is None:
            first = dict(seq=left_event["seq"], kind=left_event["kind"],
                         branch=left_event["branch"], reason="value",
                         fields=changed_fields, detail=details)
    return dict(structural_match=True, n_events_left=len(left),
                n_events_right=len(right), n_changed_events=len(changed),
                first_divergence=first, changed_events=changed[:12],
                tolerance=tolerance)


def trace_brief(trace):
    return dict(n_events=len(trace),
                event_kinds=dict(collections.Counter(
                    item["event"]["kind"] for item in trace)),
                sampler_choice_calls=sum(
                    1 for item in trace if item["event"]["kind"] == "sampler_choice"))


def attribution_of(comparison):
    """Which `lambda` change is allowed to touch which trace event.

    A correct lambda=1 must change exactly two recorded things: the sup2 head's
    `label_weights` (the reweighted channel) and the `sup2_loss_cls` entry of
    the final loss table. Everything else -- proposals, the `randperm` choices,
    the sampled indices, the positives' weights, the regression targets -- must
    be bitwise identical.
    """
    changed = comparison["changed_events"]
    head_events = [event for event in changed if event["kind"] == "head_loss_inputs"]
    loss_events = [event for event in changed if event["kind"] == "losses"]
    other_events = [event for event in changed
                    if event["kind"] not in ("head_loss_inputs", "losses")]
    head_ok = bool(head_events) and all(
        event["branch"] == "student2|bbox_head" and event["fields"] == ["label_weights"]
        for event in head_events)
    loss_ok = (len(loss_events) == 1
               and loss_events[0]["fields"] == ["sup2_loss_cls[0]"])
    return dict(
        n_changed_events=len(changed),
        head_events=[dict(branch=event["branch"], fields=event["fields"])
                     for event in head_events],
        loss_events=[dict(fields=event["fields"]) for event in loss_events],
        other_events=[dict(branch=event["branch"], kind=event["kind"],
                           fields=event["fields"]) for event in other_events],
        only_sup2_label_weights_changed=head_ok,
        only_sup2_loss_cls_changed=loss_ok,
        no_other_events_changed=not other_events,
    )


# ----------------------------------------------------------------------- runs

def run_once(cfg, dataset, batch, checkpoint, device, counters, pass_name,
             policy, fixed_snapshot=None, capture=None, backward=False,
             keep_trace=True):
    """One forward under an explicit RNG policy. Never steps an optimizer."""
    from mmcv.parallel import scatter
    from mmcv.runner import build_optimizer, load_checkpoint, wrap_fp16_model
    from mmdet.models import build_detector

    model = build_detector(cfg.model)
    from ssod.models.roi_heads.small_bkg_reweight import enable_reweight_diagnostics
    enable_reweight_diagnostics(model)
    if pass_name == "train":
        wrap_fp16_model(model)
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model._pretrained_initialized = True
    model.CLASSES = dataset.CLASSES
    model.cuda(device)
    model.train()
    model.freeze("teacher1")
    model.freeze("teacher2")
    # built so the step counter is meaningful; it is never stepped
    optimizer = build_optimizer(model, cfg.optimizer)

    TRACE["names"] = build_name_map(model)
    TRACE["choice_calls"] = 0
    events = [] if keep_trace else None
    params_before, buffers_before = state_fingerprint(model)

    if policy == "fixed":
        restore_rng(fixed_snapshot)
    elif policy == "entry":
        # the retained harness: snapshot the current state and restore it, i.e.
        # continue whatever stream the previous replay left behind
        restore_rng(snapshot_rng())
    elif policy == "cuda_only":
        restore_rng(fixed_snapshot, include=("torch_cuda",))
    else:
        raise ValueError("unknown rng policy: %s" % policy)
    rng_before = rng_fingerprint()

    if capture is not None:
        capture["target"] = getattr(model.student2, "roi_head", None)
    inputs = scatter(copy.deepcopy(batch), [device])[0]
    before_counters = dict(counters)
    TRACE["events"] = events
    started = time.time()
    try:
        with torch.cuda.amp.autocast(enabled=(pass_name == "train")):
            losses = model(return_loss=True, **inputs)
        elapsed = time.time() - started

        total_loss, parsed = model._parse_losses(losses)
        loss_values = {key: as_float(value) for key, value in parsed.items()
                       if key != "loss"}
        total_loss_value = as_float(total_loss)

        # the raw per-image losses, one tensor per key per image; recorded while
        # the trace is still armed, so the loss table is part of the same order
        if events is not None:
            fields, tensors = [], []
            for key in sorted(losses):
                items = (losses[key] if isinstance(losses[key], (list, tuple))
                         else [losses[key]])
                for index, item in enumerate(items):
                    if torch.is_tensor(item):
                        fields.append("%s[%d]" % (key, index))
                        tensors.append(item)
            emit("losses", fields, tensors, "all",
                 dict(parsed=loss_values, total_loss=total_loss_value))
    finally:
        TRACE["events"] = None

    sup2_capture = None
    if capture is not None:
        sup2_capture = capture["hits"][0] if capture["hits"] else None
        capture["hits"] = []
        capture["target"] = None

    backward_info = None
    if backward:
        total_loss.backward()
        grad_records = [(name, param.grad)
                        for name, param in model.named_parameters()
                        if param.grad is not None]
        with_inf, with_nan = [], []
        for name, grad in grad_records:
            if bool(torch.isinf(grad).any().item()):
                with_inf.append(name)
            if bool(torch.isnan(grad).any().item()):
                with_nan.append(name)
        teacher_params = [(name, param) for name, param in model.named_parameters()
                          if name.split(".")[0].startswith("teacher")]
        student_with_grad = [name for name, param in model.named_parameters()
                             if name.split(".")[0].startswith("student")
                             and param.grad is not None]
        backward_info = dict(
            pass_name=pass_name,
            total_loss_value=total_loss_value,
            total_loss_requires_grad=bool(total_loss.requires_grad),
            grad_scale_applied=False,
            grad_check_scope="every grad tensor present after one backward",
            n_grad_tensors=len(grad_records),
            n_grad_tensors_with_inf=len(with_inf),
            n_grad_tensors_with_nan=len(with_nan),
            grad_tensors_with_inf=with_inf[:10],
            grad_tensors_with_nan=with_nan[:10],
            grad_inf_sampled_this_run=bool(with_inf or with_nan),
            n_teacher_params=len(teacher_params),
            n_teacher_params_with_grad=sum(1 for _, param in teacher_params
                                          if param.grad is not None),
            n_teacher_params_requiring_grad=sum(1 for _, param in teacher_params
                                               if param.requires_grad),
            n_student_params_with_grad=len(student_with_grad),
            student_params_with_grad_example=student_with_grad[:5],
            nonfinite_loss_keys=sorted(key for key, value in loss_values.items()
                                       if not np.isfinite(value)),
        )

    params_after, buffers_after = state_fingerprint(model)
    changed_params = sorted(k for k in params_before
                            if params_before[k] != params_after.get(k))
    changed_buffers = sorted(k for k in buffers_before
                             if buffers_before[k] != buffers_after.get(k))
    counter_deltas = {key: counters[key] - before_counters.get(key, 0)
                      for key in COUNTER_KEYS}
    logs = {}
    for branch in ("student1", "student2"):
        head = getattr(getattr(model, branch), "roi_head", None)
        head = getattr(head, "bbox_head", None)
        if head is not None and hasattr(head, "reweight_log"):
            logs[branch] = copy.deepcopy(head.reweight_log)

    del losses, inputs, optimizer, model
    torch.cuda.empty_cache()
    gc.collect()
    return dict(
        pass_name=pass_name, policy=policy, losses=loss_values,
        total_loss=total_loss_value, rng_before=rng_before, trace=events,
        sup2_capture=sup2_capture, elapsed_seconds=elapsed,
        counter_deltas=counter_deltas,
        params_unchanged=(not changed_params), n_changed_buffers=len(changed_buffers),
        changed_params=changed_params[:5], changed_buffers=changed_buffers[:5],
        backward=backward_info, logs=logs,
    )


# ---------------------------------------------------------------- head level

def head_loss_stage(cfg_b0, cfg_lambda0, cfg_lambda1, checkpoint, capture,
                    lambda_value):
    """Frozen-basis classification loss and its gradient, per weight variant.

    `force_fp32(apply_to=('cls_score', 'bbox_pred'))` on `BBoxHead.loss` means
    the fp16 path's classification loss is evaluated on the fp32 upcast of the
    captured cls_score, so this CPU stage predicts the amp end-to-end value too.

    The gradient w.r.t. `cls_score` is exactly the loss weight times the softmax
    residual over `avg_factor`, so on frozen inputs the reweighted rows' logit
    gradients must equal the unweighted ones times the weight factor, row by
    row.  That is the formula this returns.
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

    cls_score = capture["cls_score"].float()
    rois = capture["rois"].float()
    sampling_results = rebuild_sampling_results(capture["sampling_results"])
    gt_bboxes, gt_labels = capture["gt_bboxes"], capture["gt_labels"]
    metas = capture["img_metas"]
    if [meta["tag"] for meta in metas] != [BASELINE_TAG] * len(metas):
        raise RuntimeError("the captured branch is not the %s stream" % BASELINE_TAG)
    reweight_ctx = dict(scale_factors=[meta["scale_factor"] for meta in metas],
                        tags=[meta["tag"] for meta in metas])

    losses, grads, weights, avg_factors = {}, {}, {}, {}
    for name, head in heads.items():
        extra = {} if name == "b0" else dict(reweight_ctx=reweight_ctx)
        targets = head.get_targets(sampling_results, gt_bboxes, gt_labels,
                                   rcnn_cfg, **extra)
        weights[name] = targets[1]
        avg_factors[name] = max(float(torch.sum(targets[1] > 0)), 1.)
        # bbox_pred=None isolates the classification loss from the regression one
        leaf = cls_score.clone().detach().requires_grad_(True)
        loss_cls = head.loss(leaf, None, rois, *targets)["loss_cls"]
        losses[name] = as_float(loss_cls)
        grads[name] = torch.autograd.grad(loss_cls, leaf)[0]

    weight_b0, weight_l0, weight_l1 = weights["b0"], weights["lambda0"], weights["lambda1"]
    contributing = weight_b0 > 0                      # ignored rows drop out
    reweighted = (weight_l1 / weight_b0.clamp(min=1e-12)) == 1.0 + lambda_value
    untouched = (weight_l1 / weight_b0.clamp(min=1e-12)) == 1.0

    # The formula, element by element: dL_cls/dz = w_i / avg_factor * (softmax - 1)
    # with the same cls_score, the same targets and the same avg_factor, so
    # lambda=1 must satisfy g_l1 == factor_i * g_b0 exactly. A ratio would divide
    # by saturated-softmax zeros, so test the residual instead.
    grad_b0 = grads["b0"][contributing]
    grad_l1 = grads["lambda1"][contributing]
    weight_ratio = weight_l1[contributing] / weight_b0[contributing]
    residual = (grad_l1 - weight_ratio.unsqueeze(1) * grad_b0).abs()
    nonzero_b0 = grad_b0 != 0
    zero_b0 = ~nonzero_b0
    n_zero_b0_elements = int(zero_b0.sum())
    zero_b0_consistent = bool((grad_l1[zero_b0] == 0).all()) if n_zero_b0_elements else True
    grad_ratio = (grad_l1[nonzero_b0] / grad_b0[nonzero_b0])
    nonzero_ratio = weight_ratio.unsqueeze(1).expand_as(grad_l1)[nonzero_b0]
    reweighted_columns = reweighted[contributing].unsqueeze(1).expand_as(grad_l1)[nonzero_b0]

    # one record per sampled image; the negative rows are the trailing `num_neg`
    # rows of the concatenated targets. Only the sup2 image is eligible here.
    records = heads["lambda1"].reweight_log
    num_pos = sum(int(r["num_pos"]) for r in records)
    num_neg = sum(int(r["num_neg"]) for r in records)
    log_n_reweighted = sum(int(r.get("n_reweighted", 0)) for r in records)
    factors_seen = sorted({value for record in records
                           for value in record.get("factors", [])})
    return dict(
        basis="frozen head-level tensors from the same fixed batch",
        formula="d(loss_cls)/d(cls_score)[i, c] = w_i / avg_factor * (softmax_i - onehot_i)",
        cls_score_shape=list(cls_score.shape),
        num_rows=int(weight_b0.numel()),
        num_pos=num_pos, num_neg=num_neg,
        num_contributing_rows=int(contributing.sum()),
        avg_factor=avg_factors,
        denominators_equal=bool(avg_factors["b0"] == avg_factors["lambda1"]
                                == avg_factors["lambda0"]),
        losses=losses,
        loss_cls_delta_lambda1_vs_b0=losses["lambda1"] - losses["b0"],
        loss_cls_bitwise_equal_lambda0_vs_b0=bool(losses["lambda0"] == losses["b0"]),
        weights_bitwise_equal_lambda0_vs_b0=bool((weight_l0 == weight_b0).all()),
        weights_bitwise_equal_lambda1_vs_b0=bool((weight_l1 == weight_b0).all()),
        grads_bitwise_equal_lambda0_vs_b0=bool(torch.equal(grads["lambda0"],
                                                           grads["b0"])),
        supervision_unchanged=dict(
            positives_weight_one_in_b0=bool((weight_b0[:num_pos] == 1.0).all()),
            positives_weight_unchanged=bool(
                (weight_l1[:num_pos] == weight_b0[:num_pos]).all()),
            reweighted_rows_inside_negatives=bool(
                int(reweighted[num_pos:].sum()) == int(reweighted.sum())),
            reweighted_rows_match_reweight_log=bool(
                int(reweighted.sum()) == log_n_reweighted),
        ),
        n_rows_with_weight_ratio_2=int(reweighted.sum()),
        n_rows_with_weight_ratio_1=int(untouched.sum()),
        factors_seen=factors_seen,
        grad_ratio_tolerance=GRAD_RATIO_TOL,
        grad_residual_max_abs=float(residual.max()),
        grad_residual_bitwise_zero=bool(float(residual.max()) == 0.0),
        n_zero_grad_b0_elements=n_zero_b0_elements,
        zero_grad_b0_rows_also_zero_under_lambda1=zero_b0_consistent,
        grad_ratio_max_abs_deviation_over_nonzero=(
            float((grad_ratio - nonzero_ratio).abs().max())
            if int(nonzero_b0.sum()) else None),
        grad_ratio_all_rows_follow_weight=bool(
            float(residual.max()) <= GRAD_RATIO_TOL and zero_b0_consistent),
        reweighted_grad_ratio_min=(float(grad_ratio[reweighted_columns].min())
                                  if int(reweighted_columns.sum()) else None),
        reweighted_grad_ratio_max=(float(grad_ratio[reweighted_columns].max())
                                  if int(reweighted_columns.sum()) else None),
        untouched_grad_ratio_min=(
            float(grad_ratio[~reweighted_columns].min())
            if int((~reweighted_columns).sum()) else None),
        untouched_grad_ratio_max=(
            float(grad_ratio[~reweighted_columns].max())
            if int((~reweighted_columns).sum()) else None),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ablation_configs/fold6_seed678/b0.py")
    parser.add_argument("--checkpoint",
                        default="work_dirs/ablation_v1/fold6_seed678/b0/iter_32000.pth")
    parser.add_argument(
        "--out-dir",
        default="ablation_configs/reweight_small_bkg_20260929_supplement")
    parser.add_argument("--passes", default="fp32,train")
    parser.add_argument("--scan-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing to write into a non-empty directory: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    passes = [name for name in args.passes.split(",") if name]
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
    from mmdet.models import build_detector  # noqa: F401 (registers pipelines)
    from mmdet.models.builder import HEADS
    from ssod.apis import set_random_seed
    from ssod.datasets import build_dataloader, build_dataset
    from ssod.utils import get_root_logger, patch_config
    from ssod.utils.ablation import differences

    importlib.import_module(_harness.NEW_MODULE)  # register the classes

    started = time.time()
    logger = get_root_logger(log_file=str(out_dir / "harness.log"), log_level="INFO")
    logger.info("[supplement] command: %s", " ".join(
        [sys.executable, "tools/verify_reweight_supplement.py"] + sys.argv[1:]))

    receipt = _entry.source_manifest()
    receipt["extensions"] = dict(
        new_module=_harness.NEW_MODULE,
        supplement_harness=dict(path=str(Path(__file__).resolve()),
                                sha256=sha256_file(__file__)),
        retained_harness=dict(path=str(_HARNESS_PATH.resolve()),
                              sha256=sha256_file(_HARNESS_PATH)),
        train_ablation=dict(path=str(ROOT / "tools" / "train_ablation.py"),
                            sha256=sha256_file(ROOT / "tools" / "train_ablation.py")),
        retained_acceptance=dict(
            path=str(RETAINED_OUT_DIR),
            verdict_sha256=(sha256_file(RETAINED_OUT_DIR / "A_acceptance.json")
                            if (RETAINED_OUT_DIR / "A_acceptance.json").exists()
                            else None)),
    )
    write_json(out_dir / "source_receipt.json", receipt)

    counters, instrumentation_proof = install_instrumentation()
    logger.info("[supplement] instrumentation self-test: %s",
                json.dumps(instrumentation_proof, sort_keys=True))

    seed = args.seed if args.seed is not None else Config.fromfile(args.config).get("seed")
    cfg_b0 = patch_config(Config.fromfile(args.config))
    if (cfg_b0.model.model.roi_head.type != "StandardRoIHead"
            or cfg_b0.model.model.roi_head.bbox_head.type != "Shared2FCBBoxHead"):
        raise SystemExit("the reference config is not an unmodified B0 roi_head")
    os.environ["WORK_DIR"] = str(out_dir / "work_dir_override")
    (out_dir / "work_dir_override").mkdir(exist_ok=True)

    cfg_lambda1 = patch_config(Config.fromfile(args.config))
    cfg_lambda1.custom_imports = dict(imports=[_harness.NEW_MODULE],
                                      allow_failed_imports=False)
    cfg_lambda1.model.model.roi_head.type = "SmallBkgReweightRoIHead"
    cfg_lambda1.model.model.roi_head.bbox_head.type = "SmallBkgReweightBBoxHead"
    cfg_lambda1.model.model.roi_head.bbox_head.reweight = dict(
        enable=True, lambda_=1.0, max_area=MAX_AREA, tag=BASELINE_TAG)
    variant_path = out_dir / VARIANT_CONFIG_NAME
    cfg_lambda1.dump(str(variant_path))
    diff_paths = sorted(row["path"] for row in
                        differences(dict(cfg_b0), dict(cfg_lambda1)))
    if set(diff_paths) != EXPECTED_CONFIG_DIFFS:
        raise SystemExit("unexpected config difference set: %s" % diff_paths)
    cfg_lambda1 = patch_config(Config.fromfile(str(variant_path)))
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
        probe = scatter(copy.deepcopy(candidate), [device])[0]
        tags = [str(meta["tag"]) for meta in probe["img_metas"]]
        del probe
        if sorted(tags) == sorted(REQUIRED_TAGS):
            batch, batch_index, batch_tags = candidate, index, tags
            break
    if batch is None:
        raise SystemExit("no scanned batch covered all four supervision streams")
    logger.info("[supplement] batch %d tags=%s", batch_index, batch_tags)

    environment = dict(
        seed=seed,
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        batch_index=batch_index, batch_tags=batch_tags,
        checkpoint=dict(path=str(Path(args.checkpoint).resolve()),
                        sha256=sha256_file(args.checkpoint)),
        config=dict(path=str(Path(args.config).resolve()),
                    sha256=sha256_file(args.config)),
        tolerance=TOLERANCE, grad_ratio_tolerance=GRAD_RATIO_TOL,
        expected_sup2_weight=EXPECTED_SUP2_WEIGHT,
        device=torch.cuda.get_device_name(device),
    )

    install_trace()
    capture = install_capture()

    # ------------------------------------------------------------------ stage 1
    # One snapshot, taken after the batch scan and before any forward; every
    # `fixed` replay restores exactly this.
    fixed_snapshot = snapshot_rng()
    fixed_fingerprint = rng_fingerprint()
    logger.info("[supplement] stage 1: fixed snapshot %s", fixed_fingerprint)

    stage1 = dict(fixed_snapshot_fingerprint=fixed_fingerprint,
                  cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                  policies={})
    for policy in POLICIES:
        replays = []
        for replay in range(2):
            logger.info("[supplement] stage 1: policy=%s replay=%d", policy, replay)
            result = run_once(cfg_b0, dataset, batch, args.checkpoint, device,
                              counters, "fp32", policy,
                              fixed_snapshot=fixed_snapshot)
            replays.append(dict(replay=replay, rng_before=result["rng_before"],
                                total_loss=result["total_loss"],
                                trace=trace_brief(result["trace"]),
                                params_unchanged=result["params_unchanged"],
                                counter_deltas=result["counter_deltas"]))
            if replay == 0:
                first_trace = result["trace"]
            else:
                second_trace = result["trace"]
            del result
        comparison = compare_traces(first_trace, second_trace, TOLERANCE)
        stage1["policies"][policy] = dict(
            replays=replays, comparison=comparison,
            rng_restored_identically=(replays[0]["rng_before"]
                                      == replays[1]["rng_before"]),
            identical=(comparison["structural_match"]
                       and comparison["n_changed_events"] == 0))
        logger.info("[supplement] stage 1: policy=%s identical=%s first_divergence=%s",
                    policy, stage1["policies"][policy]["identical"],
                    json.dumps(comparison["first_divergence"], sort_keys=True)[:400])
        del first_trace, second_trace
        gc.collect()

    repeat_comparable = stage1["policies"]["fixed"]["identical"]

    # ------------------------------------------------------------------ stage 2
    stage2 = dict(passes={}, repeat_comparable=repeat_comparable)
    captures = {}
    if repeat_comparable:
        for pass_name in passes:
            logger.info("[supplement] stage 2: pass=%s", pass_name)
            runs = {}
            for variant, cfg in (("b0", cfg_b0), ("lambda0", cfg_lambda0),
                                 ("lambda1", cfg_lambda1)):
                want_capture = (capture if variant == "b0" else None)
                result = run_once(cfg, dataset, batch, args.checkpoint, device,
                                  counters, pass_name, "fixed",
                                  fixed_snapshot=fixed_snapshot,
                                  capture=want_capture)
                runs[variant] = result
                logger.info("[supplement] stage 2: pass=%s variant=%s loss=%.9f",
                            pass_name, variant, result["total_loss"])
            b0_trace = runs["b0"]["trace"]
            built = {}
            for variant in ("lambda0", "lambda1"):
                comparison = compare_traces(b0_trace, runs[variant]["trace"],
                                            TOLERANCE)
                changed_keys = sorted(key for key in runs["b0"]["losses"]
                                      if runs[variant]["losses"].get(key)
                                      != runs["b0"]["losses"][key])
                rows = {}
                for key in sorted(runs["b0"]["losses"]):
                    left = runs["b0"]["losses"][key]
                    right = runs[variant]["losses"].get(key)
                    rows[key] = dict(left=left, right=right,
                                     abs_diff=(None if right is None
                                               else abs(right - left)),
                                     bitwise_equal=(right == left))
                built[variant] = dict(
                    trace_comparison=comparison,
                    attribution=attribution_of(comparison),
                    changed_loss_keys=changed_keys,
                    total_loss=dict(b0=runs["b0"]["total_loss"],
                                    variant=runs[variant]["total_loss"],
                                    abs_diff=abs(runs[variant]["total_loss"]
                                                 - runs["b0"]["total_loss"])),
                    loss_rows=rows,
                    sup2_reweight_log=runs[variant]["logs"].get("student2", []),
                )
            captures[pass_name] = runs["b0"]["sup2_capture"]

            # the frozen head-level basis predicts the sup2 classification loss
            head_basis = head_loss_stage(cfg_b0, cfg_lambda0, cfg_lambda1,
                                         args.checkpoint, captures[pass_name], 1.0)
            sup2_b0 = runs["b0"]["losses"].get("sup2_loss_cls")
            sup2_l0 = runs["lambda0"]["losses"].get("sup2_loss_cls")
            scaling = (sup2_l0 / head_basis["losses"]["b0"]
                       if head_basis["losses"]["b0"] else None)
            predicted = (scaling * head_basis["loss_cls_delta_lambda1_vs_b0"]
                         if scaling is not None else None)
            measured = (runs["lambda1"]["losses"]["sup2_loss_cls"] - sup2_l0
                        if "sup2_loss_cls" in runs["lambda1"]["losses"] else None)
            stage2["passes"][pass_name] = dict(
                variants={name: dict(losses=runs[name]["losses"],
                                     total_loss=runs[name]["total_loss"],
                                     rng_before=runs[name]["rng_before"],
                                     params_unchanged=runs[name]["params_unchanged"],
                                     counter_deltas=runs[name]["counter_deltas"])
                          for name in runs},
                lambda0=built["lambda0"], lambda1=built["lambda1"],
                head_level_basis=head_basis,
                sup2_loss_cls_b0=sup2_b0,
                sup2_weight_observed=scaling,
                sup2_weight_matches_expected=(
                    scaling is not None
                    and abs(scaling - EXPECTED_SUP2_WEIGHT) <= SUP2_WEIGHT_TOL),
                predicted_sup2_loss_cls_delta=predicted,
                measured_sup2_loss_cls_delta=measured,
                prediction_error=(None if (predicted is None or measured is None)
                                  else abs(measured - predicted)),
                prediction_matches=(None if (predicted is None or measured is None)
                                    else abs(measured - predicted) <= 1e-9),
            )
            del runs, b0_trace
            gc.collect()
    else:
        logger.warning("[supplement] stage 1 not consistent; stage 2 not attempted")

    # ------------------------------------------------------------------ stage 3
    stage3 = dict(head_level=None, backward={}, repeat_comparable=repeat_comparable)
    if repeat_comparable and "fp32" in captures:
        logger.info("[supplement] stage 3a: head-level gradient")
        stage3["head_level"] = head_loss_stage(cfg_b0, cfg_lambda0, cfg_lambda1,
                                               args.checkpoint, captures["fp32"],
                                               1.0)
    for pass_name in passes:
        logger.info("[supplement] stage 3b: backward pass=%s", pass_name)
        result = run_once(cfg_b0, dataset, batch, args.checkpoint, device, counters,
                          pass_name, "fixed", fixed_snapshot=fixed_snapshot,
                          backward=True, keep_trace=False)
        stage3["backward"][pass_name] = result["backward"]
        stage3["backward"][pass_name].update(
            params_unchanged_after_backward=result["params_unchanged"],
            n_changed_buffers_after_backward=result["n_changed_buffers"],
            counter_deltas=result["counter_deltas"],
            optimizer_step_executed=bool(result["counter_deltas"]["optimizer_step"]),
            ema_update_executed=bool(result["counter_deltas"]["ema_momentum_update"]),
        )
        del result
        gc.collect()

    # ------------------------------------------------------------------ verdict
    a3_end_to_end = {}
    for pass_name, block in stage2["passes"].items():
        others = [key for key in block["lambda1"]["changed_loss_keys"]
                  if key not in ("sup2_loss_cls", "loss")]
        attribution = block["lambda1"]["attribution"]
        a3_end_to_end[pass_name] = dict(
            changed_loss_keys=block["lambda1"]["changed_loss_keys"],
            changed_keys_other_than_sup2_loss_cls=others,
            trace_attribution=attribution,
            sup2_loss_cls_increased=bool(
                block["measured_sup2_loss_cls_delta"] is not None
                and block["measured_sup2_loss_cls_delta"] > 0),
            measured_delta=block["measured_sup2_loss_cls_delta"],
            predicted_delta=block["predicted_sup2_loss_cls_delta"],
            sup2_weight_observed=block["sup2_weight_observed"],
            sup2_weight_matches_expected=block["sup2_weight_matches_expected"],
            prediction_matches=block["prediction_matches"],
            passed=bool(not others
                        and attribution["only_sup2_label_weights_changed"]
                        and attribution["only_sup2_loss_cls_changed"]
                        and attribution["no_other_events_changed"]
                        and block["sup2_weight_matches_expected"]
                        and block["prediction_matches"]),
        )
    a5_end_to_end = {}
    for pass_name, block in stage2["passes"].items():
        a5_end_to_end[pass_name] = dict(
            changed_loss_keys=block["lambda0"]["changed_loss_keys"],
            trace_attribution=block["lambda0"]["attribution"],
            max_loss_abs_diff=max(
                [row["abs_diff"] for row in block["lambda0"]["loss_rows"].values()
                 if row["abs_diff"] is not None] or [0.0]),
            total_loss_abs_diff=block["lambda0"]["total_loss"]["abs_diff"],
            passed=bool(not block["lambda0"]["changed_loss_keys"]
                        and block["lambda0"]["attribution"]["n_changed_events"] == 0),
        )

    figure_correction = dict(
        quoted_figure="about 2e-4",
        provenance=dict(
            file=str(RETAINED_OUT_DIR / "loss_comparisons.json"),
            rows=[
                dict(key="/fp32/lambda0_vs_b0/rows/sup2_loss_cls/abs_diff",
                     value=0.00015968922525644302),
                dict(key="/train/b0_repeat_vs_b0/rows/sup2_loss_cls/abs_diff",
                     value=0.000195611035451293),
            ]),
        disposition=("withdrawn: both ~2e-4 numbers are residuals of the broken "
                     "replay -- a lambda0-vs-b0 difference, which is required to "
                     "be 0, and a b0-vs-b0_repeat difference of the same variant. "
                     "Neither is the lambda effect."),
        corrected_fp32_head_level_increment=(
            stage2["passes"].get("fp32", {})
            .get("head_level_basis", {}).get("loss_cls_delta_lambda1_vs_b0")),
        times_sup2_weight=(
            None if not stage2["passes"].get("fp32") else
            stage2["passes"]["fp32"]["predicted_sup2_loss_cls_delta"]),
        measured_end_to_end_delta_by_pass={
            name: block["measured_sup2_loss_cls_delta"]
            for name, block in stage2["passes"].items()},
    )

    verdict = dict(
        retained_acceptance_status=(
            "局部验收通过，端到端隔离与反向检查待补验"),
        retained_claims_retracted=[
            "A1-A6 全部通过",
            "端到端 loss 无法分辨该效应（方法结论）",
        ],
        stage1_replay_comparable=repeat_comparable,
        stage1_policies={name: dict(identical=block["identical"],
                                    rng_restored_identically=block["rng_restored_identically"],
                                    first_divergence=block["comparison"]["first_divergence"])
                         for name, block in stage1["policies"].items()},
        stage2_end_to_end_lambda1=a3_end_to_end,
        stage2_end_to_end_lambda0=a5_end_to_end,
        stage3_head_level=(None if stage3["head_level"] is None else dict(
            gradient_follows_weight=stage3["head_level"]["grad_ratio_all_rows_follow_weight"],
            grad_residual_max_abs=stage3["head_level"]["grad_residual_max_abs"],
            grad_residual_bitwise_zero=stage3["head_level"]["grad_residual_bitwise_zero"],
            reweighted_grad_ratio_min=stage3["head_level"]["reweighted_grad_ratio_min"],
            reweighted_grad_ratio_max=stage3["head_level"]["reweighted_grad_ratio_max"],
            untouched_grad_ratio_min=stage3["head_level"]["untouched_grad_ratio_min"],
            untouched_grad_ratio_max=stage3["head_level"]["untouched_grad_ratio_max"],
            n_rows_reweighted=stage3["head_level"]["n_rows_with_weight_ratio_2"],
            n_rows_untouched=stage3["head_level"]["n_rows_with_weight_ratio_1"],
            n_zero_grad_b0_elements=stage3["head_level"]["n_zero_grad_b0_elements"],
            zero_grad_b0_rows_also_zero_under_lambda1=(
                stage3["head_level"]["zero_grad_b0_rows_also_zero_under_lambda1"]),
            loss_cls_bitwise_equal_lambda0_vs_b0=(
                stage3["head_level"]["loss_cls_bitwise_equal_lambda0_vs_b0"]),
            grads_bitwise_equal_lambda0_vs_b0=(
                stage3["head_level"]["grads_bitwise_equal_lambda0_vs_b0"]),
            supervision_unchanged=stage3["head_level"]["supervision_unchanged"],
        )),
        stage3_backward_finite_and_teachers_clean={
            name: dict(n_grad_tensors=block["n_grad_tensors"],
                       n_grad_tensors_with_inf=block["n_grad_tensors_with_inf"],
                       n_grad_tensors_with_nan=block["n_grad_tensors_with_nan"],
                       n_teacher_params=block["n_teacher_params"],
                       n_teacher_params_with_grad=block["n_teacher_params_with_grad"],
                       n_teacher_params_requiring_grad=block["n_teacher_params_requiring_grad"],
                       n_student_params_with_grad=block["n_student_params_with_grad"],
                       grad_scale_applied=block["grad_scale_applied"],
                       optimizer_step_executed=block["optimizer_step_executed"],
                       ema_update_executed=block["ema_update_executed"],
                       params_unchanged_after_backward=block["params_unchanged_after_backward"])
            for name, block in stage3["backward"].items()},
    )
    outstanding = []
    if not repeat_comparable:
        outstanding.append("stage 1: replays are not comparable")
    if not stage2["passes"]:
        outstanding.append("stage 2: not attempted (stage 1 first)")
    else:
        for name, block in a3_end_to_end.items():
            if not block["passed"]:
                outstanding.append("stage 2 lambda1 end-to-end attribution (%s)" % name)
        for name, block in a5_end_to_end.items():
            if not block["passed"]:
                outstanding.append("stage 2 lambda0 reproduction (%s)" % name)
    head = stage3["head_level"]
    if head is None:
        outstanding.append("stage 3a: head-level gradient not checked")
    else:
        if not head["grad_ratio_all_rows_follow_weight"]:
            outstanding.append("stage 3a: gradient does not follow the weight factor")
        if not head["grads_bitwise_equal_lambda0_vs_b0"]:
            outstanding.append("stage 3a: lambda0 gradient differs from b0")
        if not head["loss_cls_bitwise_equal_lambda0_vs_b0"]:
            outstanding.append("stage 3a: lambda0 loss differs from b0")
        if not all(head["supervision_unchanged"].values()):
            outstanding.append("stage 3a: positives/ignored rows were touched")
    for name, block in stage3["backward"].items():
        if (block["n_teacher_params_with_grad"]
                or not block["n_student_params_with_grad"]
                or block["optimizer_step_executed"]
                or block["ema_update_executed"]
                or not block["params_unchanged_after_backward"]):
            outstanding.append("stage 3b: backward guardrails (%s)" % name)
    verdict["outstanding"] = outstanding
    verdict["overall_status"] = ("all three supplementary checks passed"
                                 if not outstanding
                                 else "未完成: " + "; ".join(outstanding))

    report = dict(environment=environment, stage1_replay_determinism=stage1,
                  stage2_end_to_end_attribution=stage2, stage3_backward=stage3,
                  figure_correction=figure_correction, verdict=verdict,
                  elapsed_seconds=time.time() - started)
    write_json(out_dir / "supplement_verdict.json", report)
    logger.info("[supplement] verdict: %s", json.dumps(verdict["overall_status"]))
    logger.info("[supplement] outstanding: %s", json.dumps(outstanding))
    print(json.dumps(verdict, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
