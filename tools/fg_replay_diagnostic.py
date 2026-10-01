"""No-update replay of a trained checkpoint over labeled batches.

Answers one question with numbers instead of display rounding: is the recorded
`sup2_loss_foreground` a real, non-zero loss with real, non-zero gradients?
Nothing is stepped: no optimizer update, no EMA update, no checkpoint written.

Both precision paths run in this one process over the same materialized batches,
so the two passes can be juxtaposed directly instead of being compared across
runs:

  - `train` mirrors the real mixed-precision path. mmcv 1.3.9 with torch >= 1.6
    `wrap_fp16_model` only sets `fp16_enabled`; it does NOT call `model.half()`.
    Recorded dtypes are measured, not assumed: FP32 parameters do not imply that
    the forward's intermediate activations or the backward pass are FP32, so
    probe modules report their activation dtypes.
  - autocast comes from mmdet's `@auto_fp16` decorators (the runner never wraps
    the forward), so the model is called directly and the ambient autocast state
    is recorded at the ROI head and inside the auxiliary head. The dynamic loss
    scaler state is restored from the checkpoint's `meta.fp16.loss_scaler`, the
    loss is scaled before `backward()` and the gradients are unscaled with
    `GradScaler.unscale_` before inspection. An unscaled backward is refused by
    default: it cannot establish whether real training underflowed. The scaler is
    rebuilt from that one recorded state for every pass (torch rejects a second
    `unscale_` before `update()`), so no pass inherits another's bookkeeping and
    the cumulative scale history of training is deliberately not replayed.
  - `fp32` runs the same batches and the same weights with every `fp16_enabled`
    flag cleared, no scaler and no autocast, as a control.

Identical batch tensors do not by themselves match the stochastic parts of a
full training forward (RoI sampling, teacher jitter), so every forward is
preceded by a re-seed of Python/NumPy/Torch/CUDA from `--rng-seed`, the RNG
state is recorded before and after each forward, and a fingerprint of the full
model state (parameters and buffers) is taken before and after every backward.
Two passes may be compared only where the pre-forward RNG states and the
model-state fingerprints agree.

Each pass runs two backwards per batch: one on the weighted total loss (what
training actually backprops, with the same scaler path) and one on
`sup2_loss_foreground` alone (the auxiliary head's own gradient, uncontaminated
by the other terms).

Data path is the training one, not an imitation: `patch_config` +
`ssod.datasets.build_dataset/build_dataloader` with the config's own
`data.sampler.train` sampler and ssod's `collate(..., flatten=True)`, exactly as
`tools/check_ablation_step.py` does it.

Scoped claim: conclusions hold for these weights and these batches, not for the
gradient history of the whole run.

Usage (run from the checkout root):
    python tools/fg_replay_diagnostic.py --config ablation_configs/fold6_seed678/fg.py \
        --checkpoint work_dirs/ablation_v1/fold6_seed678/fg/iter_32000.pth \
        --out <new-file>.json [--batches 2] [--passes train,fp32] \
        [--scan-limit 60] [--rng-seed 0]
"""

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# The environment's egg-link points at another checkout, so pin this one the
# same way the training entry point does before importing ssod.
_spec = importlib.util.spec_from_file_location(
    "_ablation_entry", ROOT / "tools" / "train_ablation.py")
_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_entry)

_FOCUS = {"branch": None, "head": None}
_CALLS = []
_HEAD_CALLS = []
_PROBES = []

FG_PARAM_PREFIXES = ("roi_head.foreground_head",)
CONTEXT_PARAM_PREFIXES = ("student1.neck", "student2.neck",
                          "student1.backbone", "student2.backbone")

PROBE_SUFFIXES = (
    "backbone.layer1.0.conv1",
    "neck.lateral_convs.0.conv",
    "neck.fpn_convs.0.conv",
    "rpn_head.rpn_conv",
    "roi_head.bbox_head.shared_fcs.0",
    "roi_head.bbox_head.fc_cls",
)


def _decompose(logits, target, valid):
    """Reproduce the auxiliary loss term by term, in FP32, next to the module's own value."""
    with torch.no_grad():
        z = logits.detach().float()
        t = target.detach().float()
        p = z.sigmoid()
        pos = t.eq(1) & valid
        neg = t.lt(1) & valid
        positive = -torch.nn.functional.logsigmoid(z) * (1 - p).pow(2) * pos.float()
        negative = -torch.nn.functional.logsigmoid(-z) * p.pow(2) * (1 - t).pow(4) * neg.float()
        peaks = pos.float().flatten(1).sum(1)
        count = peaks.clamp(min=1)
        per_image = (positive + negative).flatten(1).sum(1) / count
        return dict(
            peaks=[int(v) for v in peaks],
            images_without_peaks=[int(v) for v in peaks.eq(0).nonzero().flatten()],
            valid_pixels=[int(v) for v in valid.flatten(1).sum(1)],
            positive_sum=[float(v) for v in positive.flatten(1).sum(1)],
            negative_sum=[float(v) for v in negative.flatten(1).sum(1)],
            image_loss=[float(v) for v in per_image],
            recomputed_mean_loss=float(per_image.mean()),
            normalizer_is_peak_count=True,
            logit_min=float(z.min()),
            logit_max=float(z.max()),
            peak_prob_min=float(p[pos].min()) if bool(pos.any()) else None,
            peak_prob_max=float(p[pos].max()) if bool(pos.any()) else None,
            background_prob_max=float(p[neg].max()) if bool(neg.any()) else None,
        )


def _patch_focal_loss(roi_head_module):
    """`foreground_roi_head` imports the loss by name and calls it bare, so the
    patch has to land in that module's globals (the definition lives in
    `foreground_head`); patching a class attribute would never be called."""
    original = roi_head_module.foreground_focal_loss

    def wrapped(logits, target, valid):
        value = original(logits, target, valid)
        head = _FOCUS["head"]
        _CALLS.append(dict(
            branch=_FOCUS["branch"],
            loss_weight=getattr(head, "foreground_loss_weight", None),
            logits_shape=list(logits.shape),
            dtypes=dict(logits=str(logits.dtype), target=str(target.dtype),
                        valid=str(valid.dtype), loss=str(value.dtype)),
            raw_focal=float(value.detach().float()),
            decomposition=_decompose(logits, target, valid),
        ))
        return value

    roi_head_module.foreground_focal_loss = wrapped
    return original


def _patch_foreground_head(module):
    original = module.ForegroundHead.forward

    def forward(self, features):
        ambient = torch.is_autocast_enabled()
        out = original(self, features)
        _HEAD_CALLS.append(dict(
            branch=_FOCUS["branch"],
            autocast_ambient=ambient,
            feature_dtype=str(features.dtype),
            weight_dtype=str(self.conv.weight.dtype),
            out_dtype=str(out.dtype),
        ))
        return out

    module.ForegroundHead.forward = forward


def _patch_probes(model):
    """Record activation dtypes through the student2 detection path."""
    for name, module in model.named_modules():
        if not name.startswith("student2."):
            continue
        if not name.endswith(PROBE_SUFFIXES):
            continue
        module.register_forward_hook(_probe_hook(name))


def _probe_hook(name):
    def hook(module, inputs, output):
        def dtype_of(value):
            if isinstance(value, torch.Tensor):
                return str(value.dtype)
            if isinstance(value, (list, tuple)):
                return [dtype_of(v) for v in value]
            return None
        _PROBES.append(dict(name=name, autocast=torch.is_autocast_enabled(),
                            in_dtype=dtype_of(inputs[0] if inputs else None),
                            out_dtype=dtype_of(output)))
    return hook


def _patch_supervised_branches(model):
    rows = []
    for branch in ("student1", "student2"):
        roi_head = getattr(model, branch).roi_head
        original = roi_head.forward_train

        def wrapper(self, *args, _branch=branch, _original=original, **kwargs):
            previous = dict(_FOCUS)
            _FOCUS["branch"], _FOCUS["head"] = _branch, self
            gt_bboxes = kwargs.get("gt_bboxes", args[3] if len(args) > 3 else None)
            rows.append(dict(
                branch=_branch,
                foreground_supervised=bool(kwargs.get("foreground_supervised", False)),
                gt_boxes=[int(len(b)) for b in (gt_bboxes if gt_bboxes is not None else [])],
                autocast_enabled=torch.is_autocast_enabled(),
                p2_dtypes=[str(t.dtype) for t in args[0]] if args and hasattr(args[0], "__iter__") else None,
                roi_head_param_dtype=str(next(self.parameters()).dtype),
            ))
            try:
                return _original(*args, **kwargs)
            finally:
                _FOCUS.clear()
                _FOCUS.update(previous)

        roi_head.forward_train = types.MethodType(wrapper, roi_head)
    return rows


def _unwrap_singleton(data):
    """ssod's collate wraps a batch group in an extra list; unwrap it without
    assuming whether `scatter` has already removed that level."""
    while isinstance(data, (list, tuple)) and len(data) == 1 and \
            isinstance(data[0], (list, tuple, torch.Tensor)):
        data = data[0]
    return data


def _metas_of(container):
    data = container.data if hasattr(container, "data") else container
    return list(_unwrap_singleton(data))


def _stacked_batch(container):
    """The per-GPU batch tensor (images are padded to a common size by collate)."""
    data = container.data if hasattr(container, "data") else container
    return _unwrap_singleton(data)


def _tensor_fingerprint(value, digest, key):
    if isinstance(value, torch.Tensor):
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        flat = value.detach().reshape(-1)
        for chunk in torch.split(flat, 1 << 20):
            digest.update(chunk.cpu().float().numpy().tobytes())
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _tensor_fingerprint(item, digest, "%s[%d]" % (key, index))
    elif isinstance(value, dict):
        for sub_key in sorted(value):
            _tensor_fingerprint(value[sub_key], digest, "%s.%s" % (key, sub_key))


def _batch_fingerprint(batch):
    digest = hashlib.sha256()
    digest.update(json.dumps(sorted(m.get("filename", "")
                                    for m in _metas_of(batch["img_metas"]))).encode())
    for key in sorted(batch):
        if key == "img_metas":
            continue
        _tensor_fingerprint(batch[key], digest, key)
    return digest.hexdigest()


def _state_fingerprint(model):
    """Hash parameters and buffers in chunks, so no full CPU copy is materialized."""
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            for chunk in torch.split(tensor.detach().reshape(-1), 1 << 20):
                digest.update(chunk.cpu().float().numpy().tobytes())
    return digest.hexdigest()


def _rng_state():
    numpy_state = np.random.get_state()
    return dict(
        python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest()[:16],
        numpy=hashlib.sha256(np.asarray(numpy_state[1]).tobytes()).hexdigest()[:16],
        torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16],
        cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()[:16],
    )


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_fp16(model, enabled):
    count = 0
    for module in model.modules():
        if hasattr(module, "fp16_enabled"):
            module.fp16_enabled = enabled
            count += 1
    return count


def _clear_grads(model):
    for param in model.parameters():
        param.grad = None


def _grad_report(model, prefixes):
    rows = {}
    total_sq = 0.0
    with_grad = 0
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is not None:
            with_grad += 1
            flat = grad.detach().float()
            if torch.isfinite(flat).all():
                total_sq += float(flat.pow(2).sum())
        if not any(prefix in name for prefix in prefixes):
            continue
        if grad is None:
            rows[name] = dict(requires_grad=bool(param.requires_grad), grad=None)
            continue
        flat = grad.detach().float()
        rows[name] = dict(
            requires_grad=bool(param.requires_grad),
            param_dtype=str(param.dtype),
            grad_dtype=str(grad.dtype),
            finite=bool(torch.isfinite(flat).all()),
            abs_max=float(flat.abs().max()),
            l2_norm=float(flat.norm()),
            nonzero=int(flat.ne(0).sum()),
            numel=int(flat.numel()),
            zero_fraction=float(flat.eq(0).float().mean()),
        )
    return rows, total_sq ** 0.5, with_grad


def _grad_summary(rows, label_prefix):
    """Aggregate every parameter under `label_prefix` into one zero-fraction / norm row."""
    numel = nonzero = 0
    sq = 0.0
    abs_max = 0.0
    finite = True
    present = 0
    for name, row in rows.items():
        if label_prefix not in name:
            continue
        present += 1
        # `_grad_report` writes `grad_dtype` only for parameters that actually
        # carry a gradient; the no-grad branch writes a lone `grad: None`.
        if "grad_dtype" not in row:
            continue
        numel += row["numel"]
        nonzero += row["nonzero"]
        sq += row["l2_norm"] ** 2
        abs_max = max(abs_max, row["abs_max"])
        finite = finite and row["finite"]
    return dict(
        tensors_under_prefix=present,
        grads_present=sum(1 for n, r in rows.items()
                          if label_prefix in n and "grad_dtype" in r),
        numel=numel,
        nonzero=nonzero,
        zero_fraction=(None if numel == 0 else 1.0 - nonzero / numel),
        l2_norm=(None if numel == 0 else sq ** 0.5),
        abs_max=abs_max,
        finite=finite,
    )


def _mean_reduce(value, key):
    if isinstance(value, torch.Tensor):
        return value.mean()
    if isinstance(value, (list, tuple)):
        return sum(item.mean() for item in value)
    raise TypeError("%s is not a tensor or list of tensors" % key)


def _trainer_total_loss(losses):
    """Mirrors mmdet's BaseDetector._parse_losses: only keys containing 'loss'
    are summed, each mean-reduced. Summing every returned value instead would
    add the accuracy scalars and diverge from what training actually backprops."""
    total, used = None, []
    for key, value in losses.items():
        if "loss" not in key:
            continue
        term = _mean_reduce(value, key)
        total = term if total is None else total + term
        used.append(key)
    if total is None:
        raise RuntimeError("no loss-like key in the returned dict: %s" % sorted(losses))
    return total, used


def _scalar_losses(losses):
    out = {}
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().float().mean())
        elif isinstance(value, (list, tuple)):
            out[key] = [float(item.detach().float().mean()) for item in value]
        else:
            out[key] = str(value)
    return out


def _make_scaler(scaler_state):
    """Rebuild the scaler from one recorded state for a single pass.

    Training shares one scaler across all iterations, but the replay must not
    carry `unscale_`/`step` bookkeeping from one pass into the next (torch
    refuses a second `unscale_` before `update()`), and the checkpoint records
    only one scaler snapshot. Rebuilding per pass keeps every pass on exactly
    that state; the cumulative scale history of training is not replayed.
    """
    scaler = torch.cuda.amp.GradScaler()
    scaler.load_state_dict(scaler_state)
    return scaler


def _run_pass(model, batch, *, precision, scaler_state, optimizer, prefixes, rng_seed,
              call_rows, backward_target, pass_label):
    """One forward/backward at one precision. Backward is on `backward_target(losses)`."""
    scaler = _make_scaler(scaler_state) if scaler_state is not None else None
    gc.collect()
    torch.cuda.empty_cache()
    _clear_grads(model)
    del call_rows[:]
    del _PROBES[:]
    call_start, head_start = len(_CALLS), len(_HEAD_CALLS)
    state_before = _state_fingerprint(model)
    autocast_outside_forward = torch.is_autocast_enabled()
    _seed_all(rng_seed)
    rng_before = _rng_state()
    start = time.time()
    losses = model(**batch)  # mmdet's @auto_fp16 provides the autocast scope
    rng_after_forward = _rng_state()
    target_loss, loss_keys = backward_target(losses)
    scaled_value = float(scaler.scale(target_loss).detach().float()) if scaler else None
    if scaler is not None:
        scaler.scale(target_loss).backward()
        scaled_grad_max = max((float(p.grad.detach().float().abs().max())
                               for p in model.parameters() if p.grad is not None),
                              default=None)
        scaler.unscale_(optimizer)
    else:
        target_loss.backward()
        scaled_grad_max = None
    scaler_state_after = scaler.state_dict() if scaler is not None else None
    scaler_would_skip = None
    if scaler is not None and scaled_grad_max is not None:
        # `unscale_` leaves overflowed grads unscaled and records found_inf; the
        # resulting flag is not read here, so this is inferred from the scaled
        # gradient magnitude instead (recorded as such below).
        scaler_would_skip = not math.isfinite(scaled_grad_max)
    elapsed = time.time() - start
    grads, grad_norm, params_with_grad = _grad_report(model, prefixes)
    state_after = _state_fingerprint(model)
    record = dict(
        precision=precision,
        backward_on=pass_label,
        autocast_outside_forward=autocast_outside_forward,
        autocast_at_entry=None,  # filled from the first supervised call
        losses=_scalar_losses(losses),
        loss_keys_summed=loss_keys,
        backward_value=float(target_loss.detach().float()),
        scaled_backward_value=scaled_value,
        scaled_grad_abs_max=scaled_grad_max,
        scaler_state_used=scaler_state,
        scaler_state_after=scaler_state_after,
        scaler_would_skip=scaler_would_skip,
        scaler_note=("每个 pass 都从同一份 scaler 状态新建（scale 与 _growth_tracker 均取 checkpoint 值），"
                     "不做 step/update，故 state_after 应与 state_used 相同；"
                     "scaler_would_skip 由 scaled_grad_abs_max 是否非有限推断，未读 scaler 内部 found_inf"),
        grad_norm_after_unscale=grad_norm,
        parameters_with_any_grad=params_with_grad,
        rng_state_before_forward=rng_before,
        rng_state_after_forward=rng_after_forward,
        secs=round(elapsed, 2),
        aux_head_calls=_CALLS[call_start:],
        foreground_head_calls=_HEAD_CALLS[head_start:],
        supervised_calls=list(call_rows),
        # Copied, not aliased: the next pass clears `_PROBES` in place, so holding
        # the live list here would leave every pass showing the last pass's dtypes.
        probe_dtypes=list(_PROBES),
        grads=grads,
        grad_summary=dict(
            foreground_head=_grad_summary(grads, "roi_head.foreground_head"),
            student1_neck=_grad_summary(grads, "student1.neck"),
            student2_neck=_grad_summary(grads, "student2.neck"),
        ),
        model_state_unchanged=bool(state_before == state_after),
    )
    if record["supervised_calls"]:
        record["autocast_at_entry"] = record["supervised_calls"][0]["autocast_enabled"]
    _clear_grads(model)
    return record


def _tag_histogram(counts, tags):
    key = "+".join(sorted(tags))
    counts[key] = counts.get(key, 0) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, default=2,
                        help="number of batches containing a sup2 sample to replay")
    parser.add_argument("--passes", default="train,fp32")
    parser.add_argument("--loss-scale", type=float, default=None,
                        help="override the restored scaler; only for the train path")
    parser.add_argument("--scan-limit", type=int, default=60,
                        help="how many loader batches to scan while collecting sup2 batches")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="re-seeds python/numpy/torch/cuda before every forward")
    parser.add_argument("--select-tag", default="sup2")
    args = parser.parse_args()

    passes = [p for p in args.passes.split(",") if p]
    for name in passes:
        if name not in ("train", "fp32"):
            raise ValueError("unknown pass: %s" % name)

    out_path = Path(args.out)
    if out_path.exists():
        # Checked before any GPU work; the partial report is rewritten after
        # every batch, so a late failure still leaves the finished batches on disk.
        raise FileExistsError("refusing to overwrite an existing replay report: %s" % out_path)
    out_path.write_text("{}\n")

    _entry.pin_repository()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from this checkout root to preserve relative data paths")

    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import build_optimizer, load_checkpoint, wrap_fp16_model
    from mmdet.apis import train_detector  # noqa: F401  (registers pipelines)
    from mmdet.models import build_detector
    from ssod.datasets import build_dataloader, build_dataset
    from ssod.utils import patch_config

    from ssod.models.roi_heads import foreground_head as fg_module
    from ssod.models.roi_heads import foreground_roi_head as fg_roi_head_module

    receipt = _entry.source_manifest()
    cfg = patch_config(Config.fromfile(args.config))
    if cfg.model.model.roi_head.type != "ForegroundRoIHead":
        raise ValueError("config does not build a ForegroundRoIHead")
    sampler_cfg = copy.deepcopy(cfg.data.get("sampler", {}).get("train", {}))

    # Same data path as tools/check_ablation_step.py / ssod.apis.train.
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, cfg.data.samples_per_gpu, 0, num_gpus=1,
                              dist=True, seed=cfg.get("seed"),
                              sampler_cfg=copy.deepcopy(sampler_cfg))

    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
                           test_cfg=cfg.get("test_cfg"))
    if "train" in passes:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu", strict=True)
    model._pretrained_initialized = True  # never re-read Phase 1/2 files here
    model.CLASSES = dataset.CLASSES  # DualTeacher's box-logging path reads self.CLASSES
    model.cuda(0)
    model.train()

    scaler_source = None
    scaler_state = None
    if "train" in passes:
        stored = (checkpoint.get("meta", {}).get("fp16", {}) or {}).get("loss_scaler")
        if args.loss_scale is not None:
            # A complete 5-key spec with the CLI scale; the remaining factors keep
            # the torch defaults so the recorded state stays loadable as-is.
            scaler_state = dict(scale=float(args.loss_scale), growth_factor=2.0,
                                backoff_factor=0.5, growth_interval=2000,
                                _growth_tracker=0)
            scaler_source = dict(source="cli --loss-scale", state=scaler_state)
        elif stored is None:
            raise RuntimeError(
                "checkpoint meta has no fp16.loss_scaler; refusing to inspect an "
                "unscaled backward. Pass --loss-scale to force one and record it.")
        else:
            scaler_state = stored
            scaler_source = dict(source="checkpoint meta.fp16.loss_scaler", state=stored)
    optimizer = build_optimizer(model, cfg.optimizer)  # never stepped; only for scaler.unscale_

    _patch_focal_loss(fg_roi_head_module)
    _patch_foreground_head(fg_module)
    call_rows = _patch_supervised_branches(model)
    _patch_probes(model)

    # Collect the batches: walk the real loader on CPU, keep the ones carrying a
    # sup2 sample, and note what the sampler actually emits.
    histograms = {}
    selected = []
    scanned = 0
    for raw in loader:
        scanned += 1
        tags = [meta["tag"] for meta in _metas_of(raw["img_metas"])]
        _tag_histogram(histograms, tags)
        if args.select_tag in tags:
            selected.append((scanned, raw, tags))
        if len(selected) >= args.batches or scanned >= args.scan_limit:
            break
    if not selected:
        raise RuntimeError("no batch carrying tag %r within %d scanned batches: %s"
                           % (args.select_tag, scanned, sorted(histograms)))

    report = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        # Self-identification: the digest of this script as it ran, and the argv,
        # so the artifact does not have to be paired with an external note.
        tool_path=str(Path(__file__).resolve()),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        argv=list(sys.argv),
        checkpoint_sha256=hashlib.sha256(
            Path(args.checkpoint).read_bytes()).hexdigest(),
        passes=passes,
        loss_scaler=scaler_source,
        loss_scale_used=float(scaler_state["scale"]) if scaler_state is not None else None,
        sampler_cfg=sampler_cfg,
        samples_per_gpu=cfg.data.samples_per_gpu,
        dataset=dict(type=type(dataset).__name__, length=len(dataset),
                     sup1=len(dataset.sup1), sup2=len(dataset.sup2),
                     unsup=len(dataset.unsup)),
        loader_seed=cfg.get("seed"),
        scanned_batches=scanned,
        tag_histogram_of_scanned=histograms,
        selected=len(selected),
        rng_seed=args.rng_seed,
        allocator_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        fp16_enabled_modules=dict(
            train=sum(1 for m in model.modules() if getattr(m, "fp16_enabled", False)),
            fp32=0),
        model_state_fingerprint_before=_state_fingerprint(model),
        optimizer_updates=0,
        ema_updates=0,
        batches=[],
        scope=("结论仅限本组权重与所选批次，不外推为整段训练的梯度历史；"
               "参数 FP32 不代表中间计算与反向均为 FP32，以本文件记录的 dtype 实测为准"),
        not_claimed=("不用这些数值解释该组 mAP 的升降；不宣称每个训练步的损失或梯度非零；"
                     "这些批次由配置自带的采样器按 seed 重新抽取，不等同于正式训练第 k 步的批次。"),
        source_receipt=receipt,
    )

    fp16_by_pass = {"train": True, "fp32": False}
    for index, (batch_index, raw, tags) in enumerate(selected):
        metas = _metas_of(raw["img_metas"])
        row = dict(
            batch=index,
            loader_batch_index=batch_index,
            tags=tags,
            filenames=[meta.get("filename") for meta in metas],
            img_batch_shape=[int(v) for v in _stacked_batch(raw["img"]).shape],
            img_metas_shapes=[meta.get("pad_shape") for meta in metas],
            fingerprints_by_pass={},
            results={},
        )
        for pass_name in passes:
            # Allocator hygiene only, no numerics: four full DualTeacher
            # forward+backward graphs per batch sit close to the 16 GB card's
            # ceiling, and PyTorch otherwise carries the previous pass's cached
            # blocks into this one, where a small contiguous request can fail
            # against a fragmented cache.
            gc.collect()
            torch.cuda.empty_cache()
            _set_fp16(model, fp16_by_pass[pass_name])
            batch = scatter(copy.deepcopy(raw), [torch.cuda.current_device()])[0]
            # Per pass, so the artifact itself shows both precisions consumed the
            # same materialized batch instead of only the last pass's digest.
            row["fingerprints_by_pass"][pass_name] = _batch_fingerprint(batch)
            active_scaler_state = scaler_state if pass_name == "train" else None
            total = _run_pass(
                model, batch, precision=pass_name, scaler_state=active_scaler_state,
                optimizer=optimizer,
                prefixes=FG_PARAM_PREFIXES + CONTEXT_PARAM_PREFIXES,
                rng_seed=args.rng_seed, call_rows=call_rows,
                backward_target=_trainer_total_loss,
                pass_label="trained_total_loss")
            fg_only = _run_pass(
                model, batch, precision=pass_name, scaler_state=active_scaler_state,
                optimizer=optimizer,
                prefixes=FG_PARAM_PREFIXES + CONTEXT_PARAM_PREFIXES,
                rng_seed=args.rng_seed, call_rows=call_rows,
                backward_target=_fg_only_target,
                pass_label="sup2_loss_foreground")
            row["results"][pass_name] = dict(total_backward=total,
                                             fg_backward=fg_only)
            del batch
            torch.cuda.empty_cache()

        row["comparison"] = _compare(row["results"], passes)
        report["batches"].append(row)
        report["complete"] = False
        with open(out_path, "w") as handle:
            json.dump(report, handle, indent=1, sort_keys=True)
        _print_batch(row)

    report["model_state_fingerprint_after"] = _state_fingerprint(model)
    report["model_state_unchanged_overall"] = bool(
        report["model_state_fingerprint_before"] == report["model_state_fingerprint_after"])
    report["complete"] = True
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=1, sort_keys=True)
    print("[FG replay source] " + json.dumps(receipt, sort_keys=True))
    print("passes=%s batches=%d scanned=%d loss_scale=%s rng_seed=%s tags=%s -> %s"
          % (",".join(passes), len(report["batches"]), scanned,
             report["loss_scale_used"], args.rng_seed, sorted(histograms), out_path))
    print("model_state_unchanged_overall=%s optimizer_updates=0 ema_updates=0"
          % report["model_state_unchanged_overall"])


def _fg_only_target(losses):
    if "sup2_loss_foreground" not in losses:
        raise RuntimeError("sup2_loss_foreground missing from the loss dict; got %s"
                           % sorted(losses))
    return _mean_reduce(losses["sup2_loss_foreground"], "sup2_loss_foreground"), \
        ["sup2_loss_foreground"]


def _compare(results, passes):
    """Juxtapose the two precisions on identical inputs."""
    out = dict(compared=[],
               note=("loss_abs_diff_max 只统计键名含 loss 的项；value_abs_diff_max 覆盖"
                     "损失字典中全部数值标量（含 mmdet 的 `*_acc` 准确率）。比值表同时含这两类键。"))
    if len(passes) != 2:
        return out
    for label in ("total_backward", "fg_backward"):
        train = results.get("train", {}).get(label)
        fp32 = results.get("fp32", {}).get(label)
        if train is None or fp32 is None:
            continue
        entry = dict(
            backward_on=label,
            rng_states_match=bool(train["rng_state_before_forward"] ==
                                  fp32["rng_state_before_forward"]),
            autocast_train=train["autocast_at_entry"],
            autocast_fp32=fp32["autocast_at_entry"],
            both_model_state_unchanged=bool(train["model_state_unchanged"] and
                                            fp32["model_state_unchanged"]),
            loss_ratio_train_over_fp32={},
            fg_head_grad=dict(
                train=train["grad_summary"]["foreground_head"],
                fp32=fp32["grad_summary"]["foreground_head"]),
            # mmdet's loss dict also carries `*_acc` scores, which are not
            # losses; keep the two maxima apart so neither name overstates.
            loss_abs_diff_max=0.0,
            value_abs_diff_max=0.0,
        )
        for key in sorted(train["losses"]):
            a, b = train["losses"][key], fp32["losses"].get(key)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            entry["value_abs_diff_max"] = max(entry["value_abs_diff_max"], abs(a - b))
            if "loss" in key:
                entry["loss_abs_diff_max"] = max(entry["loss_abs_diff_max"], abs(a - b))
            if b != 0.0:
                entry["loss_ratio_train_over_fp32"][key] = a / b
        out["compared"].append(entry)
    return out


def _print_batch(row):
    print("batch %d loader_index=%d tags=%s img=%s metas=%s" %
          (row["batch"], row["loader_batch_index"], row["tags"], row["img_batch_shape"],
           row["img_metas_shapes"]))
    for pass_name, results in sorted(row["results"].items()):
        for label in ("total_backward", "fg_backward"):
            record = results[label]
            print("  %-5s %-18s value=%.6e scaled=%s grad_norm=%.6e params_with_grad=%d "
                  "state_unchanged=%s autocast_entry=%s"
                  % (pass_name, label, record["backward_value"],
                     ("%.6e" % record["scaled_backward_value"])
                     if record["scaled_backward_value"] is not None else None,
                     record["grad_norm_after_unscale"], record["parameters_with_any_grad"],
                     record["model_state_unchanged"], record["autocast_at_entry"]))
            for name, summary in sorted(record["grad_summary"].items()):
                print("      grad %-18s tensors=%d numel=%d nonzero=%d zero_frac=%s l2=%s"
                      % (name, summary["tensors_under_prefix"], summary["numel"],
                         summary["nonzero"],
                         None if summary["zero_fraction"] is None
                         else ("%.8f" % summary["zero_fraction"]),
                         None if summary["l2_norm"] is None
                         else ("%.6e" % summary["l2_norm"])))
            for call in record["aux_head_calls"]:
                dec = call["decomposition"]
                print("      fg %-9s raw=%.8e x weight=%s = %.8e peaks=%s valid_px=%s "
                      "logits=[%.4f,%.4f] dtypes=%s"
                      % (call["branch"], call["raw_focal"], call["loss_weight"],
                         call["raw_focal"] * (call["loss_weight"] or 0.0),
                         dec["peaks"], dec["valid_pixels"],
                         dec["logit_min"], dec["logit_max"], call["dtypes"]))
            for head_call in record["foreground_head_calls"]:
                print("      head %-9s feature=%s weight=%s out=%s autocast_ambient=%s"
                      % (head_call["branch"], head_call["feature_dtype"],
                         head_call["weight_dtype"], head_call["out_dtype"],
                         head_call["autocast_ambient"]))
            for probe in record["probe_dtypes"]:
                print("      probe %-40s in=%s out=%s autocast=%s"
                      % (probe["name"], probe["in_dtype"], probe["out_dtype"], probe["autocast"]))
    for entry in row["comparison"]["compared"]:
        print("  compare %-18s rng_match=%s autocast=%s/%s loss_ratio=%s"
              % (entry["backward_on"], entry["rng_states_match"],
                 entry["autocast_train"], entry["autocast_fp32"],
                 {k: round(v, 8) for k, v in sorted(entry["loss_ratio_train_over_fp32"].items())}))
        print("      fg_head zero_frac train=%s fp32=%s l2 train=%s fp32=%s"
              % (entry["fg_head_grad"]["train"]["zero_fraction"],
                 entry["fg_head_grad"]["fp32"]["zero_fraction"],
                 entry["fg_head_grad"]["train"]["l2_norm"],
                 entry["fg_head_grad"]["fp32"]["l2_norm"]))


if __name__ == "__main__":
    main()
