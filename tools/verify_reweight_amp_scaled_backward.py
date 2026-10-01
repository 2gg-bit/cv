"""AMP GradScaler backward check for the sup2 small-background ROI reweight.

Scope: exactly one item.  Both the retained acceptance and the supplement ran
`total_loss.backward()` straight on the loss, so the mixed-precision path never
went through `GradScaler`.  This script closes that gap on the optimizer path the
frozen config actually uses:

    fp16 = dict(loss_scale='dynamic')                       # b0.py
      -> mmcv Fp16OptimizerHook(loss_scale='dynamic')        # ssod/apis/train.py:130
      -> torch.cuda.amp.GradScaler()                         # mmcv runner/hooks/optimizer.py
      -> scale(loss).backward() -> unscale_(optimizer)

`lambda=0` and `lambda=1` each get the same batch, the same model state, the same
RNG snapshot and the same scaling state.  The scaler state is written down before
and after every run (source, class, init_scale / growth_factor / backoff_factor /
growth_interval, current scale, growth tracker).  After `unscale_` the gradients
must be finite, the teachers must carry none, the parameters and buffers must be
untouched, and no optimizer step, scaler update or EMA update may happen.

What this does *not* do: re-run A1-A6, touch the retained acceptance artifacts, or
step anything.  AMP B0/B0-repeat stays unrun on purpose -- the supplement's stage 1
is fp32 only -- so AMP replay repeatability is still covered only for
B0-vs-lambda=0 (all losses and all 39 traced events bitwise equal).

Run from the checkout root:
  python tools/verify_reweight_amp_scaled_backward.py \
      --config ablation_configs/fold6_seed678/b0.py \
      --checkpoint work_dirs/ablation_v1/fold6_seed678/b0/iter_32000.pth \
      --out-dir ablation_configs/reweight_small_bkg_20260929_amp_scaled_backward
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
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# Reuse the supplement's RNG helpers and the retained harness' validated helpers
# (source receipt, optimizer/EMA counters with their positive self-test, state
# fingerprints, json writer).  Both files' `main` are guarded, so importing them
# only defines names.
_HARNESS_PATH = ROOT / "tools" / "verify_small_bkg_reweight.py"
_hspec = importlib.util.spec_from_file_location("_reweight_harness", _HARNESS_PATH)
_harness = importlib.util.module_from_spec(_hspec)
_hspec.loader.exec_module(_harness)

_SUPPLEMENT_PATH = ROOT / "tools" / "verify_reweight_supplement.py"
_sspec = importlib.util.spec_from_file_location("_reweight_supplement",
                                               _SUPPLEMENT_PATH)
_supplement = importlib.util.module_from_spec(_sspec)
_sspec.loader.exec_module(_supplement)

_entry = _harness._entry
sha256_file = _harness.sha256_file
state_fingerprint = _harness.state_fingerprint
rng_fingerprint = _harness.rng_fingerprint
write_json = _harness.write_json
as_float = _harness.as_float
install_instrumentation = _harness.install_instrumentation
snapshot_rng = _supplement.snapshot_rng
restore_rng = _supplement.restore_rng
REQUIRED_TAGS = _harness.REQUIRED_TAGS
BASELINE_TAG = _harness.BASELINE_TAG
MAX_AREA = _harness.MAX_AREA
COUNTER_KEYS = _harness.COUNTER_KEYS
VARIANT_CONFIG_NAME = _harness.VARIANT_CONFIG_NAME
EXPECTED_CONFIG_DIFFS = _harness.EXPECTED_CONFIG_DIFFS

# ------------------------------------------------------------------ constants
SUPPLEMENT_OUT_DIR = ROOT / "ablation_configs" / "reweight_small_bkg_20260929_supplement"
MY_COUNTER_KEYS = tuple(COUNTER_KEYS) + ("scaler_step", "scaler_update")
VARIANTS = ("lambda0", "lambda1")
# fp16 grads are divided by the scale in fp16, so the arithmetic identity below
# is only checked where it can hold; see `unscale` in run_scaled.
UNSCALE_REL_TOL = 1e-6


# ------------------------------------------------------------------- helpers

def scaler_snapshot(scaler):
    """Public-API view of one scaler's state."""
    return dict(
        enabled=bool(_attr(scaler, "is_enabled", "_enabled")),
        state_dict=scaler.state_dict(),
        scale=float(scaler.get_scale()),
    )


def _attr(obj, public, private):
    """torch 1.7 keeps some of these private, newer versions expose them."""
    if hasattr(obj, public):
        value = getattr(obj, public)
        return value() if callable(value) else value
    return getattr(obj, private)


def scaler_provenance(scaler):
    from torch.cuda.amp import GradScaler
    return dict(
        class_path="%s.%s" % (GradScaler.__module__, GradScaler.__name__),
        source=("constructed in this script with torch.cuda.amp.GradScaler() "
                "defaults; the training path reaches the same defaults through "
                "mmcv Fp16OptimizerHook(loss_scale='dynamic') built from "
                "fp16=dict(loss_scale='dynamic') in the frozen config"),
        enabled=bool(_attr(scaler, "is_enabled", "_enabled")),
        init_scale=float(_attr(scaler, "init_scale", "_init_scale")),
        growth_factor=float(_attr(scaler, "growth_factor", "_growth_factor")),
        backoff_factor=float(_attr(scaler, "backoff_factor", "_backoff_factor")),
        growth_interval=int(_attr(scaler, "growth_interval", "_growth_interval")),
    )


def install_scaler_instrumentation(counters):
    """Count GradScaler.step / .update, then prove the counters can move.

    The control's own calls stay in `counters`; every per-run delta is measured
    against a snapshot taken after both controls, so they cancel out there.
    """
    from torch.cuda.amp import GradScaler

    original = dict(step=GradScaler.step, update=GradScaler.update)

    def scaler_step(self, optimizer, *args, **kwargs):
        counters["scaler_step"] += 1
        return original["step"](self, optimizer, *args, **kwargs)

    def scaler_update(self, new_scale=None):
        counters["scaler_update"] += 1
        return original["update"](self, new_scale)

    GradScaler.step = scaler_step
    GradScaler.update = scaler_update

    param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    param.grad = torch.zeros(1)
    toy = torch.optim.SGD([param], lr=0.1)
    # enabled=False keeps the control free of scaling, and the passthrough still
    # runs the real `step` / `update` bodies on the patched class methods
    control = GradScaler(enabled=False, init_scale=8.0)
    control.step(toy)
    control.update()
    proof = dict(
        scaler_step_seen=counters["scaler_step"],
        scaler_update_seen=counters["scaler_update"],
        step_patched=GradScaler.step is not original["step"],
        update_patched=GradScaler.update is not original["update"],
    )
    proof["passed"] = (proof["scaler_step_seen"] == 1
                       and proof["scaler_update_seen"] == 1
                       and proof["step_patched"] and proof["update_patched"])
    return proof


def grad_abs_sum(model):
    """Sum |grad| over every grad tensor, accumulated in fp32."""
    total = 0.0
    for _, param in model.named_parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().abs().sum().item())
    return total


def grad_records(model):
    """(name, grad) for every parameter that received a gradient."""
    return [(name, param.grad.detach())
            for name, param in model.named_parameters()
            if param.grad is not None]


def scan_gradients(records):
    """Non-finite accounting per grad tensor -- inf and nan counted separately."""
    with_inf, with_nan = [], []
    for name, grad in records:
        if bool(torch.isinf(grad).any().item()):
            with_inf.append(name)
        if bool(torch.isnan(grad).any().item()):
            with_nan.append(name)
    return dict(
        n_grad_tensors=len(records),
        n_grad_tensors_with_inf=len(with_inf),
        n_grad_tensors_with_nan=len(with_nan),
        grad_tensors_with_inf=with_inf,
        grad_tensors_with_nan=with_nan,
        n_grad_elements_nonfinite=sum(
            int(torch.isnan(g).sum().item()) + int(torch.isinf(g).sum().item())
            for _, g in records),
    )


def digest_grads(records):
    return {name: hashlib.sha256(grad.contiguous().cpu().numpy().tobytes())
            .hexdigest() for name, grad in records}


# --------------------------------------------------------------- scaled runs

def run_scaled(cfg, dataset, batch, checkpoint, device, counters, snapshot,
               scaler_state, variant):
    """scale(loss).backward() -> unscale_(optimizer) on the training path.

    Never calls optimizer.step, scaler.step, scaler.update, zero_grad or an EMA
    update.  Returns the unscaled gradients as detached clones.
    """
    from mmcv.parallel import scatter
    from mmcv.runner import build_optimizer, load_checkpoint, wrap_fp16_model
    from mmdet.models import build_detector

    model = build_detector(cfg.model)
    from ssod.models.roi_heads.small_bkg_reweight import enable_reweight_diagnostics
    enable_reweight_diagnostics(model)
    wrap_fp16_model(model)          # the hook does this in before_run
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model._pretrained_initialized = True
    model.CLASSES = dataset.CLASSES
    model.cuda(device)
    model.train()
    model.freeze("teacher1")
    model.freeze("teacher2")
    optimizer = build_optimizer(model, cfg.optimizer)

    scaler = torch.cuda.amp.GradScaler()
    if scaler_state is not None:
        scaler.load_state_dict(copy.deepcopy(scaler_state))
    scaler_before = scaler_snapshot(scaler)

    restore_rng(snapshot)
    rng_before = rng_fingerprint()
    params_before, buffers_before = state_fingerprint(model)
    inputs = scatter(copy.deepcopy(batch), [device])[0]
    before_counters = dict(counters)

    with torch.cuda.amp.autocast(enabled=True):
        losses = model(return_loss=True, **inputs)
    total_loss, parsed = model._parse_losses(losses)
    loss_values = {key: as_float(value) for key, value in parsed.items()
                   if key != "loss"}
    total_loss_value = as_float(total_loss)

    scaled_loss = scaler.scale(total_loss)
    scaler_scale = float(scaler.get_scale())
    scaled_loss_value = as_float(scaled_loss)
    scaled_loss.backward()
    scaled_abs_sum = grad_abs_sum(model)
    scaler.unscale_(optimizer)
    unscaled_abs_sum = grad_abs_sum(model)
    scaler_after = scaler_snapshot(scaler)

    records = grad_records(model)
    finite = scan_gradients(records)
    teacher = [(name, param) for name, param in model.named_parameters()
               if name.split(".")[0].startswith("teacher")]
    model_params = [param for _, param in model.named_parameters()]
    optimizer_params = [param for group in optimizer.param_groups
                        for param in group["params"]]
    model_param_ids = {id(param) for param in model_params}

    params_after, buffers_after = state_fingerprint(model)
    changed_params = sorted(name for name in params_before
                            if params_before[name] != params_after.get(name))
    changed_buffers = sorted(name for name in buffers_before
                             if buffers_before[name] != buffers_after.get(name))

    grad_dtypes = sorted({str(grad.dtype) for _, grad in records})
    info = dict(
        variant=variant,
        rng_before=rng_before,
        total_loss=total_loss_value,
        losses=loss_values,
        scaled_loss=scaled_loss_value,
        scaler=dict(
            provenance=scaler_provenance(scaler),
            scale_before=scaler_before["scale"],
            scale_after=scaler_after["scale"],
            scale_unchanged=(scaler_before["scale"] == scaler_after["scale"]),
            state_before=scaler_before["state_dict"],
            state_after=scaler_after["state_dict"],
            state_unchanged=(scaler_before["state_dict"]
                             == scaler_after["state_dict"]),
        ),
        scaling=dict(
            scale=scaler_scale,
            scaled_loss_equals_loss_times_scale=(
                abs(scaled_loss_value - total_loss_value * scaler_scale)
                <= UNSCALE_REL_TOL * max(abs(scaled_loss_value), 1e-30)),
            scaled_grad_abs_sum=scaled_abs_sum,
            unscaled_grad_abs_sum=unscaled_abs_sum,
            # fp16 grads are divided in fp16; `_init_scale` is 2**16, a power of
            # two, so the division is exact in fp16 except for underflow, and
            # the global ratio can only be reported, not asserted
            unscaled_over_scaled=(unscaled_abs_sum / scaled_abs_sum
                                 if scaled_abs_sum else None),
            inverse_scale=(1.0 / scaler_scale),
        ),
        gradients=finite,
        grad_dtypes=grad_dtypes,
        n_grad_elements=sum(int(grad.numel()) for _, grad in records),
        teacher=dict(
            n_teacher_params=len(teacher),
            n_teacher_params_requiring_grad=sum(1 for _, param in teacher
                                                if param.requires_grad),
            n_teacher_params_with_grad=sum(1 for _, param in teacher
                                           if param.grad is not None),
            teacher_with_grad=sorted(name for name, param in teacher
                                     if param.grad is not None)[:10],
        ),
        optimizer=dict(
            type=type(optimizer).__name__,
            n_param_tensors=len(optimizer_params),
            n_param_tensors_with_grad=sum(1 for param in optimizer_params
                                          if param.grad is not None),
            params_are_model_params=all(id(param) in model_param_ids
                                        for param in optimizer_params),
        ),
        params_unchanged=(not changed_params),
        changed_params=changed_params[:5],
        n_changed_buffers=len(changed_buffers),
        changed_buffers=changed_buffers[:5],
        counter_deltas={key: counters[key] - before_counters.get(key, 0)
                        for key in MY_COUNTER_KEYS},
        nonfinite_loss_keys=sorted(key for key, value in loss_values.items()
                                   if not np.isfinite(value)),
        reweight_log=copy.deepcopy(
            list(getattr(getattr(model.student2.roi_head, "bbox_head", None),
                         "reweight_log", []))),
        grad_digests=digest_grads(records),
    )
    kept_grads = {name: grad.detach().clone() for name, grad in records}
    del losses, inputs, optimizer, model, scaled_loss
    torch.cuda.empty_cache()
    gc.collect()
    return info, kept_grads


def compare_grads(left, right):
    """Per-parameter gradient difference between two variants of one batch."""
    names = sorted(set(left["grad_digests"]) | set(right["grad_digests"]))
    differing, max_abs = [], 0.0
    for name in names:
        a, b = left["grad_digests"].get(name), right["grad_digests"].get(name)
        if a != b:
            differing.append(name)
    rows = []
    for name in differing:
        if name in left["grads_kept"] and name in right["grads_kept"]:
            diff = (left["grads_kept"][name].float()
                    - right["grads_kept"][name].float()).abs().max().item()
            max_abs = max(max_abs, diff)
            rows.append(dict(name=name, max_abs_diff=diff))
    rows.sort(key=lambda row: -row["max_abs_diff"])
    return dict(n_grad_tensors_total=len(names),
                n_grad_tensors_differing=len(differing),
                max_abs_diff=max_abs,
                top=max_abs, top_differing=rows[:10])


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ablation_configs/fold6_seed678/b0.py")
    parser.add_argument("--checkpoint",
                        default="work_dirs/ablation_v1/fold6_seed678/b0/iter_32000.pth")
    parser.add_argument(
        "--out-dir",
        default="ablation_configs/reweight_small_bkg_20260929_amp_scaled_backward")
    parser.add_argument("--scan-limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing to write into a non-empty directory: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    importlib.import_module(_harness.NEW_MODULE)   # register the new head classes

    started = time.time()
    logger = get_root_logger(log_file=str(out_dir / "harness.log"), log_level="INFO")
    logger.info("[amp-backward] command: %s", " ".join(
        [sys.executable, "tools/verify_reweight_amp_scaled_backward.py"] + sys.argv[1:]))

    receipt = _entry.source_manifest()
    receipt["extensions"] = dict(
        new_module=_harness.NEW_MODULE,
        amp_backward_harness=dict(path=str(Path(__file__).resolve()),
                                  sha256=sha256_file(__file__)),
        retained_harness=dict(path=str(_HARNESS_PATH),
                              sha256=sha256_file(_HARNESS_PATH)),
        supplement_harness=dict(path=str(_SUPPLEMENT_PATH),
                                sha256=sha256_file(_SUPPLEMENT_PATH)),
        supplement_out_dir=dict(path=str(SUPPLEMENT_OUT_DIR)),
        train_ablation=dict(path=str(ROOT / "tools" / "train_ablation.py"),
                            sha256=sha256_file(ROOT / "tools" / "train_ablation.py")),
    )
    write_json(out_dir / "source_receipt.json", receipt)

    counters, proof_opt = install_instrumentation()
    proof_scaler = install_scaler_instrumentation(counters)
    logger.info("[amp-backward] optimizer/EMA instrumentation: %s",
                json.dumps(proof_opt, sort_keys=True))
    logger.info("[amp-backward] scaler instrumentation: %s",
                json.dumps(proof_scaler, sort_keys=True))

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
    if ("SmallBkgReweightRoIHead" not in HEADS
            or "SmallBkgReweightBBoxHead" not in HEADS):
        raise SystemExit("custom_imports did not register the new classes")
    cfg_lambda0 = patch_config(Config.fromfile(str(variant_path)))
    cfg_lambda0.model.model.roi_head.bbox_head.reweight = dict(
        enable=True, lambda_=0.0, max_area=MAX_AREA, tag=BASELINE_TAG)

    # the variant definition under test must be the one the supplement accepted
    accepted_path = SUPPLEMENT_OUT_DIR / VARIANT_CONFIG_NAME
    if not accepted_path.exists():
        raise SystemExit("accepted variant config missing: %s" % accepted_path)
    accepted_roi_head = dict(Config.fromfile(str(accepted_path)).model.model.roi_head)
    variant_same_as_accepted = (
        dict(cfg_lambda1.model.model.roi_head) == accepted_roi_head)
    if not variant_same_as_accepted:
        raise SystemExit("variant roi_head differs from the accepted one")

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
    logger.info("[amp-backward] batch %d tags=%s", batch_index, batch_tags)

    # the accepted run's batch identity: same index/tags and the same sup2 scale
    # factors (recorded in its reweight log), so "same batch" is checked, not assumed
    accepted = json.load(open(SUPPLEMENT_OUT_DIR / "supplement_verdict.json"))
    accepted_env = accepted["environment"]
    accepted_scale = (accepted["stage2_end_to_end_attribution"]["passes"]["fp32"]
                      ["lambda0"]["sup2_reweight_log"][0]["scale_factor"])

    environment = dict(
        seed=seed,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(device),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        batch_index=batch_index,
        batch_tags=batch_tags,
        accepted_batch_index=accepted_env["batch_index"],
        accepted_batch_tags=accepted_env["batch_tags"],
        accepted_sup2_scale_factor=accepted_scale,
        checkpoint=dict(path=str(Path(args.checkpoint).resolve()),
                        sha256=sha256_file(args.checkpoint)),
        config=dict(path=str(Path(args.config).resolve()),
                    sha256=sha256_file(args.config)),
        variant_same_as_accepted=variant_same_as_accepted,
        config_diff_paths=diff_paths,
        amp_training_path=dict(
            config="fp16 = dict(loss_scale='dynamic')",
            hook="mmcv.runner.hooks.Fp16OptimizerHook(loss_scale='dynamic')",
            constructed_in="ssod/apis/train.py:127-131",
            hook_body="scale(loss).backward() -> unscale_(optimizer) -> clip -> step -> update",
        ),
        unscale_rel_tol=UNSCALE_REL_TOL,
    )

    # one snapshot, taken after the batch scan and before any forward
    snapshot = snapshot_rng()
    snapshot_fingerprint = rng_fingerprint()
    logger.info("[amp-backward] fixed RNG snapshot %s", snapshot_fingerprint)

    scaler_state = torch.cuda.amp.GradScaler().state_dict()
    runs, kept = {}, {}
    for variant, cfg in (("lambda0", cfg_lambda0), ("lambda1", cfg_lambda1)):
        logger.info("[amp-backward] variant=%s", variant)
        info, grads = run_scaled(cfg, dataset, batch, args.checkpoint, device,
                                 counters, snapshot, scaler_state, variant)
        info["grads_kept"] = grads
        runs[variant] = info
        kept[variant] = grads
        logger.info(
            "[amp-backward] variant=%s loss=%.9f scaled=%.6f scale=%.1f "
            "grad_tensors=%d inf=%d nan=%d",
            variant, info["total_loss"], info["scaled_loss"],
            info["scaling"]["scale"], info["gradients"]["n_grad_tensors"],
            info["gradients"]["n_grad_tensors_with_inf"],
            info["gradients"]["n_grad_tensors_with_nan"])
        gc.collect()

    grad_comparison = compare_grads(runs["lambda0"], runs["lambda1"])
    loss_keys_changed = sorted(
        key for key in runs["lambda0"]["losses"]
        if runs["lambda1"]["losses"].get(key) != runs["lambda0"]["losses"][key])

    same_batch = (batch_index == accepted_env["batch_index"]
                  and sorted(batch_tags) == sorted(accepted_env["batch_tags"]))
    sup2_scales = [entry.get("scale_factor") for entry
                   in runs["lambda0"]["reweight_log"]]
    same_batch_scale = bool(sup2_scales) and sup2_scales[0] == accepted_scale
    same_rng = (runs["lambda0"]["rng_before"] == runs["lambda1"]["rng_before"]
                == snapshot_fingerprint)
    same_scaler_state = (runs["lambda0"]["scaler"]["state_before"]
                         == runs["lambda1"]["scaler"]["state_before"]
                         == scaler_state)

    checks = collections.OrderedDict()
    checks["batch_and_rng_shared"] = dict(
        same_batch=same_batch,
        same_batch_scale_factor=same_batch_scale,
        same_rng_state=same_rng,
        rng_before=runs["lambda0"]["rng_before"],
        snapshot_fingerprint=snapshot_fingerprint,
        passed=bool(same_batch and same_batch_scale and same_rng),
    )
    checks["scaler_state_shared_and_recorded"] = dict(
        provenance=runs["lambda0"]["scaler"]["provenance"],
        state_before=runs["lambda0"]["scaler"]["state_before"],
        equal_across_variants=same_scaler_state,
        passed=bool(same_scaler_state),
    )
    checks["scaled_backward_unscale"] = dict(
        per_variant={variant: dict(
            scale=runs[variant]["scaler"]["scale_before"],
            scaled_loss=runs[variant]["scaled_loss"],
            scaled_loss_equals_loss_times_scale=runs[variant]["scaling"][
                "scaled_loss_equals_loss_times_scale"],
            scaled_grad_abs_sum=runs[variant]["scaling"]["scaled_grad_abs_sum"],
            unscaled_grad_abs_sum=runs[variant]["scaling"]["unscaled_grad_abs_sum"],
            inverse_scale=runs[variant]["scaling"]["inverse_scale"],
            unscaled_over_scaled=runs[variant]["scaling"]["unscaled_over_scaled"],
        ) for variant in VARIANTS},
        passed=all(runs[v]["scaling"]["scaled_loss_equals_loss_times_scale"]
                   for v in VARIANTS),
    )
    checks["unscaled_gradients_finite"] = dict(
        per_variant={variant: runs[variant]["gradients"] for variant in VARIANTS},
        grad_dtypes=sorted({dtype for variant in VARIANTS
                            for dtype in runs[variant]["grad_dtypes"]}),
        passed=all(runs[v]["gradients"]["n_grad_tensors_with_inf"] == 0
                   and runs[v]["gradients"]["n_grad_tensors_with_nan"] == 0
                   and not runs[v]["nonfinite_loss_keys"] for v in VARIANTS),
    )
    checks["teacher_gradients_absent"] = dict(
        per_variant={variant: runs[variant]["teacher"] for variant in VARIANTS},
        passed=all(runs[v]["teacher"]["n_teacher_params_with_grad"] == 0
                   and runs[v]["teacher"]["n_teacher_params_requiring_grad"] == 0
                   for v in VARIANTS),
    )
    checks["parameters_and_buffers_unchanged"] = dict(
        per_variant={variant: dict(
            optimizer=runs[variant]["optimizer"],
            params_unchanged=runs[variant]["params_unchanged"],
            changed_params=runs[variant]["changed_params"],
            n_changed_buffers=runs[variant]["n_changed_buffers"],
        ) for variant in VARIANTS},
        passed=all(runs[v]["params_unchanged"]
                   and runs[v]["n_changed_buffers"] == 0 for v in VARIANTS),
    )
    no_writes = {}
    for variant in VARIANTS:
        deltas = runs[variant]["counter_deltas"]
        no_writes[variant] = dict(
            counter_deltas=deltas,
            scale_unchanged=runs[variant]["scaler"]["scale_unchanged"],
            scaler_state_unchanged=runs[variant]["scaler"]["state_unchanged"],
        )
    checks["no_step_no_scaler_update_no_ema"] = dict(
        per_variant=no_writes,
        counters_checked=list(MY_COUNTER_KEYS),
        passed=all(all(runs[v]["counter_deltas"].get(key, 0) == 0
                       for key in MY_COUNTER_KEYS)
                   and runs[v]["scaler"]["scale_unchanged"]
                   and runs[v]["scaler"]["state_unchanged"] for v in VARIANTS),
    )

    outstanding = [name for name, block in checks.items() if not block["passed"]]
    verdict = collections.OrderedDict((
        ("outstanding", outstanding),
        ("overall_status",
         "AMP GradScaler backward check passed" if not outstanding
         else "AMP GradScaler backward check incomplete: %s" % ", ".join(outstanding)),
        ("scope", "only the GradScaler gap; A1-A6 and the supplement are not re-run"),
        ("checks", checks),
        ("lambda0_vs_lambda1", dict(
            changed_loss_keys=loss_keys_changed,
            total_loss=dict(lambda0=runs["lambda0"]["total_loss"],
                            lambda1=runs["lambda1"]["total_loss"],
                            abs_diff=abs(runs["lambda1"]["total_loss"]
                                         - runs["lambda0"]["total_loss"])),
            sup2_loss_cls=dict(lambda0=runs["lambda0"]["losses"].get("sup2_loss_cls"),
                               lambda1=runs["lambda1"]["losses"].get("sup2_loss_cls")),
            unscaled_grads=grad_comparison,
            note=("reported, not a pass condition: a change in sup2_loss_cls is "
                  "expected to reach every student2 parameter its graph touches"),
        )),
        ("known_gaps", [
            "AMP B0/B0-repeat was never run (supplement stage 1 is fp32 only); "
            "AMP replay repeatability is covered only by B0-vs-lambda0, whose "
            "losses and 39 traced events are bitwise equal",
        ]),
        ("not_an_effect_claim",
         "this only shows the reweight is numerically live on the AMP training "
         "path; it says nothing about detection accuracy"),
    ))

    payload = collections.OrderedDict((
        ("environment", environment),
        ("instrumentation", dict(optimizer_ema=proof_opt, scaler=proof_scaler)),
        ("runs", {variant: {key: value for key, value in runs[variant].items()
                            if key != "grads_kept"} for variant in VARIANTS}),
        ("verdict", verdict),
        ("elapsed_seconds", time.time() - started),
    ))
    write_json(out_dir / "amp_scaled_backward_verdict.json", payload)
    logger.info("[amp-backward] outstanding=%s", outstanding)
    logger.info("[amp-backward] status=%s", verdict["overall_status"])
    del kept
    print(json.dumps(dict(out_dir=str(out_dir), outstanding=outstanding,
                          overall_status=verdict["overall_status"]),
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
