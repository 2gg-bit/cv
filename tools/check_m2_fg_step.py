"""One real training batch, FG off/on, backward, no optimizer/EMA updates.

Run through train_m2_fg.py --check-step so actual imports are root-pinned.
Uses only training data and Phase1/2 checkpoints, not dev labels or AP.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import scatter
from mmcv.runner import wrap_fp16_model
from mmdet.models import build_detector

from ssod.apis import set_random_seed
from ssod.datasets import build_dataset, build_dataloader
from ssod.utils import get_root_logger, patch_config


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_losses(losses):
    output = {}
    for key, value in losses.items():
        for i, tensor in enumerate(value if isinstance(value, (list, tuple)) else [value]):
            output[key + ":" + str(i)] = tensor.detach().float().cpu().clone()
    return output


def run(args, output, report):
    if not torch.cuda.is_available():
        raise RuntimeError("The real-step check requires the training machine's CUDA stack")
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.seed, cfg.work_dir = args.seed, str(output)
    cfg = patch_config(cfg)
    tc = cfg.model.train_cfg
    if (cfg.model.type != "DualTeacher" or not tc.get("m2_enabled") or
            tc.get("m2_force_weight_one") or tc.get("mvdt_enabled") or tc.get("m3_enabled") or
            tc.cls_pseudo_threshold != .9 or cfg.get("load_from") or cfg.get("resume_from")):
        raise ValueError("Use a fresh M2+FG config with fixed 0.9 admission")
    if cfg.model.model.roi_head.type != "ForegroundRoIHead":
        raise ValueError("ForegroundRoIHead required")
    cfg.dump(str(output / "resolved_config.py"))
    get_root_logger(log_file=str(output / "acceptance.log"), log_level="INFO")
    set_random_seed(args.seed, deterministic=True)
    model = build_detector(cfg.model)
    model.init_weights()
    model.init_from_pretrained()
    for branch in model.submodules:
        head = getattr(model, branch).roi_head
        if not head.foreground_enabled or head.foreground_loss_weight <= 0:
            raise ValueError("All branches must enable the foreground head with positive weight")
    report["initialization"] = {
        str(Path(p).resolve()): file_hash(p) for p in (model.load1_from, model.load2_from)}
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, cfg.data.samples_per_gpu, 0, num_gpus=1,
                             dist=True, seed=args.seed,
                             sampler_cfg=copy.deepcopy(cfg.data.get("sampler", {}).get("train", {})))
    batch = next(iter(loader))
    inputs = scatter(copy.deepcopy(batch), [torch.cuda.current_device()])[0]
    report["batch"] = [dict(tag=meta["tag"], filename=meta.get("filename"))
                       for meta in inputs["img_metas"]]
    if {meta["tag"] for meta in inputs["img_metas"]} != {
            "sup1", "sup2", "unsup_teacher", "unsup_student"}:
        raise ValueError("The fixed first batch must contain all supervised/unsupervised streams")
    del inputs
    model.cuda()
    fp16 = cfg.get("fp16") is not None
    if fp16:
        wrap_fp16_model(model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    report["fp16"] = fp16
    report["optimizer_updates"] = report["ema_updates"] = 0
    results = {}
    for enabled in (False, True):
        model.load_state_dict(state, strict=True)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
        model.train()
        for name in ("teacher1", "teacher2"):
            model.freeze(name)
        for name in model.submodules:
            getattr(model, name).roi_head.foreground_enabled = enabled
        for p in model.parameters():
            p.grad = None
        inputs = scatter(copy.deepcopy(batch), [torch.cuda.current_device()])[0]
        with torch.cuda.amp.autocast(enabled=fp16):
            losses = model(return_loss=True, **inputs)
            total, _ = model._parse_losses(losses)
        if not bool(torch.isfinite(total)):
            raise AssertionError("Nonfinite training loss")
        total.backward()
        gradients = {}
        for name, p in model.named_parameters():
            if name.startswith("teacher") and (p.requires_grad or p.grad is not None):
                raise AssertionError("Teacher received gradients: " + name)
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                raise AssertionError("Nonfinite gradient: " + name)
            if name.startswith("student") and ".foreground_head." in name:
                gradients[name] = None if p.grad is None else float(p.grad.float().norm())
        if enabled and any(v is None or v == 0 for v in gradients.values()):
            raise AssertionError("Foreground parameters did not all receive nonzero gradients")
        if not enabled and any(v is not None for v in gradients.values()):
            raise AssertionError("Disabled head received gradients")
        results[enabled] = flatten_losses(losses)
        report["on" if enabled else "off"] = dict(
            losses={k: float(v.mean()) for k, v in results[enabled].items()},
            foreground_gradient_norms=gradients)
        del losses, total, inputs
    off, on = results[False], results[True]
    new_keys = set(on) - set(off)
    if new_keys != {"sup1_loss_foreground:0", "sup2_loss_foreground:0"}:
        raise AssertionError("Unexpected auxiliary loss routing: " + repr(new_keys))
    if any(key not in on or not torch.allclose(value, on[key], atol=1e-6, rtol=1e-5)
           for key, value in off.items()):
        raise AssertionError("Original detection losses changed before any update")
    report["original_losses_unchanged"] = True
    model.load_state_dict(state, strict=True)
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--out-dir", required=True, help="New directory; never reused")
    parser.add_argument("--seed", type=int, default=678)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="failed")
    try:
        run(args, output, report)
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Foreground real-step acceptance PASS: " + str(output))


if __name__ == "__main__":
    main()
