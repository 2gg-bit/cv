"""Real CUDA training-batch ablation check, without optimizer or EMA updates.

Use train_ablation.py --check-step. Only training data and Phase1/2 are read.
An uncovered M2 positive branch is incomplete (exit 2), never a false PASS.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


class IncompleteCheck(RuntimeError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_losses(losses):
    return {key + ":" + str(i): tensor.detach().float().cpu().clone()
            for key, value in losses.items()
            for i, tensor in enumerate(value if isinstance(value, (list, tuple)) else [value])}


def compare_losses(kind, results, ratio=0.5):
    off, on = results["off"], results["on"]
    if kind == "fg":
        if set(on) - set(off) != {"sup1_loss_foreground:0", "sup2_loss_foreground:0"}:
            raise AssertionError("FG auxiliary losses must occur on sup1/sup2 only")
    elif set(off) != set(on):
        raise AssertionError("Unexpected detection loss keys")
    changed = []
    for key, value in off.items():
        scale = 1.0
        if kind.startswith("pg_") and "loss" in key:
            if key.startswith("sup2_") or (kind == "pg_both" and key.startswith("unsup2_")):
                scale = ratio
        if key not in on:
            raise AssertionError("Missing baseline loss: " + key)
        equal = torch.allclose(value * scale, on[key], atol=1e-6, rtol=1e-5)
        if kind == "m2" and key in ("unsup1_loss_cls:0", "unsup2_loss_cls:0"):
            if not equal:
                changed.append(key)
        elif not equal:
            raise AssertionError("Unexpected change to " + key)
    if kind == "m2" and len(changed) != 2:
        raise IncompleteCheck("M2 must change both positive classification branches; try another --batch-index")
    if "identity" in results:
        identity = results["identity"]
        if set(identity) != set(off) or any(
                not torch.allclose(v, identity[k], atol=1e-6, rtol=1e-5) for k, v in off.items()):
            raise AssertionError("Identity setting does not recover B0 losses")


def run(args, output, report):
    from mmcv import Config, DictAction
    from mmcv.parallel import scatter
    from mmcv.runner import wrap_fp16_model
    from mmdet.models import build_detector
    from ssod.apis import set_random_seed
    from ssod.datasets import build_dataset, build_dataloader
    from ssod.utils import get_root_logger, patch_config

    if not torch.cuda.is_available():
        raise RuntimeError("Run this check on the training machine with CUDA")
    cfg = Config.fromfile(args.config)
    if cfg.get("seed") is not None and cfg.seed != args.seed:
        raise ValueError("Use the seed of the frozen B0 config")
    cfg.seed, cfg.work_dir = args.seed, str(output)
    cfg = patch_config(cfg)
    tc = cfg.model.train_cfg
    fg = cfg.model.model.roi_head.type == "ForegroundRoIHead"
    pg_cfg = tc.get("pg")
    if (cfg.model.type != "DualTeacher" or sum((bool(tc.get("m2_enabled")), fg, bool(pg_cfg))) != 1
            or tc.get("m2_force_weight_one") or tc.get("mvdt_enabled") or tc.get("m3_enabled")
            or cfg.get("load_from") or cfg.get("resume_from")):
        raise ValueError("Use a fresh config with exactly one experimental feature")
    kind = "m2" if tc.get("m2_enabled") else "fg" if fg else "pg_" + pg_cfg.mode
    cfg.dump(str(output / "resolved_config.py"))
    get_root_logger(log_file=str(output / "acceptance.log"), log_level="INFO")
    set_random_seed(args.seed, deterministic=True)
    model = build_detector(cfg.model)
    model.init_weights()
    model.init_from_pretrained()
    if fg and any(not getattr(model, n).roi_head.foreground_enabled or
                  getattr(model, n).roi_head.foreground_loss_weight <= 0 for n in model.submodules):
        raise ValueError("Every FG branch must enable its auxiliary head")
    report.update(experiment=kind, config_sha256=file_hash(args.config),
                  seed=args.seed, batch_index=args.batch_index,
                  initialization={str(Path(p).resolve()): file_hash(p)
                                  for p in (model.load1_from, model.load2_from)})
    dataset = build_dataset(cfg.data.train)
    # Match tools/train.py: positive pseudo-label visualization reads CLASSES.
    model.CLASSES = dataset.CLASSES
    loader = build_dataloader(dataset, cfg.data.samples_per_gpu, 0, num_gpus=1,
                             dist=True, seed=args.seed,
                             sampler_cfg=copy.deepcopy(cfg.data.get("sampler", {}).get("train", {})))
    iterator = iter(loader)
    for _ in range(args.batch_index + 1):
        batch = next(iterator)
    inputs = scatter(copy.deepcopy(batch), [torch.cuda.current_device()])[0]
    report["batch"] = [dict(tag=m["tag"], filename=m.get("filename")) for m in inputs["img_metas"]]
    if {m["tag"] for m in inputs["img_metas"]} != {"sup1", "sup2", "unsup_teacher", "unsup_student"}:
        raise IncompleteCheck("Selected batch does not cover every supervision stream")
    del inputs
    model.cuda()
    fp16 = cfg.get("fp16") is not None
    if fp16:
        wrap_fp16_model(model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    report.update(fp16=fp16, optimizer_updates=0, ema_updates=0)
    pg = model.pg
    original_log = model._log_m2_stats
    observations = {}

    def capture(branch, num_pos, num_neg, normalizer, w_pos=None):
        observations[branch] = dict(positive=num_pos, negative=num_neg,
                                   downweighted=0 if w_pos is None else int((w_pos < 1).sum()))
        original_log(branch, num_pos, num_neg, normalizer, w_pos)
    model._log_m2_stats = capture
    results = {}
    modes = ("off", "on") if fg else ("off", "on", "identity")
    for mode in modes:
        model.pg = pg
        model.load_state_dict(state, strict=True)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
        model.train()
        for name in ("teacher1", "teacher2"):
            model.freeze(name)
        if fg:
            for name in model.submodules:
                getattr(model, name).roi_head.foreground_enabled = mode != "off"
        model.m2_enabled = kind == "m2" and mode != "off"
        model.m2_force_weight_one = kind == "m2" and mode == "identity"
        if pg is not None:
            if mode == "off":
                model.pg = None
            else:
                pg.set_iteration(0 if mode == "on" else pg.total_iters - 1)
        for parameter in model.parameters():
            parameter.grad = None
        inputs = scatter(copy.deepcopy(batch), [torch.cuda.current_device()])[0]
        observations.clear()
        with torch.cuda.amp.autocast(enabled=fp16):
            losses = model(return_loss=True, **inputs)
            total, _ = model._parse_losses(losses)
        model.pg = pg
        if not bool(torch.isfinite(total)):
            raise AssertionError("Nonfinite training loss")
        total.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if name.startswith("teacher") and (parameter.requires_grad or parameter.grad is not None):
                raise AssertionError("Teacher received gradients: " + name)
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError("Nonfinite gradient: " + name)
            if name.startswith("student") and ".foreground_head." in name:
                gradients[name] = None if parameter.grad is None else float(parameter.grad.float().norm())
        if fg:
            if mode == "on" and (not gradients or any(v is None or v == 0 for v in gradients.values())):
                raise AssertionError("Enabled foreground parameters lack nonzero gradients")
            if mode == "off" and any(v is not None for v in gradients.values()):
                raise AssertionError("Disabled foreground parameters received gradients")
        results[mode] = flatten_losses(losses)
        report[mode] = dict(losses={k: float(v.mean()) for k, v in results[mode].items()},
                            foreground_gradient_norms=gradients, m2=copy.deepcopy(observations))
        del losses, total, inputs
    compare_losses(kind, results, pg.start_ratio if pg is not None else 0.5)
    if kind == "m2" and any(report["on"]["m2"].get(n, {}).get("downweighted", 0) == 0
                           for n in ("unsup1", "unsup2")):
        raise IncompleteCheck("No downweighted positives in one M2 branch")
    report["loss_routing_verified"] = True
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--out-dir", required=True, help="New directory; never reused")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-index", type=int, default=0)
    args = parser.parse_args()
    if args.batch_index < 0:
        parser.error("--batch-index must be nonnegative")
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="failed")
    try:
        run(args, output, report)
    except IncompleteCheck as exc:
        report.update(status="incomplete", error=str(exc))
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Ablation real-step acceptance PASS: " + str(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
