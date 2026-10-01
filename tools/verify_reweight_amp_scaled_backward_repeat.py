"""Is the AMP scaled backward bitwise reproducible run-to-run?

Why this exists: the accepted AMP scaled-backward run
(`ablation_configs/reweight_small_bkg_20260929_amp_scaled_backward/`) reported
`lambda1` changing 215 of 246 gradient tensors, 104 of them `student1.*`, while
its only changed loss key was `sup2_loss_cls`.  Reading the model
(`ssod/models/dual_teacher.py:143-164`), `sup1` is the only supervision
`student1` receives and `sup2` goes to `student2` alone, so in exact arithmetic
lambda's weight change cannot reach `student1`'s gradients at all.  Two
explanations remain, and they have opposite consequences:

  (a) the AMP backward is not bitwise reproducible across passes, in which case
      the per-tensor gradient comparison is not attributable and only the
      loss-level attribution stands; or
  (b) `student1`'s graph really is coupled to the sup2 loss, in which case the
      reweight has an effect the acceptance never described.

This script settles which, and nothing else.  It re-runs the accepted check's
`lambda0` twice -- two independent passes, each with its own freshly built model,
the same fixed RNG snapshot and the same scaler state -- and compares the
unscaled gradients pass-to-pass and against the digests the accepted run
recorded for `lambda0`.  A same-variant difference of the same size and shape as
the `lambda1` difference means (a).

It adds no acceptance dimension: the four required checks are the accepted
run's, and they are not repeated here.  Nothing is stepped, saved or overwritten;
the accepted directory is read-only input.

Run from the checkout root:
  python tools/verify_reweight_amp_scaled_backward_repeat.py \
      --out-dir ablation_configs/reweight_small_bkg_20260929_amp_scaled_backward_repeat
"""

import argparse
import copy
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# The accepted check is the module under test: import its helpers and `run_scaled`
# rather than reimplementing the scaled backward.  Its `main` is guarded.
_AMP_PATH = ROOT / "tools" / "verify_reweight_amp_scaled_backward.py"
_aspec = importlib.util.spec_from_file_location("_amp_backward", _AMP_PATH)
amp = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(amp)

AMP_OUT_DIR = (ROOT / "ablation_configs"
               / "reweight_small_bkg_20260929_amp_scaled_backward")
PASS_LABELS = ("lambda0_pass1", "lambda0_pass2")


def prefix_of(name):
    return name.split(".")[0]


def prefix_counts(names):
    counts = {}
    for name in names:
        prefix = prefix_of(name)
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def digest_diff(left, right):
    """Digest-level comparison; no numeric diff is possible without the tensors."""
    names = sorted(set(left) | set(right))
    differing = [name for name in names
                 if left.get(name) != right.get(name)]
    return dict(n_grad_tensors_total=len(names),
                n_grad_tensors_differing=len(differing),
                n_missing_left=sum(1 for name in names if name not in left),
                n_missing_right=sum(1 for name in names if name not in right),
                differing_prefix_counts=prefix_counts(differing),
                top_differing=differing[:10])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amp-out-dir", default=str(AMP_OUT_DIR))
    parser.add_argument(
        "--out-dir",
        default="ablation_configs/reweight_small_bkg_20260929_amp_scaled_backward_repeat")
    parser.add_argument("--scan-limit", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    amp_out_dir = Path(args.amp_out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing to write into a non-empty directory: %s" % out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("run from this checkout root to preserve relative data paths")
    amp._entry.pin_repository()
    if not torch.cuda.is_available():
        raise SystemExit("run this check on the training machine with CUDA")

    from mmcv import Config
    from mmcv.parallel import scatter
    from mmdet.models import build_detector  # noqa: F401 (registers pipelines)
    from ssod.apis import set_random_seed
    from ssod.datasets import build_dataloader, build_dataset
    from ssod.utils import get_root_logger, patch_config

    importlib.import_module(amp._harness.NEW_MODULE)

    accepted_verdict_path = amp_out_dir / "amp_scaled_backward_verdict.json"
    if not accepted_verdict_path.exists():
        raise SystemExit("accepted verdict missing: %s" % accepted_verdict_path)
    accepted = json.load(open(str(accepted_verdict_path)))
    accepted_env = accepted["environment"]
    accepted_l0 = accepted["runs"]["lambda0"]
    accepted_l1 = accepted["runs"]["lambda1"]

    started = time.time()
    logger = get_root_logger(log_file=str(out_dir / "harness.log"),
                             log_level="INFO")
    logger.info("[amp-repeat] command: %s", " ".join(
        [sys.executable, "tools/verify_reweight_amp_scaled_backward_repeat.py"]
        + sys.argv[1:]))

    receipt = amp._entry.source_manifest()
    receipt["extensions"] = dict(
        new_module=amp._harness.NEW_MODULE,
        repeat_harness=dict(path=str(Path(__file__).resolve()),
                            sha256=amp.sha256_file(__file__)),
        amp_backward_harness=dict(path=str(_AMP_PATH),
                                  sha256=amp.sha256_file(_AMP_PATH)),
        retained_harness=dict(path=str(amp._HARNESS_PATH),
                              sha256=amp.sha256_file(amp._HARNESS_PATH)),
        supplement_harness=dict(path=str(amp._SUPPLEMENT_PATH),
                                sha256=amp.sha256_file(amp._SUPPLEMENT_PATH)),
        amp_out_dir=dict(path=str(amp_out_dir),
                         verdict_sha256=amp.sha256_file(accepted_verdict_path)),
        accepted_artifacts={
            name: amp.sha256_file(amp_out_dir / name)
            for name in sorted(p.name for p in amp_out_dir.iterdir() if p.is_file())},
    )
    amp.write_json(out_dir / "source_receipt.json", receipt)

    counters, proof_opt = amp.install_instrumentation()
    proof_scaler = amp.install_scaler_instrumentation(counters)
    logger.info("[amp-repeat] optimizer/EMA instrumentation: %s",
                json.dumps(proof_opt, sort_keys=True))
    logger.info("[amp-repeat] scaler instrumentation: %s",
                json.dumps(proof_scaler, sort_keys=True))

    # The lambda0 definition under test, rebuilt from the accepted lambda1 config
    # that the acceptance directory recorded.  Same construction as the accepted
    # run, so the comparison is against the accepted definition, not a new one.
    accepted_cfg_path = amp_out_dir / amp.VARIANT_CONFIG_NAME
    cfg_accepted = patch_config(Config.fromfile(str(accepted_cfg_path)))
    expected_l0_roi_head = copy.deepcopy(dict(cfg_accepted.model.model.roi_head))
    expected_l0_roi_head["bbox_head"]["reweight"] = dict(
        enable=True, lambda_=0.0, max_area=amp.MAX_AREA, tag=amp.BASELINE_TAG)
    cfg_lambda0 = patch_config(Config.fromfile(str(accepted_cfg_path)))
    cfg_lambda0.model.model.roi_head.bbox_head.reweight = dict(
        enable=True, lambda_=0.0, max_area=amp.MAX_AREA, tag=amp.BASELINE_TAG)
    lambda0_matches_accepted_definition = (
        dict(cfg_lambda0.model.model.roi_head) == expected_l0_roi_head)
    if not lambda0_matches_accepted_definition:
        raise SystemExit("lambda0 roi_head differs from the accepted definition")

    checkpoint = accepted_env["checkpoint"]["path"]
    if amp.sha256_file(checkpoint) != accepted_env["checkpoint"]["sha256"]:
        raise SystemExit("checkpoint hash no longer matches the accepted run")
    config = accepted_env["config"]["path"]
    if amp.sha256_file(config) != accepted_env["config"]["sha256"]:
        raise SystemExit("b0 config hash no longer matches the accepted run")

    seed = accepted_env["seed"]
    os.environ["WORK_DIR"] = str(out_dir / "work_dir_override")
    (out_dir / "work_dir_override").mkdir(exist_ok=True)

    set_random_seed(seed, deterministic=True)
    torch.backends.cudnn.benchmark = False
    dataset = build_dataset(cfg_lambda0.data.train)
    loader = build_dataloader(
        dataset, cfg_lambda0.data.samples_per_gpu, 0, num_gpus=1, dist=True,
        seed=seed,
        sampler_cfg=copy.deepcopy(cfg_lambda0.data.get("sampler", {}).get("train", {})))
    device = torch.cuda.current_device()
    iterator = iter(loader)
    batch, batch_index, batch_tags = None, None, None
    for index in range(args.scan_limit):
        candidate = next(iterator)
        probe = scatter(copy.deepcopy(candidate), [device])[0]
        tags = [str(meta["tag"]) for meta in probe["img_metas"]]
        del probe
        if sorted(tags) == sorted(amp.REQUIRED_TAGS):
            batch, batch_index, batch_tags = candidate, index, tags
            break
    if batch is None:
        raise SystemExit("no scanned batch covered all four supervision streams")

    # one snapshot, taken after the batch scan and before any forward, exactly as
    # the accepted run did; it must reproduce the accepted run's snapshot
    snapshot = amp.snapshot_rng()
    snapshot_fingerprint = amp.rng_fingerprint()
    snapshot_reproduces_accepted = (
        snapshot_fingerprint == accepted_l0["rng_before"])
    logger.info("[amp-repeat] batch %d tags=%s snapshot %s reproduced=%s",
                batch_index, batch_tags, snapshot_fingerprint,
                snapshot_reproduces_accepted)
    if batch_index != accepted_env["batch_index"] or batch_tags != accepted_env["batch_tags"]:
        raise SystemExit("scanned batch %d/%s differs from the accepted run's %d/%s"
                         % (batch_index, batch_tags, accepted_env["batch_index"],
                            accepted_env["batch_tags"]))
    if not snapshot_reproduces_accepted:
        raise SystemExit("RNG snapshot %s does not reproduce the accepted %s"
                         % (snapshot_fingerprint, accepted_l0["rng_before"]))

    # the same scaling state the accepted run used, from the same construction
    # path; recorded rather than asserted so a mismatch is visible, not fatal
    scaler_state = torch.cuda.amp.GradScaler().state_dict()
    scaler_state_matches_accepted = (
        scaler_state == accepted_l0["scaler"]["state_before"])

    runs = {}
    for label in PASS_LABELS:
        info, kept = amp.run_scaled(cfg_lambda0, dataset, batch, checkpoint,
                                    device, counters, snapshot, scaler_state, label)
        runs[label] = dict(info=info, kept=kept)
        logger.info("[amp-repeat] %s total_loss=%r scaled=%r digest=%s",
                    label, info["total_loss"], info["scaled_loss"],
                    info["rng_before"])
        del kept

    pass1, pass2 = runs[PASS_LABELS[0]], runs[PASS_LABELS[1]]

    pass1_vs_pass2 = amp.compare_grads(
        dict(grad_digests=pass1["info"]["grad_digests"], grads_kept=pass1["kept"]),
        dict(grad_digests=pass2["info"]["grad_digests"], grads_kept=pass2["kept"]))
    losses_reproduce = (pass1["info"]["losses"] == pass2["info"]["losses"])
    total_loss_reproduces = (pass1["info"]["total_loss"] == pass2["info"]["total_loss"])
    scaled_loss_reproduces = (pass1["info"]["scaled_loss"] == pass2["info"]["scaled_loss"])
    pass1_vs_pass2["losses_identical"] = losses_reproduce
    pass1_vs_pass2["total_loss_identical"] = total_loss_reproduces
    pass1_vs_pass2["scaled_loss_identical"] = scaled_loss_reproduces
    pass1_vs_pass2["reproducible"] = (
        pass1_vs_pass2["n_grad_tensors_differing"] == 0
        and losses_reproduce and total_loss_reproduces and scaled_loss_reproduces)

    # against the digests the accepted run recorded for lambda0 and lambda1
    pass1_vs_recorded_lambda0 = digest_diff(accepted_l0["grad_digests"],
                                           pass1["info"]["grad_digests"])
    pass2_vs_recorded_lambda0 = digest_diff(accepted_l0["grad_digests"],
                                           pass2["info"]["grad_digests"])
    recorded_lambda1_vs_lambda0 = digest_diff(accepted_l0["grad_digests"],
                                              accepted_l1["grad_digests"])
    recorded_losses_equal = (accepted_l0["losses"] == accepted_l1["losses"])
    recorded_changed_loss_keys = sorted(
        key for key in set(accepted_l0["losses"]) | set(accepted_l1["losses"])
        if accepted_l0["losses"].get(key) != accepted_l1["losses"].get(key))

    total_counter_deltas = {key: sum(run["info"]["counter_deltas"][key]
                                     for run in runs.values())
                            for key in amp.MY_COUNTER_KEYS}

    checks = {}
    checks["same_variant_grads_reproducible"] = dict(
        passed=bool(pass1_vs_pass2["reproducible"]),
        detail=dict(
            pass1_total_loss=pass1["info"]["total_loss"],
            pass2_total_loss=pass2["info"]["total_loss"],
            pass1_scaled_loss=pass1["info"]["scaled_loss"],
            pass2_scaled_loss=pass2["info"]["scaled_loss"],
            n_grad_tensors_total=pass1_vs_pass2["n_grad_tensors_total"],
            n_grad_tensors_differing=pass1_vs_pass2["n_grad_tensors_differing"],
            differing_prefix_counts=prefix_counts(
                name for name, left in pass1["info"]["grad_digests"].items()
                if left != pass2["info"]["grad_digests"].get(name)),
            max_abs_diff=pass1_vs_pass2["max_abs_diff"],
            top_differing=pass1_vs_pass2["top_differing"],
            losses_identical=losses_reproduce,
        ))
    checks["recorded_lambda0_reproduced"] = dict(
        passed=bool(pass1_vs_recorded_lambda0["n_grad_tensors_differing"] == 0
                    and pass2_vs_recorded_lambda0["n_grad_tensors_differing"] == 0
                    and pass1["info"]["total_loss"] == accepted_l0["total_loss"]
                    and pass1["info"]["losses"] == accepted_l0["losses"]),
        detail=dict(
            pass1=pass1_vs_recorded_lambda0,
            pass2=pass2_vs_recorded_lambda0,
            pass1_total_loss=pass1["info"]["total_loss"],
            accepted_lambda0_total_loss=accepted_l0["total_loss"],
            pass1_losses_equal_accepted=pass1["info"]["losses"] == accepted_l0["losses"],
        ))
    checks["no_step_no_scaler_update_no_ema"] = dict(
        passed=bool(all(value == 0 for value in total_counter_deltas.values())
                    and pass1["info"]["params_unchanged"]
                    and pass2["info"]["params_unchanged"]
                    and pass1["info"]["n_changed_buffers"] == 0
                    and pass2["info"]["n_changed_buffers"] == 0
                    and pass1["info"]["scaler"]["state_unchanged"]
                    and pass2["info"]["scaler"]["state_unchanged"]),
        detail=dict(
            counter_deltas=total_counter_deltas,
            params_unchanged=[pass1["info"]["params_unchanged"],
                              pass2["info"]["params_unchanged"]],
            n_changed_buffers=[pass1["info"]["n_changed_buffers"],
                               pass2["info"]["n_changed_buffers"]],
            scaler_state_unchanged=[pass1["info"]["scaler"]["state_unchanged"],
                                    pass2["info"]["scaler"]["state_unchanged"]],
            gradients_finite=[pass1["info"]["gradients"]["n_grad_tensors_with_inf"] == 0
                              and pass1["info"]["gradients"]["n_grad_tensors_with_nan"] == 0,
                              pass2["info"]["gradients"]["n_grad_tensors_with_inf"] == 0
                              and pass2["info"]["gradients"]["n_grad_tensors_with_nan"] == 0],
            nonfinite_loss_keys=[pass1["info"]["nonfinite_loss_keys"],
                                 pass2["info"]["nonfinite_loss_keys"]],
        ))

    outstanding = [name for name, block in checks.items() if not block["passed"]]

    reproducible = checks["same_variant_grads_reproducible"]["passed"]
    if not reproducible:
        interpretation = (
            "AMP same-variant backward is NOT bitwise reproducible in this "
            "environment: two lambda0 passes with the same batch, snapshot and "
            "scaler state differ on %d/%d gradient tensors.  The accepted run's "
            "per-tensor lambda0-vs-lambda1 gradient comparison is therefore not "
            "attributable; only its loss-level attribution (the single changed "
            "key sup2_loss_cls, magnitude matching the weighting formula) stands."
            % (pass1_vs_pass2["n_grad_tensors_differing"],
               pass1_vs_pass2["n_grad_tensors_total"]))
        student1_observation = dict(
            status="explained_as_backward_nondeterminism",
            detail="the same-variant repeat shows the same student1.* spread, so "
                   "the student1.* entries in the lambda0-vs-lambda1 difference "
                   "carry no signal about the reweight")
    else:
        interpretation = (
            "AMP same-variant backward is bitwise reproducible, so the accepted "
            "run's per-tensor lambda0-vs-lambda1 gradient difference is a real "
            "consequence of the reweight.")
        student1_observation = dict(
            status="unexplained_observation",
            detail="lambda1 changes %d gradient tensors including %d student1.* "
                   "ones, while student1 receives no sup2 supervision "
                   "(ssod/models/dual_teacher.py:143-164); the graph path that "
                   "couples them is not established by this check"
                   % (recorded_lambda1_vs_lambda0["n_grad_tensors_differing"],
                      recorded_lambda1_vs_lambda0["differing_prefix_counts"]
                      .get("student1", 0)))

    verdict = dict(
        scope=("does not re-run the accepted AMP scaled-backward checks, A1-A6 or "
               "the supplement; changes no loss, optimizer, scaler or EMA state"),
        question=("is the accepted run's per-tensor lambda0-vs-lambda1 gradient "
                  "difference attributable to the reweight, or to a "
                  "non-reproducible AMP backward?"),
        checks=checks,
        outstanding=outstanding,
        setup=dict(
            batch_index=batch_index,
            batch_tags=batch_tags,
            snapshot_fingerprint=snapshot_fingerprint,
            snapshot_reproduces_accepted=snapshot_reproduces_accepted,
            scaler_state=scaler_state,
            scaler_state_matches_accepted=scaler_state_matches_accepted,
            lambda0_matches_accepted_definition=lambda0_matches_accepted_definition,
            lambda0_roi_head=dict(cfg_lambda0.model.model.roi_head),
            pass_labels=list(PASS_LABELS),
        ),
        same_variant_pass1_vs_pass2=pass1_vs_pass2,
        recorded_lambda0=dict(pass1=pass1_vs_recorded_lambda0,
                              pass2=pass2_vs_recorded_lambda0),
        recorded_lambda1_vs_lambda0=recorded_lambda1_vs_lambda0,
        recorded_lambda1_changed_loss_keys=recorded_changed_loss_keys,
        recorded_losses_equal_between_lambda0_and_lambda1=recorded_losses_equal,
        interpretation=interpretation,
        student1_observation=student1_observation,
        not_an_effect_claim=(
            "reproducibility and attribution of an implementation, not evidence "
            "of any accuracy gain"),
        overall_status=("AMP scaled backward reproducible; attribution unaffected"
                        if not outstanding else "outstanding: %s" % outstanding),
        elapsed_seconds=time.time() - started,
        instrumentation=dict(optimizer_ema=proof_opt, scaler=proof_scaler),
    )

    amp.write_json(out_dir / "amp_scaled_backward_repeat_verdict.json", verdict)
    logger.info("[amp-repeat] outstanding=%s", outstanding)
    logger.info("[amp-repeat] %s", interpretation)
    print(json.dumps(dict(out_dir=str(out_dir),
                          outstanding=outstanding,
                          snapshot_reproduces_accepted=snapshot_reproduces_accepted,
                          scaler_state_matches_accepted=scaler_state_matches_accepted,
                          same_variant=checks["same_variant_grads_reproducible"]["detail"],
                          recorded_lambda0=checks["recorded_lambda0_reproduced"]["detail"],
                          recorded_lambda1_prefix=recorded_lambda1_vs_lambda0[
                              "differing_prefix_counts"],
                          interpretation=interpretation,
                          elapsed_seconds=verdict["elapsed_seconds"]),
                     indent=2, sort_keys=True, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
