#!/usr/bin/env python
"""M2 cross-seed orchestrator (v3): train -> verify -> record -> next run.

Serial, no human confirmation needed.  The queue stops on ANY of:

  * a run we started exiting non-zero, or its exit code being lost
  * the release verification failing
  * any record failing to land (verify result, VERIFIED.json, ledger)
  * the source / config snapshot changing while a run is in flight
  * insufficient free space before a start
  * an unexpected pre-existing run directory
  * an approval grant that does not match the current request or artefacts

The only run allowed to bypass the exit-code requirement is the designated
handover run (HANDOVER_RUNS): it was started by a previous orchestrator whose
exit code cannot be recovered.  It must pass the same verification and then
needs a one-time manual approval bound to a request id and the artefact
hashes.  It is recorded with exit_code=null, exit_status=unknown_due_to_handover
-- the real exit code is never fabricated as 0.

This file is a NEW version; the previous bash orchestrator is left untouched.
"""

import argparse
import datetime as dt
import json
import os
import os.path as osp
import subprocess
import sys
import time

sys.path.insert(0, "/home/xcc/dual_teacher_project/DualTeacher_m3/tools")

import crossseed_gate as G  # noqa: E402

PY = "/home/xcc/anaconda3/envs/dt/bin/python"
LOG = osp.join(G.PROTOCOL_DIR, "crossseed_orchestrator.log")
MIN_DISK_GB = 20.0
POLL_SECONDS = 60


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = "[%s] %s" % (now(), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def die(msg):
    log("FAIL: " + msg)
    log("QUEUE STOPPED; no further run will be started.")
    sys.exit(1)


# --------------------------------------------------------- process introspection

def iter_processes():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                raw = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        if not raw:
            continue
        yield int(pid), raw.split("\x00")


def training_pids(run_rel):
    """Training processes bound to THIS run (same --work-dir and --seed)."""
    found = []
    for pid, argv in iter_processes():
        if not any("tools/train.py" in a for a in argv):
            continue
        if "--work-dir" not in argv:
            continue
        try:
            wd = argv[argv.index("--work-dir") + 1]
        except IndexError:
            continue
        if osp.normpath(wd) != osp.normpath(run_rel):
            continue
        found.append(pid)
    return found


def any_training():
    return any(any("tools/train.py" in a for a in argv) for _, argv in iter_processes())


# ------------------------------------------------------------------ artefacts

def manifest_path(run_dir_abs):
    return osp.join(run_dir_abs, "run_manifest.json")


def build_manifest(card):
    def h(rel):
        p = osp.join(G.REPO, rel)
        return G.sha256_file(p) if osp.isfile(p) else None
    return {
        "run_key": card["run_key"],
        "run_dir": card["run_dir"],
        "seed": card["seed"], "ver": card["ver"], "fold": card["fold"],
        "expected_iters": card["expected_iters"],
        "created_at": now(),
        "source_dual_teacher_sha256": h(G.SOURCES["dual_teacher"]),
        "config_recipe": card["config_recipe"],
        "config_recipe_sha256": h(card["config_recipe"]),
        "init_phase1_sha256": h(card["expected_load1_from"]),
        "init_phase2_sha256": h(card["expected_load2_from"]),
        "evidence_kind": "captured_before_training_start",
    }


def check_manifest_still_valid(card, manifest):
    """The code, recipe and initialisation must not change mid-run."""
    problems = []
    cur = build_manifest(card)
    for field in ("source_dual_teacher_sha256", "config_recipe_sha256",
                  "init_phase1_sha256", "init_phase2_sha256"):
        if manifest.get(field) != cur.get(field):
            problems.append("%s changed during the run: %r -> %r"
                            % (field, manifest.get(field), cur.get(field)))
    return problems


def run_verifier(card, run_dir_abs, evidence_ts):
    out = osp.join(run_dir_abs, "verify_result.json")
    cmd = [PY, osp.join(G.REPO, "tools/crossseed_verify.py"),
           "--run-dir", run_dir_abs, "--seed", str(card["seed"]),
           "--ver", card["ver"], "--fold", str(card["fold"]),
           "--expected-iters", str(card["expected_iters"]),
           "--expected-max-keep-ckpts", str(card["expected_max_keep_ckpts"]),
           "--evidence-timestamp", evidence_ts, "--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()
    tail += (proc.stderr or "").strip().splitlines()
    for line in tail[-5:]:
        log("    verifier: " + line)
    if proc.returncode != 0:
        return None, out
    try:
        return G.load_json(out), out
    except (OSError, ValueError) as exc:
        die("verify_result.json unreadable after a passing verifier: %s" % exc)


def record(card, result, exit_code, exit_status, manual_release):
    run_dir_abs = osp.join(G.REPO, card["run_dir"])
    sha = result["checks"]["sha256"]
    row = {
        "seed": card["seed"], "ver": card["ver"], "fold": card["fold"],
        "run_dir": card["run_dir"], "sha256": sha,
        "iters": card["expected_iters"],
        "exit_code": "" if exit_code is None else exit_code,
        "exit_status": exit_status,
        "config_snapshot_sha256": result["checks"].get("config_snapshot_sha256"),
        "source_sha256": result["checks"].get("source_sha256"),
        "verified_at": now(),
    }
    try:
        G.ledger_append(row)
        G.atomic_write_json(osp.join(run_dir_abs, "VERIFIED.json"), {
            "run_dir": card["run_dir"], "seed": card["seed"], "ver": card["ver"],
            "fold": card["fold"], "iters": card["expected_iters"], "sha256": sha,
            "exit_code": exit_code, "exit_status": exit_status,
            "manual_release": manual_release,
            "config_snapshot_sha256": row["config_snapshot_sha256"],
            "source_sha256": row["source_sha256"],
            "verified_at": row["verified_at"],
        })
    except Exception as exc:
        die("could not land the records for %s (%s); reconcile before continuing"
            % (card["run_key"], exc))
    log("RECORDED %s sha256=%s exit_status=%s" % (card["run_key"], sha[:16], exit_status))


def reconcile():
    """Bring the ledger back in line with the VERIFIED.json files on disk."""
    fixed = 0
    for seed, ver, fold in G.all_runs():
        card = G.expectation(seed, ver, fold)
        run_dir_abs = osp.join(G.REPO, card["run_dir"])
        vpath = osp.join(run_dir_abs, "VERIFIED.json")
        if not osp.isfile(vpath):
            continue
        try:
            v = G.load_json(vpath)
            result = G.load_json(osp.join(run_dir_abs, "verify_result.json"))
        except (OSError, ValueError):
            log("reconcile: %s has VERIFIED.json but no readable verify_result.json" % card["run_key"])
            continue
        rows = G.ledger_rows()
        if any(r.get("run_dir") == card["run_dir"] and r.get("sha256") == v.get("sha256")
               for r in rows):
            continue
        row = {
            "seed": seed, "ver": ver, "fold": fold, "run_dir": card["run_dir"],
            "sha256": v.get("sha256"), "iters": v.get("iters"),
            "exit_code": "" if v.get("exit_code") is None else v.get("exit_code"),
            "exit_status": v.get("exit_status"),
            "config_snapshot_sha256": v.get("config_snapshot_sha256"),
            "source_sha256": v.get("source_sha256"),
            "verified_at": v.get("verified_at"),
        }
        try:
            G.ledger_append(row)
            fixed += 1
            log("reconcile: appended missing ledger row for %s" % card["run_key"])
        except Exception as exc:
            die("reconcile failed for %s: %s" % (card["run_key"], exc))
    if fixed == 0:
        log("reconcile: ledger already consistent with VERIFIED.json files")


def await_approval(card, run_dir_abs, result):
    ckpt_sha = result["checks"]["sha256"]
    snap_sha = result["checks"].get("config_snapshot_sha256")
    src_sha = result["checks"].get("source_sha256")
    req_id = G.approval_request_id(card["seed"], card["ver"], card["fold"],
                                   ckpt_sha, snap_sha, src_sha)
    req_path = osp.join(run_dir_abs, "HANDOVER_RELEASE_REQUIRED.json")
    grant_path = osp.join(run_dir_abs, "HANDOVER_RELEASED")
    try:
        G.atomic_write_json(req_path, {
            "request_id": req_id,
            "run_key": card["run_key"], "run_dir": card["run_dir"],
            "seed": card["seed"], "ver": card["ver"], "fold": card["fold"],
            "checkpoint_sha256": ckpt_sha,
            "config_snapshot_sha256": snap_sha,
            "source_sha256": src_sha,
            "exit_code": None, "exit_status": "unknown_due_to_handover",
            "verification": "passed",
            "reason": ("started by a previous orchestrator; its real exit code "
                       "cannot be recovered"),
            "release_action": ("write %s with request_id and the artefact hashes" % grant_path),
        })
    except Exception as exc:
        die("could not write the approval request for %s: %s" % (card["run_key"], exc))

    log("HANDOVER: verification PASSED for %s" % card["run_key"])
    log("HANDOVER: approve with request_id=%s" % req_id)
    log("HANDOVER: awaiting one-time manual approval at %s"
        % osp.relpath(grant_path, G.REPO))
    request = G.load_json(req_path)
    while True:
        if osp.isfile(grant_path):
            try:
                grant = G.load_json(grant_path)
            except ValueError as exc:
                die("approval file %s is not valid JSON: %s" % (grant_path, exc))
            ok, why = G.approval_grant_ok(request, grant)
            if not ok:
                die("refusing release of %s: %s" % (card["run_key"], why))
            log("HANDOVER: approval accepted for %s (%s)" % (card["run_key"], why))
            return True
        time.sleep(POLL_SECONDS)


def train(card):
    run_dir_abs = osp.join(G.REPO, card["run_dir"])
    ok, avail, msg = G.disk_ok(G.WORK, MIN_DISK_GB)
    if not ok:
        die("not enough free space before starting %s: %s" % (card["run_key"], msg))
    log("disk before start: %.1fGB free (min %.1fGB)" % (avail, MIN_DISK_GB))

    manifest = build_manifest(card)
    try:
        G.atomic_write_json(manifest_path(run_dir_abs), manifest)
    except Exception as exc:
        die("could not write the run manifest for %s: %s" % (card["run_key"], exc))
    log("=== START %s %s ===" % (card["run_key"], now()))

    train_log = osp.join(G.PROTOCOL_DIR, "crossseed_train_%s.log" % card["run_key"].replace("/", "_"))
    cmd = [PY, "-m", "torch.distributed.launch", "--nproc_per_node=1",
           "tools/train.py", card["config_recipe"], "--launcher", "pytorch",
           "--seed", str(card["seed"]), "--no-validate", "--work-dir", card["run_dir"],
           "--cfg-options", "fold=%d" % card["fold"], "percent=%d" % card["percent"],
           "auto_resume=False", "runner.max_iters=%d" % card["expected_iters"],
           "checkpoint_config.max_keep_ckpts=%d" % card["expected_max_keep_ckpts"]]
    with open(train_log, "a") as fh:
        fh.write("### %s cmd: %s\n" % (now(), " ".join(cmd)))
        fh.flush()
        rc = subprocess.call(cmd, cwd=G.REPO, stdout=fh, stderr=subprocess.STDOUT)
    log("=== END %s rc=%s %s ===" % (card["run_key"], rc, now()))

    action, why = G.decide_after_training(rc, is_handover=False)
    if action != "proceed":
        die("run %s: %s" % (card["run_key"], why))

    problems = check_manifest_still_valid(card, manifest)
    if problems:
        die("run %s: source/recipe/init changed mid-run: %s" % (card["run_key"], problems))
    return rc


def handle_handover(card):
    run_dir_abs = osp.join(G.REPO, card["run_dir"])
    pids = training_pids(card["run_dir"])
    if pids:
        log("handover: %s is in flight (pids %s); waiting" % (card["run_key"], pids))
        while training_pids(card["run_dir"]):
            time.sleep(POLL_SECONDS)
        log("handover: in-flight training for %s has ended" % card["run_key"])
    if not osp.isfile(osp.join(run_dir_abs, "iter_32000.pth")):
        die("handover run %s ended without iter_32000.pth" % card["run_key"])

    existing = osp.join(run_dir_abs, "verify_result.json")
    if osp.isfile(existing):
        try:
            result = G.load_json(existing)
        except ValueError as exc:
            die("existing verify_result.json for %s is unreadable: %s" % (card["run_key"], exc))
        if not result.get("ok"):
            die("existing verify_result.json for %s records a FAILURE; investigate"
                % card["run_key"])
        log("handover: reusing the existing passing verify_result.json for %s" % card["run_key"])
    else:
        result, _ = run_verifier(card, run_dir_abs, "handover_verification")
        if result is None:
            die("verification failed for handover run %s" % card["run_key"])
    await_approval(card, run_dir_abs, result)
    record(card, result, exit_code=None,
           exit_status="unknown_due_to_handover", manual_release=True)
    gp = osp.join(run_dir_abs, "HANDOVER_RELEASED")
    try:
        os.remove(gp)
    except OSError:
        pass


def allocate(card):
    """Decide what to do with a run and execute it."""
    run_dir_abs = osp.join(G.REPO, card["run_dir"])
    ok, reasons = G.verified_ok(run_dir_abs, card)
    if ok:
        log("skip (VERIFIED.json matches this run and its artefacts): %s" % card["run_key"])
        return
    if reasons and osp.isfile(osp.join(run_dir_abs, "VERIFIED.json")):
        # The marker exists but no longer describes this run.
        die("%s has a stale VERIFIED.json (%s); refusing to skip or overwrite"
            % (card["run_key"], "; ".join(reasons)))

    if osp.isdir(run_dir_abs):
        if card["run_key"] in G.HANDOVER_RUNS:
            handle_handover(card)
            return
        if G.dir_has_training_artifacts(run_dir_abs):
            die("unexpected pre-existing directory for %s (%s) holding training "
                "artefacts; investigate rather than re-run"
                % (card["run_key"], card["run_dir"]))
        log("%s exists but holds no training artefacts; a previous attempt "
            "never started training, so this run is re-issued" % card["run_key"])

    rc = train(card)
    result, _ = run_verifier(card, run_dir_abs, "post_training_verification")
    if result is None:
        die("verification failed for %s; queue stopped before the next start" % card["run_key"])
    record(card, result, exit_code=rc, exit_status="ok", manual_release=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the reconcile result, start nothing")
    args = ap.parse_args()

    log("orchestrator start (pid %d)%s" % (os.getpid(), " [dry-run]" if args.dry_run else ""))
    reconcile()

    for seed, ver, fold in G.all_runs():
        card = G.expectation(seed, ver, fold)
        run_dir_abs = osp.join(G.REPO, card["run_dir"])
        ok, _ = G.verified_ok(run_dir_abs, card)
        state = ("done" if ok else
                 "handover-pending" if (osp.isdir(run_dir_abs) and card["run_key"] in G.HANDOVER_RUNS)
                 else "pre-existing" if osp.isdir(run_dir_abs)
                 else "pending")
        log("plan %-16s %-18s %s" % (card["run_key"], state, card["run_dir"]))

    if args.dry_run:
        log("dry-run: no training started")
        return 0

    for seed, ver, fold in G.all_runs():
        card = G.expectation(seed, ver, fold)
        allocate(card)

    log("ALL M2 CROSSSEED DONE %s" % now())
    return 0


if __name__ == "__main__":
    sys.exit(main())
