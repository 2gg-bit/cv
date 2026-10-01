#!/usr/bin/env python
"""M2 cross-seed orchestrator (v4): train -> verify -> record -> next run.

New in v4 relative to v3
------------------------
1. **Single-instance lock.**  A second orchestrator competing for
   ``crossseed_orchestrator.lock`` exits immediately with a distinct status and
   starts nothing.
2. **The real exit code is persisted the instant training returns**, before any
   other fallible step (log sync, manifest cross-check, verification).  If the
   process dies before that write, the run keeps "exit code unknown" and the
   queue stops -- a missing exit code is never read as success.
3. **Restart recovery for ordinary runs.**  A run directory that already holds
   training artefacts is resumed according to its *persisted* exit status
   (``decide_resume``), instead of stopping the queue with no way forward.
4. **Process matching resolves each process's own cwd** and matches the seed, so
   a same-named run directory in another checkout cannot be mistaken for ours.
   Unreadable process information stops the queue; it is never read as "training
   has ended".
5. **A stale verify report is re-verified, not reused.**  At handover the
   verifier always runs again; if the stored report disagrees with the files on
   disk that disagreement is logged as evidence.
6. **Artefacts are re-checked while waiting for human approval and once more
   before the ledger write**, so weights or a config snapshot that change during
   the wait cannot be released.

The only run allowed to bypass the exit-code requirement is the designated
handover run (``HANDOVER_RUNS``): it was started by a previous orchestrator
whose exit code cannot be recovered.  It must pass the same verification and
then needs a one-time manual approval bound to a request id and the artefact
hashes.  It is recorded with exit_code=null, exit_status=unknown_due_to_handover
-- the real exit code is never fabricated as 0.

The queue stops on ANY of:

  * a run we started exiting non-zero, or its exit code being lost
  * the release verification failing
  * any record failing to land (exit status, verify result, VERIFIED.json, ledger)
  * our own code / the recipe / the initialisation changing while a run is in flight
  * insufficient free space before a start
  * an unexpected pre-existing run directory
  * an approval grant that does not match the current request or artefacts
  * process information that cannot be read

v3 and the training it is driving are left untouched; this is a separate file.
"""

import argparse
import datetime as dt
import os
import os.path as osp
import subprocess
import sys
import time

sys.path.insert(0, "/home/xcc/dual_teacher_project/DualTeacher_m3/tools")

import crossseed_gate_v4 as G  # noqa: E402

PY = "/home/xcc/anaconda3/envs/dt/bin/python"
VERIFIER = osp.join(G.REPO, "tools/crossseed_verify_v4.py")
LOG = osp.join(G.PROTOCOL_DIR, "crossseed_orchestrator_v4.log")
MIN_DISK_GB = 20.0
POLL_SECONDS = 60
EXIT_LOCK_HELD = 3


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

def training_pids_strict(card):
    """PIDs of training processes for this run; stop if we cannot tell.

    Unlike v3 this cannot silently return "no processes": if a process's cwd is
    unreadable, or a process points at this run without a usable --seed, we do
    not know whether training is still running, so the queue stops.
    """
    pids, errors, info = G.training_pids(card)
    if errors:
        die("cannot determine whether %s is still training: %s"
            % (card["run_key"], "; ".join(errors)))
    if info:
        log("%s: training processes %s"
            % (card["run_key"],
               ", ".join("pid %d (start_ticks %s)" % (i["pid"], i["start_ticks"])
                         for i in info)))
    return pids


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
    """Run the verifier as a subprocess. Returns (result_or_None, out_path)."""
    out = osp.join(run_dir_abs, "verify_result.json")
    cmd = [PY, VERIFIER,
           "--run-dir", run_dir_abs, "--seed", str(card["seed"]),
           "--ver", card["ver"], "--fold", str(card["fold"]),
           "--expected-iters", str(card["expected_iters"]),
           "--expected-max-keep-ckpts", str(card["expected_max_keep_ckpts"]),
           "--expected-code-root", G.CODE_ROOT,
           "--evidence-timestamp", evidence_ts, "--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()
    tail += (proc.stderr or "").strip().splitlines()
    for line in tail[-8:]:
        log("    verifier: " + line)
    if proc.returncode != 0:
        return None, out
    try:
        return G.load_json(out), out
    except (OSError, ValueError) as exc:
        die("verify_result.json unreadable after a passing verifier: %s" % exc)


def confirm_artifacts(result, run_dir_abs, card, when):
    """Stop if the artefacts no longer match the verification we are acting on."""
    ok, reasons = G.report_matches_artifacts(result, run_dir_abs, card)
    if not ok:
        die("%s: artefacts changed %s: %s" % (card["run_key"], when, "; ".join(reasons)))
    log("%s: artefacts still match the verification (%s)" % (card["run_key"], when))


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
            "code_root": G.CODE_ROOT,
            "loaded_implementation": result.get("evidence", {}).get("loaded_implementation"),
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
            G.load_json(osp.join(run_dir_abs, "verify_result.json"))
        except (OSError, ValueError):
            log("reconcile: %s has VERIFIED.json but no readable verify_result.json"
                % card["run_key"])
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
            # The wait may have been long; the approval speaks for the artefacts
            # as they were when the request was written, not as they are now.
            confirm_artifacts(result, run_dir_abs, card, "after the approval wait")
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

    train_log = osp.join(G.PROTOCOL_DIR,
                         "crossseed_train_%s.log" % card["run_key"].replace("/", "_"))
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

    # Persist the real exit code BEFORE anything else that can fail. If we die
    # between here and the write, the run stays "unknown" and stops the queue.
    try:
        G.persist_exit_status(run_dir_abs, card["run_key"], rc,
                              exit_status="ok" if rc == 0 else "nonzero_exit")
    except Exception as exc:
        die("training for %s returned rc=%s but the exit status could not be "
            "persisted (%s); treating the exit code as unknown" % (card["run_key"], rc, exc))
    log("=== END %s rc=%s %s (exit status persisted) ===" % (card["run_key"], rc, now()))

    action, why = G.decide_after_training(rc, is_handover=False)
    if action != "proceed":
        die("run %s: %s" % (card["run_key"], why))

    problems = check_manifest_still_valid(card, manifest)
    if problems:
        die("run %s: source/recipe/init changed mid-run: %s" % (card["run_key"], problems))
    return rc


def verify_and_record(card, exit_code, exit_status, manual_release):
    run_dir_abs = osp.join(G.REPO, card["run_dir"])
    if not osp.isfile(osp.join(run_dir_abs, "iter_32000.pth")):
        die("%s ended without iter_32000.pth" % card["run_key"])
    result, _ = run_verifier(card, run_dir_abs, "post_training_verification")
    if result is None:
        die("verification failed for %s; queue stopped before the next start"
            % card["run_key"])
    if manual_release:
        await_approval(card, run_dir_abs, result)
    # Last look before the ledger write.
    confirm_artifacts(result, run_dir_abs, card, "before recording")
    record(card, result, exit_code=exit_code, exit_status=exit_status,
           manual_release=manual_release)
    return result


def resume_run(card):
    """A run directory holding training artefacts, from a previous orchestrator."""
    run_dir_abs = osp.join(G.REPO, card["run_dir"])
    is_handover = card["run_key"] in G.HANDOVER_RUNS
    action, why = G.decide_resume(run_dir_abs, card["run_key"], is_handover)
    log("resume %s: %s (%s)" % (card["run_key"], action, why))
    if action == "stop":
        die("cannot resume %s: %s" % (card["run_key"], why))

    if is_handover:
        pids = training_pids_strict(card)
        if pids:
            log("handover: %s is in flight (pids %s); waiting" % (card["run_key"], pids))
            while training_pids_strict(card):
                time.sleep(POLL_SECONDS)
            log("handover: in-flight training for %s has ended" % card["run_key"])

    # A stored report describes the files as they were when it was written. We
    # never release on its own ok:true -- the verifier runs again.
    existing = osp.join(run_dir_abs, "verify_result.json")
    if osp.isfile(existing):
        try:
            old = G.load_json(existing)
        except ValueError as exc:
            log("existing verify_result.json for %s is unreadable (%s); re-verifying"
                % (card["run_key"], exc))
            old = None
        if old is not None:
            same, reasons = G.report_matches_artifacts(old, run_dir_abs, card)
            log("stored verify_result.json for %s: ok=%r, matches current artefacts=%r%s"
                % (card["run_key"], old.get("ok"), same,
                   "" if same else " (%s)" % "; ".join(reasons)))

    exit_code, rec, err = G.read_exit_status(run_dir_abs, card["run_key"])
    if is_handover:
        verify_and_record(card, exit_code=None,
                          exit_status="unknown_due_to_handover", manual_release=True)
        gp = osp.join(run_dir_abs, "HANDOVER_RELEASED")
        try:
            os.remove(gp)
        except OSError:
            pass
        return
    if err:
        die("cannot resume %s: %s" % (card["run_key"], err))
    verify_and_record(card, exit_code=exit_code,
                      exit_status=rec.get("exit_status", "ok"), manual_release=False)


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
            resume_run(card)
            return
        if G.dir_has_training_artifacts(run_dir_abs):
            # Training already ran here; continue from its persisted exit status
            # rather than re-running it or dead-ending the queue.
            resume_run(card)
            return
        log("%s exists but holds no training artefacts; a previous attempt "
            "never started training, so this run is re-issued" % card["run_key"])

    rc = train(card)
    verify_and_record(card, exit_code=rc, exit_status="ok", manual_release=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the reconcile result, start nothing")
    args = ap.parse_args()

    lock = G.acquire_lock()
    if lock is None:
        print("another orchestrator already holds %s; exiting without starting "
              "anything" % G.LOCK_FILE)
        return EXIT_LOCK_HELD

    log("orchestrator v4 start (pid %d) lock=%s%s"
        % (os.getpid(), osp.relpath(G.LOCK_FILE, G.REPO),
           " [dry-run]" if args.dry_run else ""))
    try:
        reconcile()

        for seed, ver, fold in G.all_runs():
            card = G.expectation(seed, ver, fold)
            run_dir_abs = osp.join(G.REPO, card["run_dir"])
            ok, _ = G.verified_ok(run_dir_abs, card)
            state = ("done" if ok else
                     "handover-pending" if (osp.isdir(run_dir_abs) and card["run_key"] in G.HANDOVER_RUNS)
                     else "resumable" if (osp.isdir(run_dir_abs)
                                          and G.dir_has_training_artifacts(run_dir_abs))
                     else "pre-existing" if osp.isdir(run_dir_abs)
                     else "pending")
            log("plan %-16s %-18s %s" % (card["run_key"], state, card["run_dir"]))

        if args.dry_run:
            log("dry-run: no training started")
            return 0

        for seed, ver, fold in G.all_runs():
            allocate(G.expectation(seed, ver, fold))

        log("ALL M2 CROSSSEED DONE %s" % now())
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
