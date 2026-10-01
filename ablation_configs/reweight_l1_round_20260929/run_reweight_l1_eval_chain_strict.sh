#!/bin/bash
# Strict (fail-fast) serial evaluation chain for the reweight_l1 exploratory paired
# round: fold6 -> fold7 -> fold8, then the unrounded paired delta.
#
# Supersedes run_reweight_l1_eval_chain.sh in this directory, which skipped a failed
# fold and carried on; that script was never launched, and this one stops at the first
# problem instead, matching the convention that any failed step ends the run.  Both
# files are kept on disk as the record of the change.
#
# Stop conditions, each logged with its reason and a non-zero exit: non-zero training
# exit code, missing final weights, conflicting output directory, a failed evaluation,
# or an inconsistent paired recomputation.  Nothing already produced is removed or
# overwritten; the evaluation tool itself refuses a non-empty output directory.
#
# Launch this only after the training chain has finished -- one evaluation at a time,
# no concurrent GPU work.
set -u

PY=/home/xcc/anaconda3/envs/dt/bin/python
ROOT=/home/xcc/dual_teacher_project/DualTeacher_ablation
WEIGHTS_ROOT=/home/xcc/dual_teacher_project/DualTeacher/work_dirs/ablation_v1
ROUND="$ROOT/ablation_configs/reweight_l1_round_20260929"
CHAIN_LOG="$ROUND/eval_chain.log"
DELTA_OUT="$ROUND/paired_coco_delta.json"
DELTA_LOG="$ROUND/paired_coco_delta.log"

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"

abort() {
    echo "[evalchain] $(date '+%F %T') ABORT: $1" >> "$CHAIN_LOG"
    exit 1
}

echo "[evalchain] $(date '+%F %T') strict eval chain started (pid $$, script sha256 $(sha256sum "$0" | cut -d' ' -f1))" >> "$CHAIN_LOG"

for fold in 6 7 8; do
    cfg="ablation_configs/fold${fold}_seed678/reweight_l1.py"
    code="ablation_configs/fold${fold}_seed678/reweight_l1_train_exit_code.txt"
    wd="$WEIGHTS_ROOT/fold${fold}_seed678/reweight_l1"
    ckpt="$wd/iter_32000.pth"
    out="ablation_configs/fold${fold}_seed678/reweight_l1_evaluation"
    log="ablation_configs/fold${fold}_seed678/reweight_l1_eval.log"
    eval_code="ablation_configs/fold${fold}_seed678/reweight_l1_eval_exit_code.txt"

    [ -f "$code" ] || abort "fold${fold}: no training exit code at $code"
    train_rc=$(cat "$code")
    [ "$train_rc" = "0" ] || abort "fold${fold}: training exit=$train_rc"
    [ -f "$ckpt" ] || abort "fold${fold}: missing final weights $ckpt"
    [ ! -e "$out" ] || abort "fold${fold}: output directory already exists ($out)"
    [ ! -e "$log" ] || abort "fold${fold}: evaluation log already exists ($log)"
    [ ! -e "$eval_code" ] || abort "fold${fold}: evaluation exit code already exists ($eval_code)"

    ckpt_sha=$(sha256sum "$ckpt" | cut -d' ' -f1)
    echo "[evalchain] $(date '+%F %T') fold${fold} checkpoint sha256=$ckpt_sha" >> "$CHAIN_LOG"
    echo "[evalchain] $(date '+%F %T') fold${fold} launching: $PY tools/train_ablation.py $cfg $ckpt --eval --fold ${fold} --out-dir $out" >> "$CHAIN_LOG"

    "$PY" tools/train_ablation.py "$cfg" "$ckpt" --eval --fold "$fold" --out-dir "$out" > "$log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$eval_code"
    echo "[evalchain] $(date '+%F %T') fold${fold} exit=$rc" >> "$CHAIN_LOG"
    [ "$rc" -eq 0 ] || abort "fold${fold}: evaluation exit=$rc (see $log)"
done

echo "[evalchain] $(date '+%F %T') all three evaluations finished; computing paired delta" >> "$CHAIN_LOG"

"$PY" tools/recompute_paired_coco_delta.py \
    --expect-pairs 3 \
    --pair fold6 "ablation_configs/fold6_seed678/b0_evaluation" "ablation_configs/fold6_seed678/reweight_l1_evaluation" \
    --pair fold7 "ablation_configs/fold7_seed678/b0_evaluation" "ablation_configs/fold7_seed678/reweight_l1_evaluation" \
    --pair fold8 "ablation_configs/fold8_seed678/b0_evaluation" "ablation_configs/fold8_seed678/reweight_l1_evaluation" \
    --out "$DELTA_OUT" > "$DELTA_LOG" 2>&1
rc=$?
echo "[evalchain] $(date '+%F %T') paired delta recompute exit=$rc" >> "$CHAIN_LOG"
[ "$rc" -eq 0 ] || abort "paired delta recompute exit=$rc -- rounds did not agree or a fold was missing (see $DELTA_LOG)"

echo "[evalchain] $(date '+%F %T') strict eval chain finished (all steps passed)" >> "$CHAIN_LOG"
