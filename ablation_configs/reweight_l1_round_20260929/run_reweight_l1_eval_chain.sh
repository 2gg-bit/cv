#!/bin/bash
# Serial evaluation of the reweight_l1 exploratory paired round (fold6 -> fold7 -> fold8).
#
# One fold at a time, no concurrent GPU work: each evaluation builds the detector,
# runs 232-image inference on GPU 0 and writes into that fold's own evaluation dir.
# Launch this only after the training chain has finished.
#
# Per fold the variant is that fold's own reweight_l1 config and its own final
# iteration weights; the comparison baseline stays that fold's existing
# b0_evaluation (untouched by this script).  A fold is skipped -- with the reason
# recorded -- if its training did not finish cleanly, so a partially completed chain
# still yields records for the folds that did.
#
# The unrounded paired delta is NOT computed here: it is a separate CPU-only step
# (tools/recompute_paired_coco_delta.py) run over the resulting directories.
set -u

PY=/home/xcc/anaconda3/envs/dt/bin/python
ROOT=/home/xcc/dual_teacher_project/DualTeacher_ablation
WEIGHTS_ROOT=/home/xcc/dual_teacher_project/DualTeacher/work_dirs/ablation_v1
ROUND="$ROOT/ablation_configs/reweight_l1_round_20260929"
CHAIN_LOG="$ROUND/eval_chain.log"

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"

echo "[evalchain] $(date '+%F %T') eval chain started (pid $$)" >> "$CHAIN_LOG"

overall=0
for fold in 6 7 8; do
    cfg="ablation_configs/fold${fold}_seed678/reweight_l1.py"
    code="ablation_configs/fold${fold}_seed678/reweight_l1_train_exit_code.txt"
    wd="$WEIGHTS_ROOT/fold${fold}_seed678/reweight_l1"
    ckpt="$wd/iter_32000.pth"
    out="ablation_configs/fold${fold}_seed678/reweight_l1_evaluation"
    log="ablation_configs/fold${fold}_seed678/reweight_l1_eval.log"
    eval_code="ablation_configs/fold${fold}_seed678/reweight_l1_eval_exit_code.txt"

    if [ ! -f "$code" ]; then
        echo "[evalchain] $(date '+%F %T') fold${fold} SKIPPED: no training exit code at $code" >> "$CHAIN_LOG"
        overall=1
        continue
    fi
    train_rc=$(cat "$code")
    if [ "$train_rc" != "0" ]; then
        echo "[evalchain] $(date '+%F %T') fold${fold} SKIPPED: training exit=$train_rc" >> "$CHAIN_LOG"
        overall=1
        continue
    fi
    if [ ! -f "$ckpt" ]; then
        echo "[evalchain] $(date '+%F %T') fold${fold} SKIPPED: missing $ckpt" >> "$CHAIN_LOG"
        overall=1
        continue
    fi
    if [ -e "$out" ]; then
        echo "[evalchain] $(date '+%F %T') fold${fold} SKIPPED: $out already exists" >> "$CHAIN_LOG"
        overall=1
        continue
    fi

    ckpt_sha=$(sha256sum "$ckpt" | cut -d' ' -f1)
    echo "[evalchain] $(date '+%F %T') fold${fold} checkpoint sha256=$ckpt_sha" >> "$CHAIN_LOG"
    echo "[evalchain] $(date '+%F %T') fold${fold} launching: $PY tools/train_ablation.py $cfg $ckpt --eval --fold ${fold} --out-dir $out" >> "$CHAIN_LOG"

    "$PY" tools/train_ablation.py "$cfg" "$ckpt" --eval --fold "$fold" --out-dir "$out" > "$log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$eval_code"
    echo "[evalchain] $(date '+%F %T') fold${fold} exit=$rc" >> "$CHAIN_LOG"
    if [ "$rc" -ne 0 ]; then
        echo "[evalchain] $(date '+%F %T') fold${fold} evaluation failed; continuing with remaining folds" >> "$CHAIN_LOG"
        overall=1
    fi
done

echo "[evalchain] $(date '+%F %T') eval chain finished (overall=$overall)" >> "$CHAIN_LOG"
exit "$overall"
