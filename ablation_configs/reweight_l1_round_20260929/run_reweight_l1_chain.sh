#!/bin/bash
# Exploratory paired experiment: sup2 small-background ROI reweight (lambda=1) on
# SSDD folds 6/7/8, one run per fold, sequential.
#
# Recipe = the frozen B0 recipe (same launch line as b0/pg_both): --seed 678,
# --no-validate, no --deterministic.  The acceptance-time determinism settings are
# deliberately NOT carried into formal training.  Each fold initializes from its own
# Phase1/Phase2 weights (config train_cfg.load1_from/load2_from), never from the
# acceptance B0 final weights, and writes into a fresh work_dir.
#
# A fold's later folds are not cancelled on the sign of an earlier fold's gain; the
# chain only stops on a failure, keeping every record produced so far.
set -u

PY=/home/xcc/anaconda3/envs/dt/bin/python
ROOT=/home/xcc/dual_teacher_project/DualTeacher_ablation
ROUND="$ROOT/ablation_configs/reweight_l1_round_20260929"
CHAIN_LOG="$ROUND/chain.log"

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT"

echo "[chain] $(date '+%F %T') chain started (pid $$)" >> "$CHAIN_LOG"

for fold in 6 7 8; do
    cfg="ablation_configs/fold${fold}_seed678/reweight_l1.py"
    log="ablation_configs/fold${fold}_seed678/reweight_l1_train.log"
    code="ablation_configs/fold${fold}_seed678/reweight_l1_train_exit_code.txt"
    echo "[chain] $(date '+%F %T') fold${fold} launching: $PY -m torch.distributed.launch --nproc_per_node=1 tools/train_ablation.py $cfg --launcher pytorch --seed 678 --no-validate" >> "$CHAIN_LOG"
    "$PY" -m torch.distributed.launch --nproc_per_node=1 tools/train_ablation.py \
        "$cfg" --launcher pytorch --seed 678 --no-validate > "$log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$code"
    echo "[chain] $(date '+%F %T') fold${fold} exit=$rc" >> "$CHAIN_LOG"
    if [ "$rc" -ne 0 ]; then
        echo "[chain] $(date '+%F %T') stopping after fold${fold} failure; records kept" >> "$CHAIN_LOG"
        exit "$rc"
    fi
done

echo "[chain] $(date '+%F %T') all folds finished" >> "$CHAIN_LOG"
