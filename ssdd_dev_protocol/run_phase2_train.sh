#!/bin/bash
# Phase2 三组正式训练（dev 协议版，seed=678，各 11200 iter，串行）
set -euo pipefail
cd /home/xcc/dual_teacher_project/DualTeacher
PY=/home/xcc/anaconda3/envs/dt/bin/python

for selection_id in 6 7 8; do
    run_dir="work_dirs/dev_ssdd/phase2_pretrain_optical_sar/3/${selection_id}"
    if [ -e "$run_dir" ]; then
        echo "停止：正式目录已存在，请先核对：$run_dir"
        exit 1
    fi
done

for selection_id in 6 7 8; do
    echo "=== Phase2 fold ${selection_id} START $(date '+%F %T') ==="
    $PY -m torch.distributed.launch --nproc_per_node=1 \
      tools/train.py \
      configs/reproduce/phase2_pretrain_optical_sar_dev.py \
      --launcher pytorch \
      --seed 678 \
      --no-validate \
      --work-dir "work_dirs/dev_ssdd/phase2_pretrain_optical_sar/3/${selection_id}" \
      --cfg-options \
      fold="$selection_id" percent=3 \
      auto_resume=False \
      runner.max_iters=11200
    echo "=== Phase2 fold ${selection_id} DONE $(date '+%F %T') ==="
done
echo "ALL PHASE2 DONE $(date '+%F %T')"
