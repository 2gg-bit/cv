#!/bin/bash
# M2 跨种子复验：12 次正式训练，串行。
#   seed 123 先（fold 6 M0→M2, fold 7 M0→M2, fold 8 M0→M2），再 seed 456 同样顺序。
# 冻结协议见 DualTeacher_m3/M2_crossseed_protocol.md。
# 仅 M2 开关不同；--no-validate；32000 iter；max_keep_ckpts=1 只留 iter_32000.pth + latest.pth 副本。
set -euo pipefail
cd /home/xcc/dual_teacher_project/DualTeacher_m3
PY=/home/xcc/anaconda3/envs/dt/bin/python
MAXITERS=32000

run_one() {
  local seed=$1 ver=$2 fold=$3
  local cfg run_dir
  if [ "$ver" = "m0" ]; then
    cfg=configs/reproduce/phase3_dual_teacher_ssdd_dev.py
  elif [ "$ver" = "m2" ]; then
    cfg=configs/reproduce/phase3_dual_teacher_ssdd_dev_m2.py
  else
    echo "未知版本：$ver"; exit 1
  fi
  run_dir="work_dirs/m2_crossseed/seed${seed}/${ver}/3/${fold}"
  if [ -e "$run_dir" ]; then
    echo "停止：目录已存在，拒绝覆盖：$run_dir"; exit 1
  fi
  echo "=== seed${seed} ${ver} fold${fold} START $(date '+%F %T') ==="
  $PY -m torch.distributed.launch --nproc_per_node=1 \
    tools/train.py "$cfg" \
    --launcher pytorch \
    --seed "$seed" \
    --no-validate \
    --work-dir "$run_dir" \
    --cfg-options \
    fold="$fold" percent=3 \
    auto_resume=False \
    runner.max_iters=$MAXITERS \
    checkpoint_config.max_keep_ckpts=1
  echo "=== seed${seed} ${ver} fold${fold} DONE $(date '+%F %T') ==="
}

for seed in 123 456; do
  for fold in 6 7 8; do
    run_one "$seed" m0 "$fold"
    run_one "$seed" m2 "$fold"
  done
done
echo "ALL M2 CROSSSEED DONE $(date '+%F %T')"
