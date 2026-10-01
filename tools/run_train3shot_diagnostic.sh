#!/usr/bin/env bash
# 训练图诊断：三折 B0 各自推理本折 sup2 的 3 张训练图（teacher2）。
#
# 性质：训练图诊断，不是开发集指标，不是泛化证据。
# 说明：只读既有权重与既有划分；不改动任何 live 脚本；
#       正式评估入口 tools/eval_teacher2_export.py 的 232 图断言保持原样。
#
# 两组产物（**不得混用**）：
#   1) fold{N}/                        —— 既定后处理（未调整，与正式评估同一 test_cfg）
#   2) fold{N}_probe_scorethr0.001/    —— 探查用：仅把 rcnn.score_thr 降到 0.001，
#                                         目的是看低分候选的分布；**不是**既定后处理的结果。
set -euo pipefail

PY=/home/xcc/anaconda3/envs/dt/bin/python
ROOT=/home/xcc/dual_teacher_project/DualTeacher_ablation
OUT_BASE="$ROOT/ablation_configs/train3shot_b0_diagnostic_20260929/results"

mkdir -p "$OUT_BASE"
cd "$ROOT"
export PYTHONPATH="$ROOT"

run_one() {  # $1 = 输出目录, 其余 = 透传参数
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "跳过：输出目录已存在（本脚本不覆盖）: $out"
    return 0
  fi
  echo "==== $(basename "$out") ===="
  "$PY" tools/diagnose_train3shot_export.py --out-dir "$out" "$@"
}

for FOLD in 6 7 8; do
  run_one "$OUT_BASE/fold${FOLD}" --fold "$FOLD"
done

for FOLD in 6 7 8; do
  run_one "$OUT_BASE/fold${FOLD}_probe_scorethr0.001" --fold "$FOLD" --score-thr 0.001
done

echo "全部完成。产物根目录：$OUT_BASE"
