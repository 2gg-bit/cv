#!/bin/bash
# ============================================================
# Dual Teacher 一键复现脚本 (DIOR + SSDD 版)
# ============================================================
#
# 前置条件:
#   1. 已安装 Dual Teacher 环境 (conda activate dt)
#   2. 已运行 prepare_data.py 处理数据
#   3. 已运行 setup_phase2_images.py 合并图像目录
#
# 用法:
#   bash run_reproduce.sh <GPU_ID> <SHOT> <FOLD>
#
# 示例:
#   bash run_reproduce.sh 0 3 1     # GPU 0, 3-shot, fold 1
#   bash run_reproduce.sh 0 1 1     # GPU 0, 1-shot, fold 1
#   bash run_reproduce.sh 0 5 1     # GPU 0, 5-shot, fold 1
#   bash run_reproduce.sh 0 10 1    # GPU 0, 10-shot, fold 1
#
# 配置文件位于: configs/reproduce/
#
# 训练时间预估 (单 GPU Quadro RTX 5000, lr=0.0025):
#   Phase 1: 8000 iter, ~1h
#   Phase 2: 11200 iter, ~1.5h
#   Phase 3: 32000 iter, ~8h
#   总计: ~10.5h
# ============================================================

set -e

GPU_ID=${1:-0}
SHOT=${2:-3}
FOLD=${3:-1}

# 路径配置 (根据实际安装修改)
REPO_DIR="."                               # DualTeacher 仓库根目录
CONFIG_DIR="configs/reproduce"             # 复现配置目录 (在仓库内)

echo "============================================================"
echo " Dual Teacher 复现训练 (DIOR + SSDD)"
echo " GPU: ${GPU_ID}, Shot: ${SHOT}, Fold: ${FOLD}"
echo " 工作目录: $(pwd)"
echo "============================================================"

# ============================================================
# Phase 1: 在 DIOR 光学数据集上预训练 T1/S1
# ============================================================
echo ""
echo "[Phase 1/3] 预训练 T1/S1 (DIOR 光学域, 8000 iter)"
echo "------------------------------------------------------------"

PHASE1_CKPT="work_dirs/phase1_pretrain_optical/100/${FOLD}/iter_8000.pth"
if [ -f "${PHASE1_CKPT}" ]; then
    echo "[Phase 1] 已存在 checkpoint, 跳过: ${PHASE1_CKPT}"
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch \
        --nproc_per_node=1 tools/train.py \
        ${CONFIG_DIR}/phase1_pretrain_optical.py \
        --launcher pytorch \
        --cfg-options fold=${FOLD} percent=100
    echo "[Phase 1] 完成: ${PHASE1_CKPT}"
fi

# ============================================================
# Phase 2: 在 DIOR ∪ Dl 上预训练 T2/S2 (DIOR + few-shot SAR)
# ============================================================
echo ""
echo "[Phase 2/3] 预训练 T2/S2 (DIOR + ${SHOT}-shot SAR, 11200 iter)"
echo "------------------------------------------------------------"

PHASE2_CKPT="work_dirs/phase2_pretrain_optical_sar/${SHOT}/${FOLD}/iter_11200.pth"
if [ -f "${PHASE2_CKPT}" ]; then
    echo "[Phase 2] 已存在 checkpoint, 跳过: ${PHASE2_CKPT}"
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch \
        --nproc_per_node=1 tools/train.py \
        ${CONFIG_DIR}/phase2_pretrain_optical_sar.py \
        --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${SHOT}
    echo "[Phase 2] 完成: ${PHASE2_CKPT}"
fi

# ============================================================
# Phase 3: Dual Teacher 半监督训练
# ============================================================
echo ""
echo "[Phase 3/3] Dual Teacher 训练 (${SHOT}-shot, fold ${FOLD}, 32k iter)"
echo "------------------------------------------------------------"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch \
    --nproc_per_node=1 tools/train.py \
    ${CONFIG_DIR}/phase3_dual_teacher_ssdd.py \
    --launcher pytorch \
    --cfg-options fold=${FOLD} percent=${SHOT}

PHASE3_DIR="work_dirs/phase3_dual_teacher/${SHOT}/${FOLD}"
echo "[Phase 3] 完成: ${PHASE3_DIR}"

# ============================================================
# 评估: 找到最新 checkpoint 并测试
# ============================================================
echo ""
echo "[评估] 测试最终模型"
echo "------------------------------------------------------------"

# 找到最新的 checkpoint
BEST_CKPT=$(ls -t ${PHASE3_DIR}/iter_*.pth 2>/dev/null | head -1)
if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: 未找到 checkpoint 文件!"
    exit 1
fi
echo "评估 checkpoint: ${BEST_CKPT}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch \
    --nproc_per_node=1 tools/test.py \
    ${CONFIG_DIR}/phase3_dual_teacher_ssdd.py \
    ${BEST_CKPT} \
    --eval bbox \
    --launcher pytorch \
    --cfg-options fold=${FOLD} percent=${SHOT} \
    --work-dir ${PHASE3_DIR}/eval_results

echo ""
echo "============================================================"
echo " 全部完成!"
echo " 结果保存在: ${PHASE3_DIR}/eval_results/"
echo "============================================================"
