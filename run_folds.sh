#!/bin/bash
set -e
GPU=0
SHOT=3
for FOLD in 2 3 4 5; do
    echo "============================================"
    echo " Fold ${FOLD}/5  $(date)"
    echo "============================================"
    
    # Phase 2
    P2_CKPT="work_dirs/phase2_pretrain_optical_sar/${SHOT}/${FOLD}/iter_11200.pth"
    if [ -f "${P2_CKPT}" ]; then
        echo "[P2 Fold ${FOLD}] 已存在, 跳过"
    else
        CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=1 \
            tools/train.py configs/reproduce/phase2_pretrain_optical_sar.py \
            --launcher pytorch --cfg-options fold=${FOLD} percent=${SHOT}
    fi
    
    # Phase 3
    CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=1 \
        tools/train.py configs/reproduce/phase3_dual_teacher_ssdd.py \
        --launcher pytorch --cfg-options fold=${FOLD} percent=${SHOT}
    
    echo "[Fold ${FOLD}] 完成 $(date)"
done
echo "全部 folds 完成!"
