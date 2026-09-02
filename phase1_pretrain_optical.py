"""
Phase 1: Pretrain T1/S1 backbone on optical dataset (ShipRSImageNet) only.

This trains a standard Faster R-CNN + ResNet-101 + FPN on ShipRSImageNet.
The resulting checkpoint initializes T1 and S1 in the Dual Teacher framework.

Usage:
    python tools/train.py configs/phase1_pretrain_optical.py \
        --cfg-options fold=1 percent=100
"""

_base_ = [
    # 部署时将本文件放入 dual_teacher_repo/configs/reproduce/ 目录下
    # 此路径指向 baseline 的基础配置
    "../baseline/base.py",
]

# ============================================================
# Model: Faster R-CNN + ResNet-101 (caffe) + FPN
# ============================================================
model = dict(
    backbone=dict(
        depth=101,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="open-mmlab://detectron2/resnet101_caffe",
        ),
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    ),
)

# ============================================================
# Z-score normalization (same for both domains per paper)
# ============================================================
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False,
)

# ============================================================
# Dataset: ShipRSImageNet (optical domain Do)
# ============================================================
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type="CocoDataset",
        classes=("ship",),
        ann_file="data/optical/dior_annotations.json",
        img_prefix="data/optical/images/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="Resize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
    val=dict(
        type="CocoDataset",
        classes=("ship",),
        ann_file="data/ssdd/annotations/test.json",
        img_prefix="data/ssdd/test_images/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type="CocoDataset",
        classes=("ship",),
        ann_file="data/ssdd/annotations/test.json",
        img_prefix="data/ssdd/test_images/",
    ),
)

# ============================================================
# Training schedule
# ============================================================
# 1 GPU: ~2-3k iters sufficient for optical pretrain
# Adjust based on dataset size
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[1500, 1800])
runner = dict(type="IterBasedRunner", max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=5)
evaluation = dict(interval=500, metric="bbox")

# ============================================================
# Output
# ============================================================
work_dir = "work_dirs/phase1_pretrain_optical/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)
