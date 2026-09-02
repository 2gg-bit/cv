"""
Phase 1: Pretrain T1/S1 backbone on DIOR (optical) only.

This trains a standard Faster R-CNN + ResNet-101 + FPN on DIOR ship subset.
The resulting checkpoint initializes T1 and S1 in the Dual Teacher framework.

DIOR 数据统计 (合并全部 ship 图像, 论文 "Only 2702 ship images"):
  - 2706 张含 ship 的图像 (trainval 1302 + test 1404 合并)
  - 62533 个 ship 标注框
  - 平均每图 23.1 个目标
  - 图像尺寸: 800x800

论文原始参数 (4× Quadro RTX 6000):
  - lr=0.01, batch=12 (3 per GPU), max_iters=2000
  - 无 lr decay (step=[120000,160000] >> max_iters)
  - 无 warmup, fp16=dynamic

单 GPU 适配 (Quadro RTX 5000, 16GB):
  - lr=0.0025 (线性缩放: 0.01 × 3/12)
  - batch=3 (samples_per_gpu=3)
  - warmup=500 (单GPU稳定性保障)
  - grad_clip=35, fp16=dynamic

Usage:
    python -m torch.distributed.launch --nproc_per_node=1 \
        tools/train.py configs/reproduce/phase1_pretrain_optical.py \
        --launcher pytorch \
        --cfg-options fold=1 percent=100
"""

_base_ = [
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
# Dataset: DIOR (optical domain Do) — ship class only
# ============================================================
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
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
# Training schedule (DIOR: 2706 images, batch=3, ~902 iter/epoch)
# ============================================================
# 论文: max_iters=2000, lr=0.01 (constant, step>>max_iters)
# 单GPU: lr=0.0025 (线性缩放 0.01 × 3/12), 无 lr decay
optimizer = dict(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(
    _delete_=True,
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[120000, 160000],  # >> max_iters → lr 永不衰减 (同论文)
)
runner = dict(type="IterBasedRunner", max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=5)
evaluation = dict(interval=2000, metric="bbox")

# ============================================================
# Output
# ============================================================
work_dir = "work_dirs/phase1_pretrain_optical/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook", by_epoch=False)],
)

# ============================================================
# Mixed precision (论文全阶段 fp16=dynamic)
# ============================================================
fp16 = dict(loss_scale="dynamic")
