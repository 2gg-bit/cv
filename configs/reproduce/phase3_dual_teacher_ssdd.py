"""
Phase 3: Dual Teacher training on DIOR (optical) + SSDD (SAR).

This is the main semi-supervised cross-domain training phase.
- T1/S1: initialized from Phase 1 (DIOR optical pretrain, 2000 iter)
- T2/S2: initialized from Phase 2 (DIOR + few-shot SAR pretrain, 2800 iter)
- sup1: DIOR labeled optical images (2706 images, 62533 ship boxes)
- sup2: few-shot labeled SAR images (1/3/5/10)
- unsup: unlabeled SAR images

论文原始参数 (4× Quadro RTX 6000):
  - lr=0.01, batch=12, max_iters=8000
  - 无 lr decay (step=[120000,160000] >> max_iters)
  - 无 warmup, fp16=dynamic, unsup_weight=2.0
  - EMA momentum=0.999 (beta)

单 GPU 适配:
  - lr=0.0025 (线性缩放 0.01 × 3/12), batch=3
  - warmup=500, grad_clip=35, fp16=dynamic

Usage:
    python -m torch.distributed.launch --nproc_per_node=1 \
        tools/train.py configs/reproduce/phase3_dual_teacher_ssdd.py \
        --launcher pytorch \
        --cfg-options fold=1 percent=3
"""

_base_ = [
    "../dual_teacher/base.py",
]

# ============================================================
# Model overrides: ResNet-101 (caffe) + 1 class (ship)
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
    test_cfg=dict(inference_on="teacher2"),
)

# ============================================================
# Dual Teacher wrapper: paper's exact hyperparameters
# ============================================================
# alpha=2.0, beta=0.999, gamma=0.2
# unsup2 weight = unsup_weight * gamma = 2.0 * 0.2 = 0.4
semi_wrapper = dict(
    type="DualTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.9,
        cls_pseudo_threshold=0.9,
        reg_pseudo_threshold=0.02,
        consensus_iou_thr=0.5,
        consensus_single_scale1=1.0,
        consensus_single_scale2=1.0,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
        # Phase 1 checkpoint -> initializes T1, S1 (DIOR pretrain)
        load1_from="work_dirs/phase1_pretrain_optical/100/1/iter_8000.pth",
        # Phase 2 checkpoint -> initializes T2, S2 (DIOR+SSDD pretrain)
        load2_from="work_dirs/phase2_pretrain_optical_sar/${percent}/${fold}/iter_11200.pth",
    ),
    test_cfg=dict(inference_on="teacher2"),
)

# ============================================================
# Dataset: override only paths and types from base.py
# ============================================================
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        sup1=dict(
            type="CocoDataset",
            classes=("ship",),
            ann_file="data/optical/dior_annotations.json",
            img_prefix="data/optical/images/",
        ),
        sup2=dict(
            type="CocoDataset",
            classes=("ship",),
            ann_file="data/ssdd/annotations/semi_supervised/"
                     "instances_train2017.${fold}@${percent}.json",
            img_prefix="data/ssdd/JPEGImages/",
        ),
        unsup=dict(
            type="CocoDataset",
            classes=("ship",),
            ann_file="data/ssdd/annotations/semi_supervised/"
                     "instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/ssdd/JPEGImages/",
            filter_empty_gt=False,
        ),
    ),
    val=dict(
        type="CocoDataset",
        classes=("ship",),
        ann_file="data/ssdd/annotations/test.json",
        img_prefix="data/ssdd/test_images/",
    ),
    test=dict(
        type="CocoDataset",
        classes=("ship",),
        ann_file="data/ssdd/annotations/test.json",
        img_prefix="data/ssdd/test_images/",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1, 1],
        )
    ),
)

# ============================================================
# EMA hook: beta=0.999, no warmup
# ============================================================
custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]

# ============================================================
# Training schedule
# ============================================================
# 论文: max_iters=8000, lr=0.01 (constant, step>>max_iters)
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
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=32000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=10)
evaluation = dict(type="SubModulesDistEvalHook", interval=4000)

# ============================================================
# Mixed precision (saves memory on 16GB GPU)
# ============================================================
fp16 = dict(loss_scale="dynamic")

# ============================================================
# Output
# ============================================================
work_dir = "work_dirs/phase3_dual_teacher/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
