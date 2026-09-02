"""
Phase 3: Dual Teacher training on ShipRSImageNet (optical) + SSDD (SAR).

This is the main semi-supervised cross-domain training phase.
- T1/S1: initialized from Phase 1 (optical pretrain)
- T2/S2: initialized from Phase 2 (optical + few-shot SAR pretrain)
- sup1: ShipRSImageNet labeled optical images
- sup2: few-shot labeled SAR images (1/3/5/10)
- unsup: unlabeled SAR images

Usage:
    python tools/train.py configs/phase3_dual_teacher_ssdd.py \
        --cfg-options fold=1 percent=3

IMPORTANT:
    This config inherits from the DualTeacher base.py which defines:
    - Faster R-CNN + FPN model structure
    - Z-score normalization: mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0]
    - Augmentation pipelines (train1, train2, strong, weak, unsup, test)
    - SemiCrossDataset with SemiCrossBalanceSampler
    - MeanTeacher EMA hook
    - DualTeacher semi_wrapper

    We only override the fields that differ from base.py.
"""

# ============================================================
# Inherit from the DualTeacher base config
# ============================================================
_base_ = [
    # 部署时将本文件放入 dual_teacher_repo/configs/reproduce/ 目录下
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
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
        # Phase 1 checkpoint -> initializes T1, S1
        load1_from="work_dirs/phase1_pretrain_optical/100/${fold}/iter_2000.pth",
        # Phase 2 checkpoint -> initializes T2, S2
        load2_from="work_dirs/phase2_pretrain_optical_sar/${percent}/${fold}/iter_2800.pth",
    ),
    test_cfg=dict(inference_on="teacher2"),
)

# ============================================================
# Dataset: override only paths and types from base.py
# ============================================================
# base.py defines:
#   data.train.type = "SemiCrossDataset"
#   data.train.sup1/sup2/unsup with ann_file=None, img_prefix=None
#   pipelines (train1_pipeline, train2_pipeline, etc.)
# We override ann_file, img_prefix, type, classes for each subset.
# The pipelines are INHERITED from base.py (DO NOT redefine them).
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
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[6000, 7500])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=8000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=10)
evaluation = dict(type="SubModulesDistEvalHook", interval=1000)

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
