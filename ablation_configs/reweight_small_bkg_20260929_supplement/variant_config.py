model = dict(
    type='DualTeacher',
    model=dict(
        type='FasterRCNN',
        backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet101_caffe')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='SmallBkgReweightRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='SmallBkgReweightBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                reweight=dict(
                    enable=True, lambda_=1.0, max_area=1024.0, tag='sup2'))),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100),
            inference_on='teacher2')),
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
        load1_from='work_dirs/phase1_pretrain_optical/100/1/iter_8000.pth',
        load2_from='work_dirs/phase2_pretrain_optical_sar/3/6/iter_11200.pth',
        m2_enabled=False,
        m2_force_weight_one=False),
    test_cfg=dict(inference_on='teacher2'))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type='SemiCrossDataset',
        sup1=dict(
            type='CocoDataset',
            ann_file='data/optical/dior_annotations.json',
            img_prefix='data/optical/images/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Sequential',
                    transforms=[
                        dict(
                            type='RandResize',
                            img_scale=[(1333, 400), (1333, 1200)],
                            multiscale_mode='range',
                            keep_ratio=True),
                        dict(type='RandFlip', flip_ratio=0.5),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='Identity'),
                                dict(type='AutoContrast'),
                                dict(type='RandEqualize'),
                                dict(type='RandSolarize'),
                                dict(type='RandColor'),
                                dict(type='RandContrast'),
                                dict(type='RandBrightness'),
                                dict(type='RandSharpness'),
                                dict(type='RandPosterize')
                            ])
                    ],
                    record=True),
                dict(type='Pad', size_divisor=32),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='ExtraAttrs', tag='sup1'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'img_norm_cfg', 'pad_shape', 'scale_factor',
                               'tag'))
            ],
            classes=('ship', )),
        sup2=dict(
            type='CocoDataset',
            ann_file=
            'data/ssdd/annotations/semi_supervised/instances_train2017.6@3.json',
            img_prefix='data/ssdd/JPEGImages/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Sequential',
                    transforms=[
                        dict(
                            type='RandResize',
                            img_scale=[(1333, 400), (1333, 1200)],
                            multiscale_mode='range',
                            keep_ratio=True),
                        dict(type='RandFlip', flip_ratio=0.5),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='Identity'),
                                dict(type='AutoContrast'),
                                dict(type='RandEqualize'),
                                dict(type='RandSolarize'),
                                dict(type='RandColor'),
                                dict(type='RandContrast'),
                                dict(type='RandBrightness'),
                                dict(type='RandSharpness'),
                                dict(type='RandPosterize')
                            ])
                    ],
                    record=True),
                dict(type='Pad', size_divisor=32),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='ExtraAttrs', tag='sup2'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'img_norm_cfg', 'pad_shape', 'scale_factor',
                               'tag'))
            ],
            classes=('ship', )),
        unsup=dict(
            type='CocoDataset',
            ann_file=
            'data/ssdd/annotations/semi_supervised/instances_train2017.6@3-unlabeled.json',
            img_prefix='data/ssdd/JPEGImages/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='PseudoSamples', with_bbox=True),
                dict(
                    type='MultiBranch',
                    unsup_student=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(1333, 400), (1333, 1200)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5),
                                dict(
                                    type='ShuffledSequential',
                                    transforms=[
                                        dict(
                                            type='OneOf',
                                            transforms=[
                                                dict(type='Identity'),
                                                dict(type='AutoContrast'),
                                                dict(type='RandEqualize'),
                                                dict(type='RandSolarize'),
                                                dict(type='RandColor'),
                                                dict(type='RandContrast'),
                                                dict(type='RandBrightness'),
                                                dict(type='RandSharpness'),
                                                dict(type='RandPosterize')
                                            ]),
                                        dict(
                                            type='OneOf',
                                            transforms=[{
                                                'type': 'RandTranslate',
                                                'x': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandTranslate',
                                                'y': (-0.1, 0.1)
                                            }, {
                                                'type': 'RandRotate',
                                                'angle': (-30, 30)
                                            },
                                                        [{
                                                            'type':
                                                            'RandShear',
                                                            'x': (-30, 30)
                                                        }, {
                                                            'type':
                                                            'RandShear',
                                                            'y': (-30, 30)
                                                        }]])
                                    ]),
                                dict(
                                    type='RandErase',
                                    n_iterations=(1, 5),
                                    size=[0, 0.2],
                                    squared=True)
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[103.53, 116.28, 123.675],
                            std=[1.0, 1.0, 1.0],
                            to_rgb=False),
                        dict(type='ExtraAttrs', tag='unsup_student'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ],
                    unsup_teacher=[
                        dict(
                            type='Sequential',
                            transforms=[
                                dict(
                                    type='RandResize',
                                    img_scale=[(1333, 400), (1333, 1200)],
                                    multiscale_mode='range',
                                    keep_ratio=True),
                                dict(type='RandFlip', flip_ratio=0.5)
                            ],
                            record=True),
                        dict(type='Pad', size_divisor=32),
                        dict(
                            type='Normalize',
                            mean=[103.53, 116.28, 123.675],
                            std=[1.0, 1.0, 1.0],
                            to_rgb=False),
                        dict(type='ExtraAttrs', tag='unsup_teacher'),
                        dict(type='DefaultFormatBundle'),
                        dict(
                            type='Collect',
                            keys=['img', 'gt_bboxes', 'gt_labels'],
                            meta_keys=('filename', 'ori_shape', 'img_shape',
                                       'img_norm_cfg', 'pad_shape',
                                       'scale_factor', 'tag',
                                       'transform_matrix'))
                    ])
            ],
            filter_empty_gt=False,
            classes=('ship', ))),
    val=dict(
        type='CocoDataset',
        ann_file='data/ssdd/annotations/test.json',
        img_prefix='data/ssdd/test_images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('ship', )),
    test=dict(
        type='CocoDataset',
        ann_file='data/ssdd/annotations/test.json',
        img_prefix='data/ssdd/test_images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('ship', )),
    sampler=dict(
        train=dict(
            type='SemiCrossBalanceSampler',
            sample_ratio=[1, 1, 1],
            by_prob=False,
            epoch_length=7330)))
evaluation = dict(interval=4000, metric='bbox', type='SubModulesDistEvalHook')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[120000, 160000])
runner = dict(type='IterBasedRunner', max_iters=32000)
checkpoint_config = dict(interval=4000, by_epoch=False, max_keep_ckpts=10)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='WeightSummary'),
    dict(type='MeanTeacher', momentum=0.999, interval=1, warm_up=0)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
mmdet_base = '../../../thirdparty/mmdetection/configs/_base_'
train1_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Identity'),
                    dict(type='AutoContrast'),
                    dict(type='RandEqualize'),
                    dict(type='RandSolarize'),
                    dict(type='RandColor'),
                    dict(type='RandContrast'),
                    dict(type='RandBrightness'),
                    dict(type='RandSharpness'),
                    dict(type='RandPosterize')
                ])
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='ExtraAttrs', tag='sup1'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag'))
]
train2_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Identity'),
                    dict(type='AutoContrast'),
                    dict(type='RandEqualize'),
                    dict(type='RandSolarize'),
                    dict(type='RandColor'),
                    dict(type='RandContrast'),
                    dict(type='RandBrightness'),
                    dict(type='RandSharpness'),
                    dict(type='RandPosterize')
                ])
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='ExtraAttrs', tag='sup2'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag'))
]
strong_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5),
            dict(
                type='ShuffledSequential',
                transforms=[
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Identity'),
                            dict(type='AutoContrast'),
                            dict(type='RandEqualize'),
                            dict(type='RandSolarize'),
                            dict(type='RandColor'),
                            dict(type='RandContrast'),
                            dict(type='RandBrightness'),
                            dict(type='RandSharpness'),
                            dict(type='RandPosterize')
                        ]),
                    dict(
                        type='OneOf',
                        transforms=[{
                            'type': 'RandTranslate',
                            'x': (-0.1, 0.1)
                        }, {
                            'type': 'RandTranslate',
                            'y': (-0.1, 0.1)
                        }, {
                            'type': 'RandRotate',
                            'angle': (-30, 30)
                        },
                                    [{
                                        'type': 'RandShear',
                                        'x': (-30, 30)
                                    }, {
                                        'type': 'RandShear',
                                        'y': (-30, 30)
                                    }]])
                ]),
            dict(
                type='RandErase',
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True)
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='ExtraAttrs', tag='unsup_student'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
weak_pipeline = [
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='RandResize',
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandFlip', flip_ratio=0.5)
        ],
        record=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='ExtraAttrs', tag='unsup_teacher'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor', 'tag', 'transform_matrix'))
]
unsup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PseudoSamples', with_bbox=True),
    dict(
        type='MultiBranch',
        unsup_student=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(1333, 400), (1333, 1200)],
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5),
                    dict(
                        type='ShuffledSequential',
                        transforms=[
                            dict(
                                type='OneOf',
                                transforms=[
                                    dict(type='Identity'),
                                    dict(type='AutoContrast'),
                                    dict(type='RandEqualize'),
                                    dict(type='RandSolarize'),
                                    dict(type='RandColor'),
                                    dict(type='RandContrast'),
                                    dict(type='RandBrightness'),
                                    dict(type='RandSharpness'),
                                    dict(type='RandPosterize')
                                ]),
                            dict(
                                type='OneOf',
                                transforms=[{
                                    'type': 'RandTranslate',
                                    'x': (-0.1, 0.1)
                                }, {
                                    'type': 'RandTranslate',
                                    'y': (-0.1, 0.1)
                                }, {
                                    'type': 'RandRotate',
                                    'angle': (-30, 30)
                                },
                                            [{
                                                'type': 'RandShear',
                                                'x': (-30, 30)
                                            }, {
                                                'type': 'RandShear',
                                                'y': (-30, 30)
                                            }]])
                        ]),
                    dict(
                        type='RandErase',
                        n_iterations=(1, 5),
                        size=[0, 0.2],
                        squared=True)
                ],
                record=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='ExtraAttrs', tag='unsup_student'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ],
        unsup_teacher=[
            dict(
                type='Sequential',
                transforms=[
                    dict(
                        type='RandResize',
                        img_scale=[(1333, 400), (1333, 1200)],
                        multiscale_mode='range',
                        keep_ratio=True),
                    dict(type='RandFlip', flip_ratio=0.5)
                ],
                record=True),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='ExtraAttrs', tag='unsup_teacher'),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'],
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'img_norm_cfg', 'pad_shape', 'scale_factor', 'tag',
                           'transform_matrix'))
        ])
]
fp16 = dict(loss_scale='dynamic')
work_dir = '/home/xcc/dual_teacher_project/DualTeacher/work_dirs/ablation_v1/fold6_seed678/b0'
auto_resume = False
fold = 6
percent = 3
cfg_name = 'b0'
seed = 678
ablation_experiment = 'b0'
custom_imports = dict(
    imports=['ssod.models.roi_heads.small_bkg_reweight'],
    allow_failed_imports=False)
