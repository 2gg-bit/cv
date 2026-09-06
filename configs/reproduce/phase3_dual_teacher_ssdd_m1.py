"""Exploratory M1: supervised RoI localization quality, no pseudo-label change.

M1-A is the default (original classification ranking). For M1-B evaluation of
the SAME trained checkpoint, set model.roi_head.quality_inference=True before
patch_config. To check M0 equivalence with a baseline checkpoint, disable both
quality_enabled and quality_inference. Do not resume a baseline checkpoint
into the enabled M1 model; start from the unchanged Phase 1/2 checkpoints.
"""

_base_ = ["phase3_dual_teacher_ssdd.py"]

model = dict(
    roi_head=dict(
        type="QualityRoIHead",
        quality_enabled=True,
        quality_inference=False,
        quality_hidden_channels=64,
        quality_loss_weight=1.0,
    ),
)

# Never overwrite baseline checkpoints, resolved configs, or training logs.
work_dir = "work_dirs/phase3_dual_teacher_m1_quality/${percent}/${fold}"
