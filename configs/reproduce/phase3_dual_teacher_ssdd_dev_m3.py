"""M3 v1: M2 classification + teacher-specific regression targets.

No change to original fused pseudo labels, uncertainty eligibility, assignment,
sampling, RPN, supervised losses or inference. Requires the EXISTING dev split.
Do not generate a new split. All switches are default-off in model code.
"""

_base_ = ["./phase3_dual_teacher_ssdd_dev_m2.py"]

model = dict(roi_head=dict(type="M3RoIHead"))
semi_wrapper = dict(train_cfg=dict(
    m2_enabled=True,
    m2_force_weight_one=False,
    m3_enabled=True,
    m3_target_mode="lower_uncertainty",
    # Geometric identity guard, not a pseudo-label admission threshold.
    # Fixed before offline audit; do not sweep to obtain a positive result.
    m3_min_anchor_iou=0.5,
))
work_dir = "work_dirs/dev_ssdd/phase3_dual_teacher_m3/3/${fold}"
auto_resume = False
resume_from = None
load_from = None
