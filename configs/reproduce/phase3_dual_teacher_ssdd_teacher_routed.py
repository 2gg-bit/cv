"""B0 + opt-in agreement-aware routing for dual-teacher pseudo labels.

Teacher detections that agree are score-weighted into a shared target. A
single-teacher detection remains in the shared target set, avoiding a false
background target in the other student branch. B0 is unchanged when this
config option is disabled.
"""

_base_ = ["./phase3_dual_teacher_ssdd.py"]

semi_wrapper = dict(
    train_cfg=dict(
        m2_enabled=False,
        m2_force_weight_one=False,
        teacher_pseudo_routing=dict(enabled=True, iou_threshold=0.5),
    )
)

work_dir = "work_dirs/ablations/teacher_pseudo_routing/${percent}/${fold}"
