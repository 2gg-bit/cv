"""M2 + exact minimum-variance pseudo-label admission (single-class SSDD).

Only the unsupervised RoI classification admission changes. Initial detection,
RPN, regression uncertainty, M2 weights/Z0 rule, and inference stay as in M2.
Start fresh from Phase1/2; do not resume an old M2/M3 checkpoint.
"""

_base_ = ["./phase3_dual_teacher_ssdd_dev_m2.py"]

semi_wrapper = dict(
    train_cfg=dict(
        m2_enabled=True,
        m2_force_weight_one=False,
        m3_enabled=False,
        mvdt_enabled=True,
        mvdt=dict(
            warmup_iters=1000,
            update_interval=1000,
            min_samples=128,
            min_group_size=2,
            max_scores=1000000,
        ),
    ),
)

work_dir = "work_dirs/dev_ssdd/phase3_dual_teacher_m2_mvdt/${percent}/${fold}"
auto_resume = False
resume_from = None
load_from = None
