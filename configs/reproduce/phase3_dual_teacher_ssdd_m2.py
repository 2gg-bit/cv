"""E1: baseline SSDD protocol + positive pseudo-label reliability only."""
_base_ = ["./phase3_dual_teacher_ssdd.py"]
semi_wrapper = dict(train_cfg=dict(m2_enabled=True, m2_force_weight_one=False))
work_dir = "work_dirs/ablations/m2/${percent}/${fold}"
