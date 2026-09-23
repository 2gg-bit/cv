"""E3: ramp supervised SAR weight only; all pseudo-label settings stay B0."""
_base_ = ["./phase3_dual_teacher_ssdd.py"]
semi_wrapper = dict(train_cfg=dict(m2_enabled=False, pg=dict(
    mode="sup2", total_iters="${runner.max_iters}", start_ratio=0.5, end_ratio=1.0)))
work_dir = "work_dirs/ablations/pg_sup2/${percent}/${fold}"
