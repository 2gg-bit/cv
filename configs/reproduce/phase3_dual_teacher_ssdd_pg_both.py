"""E4: ramp supervised/unsupervised SAR student weights together."""
_base_ = ["./phase3_dual_teacher_ssdd.py"]
semi_wrapper = dict(train_cfg=dict(m2_enabled=False, pg=dict(
    mode="both", total_iters="${runner.max_iters}", start_ratio=0.5, end_ratio=1.0)))
work_dir = "work_dirs/ablations/pg_both/${percent}/${fold}"
