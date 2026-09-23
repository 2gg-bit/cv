"""E2: baseline SSDD protocol + supervised foreground positions; M2 is OFF."""
_base_ = ["./phase3_dual_teacher_ssdd.py"]
model = dict(roi_head=dict(type="ForegroundRoIHead", foreground_enabled=True,
                          foreground_hidden_channels=32, foreground_loss_weight=0.1))
semi_wrapper = dict(train_cfg=dict(m2_enabled=False, m2_force_weight_one=False))
work_dir = "work_dirs/ablations/fg/${percent}/${fold}"
