"""M2 + supervised foreground positions; fresh Phase3 from Phase1/2 weights."""

_base_ = ["./phase3_dual_teacher_ssdd_dev_m2.py"]

model = dict(roi_head=dict(
    type="ForegroundRoIHead",
    foreground_enabled=True,
    foreground_hidden_channels=32,
    foreground_loss_weight=0.1,
))
semi_wrapper = dict(train_cfg=dict(
    m2_enabled=True, m2_force_weight_one=False,
    m3_enabled=False, mvdt_enabled=False,
))
work_dir = "work_dirs/dev_ssdd/phase3_dual_teacher_m2_fg/${percent}/${fold}"
auto_resume = False
resume_from = None
load_from = None
