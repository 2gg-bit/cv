"""Reusable reweight recipe; historical generated config bytes stay archived."""
_base_ = ["./phase3_dual_teacher_ssdd.py"]
custom_imports = dict(imports=["ssod.models.roi_heads.small_bkg_reweight"],
                      allow_failed_imports=False)
model = dict(roi_head=dict(
    type="SmallBkgReweightRoIHead",
    bbox_head=dict(type="SmallBkgReweightBBoxHead",
                   reweight=dict(enable=True, lambda_=1.0, max_area=1024.0,
                                 tag="sup2"))))
semi_wrapper = dict(train_cfg=dict(m2_enabled=False, m2_force_weight_one=False))
work_dir = "work_dirs/ablations/reweight_l1/${percent}/${fold}"
