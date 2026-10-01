"""B0 + supervised SAR positive-RoI GIoU only; original L1 stays enabled."""
_base_ = ["./phase3_dual_teacher_ssdd.py"]
custom_imports = dict(imports=["ssod.models.roi_heads.sup2_giou"],
                      allow_failed_imports=False)
model = dict(roi_head=dict(type="Sup2GIoURoIHead", sup2_giou_enabled=True,
                          sup2_giou_weight=1.0))
semi_wrapper = dict(train_cfg=dict(m2_enabled=False, m2_force_weight_one=False))
work_dir = "work_dirs/ablations/sup2_giou/${percent}/${fold}"
