"""
Phase 3 (dev protocol) + M2 positive pseudo-label reliability weight.

Only change vs phase3_dual_teacher_ssdd_dev.py: enable the M2 positive-ROI
classification weight  a_i^M2 = a_i * stopgrad(min(p_T1, p_T2)).

Training math (data, thresholds, sampling, fusion, background weights, the
normalization denominator Z_0, regression loss, inference) is unchanged.
The output directory is explicitly isolated from the old baseline runs.
"""
_base_ = ["./phase3_dual_teacher_ssdd_dev.py"]

semi_wrapper = dict(
    train_cfg=dict(
        # M2: positive pseudo-label reliability weight, min of the two teachers'
        # ship probabilities on the same aligned ROI.  Default-off in the model;
        # this config explicitly turns it on.
        m2_enabled=True,
        m2_force_weight_one=False,
    ),
)
work_dir = "work_dirs/dev_ssdd/phase3_dual_teacher_m2_new/${percent}/${fold}"
auto_resume = False
resume_from = None
load_from = None
