# Agreement-aware dual-teacher pseudo-label routing

This is an opt-in training experiment. It changes the pseudo-label construction
used for the unlabeled SAR stream; it adds no detector parameters or loss terms
and does not change inference. No long training run or accuracy evaluation is
included in this code delivery.

## Rule

For each unlabeled image, match T1 and T2 detections only when they have the
same class and IoU at least `teacher_pseudo_routing.iou_threshold` (default
`0.5`). Matching is greedy, one-to-one, and ordered by decreasing IoU.

- A matched pair becomes one shared target. Coordinates are averaged using the
  two detection scores as weights, and its score is `sqrt(score_T1 * score_T2)`.
  The score does not exceed the stronger teacher's score.
- A detection without a match is retained unchanged in the shared target set.
  This avoids making one teacher's positive candidate an implicit background
  for the other student.
- The common routed boxes are passed to both students. Each teacher separately
  recomputes augmentation-jitter localization uncertainty on those boxes; the
  existing branch-specific uncertainty filter remains in effect.
- The existing B0 NMS fusion remains byte-for-byte on the default path when the
  option is absent or disabled.

The current route counters (`matched_pairs`, `teacher1_only`, `teacher2_only`,
per-teacher inputs, and output count) are logged every 50 calls and captured by
the single-step acceptance check. These are implementation diagnostics, not
accuracy evidence.

## Configuration

Use `configs/reproduce/phase3_dual_teacher_ssdd_teacher_routed.py`. It inherits
the reproduction config and enables only this option. Do not combine it with
M2, FG, PG, small-background reweighting, or GIoU in the first comparison.

## Checks on the training machine

From the actual training checkout, with its normal `dt` environment:

```bash
python tools/train_ablation.py --check-source-only
python tools/train_ablation.py --check-init \
  configs/reproduce/phase3_dual_teacher_ssdd_teacher_routed.py \
  --cfg-options fold=6 percent=3
python tools/train_ablation.py \
  configs/reproduce/phase3_dual_teacher_ssdd_teacher_routed.py \
  --check-step --seed 678 --batch-index 4 \
  --out-dir ablation_configs/teacher_pseudo_routing_acceptance_fold6_retry_d4f66fe \
  --cfg-options fold=6 percent=3
```

The retry output directory above is intentionally distinct from the first
attempt, whose failed/incomplete evidence should be preserved.

The step check runs B0, a deterministic B0 replay, and the routed variant on
the same real training batch. It checks strict Phase1/Phase2 initialization,
bitwise B0 replay, unchanged supervised losses, finite unsupervised losses,
nonempty routed teacher inputs and outputs, no teacher gradients, unchanged
parameters/buffers, and zero optimizer/EMA updates. Unsupervised losses are
allowed to differ because pseudo-label targets and teacher-specific uncertainty
are what this experiment changes; the result records which unsupervised loss
keys changed. If the selected batch contains no teacher detections, the result
is `incomplete`; try a different batch index in a new output directory.

The NumPy-only matching and fusion tests can run on a development machine
without MMDetection or CUDA:

```bash
python -m pytest tests/test_teacher_pseudo_router.py -q
```

Passing these checks establishes the routing behavior only. It does not show
that the model improves. Existing SSDD test images have already been used for
method selection; any comparison on them remains exploratory. Do not start a
long run or claim generalization gains without an untouched SAR evaluation set.
