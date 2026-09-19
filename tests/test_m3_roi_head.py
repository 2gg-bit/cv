"""Execute the actual M3 head with CPU PyTorch and an AST framework fixture.

The fixture records the original assignment/sampling inputs. These tests cover
the integration boundary, not compiled MMDetection RoIAlign or GPU acceptance;
``tools/check_m3_step.py`` exercises that real stack on a fixed pipeline batch.
"""

import ast
import hashlib
import importlib.util
from pathlib import Path
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "m3_head_routing_fixture", str(ROOT / "ssod/models/m3_routing.py"))
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


def bbox2roi(boxes_per_image):
    return torch.cat([torch.cat((boxes.new_full((len(boxes), 1), index), boxes), 1)
                      for index, boxes in enumerate(boxes_per_image)])


class BBoxHeadFixture(nn.Module):
    def __init__(self, reg_decoded_bbox=False, num_classes=1):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.4))
        self.num_classes = num_classes
        self.reg_decoded_bbox = reg_decoded_bbox
        self.bbox_coder = SimpleNamespace(encode=lambda proposals, gt: gt - proposals)

    def get_targets(self, samples, gt_bboxes, gt_labels, train_cfg):
        self.target_inputs = (samples, gt_bboxes, gt_labels, train_cfg)
        labels, label_weights, targets, bbox_weights = [], [], [], []
        for sample in samples:
            pos, neg = len(sample.pos_bboxes), len(sample.neg_bboxes)
            labels.append(torch.tensor([0] * pos + [1] * neg, dtype=torch.long))
            label_weights.append(torch.ones(pos + neg))
            encoded = (sample.pos_gt_bboxes if self.reg_decoded_bbox else
                       self.bbox_coder.encode(sample.pos_bboxes, sample.pos_gt_bboxes))
            targets.append(torch.cat((encoded, torch.zeros(neg, 4))))
            bbox_weights.append(torch.cat((torch.ones(pos, 4), torch.zeros(neg, 4))))
        self.base_targets = tuple(torch.cat(items) for items in
                                  (labels, label_weights, targets, bbox_weights))
        return self.base_targets

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights,
             bbox_targets, bbox_weights):
        self.loss_inputs = (labels, label_weights, bbox_targets, bbox_weights)
        cls_loss = ((F.cross_entropy(cls_score, labels, reduction="none")
                     * label_weights).sum() / max(len(labels), 1))
        bbox_loss = ((bbox_pred - bbox_targets).square() * bbox_weights).sum()
        return dict(loss_cls=cls_loss, loss_bbox=bbox_loss,
                    acc=(cls_score.argmax(dim=1) == labels).float().sum())


class StandardRoIHeadFixture(nn.Module):
    def __init__(self, bbox_head=None, with_bbox=True, with_mask=False):
        super().__init__()
        self.bbox_head = bbox_head or BBoxHeadFixture()
        self.with_bbox, self.with_mask = with_bbox, with_mask
        self.train_cfg = object()
        self.parent_calls = 0

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None):
        self.assignment_inputs = (proposal_list, gt_bboxes, gt_labels,
                                  gt_bboxes_ignore, gt_masks)
        if getattr(self, "raise_in_parent", False):
            raise RuntimeError("assignment failure")
        # Fixed original sampling outcomes, including original pos_gt_bboxes.
        # Passing different regression boxes cannot alter this fixture's inputs.
        self.sampled_results = self.next_samples
        return self._bbox_forward_train(
            x, self.sampled_results, gt_bboxes, gt_labels, img_metas)["loss_bbox"]

    def _bbox_forward_train(self, x, samples, gt_bboxes, gt_labels, img_metas):
        self.parent_calls += 1
        rois = bbox2roi([sample.bboxes for sample in samples])
        result = self._bbox_forward(x, rois)
        targets = self.bbox_head.get_targets(samples, gt_bboxes, gt_labels, self.train_cfg)
        result["loss_bbox"] = self.bbox_head.loss(
            result["cls_score"], result["bbox_pred"], rois, *targets)
        return result

    def _bbox_forward(self, x, rois):
        marker = self.bbox_head.marker
        return dict(cls_score=torch.stack((rois[:, 1] * 0 + marker,
                                           rois[:, 1] * 0 - marker), 1),
                    bbox_pred=rois[:, 1:] * 0 + marker)

    def simple_test(self, *args, **kwargs):
        return "parent inference"


def actual_head_class():
    source = ROOT / "ssod/models/roi_heads/m3_roi_head.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(__name__="m3_roi_head_fixture", bbox2roi=bbox2roi,
                     HEADS=SimpleNamespace(register_module=lambda: lambda cls: cls),
                     StandardRoIHead=StandardRoIHeadFixture,
                     replace_positive_regression_targets=routing.replace_positive_regression_targets)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["M3RoIHead"]


M3RoIHead = actual_head_class()


def sample(positives, negatives, gt_boxes, assigned):
    positives = torch.tensor(positives, dtype=torch.float32).reshape(-1, 4)
    negatives = torch.tensor(negatives, dtype=torch.float32).reshape(-1, 4)
    indices = torch.tensor(assigned, dtype=torch.long)
    return SimpleNamespace(pos_bboxes=positives, neg_bboxes=negatives,
                           bboxes=torch.cat((positives, negatives)),
                           pos_gt_bboxes=gt_boxes[indices], pos_assigned_gt_inds=indices)


def fixture(reg_decoded_bbox=False, empty_gt=False):
    head = M3RoIHead(bbox_head=BBoxHeadFixture(reg_decoded_bbox))
    if empty_gt:
        gt = [torch.empty(0, 4)]
        samples = [sample([], [[20., 20., 30., 30.]], gt[0], [])]
    else:
        gt = [torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]),
              torch.tensor([[1., 1., 11., 11.]])]
        samples = [sample([[20., 20., 29., 29.], [0., 0., 9., 9.]],
                          [[50., 50., 60., 60.]], gt[0], [1, 0]),
                   sample([[0., 0., 10., 10.]], [], gt[1], [0])]
    head.next_samples = samples
    labels = [torch.zeros(len(boxes), dtype=torch.long) for boxes in gt]
    proposals = [item.bboxes.clone() for item in samples]
    return head, dict(x=[], img_metas=[{} for _ in gt], proposal_list=proposals,
                     gt_bboxes=gt, gt_labels=labels)


def test_disabled_path_delegates_and_has_no_new_state_or_inference_override():
    head, inputs = fixture()
    head.forward_train(**inputs)
    assert head.parent_calls == 1 and head._m3_targets is None
    baseline = StandardRoIHeadFixture()
    assert set(head.state_dict()) == set(baseline.state_dict())
    head.load_state_dict(baseline.state_dict(), strict=True)
    assert M3RoIHead.simple_test is StandardRoIHeadFixture.simple_test


@pytest.mark.parametrize("decoded", [False, True])
def test_original_targets_have_exact_loss_and_gradient_equivalence(decoded):
    head, inputs = fixture(decoded)
    disabled = head.forward_train(**inputs)
    sum(value for key, value in disabled.items() if "loss" in key).backward()
    old_grad = head.bbox_head.marker.grad.clone()
    head.bbox_head.marker.grad = None
    enabled = head.forward_train(**inputs, reg_target_bboxes=inputs["gt_bboxes"])
    assert all(torch.equal(value, enabled[key]) for key, value in disabled.items())
    sum(value for key, value in enabled.items() if "loss" in key).backward()
    assert torch.equal(old_grad, head.bbox_head.marker.grad)
    assert head.parent_calls == 1 and head._m3_targets is None


@pytest.mark.parametrize("decoded", [False, True])
def test_routed_targets_keep_original_sampling_labels_weights_and_cls_loss(decoded):
    head, inputs = fixture(decoded)
    original_samples = head.next_samples
    snapshots = [[value.clone() for value in (sample.pos_bboxes, sample.neg_bboxes,
                  sample.pos_gt_bboxes, sample.pos_assigned_gt_inds)]
                 for sample in original_samples]
    baseline = head.forward_train(**inputs)
    chosen = [boxes.clone() for boxes in inputs["gt_bboxes"]]
    chosen[0][1, 2:] -= 1.0
    chosen[1][0, :2] += 0.5
    for boxes in chosen:
        boxes.requires_grad_()
    routed = head.forward_train(**inputs, reg_target_bboxes=chosen)
    assert torch.equal(baseline["loss_cls"], routed["loss_cls"])
    assert torch.equal(baseline["acc"], routed["acc"])
    assert not torch.equal(baseline["loss_bbox"], routed["loss_bbox"])
    assert head.assignment_inputs[0] is inputs["proposal_list"]
    assert head.assignment_inputs[1] is inputs["gt_bboxes"]
    assert head.assignment_inputs[2] is inputs["gt_labels"]
    assert head.sampled_results is original_samples
    assert head.bbox_head.target_inputs[1] is inputs["gt_bboxes"]
    base, routed_targets = head.bbox_head.base_targets, head.bbox_head.loss_inputs
    assert all(base[index] is routed_targets[index] for index in (0, 1, 3))
    # GT row 1 belongs to sampled row 0; image 2 starts after the negative row.
    assert torch.equal(base[2][1:3], routed_targets[2][1:3])
    assert not torch.equal(base[2][0], routed_targets[2][0])
    assert not torch.equal(base[2][3], routed_targets[2][3])
    assert not routed_targets[2].requires_grad
    for sampled, before in zip(original_samples, snapshots):
        assert all(torch.equal(actual, old) for actual, old in zip(
            (sampled.pos_bboxes, sampled.neg_bboxes, sampled.pos_gt_bboxes,
             sampled.pos_assigned_gt_inds), before))
    (routed["loss_cls"] + routed["loss_bbox"]).backward()
    assert all(boxes.grad is None for boxes in chosen)
    assert torch.isfinite(head.bbox_head.marker.grad)


def test_empty_gt_and_all_negative_samples_keep_finite_unchanged_losses():
    head, inputs = fixture(empty_gt=True)
    baseline = head.forward_train(**inputs)
    routed = head.forward_train(**inputs, reg_target_bboxes=[torch.empty(0, 4)])
    assert all(torch.equal(value, routed[key]) for key, value in baseline.items())
    assert routed["loss_bbox"].item() == 0
    (routed["loss_cls"] + routed["loss_bbox"]).backward()
    assert torch.isfinite(head.bbox_head.marker.grad)


def test_target_context_restores_after_parent_exception():
    head, inputs = fixture()
    previous = object()
    head._m3_targets = previous
    head.raise_in_parent = True
    with pytest.raises(RuntimeError, match="assignment failure"):
        head.forward_train(**inputs, reg_target_bboxes=inputs["gt_bboxes"])
    assert head._m3_targets is previous


@pytest.mark.parametrize("targets", [[], [torch.empty(0, 4), torch.empty(0, 4)]])
def test_rejects_target_batch_or_row_shape_mismatch_before_sampling(targets):
    head, inputs = fixture()
    with pytest.raises(ValueError, match="batch|rows"):
        head.forward_train(**inputs, reg_target_bboxes=targets)
    assert not hasattr(head, "assignment_inputs") and head._m3_targets is None


@pytest.mark.parametrize("kwargs", [dict(with_mask=True), dict(with_bbox=False),
                                    dict(bbox_head=BBoxHeadFixture(num_classes=2))])
def test_rejects_mask_disabled_bbox_and_multiclass_heads(kwargs):
    with pytest.raises(ValueError, match="single-class bbox-only"):
        M3RoIHead(**kwargs)


def acceptance_helpers():
    source = ROOT / "tools/check_m3_step.py"
    tree = ast.parse(source.read_text())
    names = {"tensor_hash", "install_sampling_observers", "sampling_signature", "compare_tensors"}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, hashlib=hashlib, types=types)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


def test_acceptance_observer_keeps_results_and_counts_actual_encoded_changes():
    helpers = acceptance_helpers()
    first, inputs = fixture()
    second, _ = fixture()
    model = SimpleNamespace(student1=SimpleNamespace(roi_head=first),
                            student2=SimpleNamespace(roi_head=second))
    reference = first.forward_train(**inputs)
    events = []
    helpers["install_sampling_observers"](model, events)
    observed = first.forward_train(**inputs)
    assert all(torch.equal(value, observed[key]) for key, value in reference.items())
    assert events[0]["actual_bbox_target_rows_changed"] == 0
    chosen = [boxes.clone() for boxes in inputs["gt_bboxes"]]
    chosen[0][1, 2:] -= 1.0
    chosen[1][0, :2] += 0.5
    first.forward_train(**inputs, reg_target_bboxes=chosen)
    assert events[1]["actual_bbox_target_rows_changed"] == 2
    assert events[1]["labels_and_weights_unchanged"]
    assert events[1]["negative_targets_unchanged"]
    assert helpers["sampling_signature"]([events[0]]) == helpers["sampling_signature"]([events[1]])
    # Scoped bbox method wrappers have been restored after the observed call.
    assert first.bbox_head.loss.__func__ is BBoxHeadFixture.loss
    assert first.bbox_head.get_targets.__func__ is BBoxHeadFixture.get_targets


def test_acceptance_comparator_excludes_only_named_regression_loss():
    compare = acceptance_helpers()["compare_tensors"]
    left = {"unsup1_loss_bbox": torch.tensor(1.), "unsup1_loss_cls": torch.tensor(2.)}
    right = {"unsup1_loss_bbox": torch.tensor(3.), "unsup1_loss_cls": torch.tensor(2.)}
    assert not compare(left, right, 1e-6, 1e-5)["passed"]
    assert compare(left, right, 1e-6, 1e-5, exclude=("unsup1_loss_bbox",))["passed"]
    right["unsup1_loss_cls"] += 0.1
    assert not compare(left, right, 1e-6, 1e-5, exclude=("unsup1_loss_bbox",))["passed"]
    del right["unsup1_loss_cls"]
    assert not compare(left, right, 1e-6, 1e-5, exclude=("unsup1_loss_bbox",))["passed"]


@pytest.mark.parametrize("fold", [6, 7, 8])
def test_acceptance_injects_fold_percent_seed_before_real_placeholder_resolution(fold, tmp_path):
    resolver_spec = importlib.util.spec_from_file_location(
        "m3_acceptance_config_vars", str(ROOT / "ssod/utils/vars.py"))
    resolver = importlib.util.module_from_spec(resolver_spec)
    resolver_spec.loader.exec_module(resolver)
    source = ROOT / "tools/check_m3_step.py"
    tree = ast.parse(source.read_text())
    execute = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "execute")
    # Execute the actual config-preparation statements, stopping at patch_config
    # so this reproduces placeholder errors without importing MMCV or a GPU.
    start = next(index for index, node in enumerate(execute.body)
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "fromfile")
    stop = next(index for index, node in enumerate(execute.body)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "patch_config")
    tree.body = execute.body[start:stop + 1]

    class ConfigFixture(SimpleNamespace):
        def merge_from_dict(self, values):
            for key, value in values.items():
                setattr(self, key, value)

    config = ConfigFixture(path="work_dirs/${percent}/${fold}/seed${seed}")
    args = SimpleNamespace(config="fixture.py", fold=fold, seed=678,
                           cfg_options=dict(percent=99, fold=42, seed=1))
    namespace = dict(Config=SimpleNamespace(fromfile=lambda _: config), args=args,
                     out_dir=tmp_path, patch_config=lambda cfg: resolver.resolve(vars(cfg)))
    exec(compile(tree, str(source), "exec"), namespace)
    resolved = namespace["cfg"]
    assert (resolved["fold"], resolved["percent"], resolved["seed"]) == (fold, 3, 678)
    assert resolved["path"] == "work_dirs/3/{}/seed678".format(fold)
