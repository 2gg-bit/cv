"""Pure CPU tests for M3 geometry selection and regression-target replacement."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "m3_routing_under_test", str(ROOT / "ssod/models/m3_routing.py"))
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)
select_regression_targets = routing.select_regression_targets
replace_positive_regression_targets = routing.replace_positive_regression_targets


def selection_inputs(count=1, dtype=torch.float32):
    anchors = torch.tensor([[0., 0., 10., 10.]] * count, dtype=dtype).reshape(count, 4)
    first = torch.tensor([[0., 0., 9., 10.]] * count, dtype=dtype).reshape(count, 4)
    second = torch.tensor([[1., 0., 10., 10.]] * count, dtype=dtype).reshape(count, 4)
    return [anchors, first, second, torch.ones(count, 4, dtype=dtype),
            torch.full((count, 4), 2., dtype=dtype)]


def test_uses_mean_coordinate_uncertainty_and_keeps_equal_mean_anchor():
    values = selection_inputs(3)
    values[3] = torch.tensor([[0., 0., 0., 4.], [0., 0., 2., 2.], [3., 3., 3., 3.]])
    values[4] = torch.tensor([[1.1] * 4, [1.] * 4, [2.] * 4])
    boxes, sources = select_regression_targets(*values)
    assert sources.tolist() == [1, 0, 2]
    expected = torch.stack((values[1][0], values[0][1], values[2][2]))
    assert torch.equal(boxes, expected)


def test_accepts_scalar_uncertainty_and_does_not_match_across_anchor_rows():
    anchors = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])
    first = anchors.flip(0)
    second = anchors + torch.tensor([1., 0., 0., 0.])
    boxes, sources = select_regression_targets(
        anchors, first, second, torch.zeros(2), torch.ones(2))
    assert sources.tolist() == [2, 2]
    assert torch.equal(boxes, second)


@pytest.mark.parametrize("teacher_index", [1, 2])
@pytest.mark.parametrize("invalid", ["nan_box", "inf_box", "zero_width", "negative_height",
                                     "distant", "nan_uncertainty", "inf_uncertainty",
                                     "negative_uncertainty"])
def test_one_invalid_candidate_cannot_win_even_with_lower_uncertainty(teacher_index, invalid):
    values = selection_inputs()
    values[3].fill_(10.)
    values[4].fill_(10.)
    values[teacher_index + 2].fill_(0.)
    if invalid == "nan_box":
        values[teacher_index][0, 0] = float("nan")
    elif invalid == "inf_box":
        values[teacher_index][0, 2] = float("inf")
    elif invalid == "zero_width":
        values[teacher_index][0, 2] = values[teacher_index][0, 0]
    elif invalid == "negative_height":
        values[teacher_index][0, 3] = -1.
    elif invalid == "distant":
        values[teacher_index] += 100.
    elif invalid == "nan_uncertainty":
        values[teacher_index + 2][0, 2] = float("nan")
    elif invalid == "inf_uncertainty":
        values[teacher_index + 2][0, 2] = float("inf")
    else:
        values[teacher_index + 2][0, 2] = -1.
    boxes, sources = select_regression_targets(*values)
    other = 3 - teacher_index
    assert sources.tolist() == [other]
    assert torch.equal(boxes, values[other])


def test_both_invalid_fall_back_to_exact_original_anchor():
    values = selection_inputs(dtype=torch.float64)
    values[0][0, 2] += 1e-10
    values[1][0, 2] = 0.
    values[4][0, 0] = -1.
    boxes, sources = select_regression_targets(*values)
    assert torch.equal(boxes, values[0])
    assert sources.tolist() == [0]


def test_iou_uses_continuous_coordinates_and_inclusive_threshold():
    values = selection_inputs(2)
    values[1][:, 2] = torch.tensor([5., 4.9])
    values[2][:, 2] = values[2][:, 0]
    boxes, sources = select_regression_targets(*values, min_anchor_iou=0.5)
    assert sources.tolist() == [1, 0]
    assert torch.equal(boxes, torch.stack((values[1][0], values[0][1])))


@pytest.mark.parametrize("scale", [1e-20, 1e20])
def test_iou_preserves_valid_geometry_at_extreme_finite_scales(scale):
    values = selection_inputs()
    for index in (0, 1, 2):
        values[index] *= scale
    boxes, sources = select_regression_targets(*values)
    assert sources.tolist() == [1]
    assert torch.equal(boxes, values[1])


@pytest.mark.parametrize("mode,expected", [("original", [0, 0]), ("teacher1", [1, 0]),
                                          ("teacher2", [2, 2])])
def test_control_modes_respect_validity_and_never_substitute_other_teacher(mode, expected):
    values = selection_inputs(2)
    values[1][1, 2] = 0.
    values[3].fill_(100.)
    boxes, sources = select_regression_targets(*values, mode=mode)
    assert sources.tolist() == expected
    for index, source in enumerate(expected):
        assert torch.equal(boxes[index], values[source][index])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_output_dtype_device_detachment_and_no_input_mutation(dtype):
    values = [value.requires_grad_() for value in selection_inputs(2, dtype)]
    snapshots = [value.detach().clone() for value in values]
    boxes, sources = select_regression_targets(*values)
    assert boxes.dtype == dtype and boxes.device == values[0].device
    assert sources.dtype == torch.long and sources.device == values[0].device
    assert not boxes.requires_grad and not sources.requires_grad
    assert all(torch.equal(value, snapshot) for value, snapshot in zip(values, snapshots))


def test_original_mode_returns_exact_double_anchor_values():
    values = selection_inputs(dtype=torch.float64)
    values[0][0, 2] += 1e-10
    boxes, sources = select_regression_targets(*values, mode="original")
    assert torch.equal(boxes, values[0])
    assert sources.tolist() == [0]


def test_comparisons_are_float32_even_for_double_inputs():
    values = selection_inputs(dtype=torch.float64)
    values[3].fill_(1.00000001)
    values[4].fill_(1.00000002)
    boxes, sources = select_regression_targets(*values)
    assert sources.tolist() == [0]
    assert torch.equal(boxes, values[0])


def test_large_finite_coordinate_uncertainty_has_finite_mean():
    values = selection_inputs()
    values[3].fill_(2e38)
    values[4].fill_(3e38)
    _, sources = select_regression_targets(*values)
    assert sources.tolist() == [1]


def test_subnormal_coordinate_uncertainty_preserves_strict_mean_order():
    values = selection_inputs()
    values[3].fill_(1e-45)
    values[4].fill_(3e-45)
    _, sources = select_regression_targets(*values)
    assert sources.tolist() == [1]


def test_selector_is_deterministic_and_leaves_global_rng_states_unchanged():
    values = selection_inputs(3)
    values[3][1].fill_(3.)
    values[3][2].fill_(2.)
    torch_state, python_state = torch.get_rng_state().clone(), random.getstate()
    first = select_regression_targets(*values)
    second = select_regression_targets(*values)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state


@pytest.mark.parametrize("mode", ["lower_uncertainty", "original", "teacher1", "teacher2"])
def test_empty_selection(mode):
    boxes, sources = select_regression_targets(*selection_inputs(0), mode=mode)
    assert tuple(boxes.shape) == (0, 4) and tuple(sources.shape) == (0,)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), "bad", None])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="min_anchor_iou"):
        select_regression_targets(*selection_inputs(), min_anchor_iou=threshold)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        select_regression_targets(*selection_inputs(), mode="random")


@pytest.mark.parametrize("input_index,shape", [(0, (4,)), (1, (1, 5)), (2, (2, 4)),
                                              (3, (1, 1)), (4, (2,))])
def test_rejects_misaligned_shapes(input_index, shape):
    values = selection_inputs()
    values[input_index] = torch.ones(*shape)
    with pytest.raises(ValueError, match="shape|rows"):
        select_regression_targets(*values)


@pytest.mark.parametrize("coordinate,value", [(0, float("nan")), (2, float("inf")),
                                             (2, 0.), (3, -1.)])
def test_rejects_invalid_anchors_including_original_mode(coordinate, value):
    values = selection_inputs()
    values[0][0, coordinate] = value
    with pytest.raises(ValueError, match="anchors must"):
        select_regression_targets(*values, mode="original")


class DeltaCoder:
    """Standard center/log-size delta encoding with an observable call list."""
    def __init__(self):
        self.calls = []
        self.scale = torch.tensor(1., requires_grad=True)

    def encode(self, proposals, targets):
        self.calls.append((proposals.detach().clone(), targets.detach().clone()))
        proposal_size = proposals[:, 2:] - proposals[:, :2]
        target_size = targets[:, 2:] - targets[:, :2]
        proposal_center = (proposals[:, 2:] + proposals[:, :2]) * 0.5
        target_center = (targets[:, 2:] + targets[:, :2]) * 0.5
        return torch.cat(((target_center - proposal_center) / proposal_size,
                          torch.log(target_size / proposal_size)), dim=1) * self.scale


def regression_fixture(decoded=False):
    anchors0 = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.],
                             [40., 40., 50., 50.]], requires_grad=True)
    anchors1 = torch.tensor([[60., 60., 70., 70.], [80., 80., 90., 90.]],
                            requires_grad=True)
    selected0 = anchors0.detach().clone()
    selected0[2] = torch.tensor([40., 40., 48., 49.])
    selected1 = anchors1.detach().clone()
    selected1[1] = torch.tensor([81., 80., 90., 89.])
    selected = [selected0.requires_grad_(), selected1.requires_grad_()]
    results = []
    for anchors, assigned, num_neg in ((anchors0, [2, 0, 2], 2), (anchors1, [1], 1)):
        indices = torch.tensor(assigned, dtype=torch.long)
        pos_gt = anchors[indices]
        pos_boxes = (pos_gt.detach() + torch.tensor([-1., -1., 1., 1.])).requires_grad_()
        results.append(SimpleNamespace(
            pos_bboxes=pos_boxes, neg_bboxes=torch.ones(num_neg, 4),
            pos_gt_bboxes=pos_gt, pos_assigned_gt_inds=indices,
            pos_inds=torch.arange(len(assigned), dtype=torch.long),
            gt_flags=torch.zeros(len(assigned) + num_neg, dtype=torch.uint8)))
    head = SimpleNamespace(reg_decoded_bbox=decoded, bbox_coder=DeltaCoder())
    labels = torch.tensor([0, 0, 0, 1, 1, 0, 1])
    label_weights = torch.tensor([0.6, 0.7, 0.8, 0.1, 0.2, 0.9, 0.3])
    # Deliberate sentinel values make fallback re-encoding detectable.
    targets = torch.arange(28, dtype=torch.float32).reshape(7, 4).requires_grad_()
    weights = torch.ones(7, 4)
    return head, results, (labels, label_weights, targets, weights), selected


@pytest.mark.parametrize("decoded", [False, True])
def test_replaces_only_changed_positive_rows_with_original_gt_index_mapping(decoded):
    head, results, targets, selected = regression_fixture(decoded)
    target_snapshot = [tensor.detach().clone() for tensor in targets]
    sample_snapshot = [{key: value.detach().clone() for key, value in vars(result).items()}
                       for result in results]
    result = replace_positive_regression_targets(head, results, targets, selected)
    assert isinstance(result, tuple)
    assert all(result[index] is targets[index] for index in (0, 1, 3))
    assert result[2] is not targets[2] and not result[2].requires_grad
    assert all(torch.equal(before, after) for before, after in zip(target_snapshot, targets))
    # Image 0 is 3 positives + 2 negatives; image 1 is 1 positive + 1 negative.
    assert torch.equal(result[2][[1, 3, 4, 6]], targets[2][[1, 3, 4, 6]])
    chosen = torch.stack((selected[0][2], selected[0][2], selected[1][1]))
    proposals = torch.stack((results[0].pos_bboxes[0], results[0].pos_bboxes[2],
                             results[1].pos_bboxes[0]))
    expected = chosen if decoded else DeltaCoder().encode(proposals, chosen)
    assert torch.equal(result[2][[0, 2, 5]], expected)
    assert len(head.bbox_coder.calls) == (0 if decoded else 2)
    if not decoded:
        assert [len(call[0]) for call in head.bbox_coder.calls] == [2, 1]
    for sample, snapshot in zip(results, sample_snapshot):
        assert all(torch.equal(getattr(sample, name), value) for name, value in snapshot.items())


def test_encoded_delta_formula_uses_original_sampled_proposal():
    head, results, targets, selected = regression_fixture()
    result = replace_positive_regression_targets(head, results, targets, selected)
    # Original sampled proposal is [39,39,51,51], chosen GT is [40,40,48,49].
    expected = torch.tensor([-1. / 12., -0.5 / 12., 0., 0.])
    expected[2:] = torch.log(torch.tensor([8. / 12., 9. / 12.]))
    assert torch.equal(result[2][0], expected)


@pytest.mark.parametrize("decoded", [False, True])
def test_regression_loss_gradient_reaches_only_student_predictions(decoded):
    head, results, targets, selected = regression_fixture(decoded)
    result = replace_positive_regression_targets(head, results, targets, selected)
    prediction = torch.zeros_like(result[2], requires_grad=True)
    ((prediction - result[2]) ** 2).sum().backward()
    assert prediction.grad is not None and bool((prediction.grad != 0).any())
    assert targets[2].grad is None and head.bbox_coder.scale.grad is None
    assert all(boxes.grad is None for boxes in selected)
    assert all(sample.pos_bboxes.grad is None for sample in results)


def test_original_targets_never_invoke_coder_or_change_any_target_bits():
    head, results, targets, _ = regression_fixture()
    original = [torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.],
                              [40., 40., 50., 50.]]),
                torch.tensor([[60., 60., 70., 70.], [80., 80., 90., 90.]])]
    result = replace_positive_regression_targets(head, results, targets, original)
    assert torch.equal(result[2], targets[2])
    assert head.bbox_coder.calls == []


@pytest.mark.parametrize("num_neg", [0, 3])
def test_images_without_positives_keep_empty_or_negative_targets(num_neg):
    result = SimpleNamespace(pos_bboxes=torch.empty(0, 4), neg_bboxes=torch.ones(num_neg, 4),
                             pos_gt_bboxes=torch.empty(0, 4),
                             pos_assigned_gt_inds=torch.empty(0, dtype=torch.long))
    targets = (torch.ones(num_neg, dtype=torch.long), torch.ones(num_neg),
               torch.ones(num_neg, 4), torch.ones(num_neg, 4))
    head = SimpleNamespace(reg_decoded_bbox=False, bbox_coder=DeltaCoder())
    output = replace_positive_regression_targets(head, [result], targets, [torch.empty(0, 4)])
    assert torch.equal(output[2], targets[2])
    assert all(output[index] is targets[index] for index in (0, 1, 3))
    assert head.bbox_coder.calls == []


def test_zero_images_accepts_empty_targets():
    targets = (torch.empty(0, dtype=torch.long), torch.empty(0),
               torch.empty(0, 4), torch.empty(0, 4))
    output = replace_positive_regression_targets(None, [], targets, [])
    assert torch.equal(output[2], targets[2])


@pytest.mark.parametrize("malformed", ["image_count", "target_shape", "sample_count",
                                       "positive_count", "assignment_shape", "assignment_type",
                                       "out_of_bounds", "negative_index", "labels_shape"])
def test_replacement_rejects_shape_or_index_mismatches(malformed):
    head, results, targets, selected = regression_fixture()
    if malformed == "image_count":
        selected.pop()
    elif malformed == "target_shape":
        selected[0] = torch.ones(3, 5)
    elif malformed == "sample_count":
        results[0].neg_bboxes = torch.ones(1, 4)
    elif malformed == "positive_count":
        results[0].pos_gt_bboxes = torch.ones(2, 4)
    elif malformed == "assignment_shape":
        results[0].pos_assigned_gt_inds = torch.zeros(3, 1, dtype=torch.long)
    elif malformed == "assignment_type":
        results[0].pos_assigned_gt_inds = torch.zeros(3)
    elif malformed == "out_of_bounds":
        results[0].pos_assigned_gt_inds[0] = 3
    elif malformed == "negative_index":
        results[0].pos_assigned_gt_inds[0] = -1
    else:
        targets = (torch.ones(7, 1),) + targets[1:]
    with pytest.raises(ValueError):
        replace_positive_regression_targets(head, results, targets, selected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cross-device validation needs CUDA")
def test_selector_rejects_different_devices():
    values = selection_inputs()
    values[1] = values[1].cuda()
    with pytest.raises(ValueError, match="device"):
        select_regression_targets(*values)
