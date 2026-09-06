"""Controlled M1 initialization tests with real CPU tensors and checkpoint IO.

No MMDetection/CUDA installation or Phase 1/2 training files are required.
The final EMA test executes the existing hook method extracted from its AST.
"""

import ast
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quality_checkpoint_utils", str(ROOT / "ssod/utils/checkpoint.py")
)
CHECKPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT)
LOGGER = logging.getLogger("quality_checkpoint_test")
QUALITY_KEYS = CHECKPOINT.QUALITY_INITIALIZATION_KEYS


class TinyQualityHead(nn.Module):
    def __init__(self, hidden=3):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)


class TinyRoIHead(nn.Module):
    def __init__(self, enabled=True, hidden=3):
        super().__init__()
        self.bbox_head = nn.Linear(2, 1)
        self.quality_enabled = enabled
        if enabled:
            self.quality_head = TinyQualityHead(hidden)

    def quality_initialization_keys(self):
        return tuple(sorted(QUALITY_KEYS)) if self.quality_enabled else ()


class TinyDetector(nn.Module):
    def __init__(self, enabled=True, hidden=3):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.roi_head = TinyRoIHead(enabled, hidden)
        self.register_buffer("running_stat", torch.zeros(2))
        self.register_buffer("num_batches_tracked", torch.tensor(0))
        self.loads = []

    def load_state_dict(self, state_dict, strict=True):
        self.loads.append(strict)
        return super().load_state_dict(state_dict, strict=strict)


def snapshot(model):
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def filled_state(enabled=False, value=7):
    state = snapshot(TinyDetector(enabled=enabled))
    for tensor in state.values():
        tensor.fill_(value)
    return state


def save_checkpoint(tmp_path, state, prefix="", raw=False):
    path = tmp_path / "detector.pth"
    stored = {prefix + key: tensor for key, tensor in state.items()}
    torch.save(stored if raw else {"state_dict": stored}, path)
    return str(path)


def assert_state_equal(model, state):
    assert set(model.state_dict()) == set(state)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, state[key].to(dtype=tensor.dtype)), key


@pytest.mark.parametrize("prefix,raw", [("", False), ("module.", False), ("", True)])
def test_old_checkpoint_initializes_only_new_head_identically(tmp_path, caplog, prefix, raw):
    teacher, student = TinyDetector(), TinyDetector()
    teacher_before = snapshot(teacher)
    old_state = filled_state()
    checkpoint = save_checkpoint(tmp_path, old_state, prefix=prefix, raw=raw)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        CHECKPOINT.load_branch_weights(
            checkpoint, [("teacher2", teacher), ("student2", student)], LOGGER
        )
    expected = dict(old_state)
    expected.update({key: teacher_before[key] for key in QUALITY_KEYS})
    assert_state_equal(teacher, expected)
    assert_state_equal(student, expected)
    assert teacher.loads == student.loads == [True]
    assert "initialized new quality-head tensors" in caplog.text
    for key in QUALITY_KEYS:
        assert key in caplog.text
    assert caplog.text.count("state tensors (strict)") == 2


def test_complete_quality_checkpoint_loads_all_tensors_strictly(tmp_path, caplog):
    teacher, student = TinyDetector(), TinyDetector()
    state = filled_state(enabled=True)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, state), [("teacher", teacher), ("student", student)], LOGGER
        )
    assert_state_equal(teacher, state)
    assert_state_equal(student, state)
    assert teacher.loads == student.loads == [True]
    assert "initialized new quality-head tensors" not in caplog.text


@pytest.mark.parametrize("count", [1, 2, 3])
def test_partial_quality_checkpoint_is_rejected_without_copy(tmp_path, count):
    state = filled_state()
    quality = filled_state(enabled=True)
    for key in sorted(QUALITY_KEYS)[:count]:
        state[key] = quality[key]
    teacher, student = TinyDetector(), TinyDetector()
    before = [snapshot(model) for model in (teacher, student)]
    with pytest.raises(RuntimeError, match="partial quality-head checkpoint"):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, state), [("teacher", teacher), ("student", student)], LOGGER
        )
    for model, original in zip((teacher, student), before):
        assert_state_equal(model, original)
        assert model.loads == []


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "nan", "inf", "not_tensor"])
def test_old_parameters_remain_mandatory_and_valid(tmp_path, problem):
    state = filled_state()
    if problem == "missing":
        del state["backbone.bias"]
    elif problem == "unexpected":
        state["surprise.weight"] = torch.zeros(1)
    elif problem == "shape":
        state["backbone.bias"] = torch.zeros(5)
    elif problem in ("nan", "inf"):
        state["backbone.bias"][0] = float(problem)
    else:
        state["backbone.bias"] = "not a tensor"
    teacher, student = TinyDetector(), TinyDetector()
    before = [snapshot(model) for model in (teacher, student)]
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, state), [("teacher", teacher), ("student", student)], LOGGER
        )
    for model, original in zip((teacher, student), before):
        assert_state_equal(model, original)
        assert model.loads == []


@pytest.mark.parametrize("destination", [0, 1])
def test_nonfinite_initialized_quality_tensors_rejected_before_copy(tmp_path, destination):
    models = [TinyDetector(), TinyDetector()]
    with torch.no_grad():
        models[destination].roi_head.quality_head.fc1.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="initialized .* contains NaN/Inf"):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, filled_state()), list(zip(("teacher", "student"), models)), LOGGER
        )
    assert all(model.loads == [] for model in models)


@pytest.mark.parametrize("problem", ["old_shape", "quality_shape", "extra_buffer"])
def test_second_destination_is_validated_before_first_is_changed(tmp_path, problem):
    teacher = TinyDetector()
    student = TinyDetector(hidden=5 if problem == "quality_shape" else 3)
    if problem == "old_shape":
        student.backbone = nn.Linear(2, 4)
    elif problem == "extra_buffer":
        student.register_buffer("extra_stat", torch.zeros(1))
    before = [snapshot(model) for model in (teacher, student)]
    with pytest.raises(RuntimeError):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, filled_state()), [("teacher", teacher), ("student", student)], LOGGER
        )
    for model, original in zip((teacher, student), before):
        assert_state_equal(model, original)
        assert model.loads == []


def test_disabled_quality_head_keeps_baseline_strictness(tmp_path):
    model = TinyDetector(enabled=False)
    state = filled_state()
    CHECKPOINT.load_branch_weights(save_checkpoint(tmp_path, state), [("baseline", model)], LOGGER)
    assert_state_equal(model, state)
    assert model.loads == [True]
    with pytest.raises(RuntimeError, match="unexpected"):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, filled_state(enabled=True)), [("baseline", model)], LOGGER
        )
    assert model.loads == [True]


@pytest.mark.parametrize("declaration", [
    ("backbone.bias",),
    ("roi_head.quality_head.fc1.weight",),
    tuple(sorted(QUALITY_KEYS)) + ("roi_head.quality_head.fc1.weight",),
    "roi_head.quality_head.fc1.weight",
    (42,),
])
def test_whitelist_cannot_be_extended_or_partially_enabled(tmp_path, declaration):
    model = TinyDetector()
    model.roi_head.quality_initialization_keys = lambda: declaration
    with pytest.raises((RuntimeError, TypeError)):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, filled_state()), [("teacher", model)], LOGGER
        )
    assert model.loads == []


def test_destination_dtype_overflow_is_rejected_before_copy(tmp_path):
    teacher, student = TinyDetector(), TinyDetector().half()
    state = filled_state()
    state["backbone.weight"].fill_(100000)
    with pytest.raises(ValueError, match="destination dtype"):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, state), [("teacher", teacher), ("student", student)], LOGGER
        )
    assert teacher.loads == student.loads == []


def test_mixed_enabled_and_disabled_destinations_are_rejected_before_copy(tmp_path):
    teacher, student = TinyDetector(), TinyDetector(enabled=False)
    with pytest.raises(RuntimeError, match="disagree"):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, filled_state()), [("teacher", teacher), ("student", student)], LOGGER
        )
    assert teacher.loads == student.loads == []


def test_missing_declared_parameters_are_rejected(tmp_path):
    model = TinyDetector()
    del model.roi_head.quality_head.fc2
    with pytest.raises(RuntimeError, match="declared quality-head tensors do not exist"):
        CHECKPOINT.load_branch_weights(
            save_checkpoint(tmp_path, filled_state()), [("teacher", model)], LOGGER
        )
    assert model.loads == []


def make_actual_dual_teacher(tmp_path, first_state, second_state):
    """Use the existing AST fixture, replacing only its detector builder."""
    baseline_spec = importlib.util.spec_from_file_location(
        "quality_dual_baseline_fixture", str(ROOT / "tests/test_dual_teacher_baseline.py")
    )
    baseline_fixture = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(baseline_fixture)
    namespace = baseline_fixture.actual_model_namespace()
    namespace["build_detector"] = lambda cfg: TinyDetector()
    first, second = tmp_path / "phase1.pth", tmp_path / "phase2.pth"
    torch.save({"state_dict": first_state}, first)
    torch.save({"state_dict": second_state}, second)
    config = baseline_fixture.TrainConfig(
        unsup_weight=2.0, load1_from=str(first), load2_from=str(second)
    )
    model = namespace["DualTeacher"]({}, config, dict(inference_on="teacher2"))
    # Make the accidental source of inequality explicit, not probabilistic.
    with torch.no_grad():
        for teacher, value in ((model.teacher1, 1.), (model.teacher2, 2.)):
            for parameter in teacher.roi_head.quality_head.parameters():
                parameter.fill_(value)
    return model


def test_random_quality_heads_cannot_hide_identical_phase_checkpoints(tmp_path):
    state = filled_state()
    model = make_actual_dual_teacher(tmp_path, state, state)
    with pytest.raises(RuntimeError, match="identical branches"):
        model.init_from_pretrained()
    assert not model._pretrained_initialized


def test_distinct_phase_checkpoints_pass_with_quality_heads(tmp_path):
    model = make_actual_dual_teacher(tmp_path, filled_state(value=7), filled_state(value=8))
    model.init_from_pretrained()
    assert model._pretrained_initialized
    assert_state_equal(model.teacher1, snapshot(model.student1))
    assert_state_equal(model.teacher2, snapshot(model.student2))
    assert not torch.equal(model.teacher1.backbone.weight, model.teacher2.backbone.weight)


def test_full_head_checkpoints_differing_only_in_quality_are_rejected(tmp_path):
    first, second = filled_state(enabled=True), filled_state(enabled=True)
    for key in QUALITY_KEYS:
        first[key].fill_(1.)
        second[key].fill_(2.)
    model = make_actual_dual_teacher(tmp_path, first, second)
    with pytest.raises(RuntimeError, match="identical branches"):
        model.init_from_pretrained()
    assert not model._pretrained_initialized


@pytest.mark.parametrize("momentum", [0.0, 0.75, 1.0])
def test_existing_ema_updates_every_new_quality_parameter(momentum):
    source_path = ROOT / "ssod/utils/hooks/mean_teacher.py"
    tree = ast.parse(source_path.read_text())
    hook_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MeanTeacher")
    hook_node.decorator_list = []
    module = ast.Module(body=[hook_node], type_ignores=[])
    namespace = {"Hook": object}
    exec(compile(module, str(source_path), "exec"), namespace)
    hook = namespace["MeanTeacher"]()
    models = {name: TinyDetector() for name in ("teacher1", "student1", "teacher2", "student2")}
    values = {"teacher1": 2., "student1": 6., "teacher2": 3., "student2": 11.}
    with torch.no_grad():
        for name, model in models.items():
            for parameter in model.parameters():
                parameter.fill_(values[name])
                if name.startswith("teacher"):
                    parameter.requires_grad_(False)
    for pair in (1, 2):
        assert list(dict(models["teacher{}".format(pair)].named_parameters())) == list(
            dict(models["student{}".format(pair)].named_parameters())
        )
    hook.momentum_update(SimpleNamespace(**models), momentum)
    for pair in (1, 2):
        teacher_name, student_name = "teacher{}".format(pair), "student{}".format(pair)
        expected = momentum * values[teacher_name] + (1. - momentum) * values[student_name]
        parameters = dict(models[teacher_name].named_parameters())
        assert QUALITY_KEYS.issubset(parameters)
        for key, parameter in parameters.items():
            assert torch.equal(parameter, torch.full_like(parameter, expected)), key
            assert not parameter.requires_grad
