"""CPU regression tests; no MMDetection/CUDA operators or real checkpoints.

The actual DualTeacher class/functions are compiled from their source AST;
only the external registry, detector builder and multi-stream base are replaced
with small torch.nn.Module fixtures. Checkpoint IO, tensor copies, NMS and the
new initialization/restore routing all execute their real implementations.
Run the separate tools/check_dual_teacher_init.py on the training machine for
the full MMDetection model and real Phase 1/2 files.
"""

import ast
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("dual_teacher_test")


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint_utils = load_file("checkpoint_utils", "ssod/utils/checkpoint.py")
nms_module = load_file("baseline_nms", "ssod/utils/ensemble_boxes/ensemble_boxes_nms.py")


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        self.register_buffer("running_stat", torch.zeros(2))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def init_weights(self):
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()


class MultiStreamFixture(nn.Module):
    def __init__(self, models, train_cfg=None, test_cfg=None):
        super().__init__()
        self.submodules = list(models)
        for name, model in models.items():
            setattr(self, name, model)
        self.train_cfg, self.test_cfg = train_cfg, test_cfg

    def freeze(self, name):
        model = getattr(self, name)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def init_weights(self):
        for name in self.submodules:
            getattr(self, name).init_weights()


class TrainConfig(dict):
    __getattr__ = dict.__getitem__


def actual_model_namespace():
    path = ROOT / "ssod/models/dual_teacher.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    namespace = dict(
        __name__="dual_teacher_under_test",
        torch=torch, np=np,
        DETECTORS=SimpleNamespace(register_module=lambda: lambda cls: cls),
        build_detector=lambda cfg: TinyDetector(),
        MultiSteamDetector=MultiStreamFixture,
        force_fp32=lambda **kwargs: lambda func: func,
        get_root_logger=lambda: LOGGER,
        load_branch_weights=checkpoint_utils.load_branch_weights,
        nms=nms_module.nms,
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


NAMESPACE = actual_model_namespace()
DualTeacher = NAMESPACE["DualTeacher"]
fuse = NAMESPACE["fuse_teacher_proposals"]


def save_detector(path, value, prefix="", raw=False):
    state = TinyDetector().state_dict()
    for tensor in state.values():
        tensor.fill_(value)
    stored = {prefix + key: tensor for key, tensor in state.items()}
    torch.save(stored if raw else {"state_dict": stored}, path)
    return state


@pytest.fixture
def phase_files(tmp_path):
    first, second = tmp_path / "phase1.pth", tmp_path / "phase2.pth"
    save_detector(first, 1)
    save_detector(second, 2)
    return first, second


def make_model(files):
    return DualTeacher({}, TrainConfig(
        unsup_weight=2.0, load1_from=str(files[0]), load2_from=str(files[1])
    ), dict(inference_on="teacher2"))


def assert_value(model, value):
    for tensor in model.state_dict().values():
        assert torch.equal(tensor, torch.full_like(tensor, value))


def test_fresh_training_loads_both_pairs_after_generic_init(phase_files, caplog):
    model = make_model(phase_files)
    model.init_weights()
    assert_value(model.student1, 0)
    runner = SimpleNamespace(model=SimpleNamespace(module=model))
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        assert checkpoint_utils.load_or_initialize_model(runner) == "pretrained"
    assert_value(model.teacher1, 1)
    assert_value(model.student1, 1)
    assert_value(model.teacher2, 2)
    assert_value(model.student2, 2)
    assert all(not p.requires_grad for p in model.teacher1.parameters())
    assert all(p.requires_grad for p in model.student1.parameters())
    assert caplog.text.count("state tensors (strict)") == 4
    assert "T1=S1, T2=S2, T1!=T2" in caplog.text
    with torch.no_grad():
        model.student1.head.weight.fill_(7)
    model.init_from_pretrained()  # a second call cannot reset a trained model
    assert torch.all(model.student1.head.weight == 7)


@pytest.mark.parametrize("mode", ["resume", "load"])
def test_full_checkpoint_skips_phase_files(tmp_path, phase_files, mode):
    trained = make_model(phase_files)
    trained.init_from_pretrained()
    state = trained.state_dict()
    state["teacher2.head.bias"].fill_(9)
    # No original pretraining files are available for the restored model.
    restored = make_model((tmp_path / "missing1.pth", tmp_path / "missing2.pth"))
    runner = SimpleNamespace(model=restored)
    runner.resume = Mock(side_effect=lambda path: restored.load_state_dict(state))
    runner.load_checkpoint = Mock(side_effect=lambda path: restored.load_state_dict(state))
    kwargs = dict(resume_from="full.pth", load_from="ignored.pth") if mode == "resume" else dict(load_from="full.pth")
    assert checkpoint_utils.load_or_initialize_model(runner, **kwargs) == mode
    getattr(runner, "resume" if mode == "resume" else "load_checkpoint").assert_called_once_with("full.pth")
    getattr(runner, "load_checkpoint" if mode == "resume" else "resume").assert_not_called()
    assert torch.all(restored.teacher2.head.bias == 9)


def test_inference_needs_no_training_config_or_phase_files(phase_files):
    trained = make_model(phase_files)
    trained.init_from_pretrained()
    inference = DualTeacher({}, train_cfg=None, test_cfg=dict(inference_on="teacher2"))
    inference.load_state_dict(trained.state_dict(), strict=True)
    assert_value(inference.teacher2, 2)


def test_single_detector_is_not_a_full_dual_checkpoint():
    model = DualTeacher({}, train_cfg=None, test_cfg={})
    with pytest.raises(RuntimeError, match="full four-branch checkpoint"):
        model.load_state_dict(TinyDetector().state_dict(), strict=False)


def test_identical_pretraining_branches_stop(phase_files):
    model = make_model((phase_files[0], phase_files[0]))
    with pytest.raises(RuntimeError, match="identical branches"):
        model.init_from_pretrained()


@pytest.mark.parametrize("raw,prefix", [(False, ""), (False, "module."), (True, "")])
def test_supported_checkpoint_formats(tmp_path, raw, prefix):
    path = tmp_path / "phase.pth"
    save_detector(path, 3, raw=raw, prefix=prefix)
    teacher, student = TinyDetector(), TinyDetector()
    checkpoint_utils.load_branch_weights(path, [("teacher", teacher), ("student", student)], LOGGER)
    assert_value(teacher, 3)
    assert_value(student, 3)


@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape", "nan", "inf"])
def test_invalid_checkpoint_fails_before_copying(tmp_path, fault):
    state = TinyDetector().state_dict()
    if fault == "missing":
        state.pop("head.bias")
    elif fault == "unexpected":
        state["wrong.key"] = torch.ones(1)
    elif fault == "shape":
        state["head.bias"] = torch.ones(20)
    else:
        state["head.bias"].fill_(float(fault))
    path = tmp_path / "bad.pth"
    torch.save({"state_dict": state}, path)
    model = TinyDetector()
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    with pytest.raises((RuntimeError, ValueError)):
        checkpoint_utils.load_branch_weights(path, [("teacher", model)], LOGGER)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key])


def test_missing_file_stops_fresh_training(tmp_path):
    model = make_model((tmp_path / "missing1", tmp_path / "missing2"))
    with pytest.raises(FileNotFoundError):
        checkpoint_utils.load_or_initialize_model(SimpleNamespace(model=model))


def test_non_dual_model_keeps_normal_initialization():
    assert checkpoint_utils.load_or_initialize_model(SimpleNamespace(model=TinyDetector())) == "none"


def test_training_entry_calls_initializer_before_runner_run():
    tree = ast.parse((ROOT / "ssod/apis/train.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_detector")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    init = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "load_or_initialize_model")
    run = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "run")
    assert init.lineno < run.lineno


def proposals(rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)


def test_nms_does_not_boost_consensus_scores():
    first = proposals([[0, 0, 10, 10, 0.7]])
    second = proposals([[1, 1, 11, 11, 0.7]])
    boxes, labels = fuse([first], [torch.tensor([0])], [second], [torch.tensor([0])])
    assert boxes[0].shape == (1, 5)
    assert boxes[0][0, 4].item() == pytest.approx(0.7)
    assert not (boxes[0][:, 4] > 0.9).any()
    assert labels[0].dtype == torch.long


def test_nms_keeps_original_higher_score_box():
    first = proposals([[0, 0, 10, 10, 0.8]])
    second = proposals([[9, 9, 19, 19, 0.7]])  # low overlap, but author IoU threshold is zero
    boxes, _ = fuse([first], [torch.tensor([0])], [second], [torch.tensor([0])])
    assert torch.equal(boxes[0], first)


def test_nms_is_class_aware():
    first = proposals([[0, 0, 10, 10, 0.8]])
    second = proposals([[0, 0, 10, 10, 0.7]])
    boxes, labels = fuse([first], [torch.tensor([0])], [second], [torch.tensor([1])])
    assert boxes[0].shape == (2, 5)
    assert set(labels[0].tolist()) == {0, 1}


@pytest.mark.parametrize("empty_side", ["first", "second", "both"])
def test_nms_empty_side_passthrough(empty_side):
    empty, empty_labels = proposals([]), torch.empty(0, dtype=torch.long)
    populated = proposals([[0, 0, 10, 10, 0.8]])
    populated_labels = torch.tensor([0])
    first = empty if empty_side in ("first", "both") else populated
    second = empty if empty_side in ("second", "both") else populated
    first_labels = empty_labels if len(first) == 0 else populated_labels
    second_labels = empty_labels if len(second) == 0 else populated_labels
    boxes, labels = fuse([first], [first_labels], [second], [second_labels])
    assert torch.equal(boxes[0], empty if empty_side == "both" else populated)
    assert labels[0].shape == (len(boxes[0]),)


def test_nms_keeps_images_separate():
    first = proposals([[0, 0, 10, 10, 0.8]])
    second = proposals([[1, 1, 11, 11, 0.7]])
    empty_labels = torch.empty(0, dtype=torch.long)
    boxes, _ = fuse([first, second], [torch.tensor([0]), torch.tensor([0])],
                    [proposals([]), proposals([])], [empty_labels, empty_labels])
    assert torch.equal(boxes[0], first)
    assert torch.equal(boxes[1], second)


def test_nms_rejects_different_batch_sizes():
    with pytest.raises(ValueError, match="batch sizes"):
        fuse([], [], [proposals([])], [torch.empty(0, dtype=torch.long)])


def test_reproduce_config_cannot_auto_resume_old_run():
    namespace = {}
    exec(compile((ROOT / "configs/reproduce/phase3_dual_teacher_ssdd.py").read_text(), "config", "exec"), namespace)
    assert namespace["auto_resume"] is False
    assert namespace["load_from"] is None
    assert namespace["resume_from"] is None
    assert "phase3_dual_teacher_baseline_nms/" in namespace["work_dir"]
    assert not any("consensus" in key for key in namespace["semi_wrapper"]["train_cfg"])
