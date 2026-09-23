"""PG schedule, actual forward routing, resume guards and isolated suite tests."""
import ast
import copy
import logging
from types import SimpleNamespace

import pytest
import torch

from test_dual_teacher_baseline import ROOT, load_file, actual_model_namespace, TrainConfig
from routing_fixture import load_forward_class

PG = load_file("pg_under_test", "ssod/models/progressive_gamma.py").ProgressiveGamma
suite_tools = load_file("ablation_under_test", "ssod/utils/ablation.py")
checker = load_file("checker_under_test", "tools/check_ablation_step.py")


@pytest.mark.parametrize("mode", ["sup2", "both"])
def test_schedule_endpoints_idempotence_and_roundtrip(mode):
    pg = PG(mode, 32000)
    with pytest.raises(RuntimeError):
        pg.weights()
    for iteration, expected in ((0, .1), (16000, .2 * (.5 + .5 * 16000 / 31999)), (31999, .2)):
        pg.set_iteration(iteration)
        a = pg.weights()
        assert a == pg.weights()
        assert a[0] == pytest.approx(expected)
        assert a[1] == pytest.approx(expected if mode == "both" else .2)
    pg.half()
    assert pg.settings.dtype == torch.float64 and pg.last_iter.dtype == torch.long
    restored = PG(mode, 32000)
    restored.load_state_dict(pg.state_dict())
    assert restored.weights() == pg.weights()
    with pytest.raises(ValueError):
        pg.set_iteration(32000)


@pytest.mark.parametrize("problem", ["missing", "mode", "duration", "ratio", "iteration"])
def test_pg_restore_fails_even_with_non_strict_loading(problem):
    pg = PG("sup2", 10)
    state = copy.deepcopy(pg.state_dict())
    if problem == "missing":
        del state["last_iter"]
    elif problem == "iteration":
        state["last_iter"].fill_(10)
    else:
        replacement = dict(mode="sup2", total_iters=10)
        replacement.update({"mode": "both"} if problem == "mode" else
                           {"total_iters": 11} if problem == "duration" else {"start_ratio": .25})
        state = PG(**replacement).state_dict()
    with pytest.raises(RuntimeError):
        pg.load_state_dict(state, strict=False)


def hook_class():
    tree = ast.parse((ROOT / "ssod/utils/hooks/progressive_gamma.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    tree.body[0].decorator_list = []
    ns = dict(Hook=object, is_module_wrapper=lambda model: hasattr(model, "module"))
    exec(compile(tree, "pg_hook", "exec"), ns)
    return ns["ProgressiveGammaHook"]


def test_hook_uses_runner_iteration_and_checks_resume_position():
    model = SimpleNamespace(pg=PG("both", 10), unsup_weight=2.)
    runner = SimpleNamespace(model=SimpleNamespace(module=model), iter=0, max_iters=10,
                             logger=logging.getLogger(), log_buffer=SimpleNamespace(output={}))
    hook = hook_class()()
    hook.before_run(runner)
    hook.before_train_iter(runner)
    assert runner.log_buffer.output["pg_sup2_weight"] == .1
    assert runner.log_buffer.output["pg_unsup2_weight"] == .2
    model.pg.set_iteration(4)  # checkpoint after five completed iterations
    runner.iter = 5
    hook.before_run(runner)
    hook.before_train_iter(runner)
    assert int(model.pg.last_iter) == 5
    runner.iter = 0
    with pytest.raises(RuntimeError, match="iteration mismatch"):
        hook.before_run(runner)
    runner.max_iters = 11
    with pytest.raises(RuntimeError, match="total_iters"):
        hook.before_run(runner)


@pytest.mark.parametrize("mode,expected", [("sup2", (.2, .8)), ("both", (.2, .4))])
def test_actual_forward_routes_pg_to_requested_sar_branches(mode, expected):
    model = load_forward_class()()
    model.unsup_weight, model.sup2_weight = 2., .2
    model.pg = PG(mode, 10)
    model.pg.set_iteration(0)
    model.student1 = model.student2 = SimpleNamespace(forward_train=lambda **kw: {"loss_cls": torch.tensor(2.)})
    model.extract_teacher_info = lambda *args: ({}, {})
    model.foward_unsup1_train = model.foward_unsup2_train = lambda *args: {"loss_cls": torch.tensor(2.)}
    tags = ["sup1", "sup2", "unsup_teacher", "unsup_student"]
    losses = model.forward_train(
        torch.zeros(4, 3, 2, 2), [dict(tag=t, filename="image") for t in tags],
        gt_bboxes=[[], [], [], []])
    assert losses["sup1_loss_cls"] == 2.
    assert losses["unsup1_loss_cls"] == 4.
    assert float(losses["sup2_loss_cls"]) == pytest.approx(expected[0])
    assert float(losses["unsup2_loss_cls"]) == pytest.approx(expected[1])


def test_actual_full_checkpoint_pg_inference_and_training_guards():
    ns = actual_model_namespace()
    ns["ProgressiveGamma"] = PG
    cls = ns["DualTeacher"]
    cfg = TrainConfig(unsup_weight=2., load1_from="p1", load2_from="p2",
                      pg=TrainConfig(mode="both", total_iters=10))
    model = cls({}, cfg, dict(inference_on="teacher2"))
    model.pg.set_iteration(4)
    state = copy.deepcopy(model.state_dict())
    restored = cls({}, cfg, dict(inference_on="teacher2"))
    restored.load_state_dict(state, strict=True)
    assert int(restored.pg.last_iter) == 4
    inference = cls({}, None, dict(inference_on="teacher2"))
    inference.load_state_dict(state, strict=True)
    assert "pg.settings" in state  # caller's checkpoint is not modified
    baseline_cfg = TrainConfig(unsup_weight=2., load1_from="p1", load2_from="p2")
    with pytest.raises(RuntimeError, match="PG checkpoint"):
        cls({}, baseline_cfg, {}).load_state_dict(state, strict=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        restored.load_state_dict(inference.state_dict(), strict=False)


def baseline():
    return dict(
        model=dict(type="DualTeacher", model=dict(roi_head=dict(
            type="StandardRoIHead", bbox_head=dict(num_classes=1))),
                   train_cfg=dict(load1_from="p1", load2_from="p2", unsup_weight=2)),
        runner=dict(type="IterBasedRunner", max_iters=32000),
        custom_hooks=[dict(type="MeanTeacher", momentum=.999)],
        optimizer=dict(type="SGD", lr=.0025), data=dict(train=dict(ann_file="correct.json")),
        evaluation=dict(interval=2000), seed=123, work_dir="original",
        load_from=None, resume_from=None)


def test_suite_preserves_b0_protocol_and_isolates_features(tmp_path):
    cfg = baseline()
    before = copy.deepcopy(cfg)
    suite = suite_tools.make_suite(cfg, tmp_path / "new", 123)
    assert cfg == before
    for name, item in suite.items():
        for key in ("data", "optimizer", "runner", "evaluation", "custom_hooks", "seed"):
            assert item[key] == cfg[key]
        tc, head = item["model"]["train_cfg"], item["model"]["model"]["roi_head"]
        assert tc["m2_enabled"] == (name == "m2")
        assert (head["type"] == "ForegroundRoIHead") == (name == "fg")
        assert ("pg" in tc) == name.startswith("pg_")
        assert item["auto_resume"] is False and item["load_from"] is None
        assert item["work_dir"] != cfg["work_dir"]
    suite["m2"]["data"]["train"]["ann_file"] = "mutated"
    assert suite["fg"]["data"] == cfg["data"]


@pytest.mark.parametrize("problem", ["m2", "fg", "pg", "seed", "hook"])
def test_suite_rejects_contaminated_baseline(tmp_path, problem):
    cfg = baseline()
    if problem == "m2":
        cfg["model"]["train_cfg"]["m2_enabled"] = True
    elif problem == "fg":
        cfg["model"]["model"]["roi_head"]["type"] = "ForegroundRoIHead"
    elif problem == "pg":
        cfg["model"]["train_cfg"]["pg"] = {"mode": "both"}
    elif problem == "seed":
        cfg["seed"] = 456
    else:
        cfg["custom_hooks"].append(dict(type="Weighter"))
    with pytest.raises(ValueError):
        suite_tools.make_suite(cfg, tmp_path, 123)


def test_acceptance_detects_wrong_routing_and_uncovered_m2():
    off = {k: torch.tensor(2.) for k in (
        "sup1_loss_cls:0", "sup2_loss_cls:0", "unsup1_loss_cls:0", "unsup2_loss_cls:0")}
    on = {k: v * (.5 if k.startswith("sup2_") else 1) for k, v in off.items()}
    checker.compare_losses("pg_sup2", dict(off=off, on=on, identity=off))
    with pytest.raises(AssertionError):
        checker.compare_losses("pg_both", dict(off=off, on=on, identity=off))
    with pytest.raises(checker.IncompleteCheck):
        checker.compare_losses("m2", dict(off=off, on=off, identity=off))
