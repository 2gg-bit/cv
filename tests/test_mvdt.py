"""CPU tests for exact MVDT math, checkpoint continuity and real collectives."""

import copy
import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mvdt_under_test", ROOT / "ssod/models/mvdt.py")
mvdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mvdt)


def controller(**kwargs):
    options = dict(warmup_iters=2, update_interval=2, min_samples=4,
                   min_group_size=2, max_scores=64)
    options.update(kwargs)
    return mvdt.MVDTThreshold(**options)


def brute_force(scores):
    values = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    choices = []
    for m in range(2, len(values) - 1):
        if values[m-1] == values[m]:
            continue
        a, b = values[:m], values[m:]
        cost = (((a-a.mean())**2).sum() + ((b-b.mean())**2).sum()) / len(values)
        choices.append((cost, m))
    if not choices:
        return None
    return float(values[min(choices)[1] - 1])


@pytest.mark.parametrize("seed", range(12))
def test_exact_objective_matches_brute_force(seed):
    scores = np.random.RandomState(seed).uniform(.5, 1, 50)
    assert mvdt.minimum_variance_threshold(scores, 4) == brute_force(scores)


def test_tied_confidences_never_split_and_invalid_values_are_ignored():
    scores = [.55, .55, .6, .6, .8, .8, .95, .95]
    expected = brute_force(scores)
    assert mvdt.minimum_variance_threshold(scores + [np.nan, np.inf, -.1, 1.1], 4) == expected
    assert mvdt.minimum_variance_threshold([.7] * 10, 4) is None
    assert mvdt.minimum_variance_threshold([.6, .8, .9], 4) is None


def test_nearly_equal_scores_still_match_direct_centered_objective():
    scores = np.array([.95, .95, .95, .95]) + np.array([0, 1, 5, 6]) * 1e-8
    assert mvdt.minimum_variance_threshold(scores, 4) == brute_force(scores)


def test_warmup_boundary_and_update_use_all_candidates_once():
    model = controller()
    assert model.eligible(torch.tensor([.89, .9, .91])).tolist() == [False, False, True]
    assert model.observe(torch.tensor([.55, .56])) is None
    assert model.value == .9
    report = model.observe(torch.tensor([.8, .81]))
    assert report["step"] == 2 and report["samples"] == 4 and report["updated"]
    assert model.value == pytest.approx(.8)
    assert model.eligible(torch.tensor([.56, .8, .81])).tolist() == [False, True, True]
    assert int(model.count) == 0 and int(model.steps) == 2
    assert not bool(model.scores.bool().any())


def test_empty_sparse_equal_score_windows_keep_previous_threshold():
    model = controller()
    for scores in ([.55, .56], [.8, .81]):
        model.observe(torch.tensor(scores))
    previous = model.value
    for batch in ([], [], [.99, .99], [.99, .99]):
        model.observe(torch.tensor(batch))
    assert model.value == previous and int(model.updates) == 1


def test_resume_restores_partial_window_and_next_update(tmp_path):
    model = controller()
    model.observe(torch.tensor([.55, .56]))
    checkpoint = tmp_path / "partial.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = controller()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    for batch in ([.8, .81], [.6], [.61, .85, .86], [], []):
        assert model.observe(torch.tensor(batch)) == restored.observe(torch.tensor(batch))
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key]), key


def test_missing_mismatched_or_corrupted_state_fails_even_non_strict():
    good = controller().state_dict()
    for damage in ("missing", "settings", "nan", "count", "bank"):
        broken = copy.deepcopy(good)
        if damage == "missing":
            broken.pop("steps")
        elif damage == "settings":
            broken["settings"][3] += 1
        elif damage == "nan":
            broken["threshold"].fill_(float("nan"))
        elif damage == "count":
            broken["count"].fill_(65)
        else:
            broken["count"].fill_(1)
            broken["scores"][0] = float("nan")
        with pytest.raises(RuntimeError, match="MVDT"):
            controller().load_state_dict(broken, strict=False)


def test_half_conversion_keeps_exact_statistics_and_no_learnable_parameters():
    model = controller()
    model.observe(torch.tensor([.5554321, .5678912]))
    before = copy.deepcopy(model.state_dict())
    model.half().float().to("cpu")
    assert not list(model.parameters())
    for key, value in before.items():
        assert value.dtype == model.state_dict()[key].dtype
        assert torch.equal(value, model.state_dict()[key])


def test_eval_does_not_change_history_and_capacity_is_not_silent_sampling():
    model = controller(max_scores=4)
    model.eval()
    assert model.observe(torch.ones(10)) is None
    assert int(model.steps) == 0
    model.train()
    with pytest.raises(RuntimeError, match="exceeds max_scores"):
        model.observe(torch.ones(5))
    assert int(model.steps) == 0 and int(model.count) == 0


@pytest.mark.parametrize("kwargs", [dict(warmup_iters=0), dict(update_interval=0),
                                     dict(min_group_size=1), dict(min_samples=3),
                                     dict(score_floor=.95), dict(max_scores=3),
                                     dict(update_interval=1.5)])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        controller(**kwargs)


def distributed_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        model = controller()
        model.observe(torch.tensor([.55, .56] if rank == 0 else [.8, .81]))
        partial = model.scores[:int(model.count)].tolist()
        # All ranks can resume the same rank-zero style checkpoint at mid-window.
        restored = controller()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        result = restored.observe(torch.tensor([] if rank == 0 else [.82, .83]))
        Path(output, "rank{}.json".format(rank)).write_text(json.dumps(
            dict(partial=partial, result=result, threshold=restored.value)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_real_ranks_share_uneven_empty_batches_and_resume(tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(distributed_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    first = json.loads((tmp_path / "rank0.json").read_text())
    second = json.loads((tmp_path / "rank1.json").read_text())
    assert first == second
    assert first["partial"] == pytest.approx([.55, .56, .8, .81])
    assert first["result"]["samples"] == 6
    assert first["threshold"] == pytest.approx(.8)
