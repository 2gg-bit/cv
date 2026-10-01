from types import SimpleNamespace

import pytest
import torch
from roi_fixture import load_heads, ParentBBox, sample

module = load_heads("ssod/models/roi_heads/small_bkg_reweight.py")
Head = module.SmallBkgReweightBBoxHead


def targets(head, tag="sup2"):
    s = sample([[0, 0, 4, 4]], [[0, 0, 63, 32], [0, 0, 64, 32], [0, 0, 80, 40]])
    ctx = dict(tags=[tag], scale_factors=[[2., 1., 2., 1.]])
    return head.get_targets([s], [], [], SimpleNamespace(pos_weight=-1), reweight_ctx=ctx)


def test_area_boundary_branch_gate_and_no_training_diagnostics():
    head = Head(reweight=dict(enable=True, lambda_=1., max_area=1024., tag="sup2"))
    # A training forward must not perform any diagnostic tensor-to-CPU conversion.
    head._as_list = lambda value: pytest.fail("diagnostic copy during normal training")
    out = targets(head)
    assert out[1].tolist() == [1, 2, 1, 1]  # original areas 1008, 1024, 1600
    for tag in ("sup1", "unsup_student", "unsup_teacher"):
        assert targets(head, tag)[1].tolist() == [1, 1, 1, 1]
    assert head.reweight_log == []


def test_diagnostic_toggle_is_bounded_and_preserves_targets_loss_gradient_rng():
    head = Head(reweight=dict(enable=True))
    before = torch.get_rng_state().clone()
    a = targets(head)
    head.enable_reweight_diagnostics(2)
    for _ in range(20):
        b = targets(head)
    assert len(head.reweight_log) == 2
    assert head.reweight_log[-1]["n_reweighted"] == 1
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(before, torch.get_rng_state())
    logits = torch.tensor([[.1, -.1]] * 4, requires_grad=True)
    losses, grads = [], []
    for labels, weights, _, _ in (a, b):
        loss = (torch.nn.functional.cross_entropy(logits, labels, reduction="none") * weights).sum() / (weights > 0).sum()
        losses.append(loss)
        grads.append(torch.autograd.grad(loss, logits)[0])
    assert torch.equal(losses[0], losses[1]) and torch.equal(grads[0], grads[1])
    head.enable_reweight_diagnostics(0)
    targets(head)
    assert head.reweight_log == []


@pytest.mark.parametrize("enabled,weight", [(False, 1), (True, 0)])
def test_identity_and_checkpoint_keys(enabled, weight):
    head = Head(reweight=dict(enable=enabled, lambda_=weight))
    parent = ParentBBox()
    head.load_state_dict(parent.state_dict(), strict=True)
    assert set(head.state_dict()) == set(parent.state_dict())
    s = sample([[0, 0, 4, 4]], [[0, 0, 5, 5]])
    cfg = SimpleNamespace(pos_weight=-1)
    a = parent.get_targets([s], [], [], cfg)
    b = head.get_targets([s], [], [], cfg, reweight_ctx=dict(tags=["sup2"], scale_factors=[[1]*4]))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    # Pseudo-label classification calls get_targets without context.
    c = head.get_targets([s], [], [], cfg)
    assert all(torch.equal(x, y) for x, y in zip(a, c))
