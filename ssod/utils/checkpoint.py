"""Explicit, verified initialization for fresh DualTeacher training runs."""

from collections import OrderedDict
from collections.abc import Mapping

import torch


def load_branch_weights(filename, branches, logger):
    """Load one trusted, local detector checkpoint into a teacher/student pair.

    Validate all keys and shapes before copying. Unlike MMCV's non-strict
    checkpoint loading, a partial detector checkpoint must stop this run.
    Checkpoints are read on CPU and are not retained on the model.
    """
    if not filename:
        raise ValueError("DualTeacher requires both load1_from and load2_from")
    checkpoint = torch.load(filename, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Not a detector checkpoint: {}".format(filename))
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Empty or invalid state_dict: {}".format(filename))
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Non-string checkpoint keys: {}".format(filename))
    if all(key.startswith("module.") for key in state_dict):
        state_dict = OrderedDict(
            (key[len("module."):], value) for key, value in state_dict.items()
        )

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError("{}: {} is not a tensor".format(filename, key))
        if not torch.isfinite(value).all():
            raise ValueError("{}: {} contains NaN/Inf".format(filename, key))

    for name, model in branches:
        expected = model.state_dict()
        missing = sorted(set(expected) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected))
        mismatched = [
            key for key in expected.keys() & state_dict.keys()
            if expected[key].shape != state_dict[key].shape
        ]
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "{} -> {}: incompatible detector checkpoint; missing={}, "
                "unexpected={}, shape_mismatch={}".format(
                    filename, name, missing[:10], unexpected[:10], mismatched[:10]
                )
            )

    for name, model in branches:
        model.load_state_dict(state_dict, strict=True)
        for key, actual in model.state_dict().items():
            # A saved FP16 checkpoint may be restored into an FP32 model.
            expected = state_dict[key].to(dtype=actual.dtype)
            if not torch.equal(actual.detach().cpu(), expected):
                raise RuntimeError("{}: loaded tensor differs: {}".format(name, key))
        logger.info(
            "[DualTeacher init] %s -> %s: verified %d state tensors (strict)",
            filename, name, len(state_dict),
        )


def load_or_initialize_model(runner, resume_from=None, load_from=None):
    """Full-run checkpoints take precedence over fresh branch initialization.

    Called after generic init_weights and before runner.run / EMA hooks.
    Inference does not call this function and needs no Phase 1/2 files.
    """
    if resume_from:
        runner.resume(resume_from)
        return "resume"
    if load_from:
        runner.load_checkpoint(load_from)
        return "load"
    model = getattr(runner.model, "module", runner.model)
    if hasattr(model, "init_from_pretrained"):
        model.init_from_pretrained()
        return "pretrained"
    return "none"
