"""Explicit, verified initialization for fresh DualTeacher training runs."""

from collections import OrderedDict
from collections.abc import Mapping

import torch


# Only this versioned, opt-in extension may be initialized from fresh weights.
# In particular, this is not a suffix match or a blanket non-strict load.
QUALITY_INITIALIZATION_KEYS = frozenset((
    "roi_head.quality_head.fc1.weight",
    "roi_head.quality_head.fc1.bias",
    "roi_head.quality_head.fc2.weight",
    "roi_head.quality_head.fc2.bias",
))


def _quality_initialization_keys(model):
    roi_head = getattr(model, "roi_head", None)
    declare_keys = getattr(roi_head, "quality_initialization_keys", None)
    if declare_keys is None:
        return frozenset()
    if not callable(declare_keys):
        raise TypeError("quality_initialization_keys must be callable")
    keys = declare_keys()
    if not isinstance(keys, (tuple, list, set, frozenset)):
        raise TypeError("quality_initialization_keys must return a key collection")
    if not all(isinstance(key, str) for key in keys):
        raise TypeError("quality_initialization_keys must contain string keys")
    declared = frozenset(keys)
    if len(declared) != len(keys) or (declared and declared != QUALITY_INITIALIZATION_KEYS):
        raise RuntimeError("Invalid quality-head initialization whitelist")
    return declared


def load_branch_weights(filename, branches, logger):
    """Load one trusted, local detector checkpoint into a teacher/student pair.

    Validate every destination before copying. A quality-aware detector may
    explicitly opt in to initializing the four new quality-head tensors when
    loading an old Phase 1/2 checkpoint. All four must be absent together; old
    detector tensors remain mandatory. The same initialized head is copied to
    both members of a pair and the merged state is loaded strictly. This helper
    is not used to restore full four-branch training/inference checkpoints.
    Checkpoints are read on CPU and are not retained on the model.
    """
    if not filename:
        raise ValueError("DualTeacher requires both load1_from and load2_from")
    branches = list(branches)
    if not branches:
        raise ValueError("At least one destination branch is required")
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

    destinations = []
    for name, model in branches:
        expected = model.state_dict()
        allowed = _quality_initialization_keys(model)
        if not allowed.issubset(expected):
            raise RuntimeError("{}: declared quality-head tensors do not exist".format(name))
        present_quality = allowed & set(state_dict)
        if present_quality and present_quality != allowed:
            raise RuntimeError(
                "{} -> {}: partial quality-head checkpoint; all four new "
                "tensors must be present or absent together".format(filename, name)
            )
        initialize = allowed if not present_quality else frozenset()
        missing = sorted(set(expected) - set(state_dict) - initialize)
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
        for key in initialize:
            if not torch.isfinite(expected[key]).all():
                raise ValueError("{}: initialized {} contains NaN/Inf".format(name, key))
        destinations.append((name, model, expected, initialize))

    initialize = destinations[0][3]
    if any(item[3] != initialize for item in destinations):
        raise RuntimeError("Destination branches disagree on quality-head initialization")
    merged = OrderedDict(state_dict)
    if initialize:
        first_state = destinations[0][2]
        for key in sorted(initialize):
            merged[key] = first_state[key].detach().cpu().clone()

    # New tensors must also match every destination; e.g. teacher and student
    # may not silently use different quality hidden dimensions. Check dtype
    # conversions before changing any branch, including FP32 -> FP16 overflow.
    for name, model, expected, _ in destinations:
        for key, value in merged.items():
            if value.shape != expected[key].shape:
                raise RuntimeError("{}: quality-head shape_mismatch: {}".format(name, key))
            if not torch.isfinite(value.to(dtype=expected[key].dtype)).all():
                raise ValueError("{}: {} becomes NaN/Inf in destination dtype".format(name, key))

    for name, model, _, _ in destinations:
        model.load_state_dict(merged, strict=True)
        for key, actual in model.state_dict().items():
            # A saved FP16 checkpoint may be restored into an FP32 model.
            expected = merged[key].to(dtype=actual.dtype)
            if not torch.equal(actual.detach().cpu(), expected):
                raise RuntimeError("{}: loaded tensor differs: {}".format(name, key))
        logger.info(
            "[DualTeacher init] %s -> %s: verified %d state tensors (strict)",
            filename, name, len(merged),
        )
    if initialize:
        logger.info(
            "[DualTeacher init] initialized new quality-head tensors from %s "
            "and copied identically to %s: %s",
            destinations[0][0], ", ".join(item[0] for item in destinations),
            ", ".join(sorted(initialize)),
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
