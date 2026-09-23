"""Runner-driven, checkpointed progressive SAR loss coefficients."""

import math

import torch
from torch import nn


class ProgressiveGamma(nn.Module):
    def __init__(self, mode, total_iters, base_gamma=0.2, unsup_weight=2.0,
                 start_ratio=0.5, end_ratio=1.0):
        super().__init__()
        if mode not in ("sup2", "both"):
            raise ValueError("PG mode must be sup2 or both")
        if isinstance(total_iters, bool) or not isinstance(total_iters, int) or total_iters < 2:
            raise ValueError("PG total_iters must be an integer >= 2")
        values = (base_gamma, unsup_weight, start_ratio, end_ratio)
        if any(not math.isfinite(float(x)) or x <= 0 for x in values):
            raise ValueError("PG coefficients must be finite and positive")
        self.mode, self.total_iters = mode, total_iters
        self.base_gamma, self.unsup_weight = float(base_gamma), float(unsup_weight)
        self.start_ratio, self.end_ratio = float(start_ratio), float(end_ratio)
        self.register_buffer("settings", torch.tensor(
            [0 if mode == "sup2" else 1, total_iters, *values], dtype=torch.float64))
        self.register_buffer("last_iter", torch.tensor(-1, dtype=torch.long))

    def _apply(self, fn):
        # Legacy wrap_fp16_model may call half(); do not round schedule state.
        for name, buffer in self._buffers.items():
            self._buffers[name] = buffer.to(device=fn(buffer.new_empty(0)).device)
        return self

    def set_iteration(self, iteration):
        if isinstance(iteration, bool) or not isinstance(iteration, int) or not 0 <= iteration < self.total_iters:
            raise ValueError("PG iteration is outside the configured training schedule")
        self.last_iter.fill_(iteration)

    def weights(self):
        iteration = int(self.last_iter.item())
        if iteration < 0:
            raise RuntimeError("PG is not initialized by ProgressiveGammaHook")
        progress = iteration / float(self.total_iters - 1)
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress
        gamma = self.base_gamma * ratio
        return gamma, gamma if self.mode == "both" else self.base_gamma

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if any(prefix + key not in state_dict for key in self._buffers):
            raise RuntimeError("PG checkpoint state is incomplete; start fresh or resume a full PG checkpoint")
        settings = state_dict[prefix + "settings"].detach().cpu()
        if not torch.equal(settings, self.settings.detach().cpu()):
            raise RuntimeError("PG checkpoint settings differ from the training config")
        iteration = state_dict[prefix + "last_iter"]
        if iteration.dtype != torch.long or iteration.numel() != 1 or not -1 <= int(iteration.item()) < self.total_iters:
            raise RuntimeError("Invalid PG checkpoint iteration")
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
