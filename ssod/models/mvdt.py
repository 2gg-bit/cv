"""Exact minimum-variance pseudo-label thresholds, without MMDetection imports.

The DDT objective is evaluated on a window of fused detection scores, not on
the aligned RoI probabilities used by M2. This implementation is single-class.
"""

import math

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


def minimum_variance_threshold(scores, min_samples=128, min_group_size=2):
    """Return the smallest score in the best high-score group, or None.

    Minimize the sum of within-group squared deviations (DDT Eq. 4).
    Sorting and cumulative moments use CPU float64. Equal scores are never
    split across groups. A tied objective favors the higher threshold.
    """
    if min_group_size < 2 or min_samples < 2 * min_group_size:
        raise ValueError("MVDT requires at least two samples per group")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values >= 0) & (values <= 1)]
    if len(values) < min_samples:
        return None
    values = np.sort(values)[::-1].copy()
    # Centering avoids cancellation for nearly equal, high confidence scores.
    centered = values - values.mean()
    sums = np.cumsum(centered)
    squares = np.cumsum(centered * centered)
    split = np.arange(min_group_size, len(values) - min_group_size + 1)
    split = split[values[split - 1] > values[split]]
    if not len(split):
        return None
    high_sum, high_sq = sums[split - 1], squares[split - 1]
    low_sum, low_sq = sums[-1] - high_sum, squares[-1] - high_sq
    cost = (high_sq - high_sum ** 2 / split
            + low_sq - low_sum ** 2 / (len(values) - split)) / len(values)
    return float(values[split[int(np.argmin(cost))] - 1])


def gather_scores(scores):
    """All ranks receive the same candidates, including empty local batches.

    Sync each training step so a rank-zero checkpoint contains the COMPLETE
    partially filled window, not only rank zero's scores. Works with Gloo/CPU
    and NCCL/CUDA using the existing model device; no object collectives.
    """
    scores = scores.detach().float().reshape(-1)
    valid = torch.isfinite(scores) & (scores >= 0) & (scores <= 1)
    scores = scores[valid]
    if not dist.is_available() or not dist.is_initialized():
        return scores
    size = torch.tensor([scores.numel()], dtype=torch.long, device=scores.device)
    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    counts = [int(item.item()) for item in sizes]
    width = max(counts)
    if width == 0:
        return scores
    padded = scores.new_zeros(width)
    padded[:scores.numel()] = scores
    gathered = [torch.zeros_like(padded) for _ in counts]
    dist.all_gather(gathered, padded)
    return torch.cat([item[:count] for item, count in zip(gathered, counts)])


class MVDTThreshold(nn.Module):
    """Checkpointable, parameter-free threshold controller.

    ``observe`` runs ONCE after both students finish a training forward.
    An update therefore first affects the NEXT training forward. Windows use
    a fixed iteration interval (an adaptation of the paper's epoch schedule).
    """

    def __init__(self, initial_threshold=0.9, score_floor=0.5,
                 warmup_iters=1000, update_interval=1000, min_samples=128,
                 min_group_size=2, max_scores=1000000):
        super().__init__()
        if not (math.isfinite(initial_threshold) and math.isfinite(score_floor)
                and 0 <= score_floor <= initial_threshold <= 1):
            raise ValueError("MVDT requires 0 <= score_floor <= initial_threshold <= 1")
        for name, value in (("warmup_iters", warmup_iters),
                            ("update_interval", update_interval),
                            ("min_samples", min_samples),
                            ("min_group_size", min_group_size),
                            ("max_scores", max_scores)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("MVDT {} must be a positive integer".format(name))
        if min_group_size < 2 or min_samples < 2 * min_group_size or max_scores < min_samples:
            raise ValueError("Invalid MVDT sample limits")
        self.warmup_iters, self.update_interval = warmup_iters, update_interval
        self.min_samples, self.min_group_size = min_samples, min_group_size
        self.max_scores, self.score_floor = max_scores, float(score_floor)
        self.register_buffer("threshold", torch.tensor(initial_threshold, dtype=torch.float64))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("scores", torch.zeros(max_scores, dtype=torch.float32))
        # Keep the schedule with the bank: resuming with different settings is
        # an error, even when MMCV requests a non-strict checkpoint load.
        self.register_buffer("settings", torch.tensor(
            [initial_threshold, score_floor, warmup_iters, update_interval,
             min_samples, min_group_size, max_scores], dtype=torch.float64))

    def _apply(self, fn):
        # Legacy FP16 wrapping may call model.half(). Move statistics to the
        # requested device but never round score/history buffers to float16.
        for name, buffer in self._buffers.items():
            device = fn(buffer.new_empty(0)).device
            self._buffers[name] = buffer.to(device=device)
        return self

    @property
    def value(self):
        return float(self.threshold.item())

    def eligible(self, scores):
        # The old fixed threshold uses >. Retain that during warmup/fallback;
        # a successful MVDT split uses >= s_m so its boundary group is kept.
        scores = scores.detach().float()
        if int(self.updates.item()):
            return scores >= self.value
        return scores > self.value

    @torch.no_grad()
    def observe(self, scores):
        if not self.training:
            return None
        scores = gather_scores(scores.to(device=self.scores.device))
        scores = scores[scores >= self.score_floor]
        count = int(self.count.item())
        end = count + scores.numel()
        if end > self.max_scores:
            raise RuntimeError(
                "MVDT window exceeds max_scores={}; increase mvdt.max_scores "
                "or shorten the update interval (no candidates were sampled/dropped)."
                .format(self.max_scores))
        self.scores[count:end].copy_(scores)
        self.count.fill_(end)
        self.steps.add_(1)
        step = int(self.steps.item())
        if step < self.warmup_iters:
            # Bound early history to the last complete warmup window. This is
            # relevant when warmup_iters is longer than update_interval.
            if step % self.update_interval == 0:
                self._clear_window()
            return None
        if (step - self.warmup_iters) % self.update_interval:
            return None
        next_threshold = minimum_variance_threshold(
            self.scores[:end].cpu().numpy(), self.min_samples, self.min_group_size)
        if next_threshold is not None:
            self.threshold.fill_(max(self.score_floor, next_threshold))
            self.updates.add_(1)
        report = dict(step=step, samples=end, threshold=self.value,
                      updated=next_threshold is not None)
        self._clear_window()
        return report

    def _clear_window(self):
        self.scores.zero_()
        self.count.zero_()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        absent = [name for name in self._buffers if prefix + name not in state_dict]
        if absent:
            raise RuntimeError("MVDT checkpoint state is incomplete: {}. Start a fresh "
                               "M2+MVDT run from Phase1/2, or resume its full checkpoint."
                               .format(", ".join(absent)))
        settings = state_dict[prefix + "settings"].detach().cpu()
        if not torch.equal(settings, self.settings.detach().cpu()):
            raise RuntimeError("MVDT checkpoint settings do not match the training config")
        count = int(state_dict[prefix + "count"].item())
        steps = int(state_dict[prefix + "steps"].item())
        updates = int(state_dict[prefix + "updates"].item())
        threshold = float(state_dict[prefix + "threshold"].item())
        if not (0 <= count <= self.max_scores and steps >= 0 and 0 <= updates <= steps
                and math.isfinite(threshold) and self.score_floor <= threshold <= 1):
            raise RuntimeError("Invalid MVDT checkpoint counters/threshold")
        stored = state_dict[prefix + "scores"]
        if stored.shape != self.scores.shape or stored.dtype != torch.float32:
            raise RuntimeError("MVDT score bank has incompatible shape/dtype")
        active = stored[:count]
        if not bool((torch.isfinite(active) & (active >= self.score_floor) & (active <= 1)).all()):
            raise RuntimeError("Invalid candidates in MVDT checkpoint")
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                           missing_keys, unexpected_keys, error_msgs)
