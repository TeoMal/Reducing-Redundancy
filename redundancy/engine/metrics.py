"""Small metric helpers."""

from __future__ import annotations

from typing import Sequence

import torch


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)):
    """Top-``k`` accuracy (as a fraction in ``[0, 1]``) for each ``k`` in ``topk``."""
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append((correct_k / batch_size).item())
    return results


class AverageMeter:
    """Running average of a scalar."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0
