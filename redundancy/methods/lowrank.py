"""Low-rank gradient approximation.

The second family of techniques in the thesis works on the *structure* of the
gradients rather than on which examples produce them. Empirically, the per-layer
gradient of a mini-batch is often close to low rank: a few directions carry most
of its energy. Projecting each parameter's gradient onto its top singular
directions before the optimiser step discards the low-energy tail -- the
"redundant" part of the update -- while keeping the informative components.

This module provides a self-contained, in-place implementation applied *after*
``backward()`` and *before* ``optimizer.step()``. It is a simplified, portable
version of the adaptive low-rank scheme discussed in the thesis.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def low_rank_approx(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Return the best rank-``rank`` approximation of a 2-D ``matrix`` (SVD)."""
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(matrix.shape)}")
    max_rank = min(matrix.shape)
    if rank >= max_rank:
        return matrix
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank, :]


def rank_for_energy(singular_values: torch.Tensor, energy: float) -> int:
    """Smallest rank whose singular values retain ``energy`` of the total energy.

    "Energy" is the sum of squared singular values (the squared Frobenius norm).
    """
    if not 0.0 < energy <= 1.0:
        raise ValueError(f"energy must be in (0, 1], got {energy}")
    squared = singular_values**2
    total = squared.sum()
    if total <= 0:
        return 1
    cumulative = torch.cumsum(squared, dim=0) / total
    reached = (cumulative >= energy).nonzero(as_tuple=False)
    return int(reached[0].item()) + 1 if reached.numel() else singular_values.numel()


class LowRankGradient:
    """Project parameter gradients onto a low-rank subspace, in place.

    Parameters
    ----------
    rank:
        Fixed target rank. Ignored when ``energy`` is given.
    energy:
        If set (a fraction in ``(0, 1]``), the rank is chosen *adaptively* per
        tensor as the smallest one retaining this fraction of the gradient's
        spectral energy.
    min_dim:
        Only tensors whose smaller 2-D dimension exceeds this are approximated;
        biases and 1-D gradients are left untouched.

    Notes
    -----
    Gradient tensors with more than two dimensions (e.g. conv weights of shape
    ``[out, in, kh, kw]``) are reshaped to ``[out, in*kh*kw]`` before the SVD and
    restored afterwards.
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        energy: Optional[float] = None,
        min_dim: int = 4,
    ):
        if (rank is None) == (energy is None):
            raise ValueError("provide exactly one of `rank` or `energy`")
        self.rank = rank
        self.energy = energy
        self.min_dim = min_dim
        self.last_ranks: dict[str, int] = {}

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        """Replace each eligible parameter's ``.grad`` with a low-rank version."""
        self.last_ranks.clear()
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None or grad.ndim < 2:
                continue
            matrix = grad.reshape(grad.shape[0], -1)
            if min(matrix.shape) <= self.min_dim:
                continue
            if self.energy is not None:
                singular = torch.linalg.svdvals(matrix)
                rank = rank_for_energy(singular, self.energy)
            else:
                rank = int(self.rank)
            approx = low_rank_approx(matrix, rank)
            grad.copy_(approx.reshape(grad.shape))
            self.last_ranks[name] = rank

    def mean_rank(self) -> float:
        """Average rank used at the last :meth:`apply` call (0 if none)."""
        if not self.last_ranks:
            return 0.0
        return sum(self.last_ranks.values()) / len(self.last_ranks)
