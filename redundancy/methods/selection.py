"""Loss-based example-selection strategies.

The central idea studied in the thesis is that, especially late in training, many
examples in a mini-batch contribute little to the gradient: their loss (and hence
gradient magnitude) is small. Instead of back-propagating the full batch we can
select a subset of high-loss examples and only train on those, reducing redundant
computation while preserving the training signal.

Each selector receives the vector of **per-example** losses (shape ``[B]``,
typically from ``CrossEntropyLoss(reduction="none")``) and returns a
:class:`SelectionResult` holding

* ``loss``          -- the scalar to call ``.backward()`` on, and
* ``num_effective`` -- how many examples actually contributed.

All selectors reduce the selected losses with a **mean** so that the gradient
magnitude (and therefore the effective learning rate) stays comparable no matter
how many examples are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SelectionResult:
    loss: torch.Tensor
    num_effective: int
    num_total: int
    indices: Optional[torch.Tensor] = None
    """Positions of the selected examples within the batch.

    ``None`` means "every example" (the :class:`FullSelector` case). Two-pass
    training uses this to build the subset that actually receives a backward
    pass; masked training ignores it and back-propagates ``loss`` directly.
    """

    @property
    def effective_ratio(self) -> float:
        return self.num_effective / self.num_total if self.num_total else 0.0


class Selector:
    """Base class: turn per-example losses into a scalar training loss."""

    name = "base"

    def __call__(self, losses: torch.Tensor) -> SelectionResult:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}()"


class FullSelector(Selector):
    """Standard training: use every example (plain cross-entropy)."""

    name = "full"

    def __call__(self, losses: torch.Tensor) -> SelectionResult:
        return SelectionResult(losses.mean(), losses.numel(), losses.numel())


class TopKSelector(Selector):
    """Keep the top-``k`` fraction of examples by loss.

    ``k`` is a fraction in ``(0, 1]``; at least one example is always kept.
    """

    name = "topk"

    def __init__(self, k: float = 0.5):
        if not 0.0 < k <= 1.0:
            raise ValueError(f"k must be in (0, 1], got {k}")
        self.k = k

    def __call__(self, losses: torch.Tensor) -> SelectionResult:
        n = losses.numel()
        m = max(1, int(round(self.k * n)))
        selected, indices = torch.topk(losses, m)
        return SelectionResult(selected.mean(), m, n, indices)

    def __repr__(self) -> str:
        return f"TopKSelector(k={self.k})"


class AdaptiveKSelector(Selector):
    """Smallest set of highest-loss examples covering a fraction of total loss.

    Sort losses in descending order and keep the fewest examples whose cumulative
    loss reaches ``fraction`` of the batch's total loss (default ``2/3``). The
    number kept therefore adapts to how concentrated the loss is across the batch.
    """

    name = "adaptive_k"

    def __init__(self, fraction: float = 2.0 / 3.0):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction

    def __call__(self, losses: torch.Tensor) -> SelectionResult:
        n = losses.numel()
        total = losses.sum()
        sorted_losses, order = torch.sort(losses, descending=True)
        cumulative = torch.cumsum(sorted_losses, dim=0)
        threshold = self.fraction * total
        reached = (cumulative >= threshold).nonzero(as_tuple=False)
        m = n if reached.numel() == 0 else int(reached[0].item()) + 1
        selected = sorted_losses[:m]
        return SelectionResult(selected.mean(), m, n, order[:m])

    def __repr__(self) -> str:
        return f"AdaptiveKSelector(fraction={self.fraction:.4f})"


class MeanAdaptiveSelector(Selector):
    """Keep examples whose loss exceeds ``c`` times the batch mean loss.

    With ``c = 1`` this keeps every above-average example. If the threshold
    excludes everything (e.g. a near-uniform batch) it falls back to the full
    batch so a training step is never wasted.
    """

    name = "mean_adaptive"

    def __init__(self, c: float = 1.0):
        if c < 0.0:
            raise ValueError(f"c must be non-negative, got {c}")
        self.c = c

    def __call__(self, losses: torch.Tensor) -> SelectionResult:
        n = losses.numel()
        threshold = self.c * losses.mean()
        mask = losses > threshold
        m = int(mask.sum().item())
        if m == 0:
            return SelectionResult(losses.mean(), n, n)
        return SelectionResult(
            losses[mask].mean(), m, n, mask.nonzero(as_tuple=True)[0]
        )

    def __repr__(self) -> str:
        return f"MeanAdaptiveSelector(c={self.c})"


_SELECTORS = {
    FullSelector.name: FullSelector,
    TopKSelector.name: TopKSelector,
    AdaptiveKSelector.name: AdaptiveKSelector,
    MeanAdaptiveSelector.name: MeanAdaptiveSelector,
}


def build_selector(name: str, **kwargs) -> Selector:
    """Construct a selector by name, forwarding any keyword arguments.

    Unknown keyword arguments are ignored so a single argument namespace (e.g.
    both ``k`` and ``c``) can be passed regardless of which selector is chosen.
    """
    if name not in _SELECTORS:
        raise KeyError(
            f"Unknown selector '{name}'. Available: {sorted(_SELECTORS)}"
        )
    cls = _SELECTORS[name]
    import inspect

    valid = inspect.signature(cls.__init__).parameters
    filtered = {key: value for key, value in kwargs.items() if key in valid}
    return cls(**filtered)


def available_selectors() -> list[str]:
    return sorted(_SELECTORS)
