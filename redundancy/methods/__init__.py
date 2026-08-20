"""Redundancy-reduction techniques.

* :mod:`~redundancy.methods.selection` -- loss-based example selection.
* :mod:`~redundancy.methods.lowrank`   -- low-rank gradient approximation.
"""

from .lowrank import LowRankGradient, low_rank_approx, rank_for_energy
from .selection import (
    AdaptiveKSelector,
    FullSelector,
    MeanAdaptiveSelector,
    SelectionResult,
    Selector,
    TopKSelector,
    available_selectors,
    build_selector,
)

__all__ = [
    "Selector",
    "SelectionResult",
    "FullSelector",
    "TopKSelector",
    "AdaptiveKSelector",
    "MeanAdaptiveSelector",
    "build_selector",
    "available_selectors",
    "LowRankGradient",
    "low_rank_approx",
    "rank_for_energy",
]
