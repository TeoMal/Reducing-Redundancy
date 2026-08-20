"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    ``deterministic=True`` also forces cuDNN into a reproducible (slower) mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
