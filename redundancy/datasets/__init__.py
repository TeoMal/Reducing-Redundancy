"""Dataset registry.

Each dataset lives in its own subpackage exposing ``build``, ``META``,
``MODELS`` and ``DEFAULT_MODEL``. This module is the single lookup point the rest
of the code (and ``train.py``) uses, so adding a dataset is just:

1. create ``redundancy/datasets/<name>/`` with ``data.py`` + ``models.py``
   following the existing folders, and
2. add it to the ``_DATASETS`` mapping below.
"""

from __future__ import annotations

from types import ModuleType
from typing import Dict, List

import torch.nn as nn

from . import cifar10, cifar100, imagenet, mnist, svhn
from .base import DataBundle, DatasetMeta, build_dataloaders

_DATASETS: Dict[str, ModuleType] = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "imagenet": imagenet,
    "mnist": mnist,
    "svhn": svhn,
}


def available_datasets() -> List[str]:
    return sorted(_DATASETS)


def _get(name: str) -> ModuleType:
    if name not in _DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available_datasets()}"
        )
    return _DATASETS[name]


def build_dataset(
    name: str,
    root: str = "data",
    val_ratio: float = 0.0,
    seed: int = 0,
    download: bool = True,
) -> DataBundle:
    """Build the train/val/test datasets for ``name``."""
    return _get(name).build(
        root=root, val_ratio=val_ratio, seed=seed, download=download
    )


def get_meta(name: str) -> DatasetMeta:
    return _get(name).META


def available_models(name: str) -> List[str]:
    return sorted(_get(name).MODELS)


def default_model(name: str) -> str:
    return _get(name).DEFAULT_MODEL


def build_model(dataset: str, model: str | None = None, **kwargs) -> nn.Module:
    """Instantiate an architecture registered for ``dataset``.

    ``model=None`` uses the dataset's default architecture. Extra keyword
    arguments (e.g. ``dropout``) are forwarded to the constructor; passing one
    an architecture does not accept is a ``TypeError`` from that constructor.
    """
    module = _get(dataset)
    model = model or module.DEFAULT_MODEL
    if model not in module.MODELS:
        raise KeyError(
            f"Model '{model}' is not registered for dataset '{dataset}'. "
            f"Available: {available_models(dataset)}"
        )
    return module.MODELS[model](num_classes=module.META.num_classes, **kwargs)


__all__ = [
    "available_datasets",
    "build_dataset",
    "get_meta",
    "available_models",
    "default_model",
    "build_model",
    "build_dataloaders",
    "DataBundle",
    "DatasetMeta",
]
