"""Shared dataset scaffolding.

Every dataset subpackage (``cifar10``, ``cifar100``, ``mnist``, ``svhn``, ...)
exposes the same small contract so the rest of the code never has to special-case
a dataset:

* ``META``          -- a :class:`DatasetMeta` describing shape / normalisation.
* ``MODELS``        -- ``{name: constructor}`` of the architectures appropriate
                       for this dataset.
* ``DEFAULT_MODEL`` -- the name of the model used when none is requested.
* ``build(...)``    -- returns a :class:`DataBundle` of train/val/test datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class DatasetMeta:
    """Static description of a dataset."""

    name: str
    num_classes: int
    num_channels: int
    image_size: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


@dataclass
class DataBundle:
    """The datasets produced by a dataset subpackage's ``build`` function."""

    train: Dataset
    val: Optional[Dataset]
    test: Dataset
    meta: DatasetMeta


def split_train_val(
    train_ds: Dataset, val_ratio: float, seed: int = 0
) -> Tuple[Dataset, Optional[Dataset]]:
    """Carve a validation split out of the training set.

    ``val_ratio == 0`` returns ``(train_ds, None)`` so callers can simply
    evaluate on the held-out test set instead.
    """
    if val_ratio <= 0.0:
        return train_ds, None
    n_total = len(train_ds)
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(
        train_ds, [n_train, n_val], generator=generator
    )
    return train_split, val_split


def build_dataloaders(
    bundle: DataBundle,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Wrap a :class:`DataBundle` into train/val/test :class:`DataLoader`\\ s.

    The evaluation loader used by the trainer is ``val`` when a validation split
    exists, otherwise ``test``.

    ``batch_size`` is the **per-process** batch size: under DDP the global batch
    is ``batch_size * world_size``. When ``distributed`` is set, each split gets a
    :class:`DistributedSampler` so ranks see disjoint shards; the trainer calls
    ``set_epoch`` on the training sampler so shuffling differs each epoch.

    ``drop_last`` is worth enabling for large-scale training: a short final batch
    makes the per-batch selection statistics noisy and, under DDP, can leave ranks
    with uneven work.
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    extra = {}
    if num_workers > 0:
        extra["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            extra["prefetch_factor"] = prefetch_factor

    def _sampler(dataset: Dataset, shuffle: bool):
        if not distributed:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    train_sampler = _sampler(bundle.train, shuffle=True)
    train_loader = DataLoader(
        bundle.train,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **extra,
    )
    val_loader = None
    if bundle.val is not None:
        val_loader = DataLoader(
            bundle.val,
            batch_size=batch_size,
            shuffle=False,
            sampler=_sampler(bundle.val, shuffle=False),
            num_workers=num_workers,
            pin_memory=pin_memory,
            **extra,
        )
    test_loader = DataLoader(
        bundle.test,
        batch_size=batch_size,
        shuffle=False,
        sampler=_sampler(bundle.test, shuffle=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
        **extra,
    )
    return train_loader, val_loader, test_loader
