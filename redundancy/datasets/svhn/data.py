"""SVHN (Street View House Numbers) data loaders and transforms.

Note SVHN's torchvision API differs from CIFAR/MNIST: it takes ``split=`` rather
than ``train=`` and stores labels in ``.labels``. The ``build`` function hides
that difference so it matches every other dataset in the package.
"""

from __future__ import annotations

from torchvision import transforms
from torchvision.datasets import SVHN

from ..base import DataBundle, DatasetMeta, split_train_val

MEAN = (0.4377, 0.4438, 0.4728)
STD = (0.1980, 0.2010, 0.1970)

META = DatasetMeta(
    name="svhn",
    num_classes=10,
    num_channels=3,
    image_size=32,
    mean=MEAN,
    std=STD,
)


def _transforms():
    train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train, test


def build(
    root: str = "data",
    val_ratio: float = 0.0,
    seed: int = 0,
    download: bool = True,
) -> DataBundle:
    train_t, test_t = _transforms()
    train_full = SVHN(root, split="train", download=download, transform=train_t)
    test = SVHN(root, split="test", download=download, transform=test_t)
    train, val = split_train_val(train_full, val_ratio, seed)
    return DataBundle(train=train, val=val, test=test, meta=META)
