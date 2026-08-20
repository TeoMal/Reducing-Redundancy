"""MNIST data loaders and transforms."""

from __future__ import annotations

from torchvision import transforms
from torchvision.datasets import MNIST

from ..base import DataBundle, DatasetMeta, split_train_val

MEAN = (0.1307,)
STD = (0.3081,)

META = DatasetMeta(
    name="mnist",
    num_classes=10,
    num_channels=1,
    image_size=28,
    mean=MEAN,
    std=STD,
)


def _transforms():
    train = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=2),
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
    train_full = MNIST(root, train=True, download=download, transform=train_t)
    test = MNIST(root, train=False, download=download, transform=test_t)
    train, val = split_train_val(train_full, val_ratio, seed)
    return DataBundle(train=train, val=val, test=test, meta=META)
