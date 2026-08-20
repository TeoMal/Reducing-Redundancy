"""CIFAR-10 data loaders and transforms."""

from __future__ import annotations

from torchvision import transforms
from torchvision.datasets import CIFAR10

from ..base import DataBundle, DatasetMeta, split_train_val

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

META = DatasetMeta(
    name="cifar10",
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
            transforms.RandomHorizontalFlip(),
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
    train_full = CIFAR10(root, train=True, download=download, transform=train_t)
    test = CIFAR10(root, train=False, download=download, transform=test_t)
    train, val = split_train_val(train_full, val_ratio, seed)
    return DataBundle(train=train, val=val, test=test, meta=META)
