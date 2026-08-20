"""CIFAR-100 data loaders and transforms."""

from __future__ import annotations

from torchvision import transforms
from torchvision.datasets import CIFAR100

from ..base import DataBundle, DatasetMeta, split_train_val

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

META = DatasetMeta(
    name="cifar100",
    num_classes=100,
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
    train_full = CIFAR100(root, train=True, download=download, transform=train_t)
    test = CIFAR100(root, train=False, download=download, transform=test_t)
    train, val = split_train_val(train_full, val_ratio, seed)
    return DataBundle(train=train, val=val, test=test, meta=META)
