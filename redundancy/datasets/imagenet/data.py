"""ImageNet-1k data loaders and transforms.

Unlike the small datasets in this repo, ImageNet cannot be downloaded
automatically -- it needs a (free) registration at https://image-net.org. Point
``--data-root`` at a directory laid out in the usual ``ImageFolder`` form::

    <root>/
      train/
        n01440764/*.JPEG
        n01443537/*.JPEG
        ...
      val/
        n01440764/*.JPEG
        ...

The raw validation archive ships as a flat directory of 50k images; the standard
``valprep.sh`` script (or ``torchvision.datasets.ImageNet``) reorganises it into
the per-synset layout above.

The augmentation recipe is the plain Inception-style
``RandomResizedCrop`` + horizontal flip. That is deliberately *not* the heavy
RandAugment/mixup/erasing recipe used to chase state-of-the-art ViT accuracy:
the question this repo asks is how the selectors compare **to each other**, and
a simple, identical pipeline across runs makes that comparison clean. If you
later want competitive absolute accuracy, strengthen it here -- but re-run every
selector so the comparison stays like-for-like.
"""

from __future__ import annotations

import os

from torchvision import transforms
from torchvision.datasets import ImageFolder

from ..base import DataBundle, DatasetMeta, split_train_val

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

META = DatasetMeta(
    name="imagenet",
    num_classes=1000,
    num_channels=3,
    image_size=IMAGE_SIZE,
    mean=MEAN,
    std=STD,
)


def _transforms(image_size: int = IMAGE_SIZE):
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    # Resize the short side to 256/224 of the crop, then centre-crop: the
    # conventional evaluation protocol.
    resize = int(round(image_size * 256 / 224))
    test = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train, test


def _require_dir(path: str, what: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"ImageNet {what} directory not found at '{path}'.\n"
            "ImageNet-1k cannot be downloaded automatically. Register at "
            "https://image-net.org, then arrange the data as "
            "<root>/train/<synset>/*.JPEG and <root>/val/<synset>/*.JPEG."
        )
    return path


def build(
    root: str = "data/imagenet",
    val_ratio: float = 0.0,
    seed: int = 0,
    download: bool = True,  # accepted for API parity; ImageNet is never fetched
) -> DataBundle:
    train_t, test_t = _transforms()
    train_dir = _require_dir(os.path.join(root, "train"), "train")
    val_dir = _require_dir(os.path.join(root, "val"), "val")

    train_full = ImageFolder(train_dir, transform=train_t)
    test = ImageFolder(val_dir, transform=test_t)
    train, val = split_train_val(train_full, val_ratio, seed)
    return DataBundle(train=train, val=val, test=test, meta=META)
