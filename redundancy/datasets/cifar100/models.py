"""Architectures appropriate for CIFAR-100 (same family as CIFAR-10)."""

from __future__ import annotations

from functools import partial

from ...models import backbones

NUM_CHANNELS = 3
IMAGE_SIZE = 32

MODELS = {
    "resnet18": partial(backbones.resnet18, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "vgg16": partial(backbones.vgg16, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "cnn": partial(backbones.small_cnn, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
}

DEFAULT_MODEL = "resnet18"
