"""Architectures appropriate for ImageNet-1k.

Implementations are shared (see :mod:`redundancy.models.backbones`); this module
declares which ones apply to ImageNet and wires in the 224x224 input size.

At 224x224 the ResNet keeps its stock 7x7 stride-2 stem (the CIFAR small-image
substitution in :func:`~redundancy.models.backbones.resnet18` only triggers for
inputs of 64px or less), so this is the standard ImageNet ResNet-18.
"""

from __future__ import annotations

from functools import partial

from ...models import backbones

NUM_CHANNELS = 3
IMAGE_SIZE = 224

MODELS = {
    "vit_ti_16": partial(backbones.vit_ti_16, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "vit_s_16": partial(backbones.vit_s_16, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "vit_b_16": partial(backbones.vit_b_16, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "vit_l_16": partial(backbones.vit_l_16, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "resnet18": partial(backbones.resnet18, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
}

DEFAULT_MODEL = "vit_b_16"
