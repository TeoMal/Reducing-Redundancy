"""Architectures appropriate for MNIST (single-channel, 28x28).

VGG-16 is intentionally omitted: its five 2x2 max-pools would collapse a 28x28
feature map to nothing.
"""

from __future__ import annotations

from functools import partial

from ...models import backbones

NUM_CHANNELS = 1
IMAGE_SIZE = 28

MODELS = {
    "cnn": partial(backbones.small_cnn, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "mlp": partial(backbones.mlp, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
    "resnet18": partial(backbones.resnet18, num_channels=NUM_CHANNELS, image_size=IMAGE_SIZE),
}

DEFAULT_MODEL = "cnn"
