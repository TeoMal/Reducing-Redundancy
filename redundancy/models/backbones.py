"""Reusable architecture constructors.

These are the concrete implementations shared across datasets. Each dataset
subpackage declares *which* of these are appropriate for it (and with what
number of input channels) in its own ``models.py`` registry, so the per-dataset
folders remain the source of truth for "the architectures for this dataset"
while the implementations live in one place.

Every constructor has the signature ``fn(num_classes, num_channels, image_size)``
and returns an ``nn.Module`` mapping ``(N, C, H, W) -> (N, num_classes)`` logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def resnet18(num_classes: int, num_channels: int = 3, image_size: int = 32) -> nn.Module:
    """A ResNet-18 adapted for small (CIFAR-sized) images.

    The stock torchvision ResNet-18 aggressively downsamples with a 7x7 stride-2
    stem plus max-pool, which destroys 32x32 inputs. For small images we swap in
    a 3x3 stride-1 stem and drop the initial max-pool, the standard CIFAR recipe.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    if image_size <= 64:
        model.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
    elif num_channels != 3:
        model.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    return model


_VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"]


class _VGG(nn.Module):
    def __init__(self, features: nn.Module, final_channels: int, num_classes: int):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def vgg16(num_classes: int, num_channels: int = 3, image_size: int = 32) -> nn.Module:
    """A compact, batch-normalised VGG-16 for small images.

    An adaptive average pool before the classifier makes it robust to the exact
    spatial size, so it works for any input at least 32x32.
    """
    layers = []
    channels = num_channels
    for v in _VGG16_CFG:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [
                nn.Conv2d(channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ]
            channels = v
    return _VGG(nn.Sequential(*layers), channels, num_classes)


class _SmallCNN(nn.Module):
    def __init__(self, num_classes: int, num_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def small_cnn(num_classes: int, num_channels: int = 3, image_size: int = 32) -> nn.Module:
    """A lightweight 3-block convolutional baseline."""
    return _SmallCNN(num_classes, num_channels)


class _MLP(nn.Module):
    def __init__(self, num_classes: int, num_channels: int, image_size: int):
        super().__init__()
        in_features = num_channels * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mlp(num_classes: int, num_channels: int = 1, image_size: int = 28) -> nn.Module:
    """A plain multi-layer perceptron, handy for MNIST-style inputs."""
    return _MLP(num_classes, num_channels, image_size)


# --------------------------------------------------------------------- ViTs
#
# Vision Transformers for ImageNet-scale runs. torchvision ships B/L/H but not
# the Ti and S variants, so all four are built from the same generic
# ``VisionTransformer`` with the standard per-variant hyper-parameters
# (Dosovitskiy et al., 2021; Ti/S widths from Touvron et al., 2021).

_VIT_VARIANTS = {
    #          layers, heads, hidden, mlp
    "ti": dict(num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768),
    "s": dict(num_layers=12, num_heads=6, hidden_dim=384, mlp_dim=1536),
    "b": dict(num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072),
    "l": dict(num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096),
}


def vision_transformer(
    num_classes: int,
    num_channels: int = 3,
    image_size: int = 224,
    variant: str = "b",
    patch_size: int = 16,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
) -> nn.Module:
    """A ViT-``variant``/``patch_size`` operating on ``image_size`` inputs."""
    if variant not in _VIT_VARIANTS:
        raise ValueError(
            f"unknown ViT variant '{variant}'; choose from {sorted(_VIT_VARIANTS)}"
        )
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size {image_size} is not divisible by patch_size {patch_size}"
        )
    from torchvision.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dropout=dropout,
        attention_dropout=attention_dropout,
        **_VIT_VARIANTS[variant],
    )
    if num_channels != 3:
        # The patch-embedding stem is a stride-`patch_size` conv over RGB; swap
        # it for one matching the dataset's channel count (e.g. grayscale).
        hidden = _VIT_VARIANTS[variant]["hidden_dim"]
        model.conv_proj = nn.Conv2d(
            num_channels, hidden, kernel_size=patch_size, stride=patch_size
        )
    return model


def vit_ti_16(num_classes: int, num_channels: int = 3, image_size: int = 224, **kwargs):
    """ViT-Tiny/16 (~5.7M params) -- the cheap variant for pipeline shakedowns."""
    return vision_transformer(num_classes, num_channels, image_size, "ti", 16, **kwargs)


def vit_s_16(num_classes: int, num_channels: int = 3, image_size: int = 224, **kwargs):
    """ViT-Small/16 (~22M params)."""
    return vision_transformer(num_classes, num_channels, image_size, "s", 16, **kwargs)


def vit_b_16(num_classes: int, num_channels: int = 3, image_size: int = 224, **kwargs):
    """ViT-Base/16 (~86M params) -- the standard ImageNet reference point."""
    return vision_transformer(num_classes, num_channels, image_size, "b", 16, **kwargs)


def vit_l_16(num_classes: int, num_channels: int = 3, image_size: int = 224, **kwargs):
    """ViT-Large/16 (~304M params)."""
    return vision_transformer(num_classes, num_channels, image_size, "l", 16, **kwargs)
