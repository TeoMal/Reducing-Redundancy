"""Shared architecture implementations (see :mod:`redundancy.models.backbones`)."""

from .backbones import mlp, resnet18, small_cnn, vgg16

__all__ = ["resnet18", "vgg16", "small_cnn", "mlp"]
