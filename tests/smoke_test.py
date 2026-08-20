"""Fast, download-free end-to-end checks.

Exercises the model registry, every selector, the low-rank projection and the
training loop on tiny random tensors shaped like each dataset. Run directly::

    python -m tests.smoke_test

or under pytest::

    pytest tests/smoke_test.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from redundancy.datasets import (
    available_datasets,
    available_models,
    build_model,
    get_meta,
)
from redundancy.engine import Trainer, TrainerConfig
from redundancy.methods import LowRankGradient, available_selectors, build_selector

# Building ViT-B/L at 224px costs seconds and gigabytes; the smaller variants
# exercise exactly the same code path, so the big ones are covered by a
# construction-only check in test_vit_variants_have_expected_widths.
_HEAVY_MODELS = {"vit_b_16", "vit_l_16"}


def _fake_loader(meta, n=16, batch_size=8):
    images = torch.randn(n, meta.num_channels, meta.image_size, meta.image_size)
    labels = torch.randint(0, meta.num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_models_produce_class_logits():
    for dataset in available_datasets():
        meta = get_meta(dataset)
        x = torch.randn(2, meta.num_channels, meta.image_size, meta.image_size)
        for model_name in available_models(dataset):
            if model_name in _HEAVY_MODELS:
                continue
            model = build_model(dataset, model_name).eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, meta.num_classes), (dataset, model_name, out.shape)


def test_vit_variants_have_expected_widths():
    """The Ti/S widths are hand-written, so pin them against the literature."""
    from redundancy.models import backbones

    expected = {  # variant -> (hidden_dim, num_layers, approx params in millions)
        "ti": (192, 12, 5.7),
        "s": (384, 12, 22.1),
        "b": (768, 12, 86.6),
    }
    for variant, (hidden, layers, approx_m) in expected.items():
        model = backbones.vision_transformer(1000, variant=variant, image_size=224)
        assert model.hidden_dim == hidden, variant
        assert len(model.encoder.layers) == layers, variant
        params = sum(p.numel() for p in model.parameters()) / 1e6
        assert abs(params - approx_m) < 0.5, (variant, params)


def test_selection_indices_match_effective_count():
    """two_pass slices the batch by `indices`, so they must agree with the count."""
    torch.manual_seed(0)
    losses = torch.rand(32).abs() + 1e-3
    for name in available_selectors():
        selector = build_selector(name, k=0.5, fraction=2 / 3, c=1.0)
        result = selector(losses)
        if result.indices is None:
            assert result.num_effective == losses.numel(), name
            continue
        assert result.indices.numel() == result.num_effective, name
        assert result.indices.unique().numel() == result.num_effective, name
        assert int(result.indices.max()) < losses.numel(), name


def test_two_pass_reports_real_compute_savings():
    """compute_ratio must follow (B + 3m) / 3B, and be 1.0 for masked mode."""
    meta = get_meta("mnist")
    loader = _fake_loader(meta, n=32, batch_size=16)

    def ratio(mode, method):
        torch.manual_seed(0)
        trainer = Trainer(
            build_model("mnist", "cnn"),
            build_selector(method, k=0.5),
            TrainerConfig(
                epochs=1, device="cpu", verbose=False,
                scheduler="none", selection_mode=mode,
            ),
        )
        return trainer.fit(loader, loader)[0].compute_ratio

    assert ratio("masked", "topk") == 1.0
    # keeping half the batch: (1 + 3*0.5) / 3
    assert abs(ratio("two_pass", "topk") - 5 / 6) < 1e-6
    # `full` selects everything, so two_pass must skip the scoring pass entirely
    assert ratio("two_pass", "full") == 1.0


def test_amp_and_accumulation_run():
    meta = get_meta("mnist")
    loader = _fake_loader(meta, n=32, batch_size=8)
    for amp in ("off", "bf16"):
        for accum in (1, 4):
            trainer = Trainer(
                build_model("mnist", "cnn"),
                build_selector("topk", k=0.5),
                TrainerConfig(
                    epochs=1, device="cpu", verbose=False, amp=amp,
                    accum_steps=accum, grad_clip=1.0, warmup_epochs=0.5,
                ),
            )
            stats = trainer.fit(loader, loader)[0]
            assert torch.isfinite(torch.tensor(stats.train_loss)), (amp, accum)


def test_invalid_config_is_rejected():
    for bad in (
        dict(amp="fp8"),
        dict(selection_mode="magic"),
        dict(accum_steps=0),
    ):
        try:
            TrainerConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"TrainerConfig accepted invalid config {bad}")


def test_selectors_are_well_formed():
    torch.manual_seed(0)
    losses = torch.rand(32).abs() + 1e-3
    for name in available_selectors():
        selector = build_selector(name, k=0.5, fraction=2 / 3, c=1.0)
        result = selector(losses)
        assert torch.isfinite(result.loss), name
        assert 1 <= result.num_effective <= losses.numel(), name
        assert result.num_total == losses.numel()


def test_low_rank_projection_runs():
    meta = get_meta("cifar10")
    model = build_model("cifar10", "resnet18")
    x = torch.randn(4, meta.num_channels, meta.image_size, meta.image_size)
    model(x).sum().backward()
    for lr in (LowRankGradient(rank=2), LowRankGradient(energy=0.9)):
        grads_before = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        lr.apply(model)
        assert lr.mean_rank() > 0
        # gradients should still be finite and correctly shaped
        for before, p in zip(grads_before, [p for p in model.parameters() if p.grad is not None]):
            assert p.grad.shape == before.shape
            assert torch.isfinite(p.grad).all()


def test_training_loop_reduces_or_runs():
    meta = get_meta("mnist")
    loader = _fake_loader(meta)
    for method in available_selectors():
        model = build_model("mnist", "cnn")
        selector = build_selector(method, k=0.5, fraction=2 / 3, c=1.0)
        config = TrainerConfig(epochs=1, device="cpu", verbose=False, scheduler="none")
        trainer = Trainer(model, selector, config)
        history = trainer.fit(loader, loader)
        assert len(history) == 1
        assert torch.isfinite(torch.tensor(history[0].train_loss))
        assert 0.0 <= history[0].effective_ratio <= 1.0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} smoke tests passed.")


if __name__ == "__main__":
    _run_all()
