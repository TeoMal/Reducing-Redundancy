#!/usr/bin/env python
"""Train a model on a dataset with a chosen redundancy-reduction method.

Examples
--------
Baseline ResNet-18 on CIFAR-10::

    python train.py --dataset cifar10 --model resnet18 --method full

MeanAdaptive selection (keep above-average-loss examples), with the last 10
epochs falling back to plain cross-entropy::

    python train.py --dataset cifar10 --method mean_adaptive --c 1.0 --ce-tail-epochs 10

AdaptiveK selection on CIFAR-100 with a VGG-16::

    python train.py --dataset cifar100 --model vgg16 --method adaptive_k --fraction 0.667

Low-rank gradient approximation, keeping 90% of each gradient's spectral energy::

    python train.py --dataset svhn --method full --low-rank-energy 0.9

ImageNet-1k / ViT-B/16 on 8 GPUs, measuring what TopK selection actually saves::

    torchrun --nproc_per_node=8 train.py \\
        --dataset imagenet --data-root /data/imagenet --model vit_b_16 \\
        --method topk --k 0.5 --selection-mode two_pass \\
        --optimizer adamw --lr 3e-3 --weight-decay 0.05 \\
        --warmup-epochs 5 --epochs 90 --batch-size 128 --amp bf16 \\
        --grad-clip 1.0 --drop-last --ckpt-dir ckpt/vit_b16_topk
"""

from __future__ import annotations

import argparse
import os

import torch

from redundancy.datasets import (
    available_datasets,
    available_models,
    build_dataloaders,
    build_dataset,
    build_model,
    default_model,
)
from redundancy.engine import Trainer, TrainerConfig
from redundancy.engine.trainer import CSV_FIELDNAMES
from redundancy.engine.distributed import (
    cleanup_distributed,
    init_distributed,
    main_print,
)
from redundancy.methods import LowRankGradient, available_selectors, build_selector
from redundancy.utils import CSVLogger, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reducing redundancy in neural network training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data / model")
    data.add_argument("--dataset", default="cifar10", choices=available_datasets())
    data.add_argument("--model", default=None,
                      help="architecture name; default = the dataset's default model")
    data.add_argument("--data-root", default="data",
                      help="where datasets are stored; for imagenet this is the "
                           "directory containing train/ and val/")
    data.add_argument("--val-ratio", type=float, default=0.0,
                      help="fraction of the training set held out for validation "
                           "(0 = evaluate on the test set)")
    data.add_argument("--no-download", action="store_true",
                      help="do not download the dataset if it is missing")
    data.add_argument("--dropout", type=float, default=0.0,
                      help="dropout inside the transformer blocks (ViT only)")
    data.add_argument("--attention-dropout", type=float, default=0.0,
                      help="attention dropout (ViT only)")
    data.add_argument("--drop-last", action="store_true",
                      help="drop the last incomplete training batch (recommended "
                           "at scale: keeps per-batch selection statistics stable)")

    method = parser.add_argument_group("redundancy-reduction method")
    method.add_argument("--method", default="full", choices=available_selectors(),
                        help="example-selection strategy")
    method.add_argument("--selection-mode", default="masked",
                        choices=["masked", "two_pass"],
                        help="'masked' back-propagates a masked loss over the full "
                             "batch (no wall-clock saving); 'two_pass' scores under "
                             "no_grad then re-forwards only the selected subset "
                             "(real saving, break-even at a kept fraction of 2/3)")
    method.add_argument("--k", type=float, default=0.5,
                        help="fraction kept by the 'topk' selector")
    method.add_argument("--fraction", type=float, default=2.0 / 3.0,
                        help="loss-mass fraction for the 'adaptive_k' selector")
    method.add_argument("--c", type=float, default=1.0,
                        help="threshold multiplier for the 'mean_adaptive' selector")
    method.add_argument("--ce-tail-epochs", type=int, default=0,
                        help="use plain cross-entropy for the final N epochs")
    method.add_argument("--low-rank", type=int, default=None,
                        help="project gradients to this fixed rank after backward")
    method.add_argument("--low-rank-energy", type=float, default=None,
                        help="adaptive low-rank: keep this fraction of spectral energy")

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=30)
    optim.add_argument("--batch-size", type=int, default=128,
                       help="per-process batch size; the global batch under "
                            "torchrun is this times the number of processes")
    optim.add_argument("--lr", type=float, default=0.1)
    optim.add_argument("--momentum", type=float, default=0.9)
    optim.add_argument("--weight-decay", type=float, default=5e-4)
    optim.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    optim.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    optim.add_argument("--warmup-epochs", type=float, default=0.0,
                       help="linear LR warmup; ViTs generally need 5+")
    optim.add_argument("--grad-clip", type=float, default=None)
    optim.add_argument("--label-smoothing", type=float, default=0.0,
                       help="cross-entropy label smoothing (0 = off); ViTs on "
                            "ImageNet conventionally use 0.1")
    optim.add_argument("--accum-steps", type=int, default=1,
                       help="micro-batches per optimiser step (raises the effective "
                            "batch without raising memory)")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--seed", type=int, default=2021)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--prefetch-factor", type=int, default=None,
                         help="batches prefetched per worker (needs --num-workers > 0)")
    runtime.add_argument("--device", default="auto",
                         help="'auto', 'cpu', 'cuda', 'cuda:0', ...")
    runtime.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"],
                         help="mixed precision; bf16 needs Ampere or newer")
    runtime.add_argument("--channels-last", action="store_true",
                         help="NHWC memory format (helps convnets on tensor cores)")
    runtime.add_argument("--compile", action="store_true",
                         help="wrap the model in torch.compile")
    runtime.add_argument("--output-dir", default="outputs",
                         help="directory for the per-epoch CSV log")
    runtime.add_argument("--ckpt-dir", default=None,
                         help="directory for checkpoints (default: no checkpointing)")
    runtime.add_argument("--ckpt-every", type=int, default=0,
                         help="save a checkpoint every N epochs (0 = only at the end)")
    runtime.add_argument("--resume", default=None, help="path to a checkpoint to resume")
    runtime.add_argument("--quiet", action="store_true", help="reduce console output")

    return parser.parse_args()


def resolve_device(choice: str, local_rank: int, distributed: bool) -> str:
    if distributed and torch.cuda.is_available():
        # torchrun gives each process one GPU; pin it explicitly.
        return f"cuda:{local_rank}"
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def main() -> None:
    args = parse_args()
    dist_info = init_distributed()
    # Identical seed on every rank so all processes start from the same weights;
    # DistributedSampler is what makes the ranks see different data.
    set_seed(args.seed)
    device = resolve_device(args.device, dist_info.local_rank, dist_info.enabled)

    try:
        if args.model is not None and args.model not in available_models(args.dataset):
            raise SystemExit(
                f"Model '{args.model}' is not available for '{args.dataset}'. "
                f"Choices: {available_models(args.dataset)}"
            )

        bundle = build_dataset(
            args.dataset,
            root=args.data_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
            download=not args.no_download,
        )
        train_loader, val_loader, test_loader = build_dataloaders(
            bundle,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
            distributed=dist_info.enabled,
            rank=dist_info.rank,
            world_size=dist_info.world_size,
            seed=args.seed,
            drop_last=args.drop_last,
            prefetch_factor=args.prefetch_factor,
        )
        eval_loader = val_loader if val_loader is not None else test_loader

        # Only forward the regularisation knobs when they are actually set, so
        # architectures that do not take them (ResNet, VGG) keep working.
        model_kwargs = {}
        if args.dropout:
            model_kwargs["dropout"] = args.dropout
        if args.attention_dropout:
            model_kwargs["attention_dropout"] = args.attention_dropout
        model = build_model(args.dataset, args.model, **model_kwargs)
        selector = build_selector(args.method, k=args.k, fraction=args.fraction, c=args.c)

        low_rank = None
        if args.low_rank is not None or args.low_rank_energy is not None:
            low_rank = LowRankGradient(rank=args.low_rank, energy=args.low_rank_energy)

        model_name = args.model or default_model(args.dataset)
        tag = f"{args.dataset}_{model_name}_{args.method}"
        if args.selection_mode != "masked":
            tag += f"_{args.selection_mode}"
        log_path = os.path.join(args.output_dir, f"{tag}.csv")
        logger = (
            CSVLogger(log_path, fieldnames=CSV_FIELDNAMES) if dist_info.is_main else None
        )

        config = TrainerConfig(
            epochs=args.epochs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            ce_tail_epochs=args.ce_tail_epochs,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            device=device,
            verbose=not args.quiet,
            amp=args.amp,
            accum_steps=args.accum_steps,
            channels_last=args.channels_last,
            compile=args.compile,
            selection_mode=args.selection_mode,
            ckpt_dir=args.ckpt_dir,
            ckpt_every=args.ckpt_every,
            resume=args.resume,
        )

        global_batch = args.batch_size * dist_info.world_size * args.accum_steps
        main_print(dist_info, f"Dataset : {args.dataset}  ({bundle.meta.num_classes} classes)")
        main_print(dist_info, f"Model   : {model_name}")
        main_print(dist_info, f"Method  : {selector}  [{args.selection_mode}]"
                   + (f" + low-rank({low_rank.rank or f'energy={low_rank.energy}'})"
                      if low_rank else ""))
        main_print(dist_info, f"Batch   : {args.batch_size}/proc x {dist_info.world_size} proc "
                   f"x {args.accum_steps} accum = {global_batch} global")
        main_print(dist_info, f"Device  : {device}  (amp={args.amp})")
        main_print(dist_info, f"Log     : {log_path}")
        main_print(dist_info, "-" * 60)

        trainer = Trainer(
            model, selector, config,
            low_rank=low_rank, logger=logger, dist_info=dist_info,
        )
        trainer.fit(train_loader, eval_loader)

        if trainer.history:
            best = max(trainer.history, key=lambda s: s.eval_acc_top1)
            mean_compute = sum(s.compute_ratio for s in trainer.history) / len(trainer.history)
            mean_used = sum(s.effective_ratio for s in trainer.history) / len(trainer.history)
            main_print(dist_info, "-" * 60)
            main_print(dist_info, f"Best eval top-1 : {best.eval_acc_top1:.4f} (epoch {best.epoch})")
            main_print(dist_info, f"Mean examples used : {mean_used:.1%}")
            main_print(dist_info, f"Mean compute vs baseline : {mean_compute:.1%}")
            main_print(dist_info, f"Log written to : {log_path}")
    finally:
        cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
