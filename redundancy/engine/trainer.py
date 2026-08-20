"""The training / evaluation loop.

The loop stays deliberately readable -- plain PyTorch, no framework -- while
supporting the machinery an ImageNet-scale run actually needs: mixed precision,
gradient accumulation, DistributedDataParallel and checkpoint/resume.

For every mini-batch it

1. computes **per-example** cross-entropy losses,
2. hands them to a :class:`~redundancy.methods.selection.Selector` which decides
   which examples contribute,
3. optionally projects the gradients to low rank
   (:class:`~redundancy.methods.lowrank.LowRankGradient`), and
4. steps the optimiser.

Selection modes
---------------
``selection_mode`` controls *how* step 2 is realised, and the choice determines
whether selection saves any real compute:

``masked``
    Forward the whole batch, reduce the selected losses to a scalar and
    back-propagate that. The unselected examples contribute zero rows to the
    gradient of the logits, but every matmul in the backward pass is still
    full-size, so **wall-clock cost is unchanged**. ``effective_ratio`` measures
    a hypothetical saving. This is the original behaviour and stays the default
    so existing CIFAR results remain reproducible.

``two_pass``
    Score the batch under ``no_grad`` (forward only), select a subset, then run
    a *second* forward and the backward on that subset alone. Every tensor in
    the backward pass is genuinely smaller, so the saving is real -- but it is
    smaller than ``1 - effective_ratio`` because of the scoring pass. Counting a
    forward as 1 unit and a backward as 2, a baseline step on ``B`` examples
    costs ``3B`` while a two-pass step keeping ``m`` costs ``B + 3m``. Break-even
    is therefore at ``m/B = 2/3``: keeping more than two thirds of the batch is
    *slower* than not selecting at all. The trainer records this as
    ``compute_ratio`` alongside measured images/second.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..methods.lowrank import LowRankGradient
from ..methods.selection import FullSelector, Selector
from ..utils.csv_logger import CSVLogger
from .distributed import DistInfo, all_reduce_sum, main_print, unwrap

try:  # optional progress bar
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


# Relative cost of a backward pass to a forward pass, used for the analytic
# `compute_ratio`. Backward computes both input- and weight-gradients, so it is
# conventionally counted as ~2x the forward FLOPs.
_BACKWARD_COST = 2.0


@dataclass
class TrainerConfig:
    epochs: int = 30
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    optimizer: str = "sgd"          # "sgd" | "adam" | "adamw"
    scheduler: str = "cosine"       # "cosine" | "none"
    warmup_epochs: float = 0.0      # linear LR warmup; ViTs need this
    ce_tail_epochs: int = 0         # use plain CE for the final N epochs
    grad_clip: Optional[float] = None
    device: str = "cpu"
    verbose: bool = True

    # --- scale / throughput -------------------------------------------------
    amp: str = "off"                # "off" | "bf16" | "fp16"
    accum_steps: int = 1            # micro-batches per optimiser step
    channels_last: bool = False     # NHWC memory format (helps convnets on TC)
    compile: bool = False           # torch.compile the model
    selection_mode: str = "masked"  # "masked" | "two_pass"

    # --- checkpointing ------------------------------------------------------
    ckpt_dir: Optional[str] = None
    ckpt_every: int = 0             # save every N epochs (0 = only at the end)
    resume: Optional[str] = None

    def __post_init__(self) -> None:
        if self.amp not in ("off", "bf16", "fp16"):
            raise ValueError(f"amp must be off|bf16|fp16, got {self.amp!r}")
        if self.selection_mode not in ("masked", "two_pass"):
            raise ValueError(
                f"selection_mode must be masked|two_pass, got {self.selection_mode!r}"
            )
        if self.accum_steps < 1:
            raise ValueError(f"accum_steps must be >= 1, got {self.accum_steps}")


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    eval_acc_top1: float
    eval_acc_top5: float
    examples_seen: int
    effective_examples: int
    effective_ratio: float
    compute_ratio: float = 1.0
    train_seconds: float = 0.0
    images_per_sec: float = 0.0
    lr: float = 0.0
    mean_rank: float = 0.0

    def as_row(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "train_acc": round(self.train_acc, 6),
            "eval_acc_top1": round(self.eval_acc_top1, 6),
            "eval_acc_top5": round(self.eval_acc_top5, 6),
            "examples_seen": self.examples_seen,
            "effective_examples": self.effective_examples,
            "effective_ratio": round(self.effective_ratio, 6),
            "compute_ratio": round(self.compute_ratio, 6),
            "train_seconds": round(self.train_seconds, 3),
            "images_per_sec": round(self.images_per_sec, 2),
            "lr": round(self.lr, 8),
            "mean_rank": round(self.mean_rank, 4),
        }


CSV_FIELDNAMES = [
    "epoch", "train_loss", "train_acc", "eval_acc_top1", "eval_acc_top5",
    "examples_seen", "effective_examples", "effective_ratio", "compute_ratio",
    "train_seconds", "images_per_sec", "lr", "mean_rank",
]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        selector: Selector,
        config: TrainerConfig,
        low_rank: Optional[LowRankGradient] = None,
        logger: Optional[CSVLogger] = None,
        dist_info: Optional[DistInfo] = None,
    ):
        self.config = config
        self.dist = dist_info or DistInfo()
        self.device = torch.device(config.device)
        self.selector = selector
        self.low_rank = low_rank
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.history: List[EpochStats] = []
        self.start_epoch = 0

        model = model.to(self.device)
        if config.channels_last:
            model = model.to(memory_format=torch.channels_last)
        if config.compile:
            model = torch.compile(model)
        self.model = self._wrap_distributed(model)

        self.optimizer = self._build_optimizer()
        self.scaler = self._build_scaler()
        self.scheduler = None  # built in fit(), once steps-per-epoch is known

        if config.resume:
            self._load_checkpoint(config.resume)

    # ------------------------------------------------------------------ setup

    def _wrap_distributed(self, model: nn.Module) -> nn.Module:
        if not self.dist.enabled:
            return model
        from torch.nn.parallel import DistributedDataParallel

        device_ids = [self.dist.local_rank] if self.device.type == "cuda" else None
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=self.dist.local_rank if device_ids else None,
            # Selection can leave some parameters without a gradient in a given
            # step (e.g. an unused head); tolerate that rather than crashing.
            find_unused_parameters=False,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        params = self.model.parameters()
        if cfg.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.momentum > 0,
            )
        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        raise ValueError(f"unknown optimizer '{cfg.optimizer}'")

    def _build_scaler(self):
        """GradScaler is only needed for fp16; bf16 has enough exponent range."""
        enabled = self.config.amp == "fp16" and self.device.type == "cuda"
        try:  # torch >= 2.4 generic API
            return torch.amp.GradScaler(self.device.type, enabled=enabled)
        except (AttributeError, TypeError):  # pragma: no cover - older torch
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast(self):
        if self.config.amp == "off":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.config.amp == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _build_scheduler(self, steps_per_epoch: int):
        """Cosine decay with optional linear warmup, stepped per optimiser step.

        Per-step (rather than per-epoch) stepping matters for short ImageNet
        schedules, where a warmup measured in epochs is only a handful of points
        of resolution.
        """
        cfg = self.config
        if cfg.scheduler in ("none", None):
            return None
        if cfg.scheduler != "cosine":
            raise ValueError(f"unknown scheduler '{cfg.scheduler}'")

        total_steps = max(1, steps_per_epoch * cfg.epochs)
        warmup_steps = int(steps_per_epoch * cfg.warmup_epochs)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _selector_for_epoch(self, epoch: int) -> Selector:
        """Fall back to plain cross-entropy for the configured tail of epochs."""
        tail = self.config.ce_tail_epochs
        if tail > 0 and epoch >= self.config.epochs - tail:
            return FullSelector()
        return self.selector

    # --------------------------------------------------------------- training

    def _forward_batch(self, inputs, targets, selector):
        """Run one micro-batch and return (loss_to_scale, result, logits, cost).

        ``cost`` is in forward-equivalent units per the module docstring, used to
        build ``compute_ratio``.
        """
        batch_size = targets.size(0)
        full_cost = batch_size * (1.0 + _BACKWARD_COST)

        two_pass = (
            self.config.selection_mode == "two_pass"
            and not isinstance(selector, FullSelector)
        )

        if not two_pass:
            with self._autocast():
                logits = self.model(inputs)
                per_example = self.criterion(logits, targets)
            result = selector(per_example)
            return result.loss, result, logits.detach(), full_cost

        # Pass 1: score the batch without building a graph.
        with torch.no_grad(), self._autocast():
            scoring_logits = self.model(inputs)
            per_example = self.criterion(scoring_logits, targets)
        result = selector(per_example)

        indices = result.indices
        if indices is None:  # selector kept everything
            indices = torch.arange(batch_size, device=inputs.device)

        # Pass 2: forward + backward on the selected subset only.
        with self._autocast():
            sub_logits = self.model(inputs[indices])
            sub_loss = self.criterion(sub_logits, targets[indices]).mean()

        m = int(indices.numel())
        cost = batch_size * 1.0 + m * (1.0 + _BACKWARD_COST)
        # Report the full-batch logits so train accuracy stays comparable across
        # modes (it describes the whole epoch, not just the selected subset).
        return sub_loss, result, scoring_logits.detach(), cost

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        selector = self._selector_for_epoch(epoch)
        cfg = self.config

        loss_sum = 0.0
        correct = 0
        examples_seen = 0
        effective_examples = 0
        cost_units = 0.0
        baseline_units = 0.0
        rank_sum = 0.0
        rank_steps = 0

        iterator = tqdm(
            loader,
            desc=f"epoch {epoch:>3} [{selector.name}]",
            disable=not (cfg.verbose and self.dist.is_main),
            leave=False,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Record the LR the epoch actually ran at, not the post-decay value.
        epoch_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader) if hasattr(loader, "__len__") else None

        for step, (inputs, targets) in enumerate(iterator):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if cfg.channels_last and inputs.ndim == 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            is_last_micro = ((step + 1) % cfg.accum_steps == 0) or (
                num_batches is not None and step + 1 == num_batches
            )

            # Skip DDP's gradient all-reduce on non-final micro-batches.
            sync_ctx = (
                self.model.no_sync()
                if (self.dist.enabled and not is_last_micro)
                else contextlib.nullcontext()
            )
            with sync_ctx:
                loss, result, logits, cost = self._forward_batch(
                    inputs, targets, selector
                )
                self.scaler.scale(loss / cfg.accum_steps).backward()

            if is_last_micro:
                if self.low_rank is not None:
                    self.scaler.unscale_(self.optimizer)
                    self.low_rank.apply(unwrap(self.model))
                    rank_sum += self.low_rank.mean_rank()
                    rank_steps += 1
                elif cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                if cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

            batch_size = targets.size(0)
            examples_seen += batch_size
            effective_examples += result.num_effective
            cost_units += cost
            baseline_units += batch_size * (1.0 + _BACKWARD_COST)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Reduce across ranks so the logged numbers describe the whole epoch.
        totals = {
            "loss_sum": all_reduce_sum(loss_sum, self.dist, self.device),
            "correct": all_reduce_sum(correct, self.dist, self.device),
            "examples_seen": all_reduce_sum(examples_seen, self.dist, self.device),
            "effective": all_reduce_sum(effective_examples, self.dist, self.device),
            "cost": all_reduce_sum(cost_units, self.dist, self.device),
            "baseline": all_reduce_sum(baseline_units, self.dist, self.device),
        }
        seen = totals["examples_seen"]
        return {
            "loss": totals["loss_sum"] / seen if seen else 0.0,
            "acc": totals["correct"] / seen if seen else 0.0,
            "examples_seen": int(seen),
            "effective": int(totals["effective"]),
            "effective_ratio": totals["effective"] / seen if seen else 0.0,
            "compute_ratio": (
                totals["cost"] / totals["baseline"] if totals["baseline"] else 1.0
            ),
            "seconds": elapsed,
            "images_per_sec": seen / elapsed if elapsed > 0 else 0.0,
            "mean_rank": rank_sum / rank_steps if rank_steps else 0.0,
            "lr": epoch_lr,
        }

    # ------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple:
        self.model.eval()
        correct1 = 0
        correct5 = 0
        total = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.config.channels_last and inputs.ndim == 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with self._autocast():
                logits = self.model(inputs)
            maxk = min(5, logits.size(1))
            _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
            hits = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct1 += hits[:, :1].sum().item()
            correct5 += hits.sum().item()
            total += targets.size(0)

        correct1 = all_reduce_sum(correct1, self.dist, self.device)
        correct5 = all_reduce_sum(correct5, self.dist, self.device)
        total = all_reduce_sum(total, self.dist, self.device)
        if not total:
            return 0.0, 0.0
        return correct1 / total, correct5 / total

    # ---------------------------------------------------------- checkpointing

    def _checkpoint_path(self, epoch: int) -> Optional[str]:
        if not self.config.ckpt_dir:
            return None
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        return os.path.join(self.config.ckpt_dir, f"epoch_{epoch:04d}.pt")

    def _save_checkpoint(self, epoch: int) -> None:
        path = self._checkpoint_path(epoch)
        if path is None or not self.dist.is_main:
            return
        torch.save(
            {
                "epoch": epoch,
                "model": unwrap(self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_type": type(self.optimizer).__name__,
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "scaler": self.scaler.state_dict(),
                "config": vars(self.config),
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        unwrap(self.model).load_state_dict(ckpt["model"])

        # Loading e.g. AdamW state into an SGD instance fails deep inside the
        # optimiser with an opaque KeyError; catch the mismatch up front.
        saved_optimizer = ckpt.get("optimizer_type")
        current_optimizer = type(self.optimizer).__name__
        if saved_optimizer is not None and saved_optimizer != current_optimizer:
            raise ValueError(
                f"checkpoint '{path}' was written with {saved_optimizer} but this "
                f"run uses {current_optimizer}. Pass the same --optimizer used for "
                f"the original run (and the same --lr/--weight-decay to match it)."
            )
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        self._pending_scheduler_state = ckpt.get("scheduler")
        self.start_epoch = int(ckpt["epoch"]) + 1
        main_print(self.dist, f"resumed from {path} at epoch {self.start_epoch}")

    # --------------------------------------------------------------- main fit

    def fit(self, train_loader: DataLoader, eval_loader: DataLoader) -> List[EpochStats]:
        steps_per_epoch = max(
            1, math.ceil(len(train_loader) / self.config.accum_steps)
        )
        self.scheduler = self._build_scheduler(steps_per_epoch)
        pending = getattr(self, "_pending_scheduler_state", None)
        if pending is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(pending)
            self._pending_scheduler_state = None

        for epoch in range(self.start_epoch, self.config.epochs):
            # Reshuffle differently each epoch on every rank.
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            stats_dict = self._train_epoch(train_loader, epoch)
            eval_top1, eval_top5 = self.evaluate(eval_loader)

            stats = EpochStats(
                epoch=epoch,
                train_loss=stats_dict["loss"],
                train_acc=stats_dict["acc"],
                eval_acc_top1=eval_top1,
                eval_acc_top5=eval_top5,
                examples_seen=stats_dict["examples_seen"],
                effective_examples=stats_dict["effective"],
                effective_ratio=stats_dict["effective_ratio"],
                compute_ratio=stats_dict["compute_ratio"],
                train_seconds=stats_dict["seconds"],
                images_per_sec=stats_dict["images_per_sec"],
                lr=stats_dict["lr"],
                mean_rank=stats_dict["mean_rank"],
            )
            self.history.append(stats)

            if self.logger is not None and self.dist.is_main:
                self.logger.log(stats.as_row())
            if self.config.verbose:
                main_print(
                    self.dist,
                    f"epoch {epoch:>3} | train_loss {stats.train_loss:.4f} "
                    f"train_acc {stats.train_acc:.4f} | eval_top1 {eval_top1:.4f} "
                    f"eval_top5 {eval_top5:.4f} | used {stats.effective_ratio:5.1%} "
                    f"| compute {stats.compute_ratio:5.1%} "
                    f"| {stats.images_per_sec:.0f} img/s"
                    + (
                        f" | rank {stats.mean_rank:.1f}"
                        if self.low_rank is not None
                        else ""
                    ),
                )

            ckpt_every = self.config.ckpt_every
            if ckpt_every and (epoch + 1) % ckpt_every == 0:
                self._save_checkpoint(epoch)

        if self.config.ckpt_dir and self.history:
            self._save_checkpoint(self.history[-1].epoch)
        return self.history
