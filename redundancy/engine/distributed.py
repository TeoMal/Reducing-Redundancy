"""Single-process / multi-process plumbing.

Everything here degrades to a no-op when the script is run normally (one
process, no ``torchrun``), so the rest of the code can call these helpers
unconditionally and stay readable.

Launch a distributed run with::

    torchrun --nproc_per_node=8 train.py --dataset imagenet --model vit_b_16 ...

``torchrun`` sets ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` in the environment;
:func:`init_distributed` picks them up. Without them we report world size 1 and
rank 0, which is exactly what the single-GPU and CPU paths want.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    """Where this process sits in the job."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def env_is_distributed() -> bool:
    """True when the process was launched by ``torchrun`` with >1 process."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(
        os.environ.get("WORLD_SIZE", "1")
    ) > 1


def init_distributed(backend: Optional[str] = None) -> DistInfo:
    """Join the process group if we were launched distributed, else no-op.

    The backend defaults to NCCL when CUDA is available and Gloo otherwise, so
    CPU-only smoke runs still work under ``torchrun``.
    """
    if not env_is_distributed():
        return DistInfo()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return DistInfo(
        rank=rank, local_rank=local_rank, world_size=world_size, enabled=True
    )


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(value: float, info: DistInfo, device: torch.device) -> float:
    """Sum a Python scalar across ranks (identity when not distributed).

    Used to turn per-rank metric accumulators into global ones so the logged
    accuracy/loss describe the whole epoch rather than rank 0's shard.
    """
    if not info.enabled or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def main_print(info: DistInfo, *args, **kwargs) -> None:
    """``print`` only on rank 0."""
    if info.is_main:
        print(*args, **kwargs)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module from a DDP / DataParallel wrapper."""
    return getattr(model, "module", model)
