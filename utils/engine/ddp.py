from __future__ import annotations

import os
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def setup_ddp_from_env() -> tuple[bool, int]:
    """Single GPU/CPU by default; enable DDP via torchrun env (WORLD_SIZE, LOCAL_RANK)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return distributed, local_rank


def cleanup_ddp() -> None:
    if is_distributed():
        dist.destroy_process_group()


def ddp_reduce_mean(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not is_distributed():
        return metrics
    keys = sorted(metrics.keys())
    t = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / float(get_world_size())
    return {k: float(v) for k, v in zip(keys, t.tolist())}
