from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
from torch.cuda.amp import GradScaler


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    ckpt_dir: str,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{tag}.pt")
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (scheduler.state_dict() if scheduler is not None else None),
        "scaler": scaler.state_dict(),
        "extra": extra or {},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    unwrap_model(model).load_state_dict(state, strict=strict)

    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
