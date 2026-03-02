from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .ddp import ddp_reduce_mean, is_main_process
from .early_stop import EarlyStopping
from .metrics import mae3d, mse3d, psnr_from_mse, slice_metrics_2d


@dataclass
class TrainConfig:
    max_epochs: int
    logdir: str
    ckpt_dir: str
    lr: float
    batch_size: int

    num_workers: int = 4
    prefetch_factor: int = 2
    log_every_steps: int = 50
    save_every_epochs: int = 5
    val_every_epochs: int = 1
    max_val_batches: int = 0

    data_range: float = 1.0
    perceptual_weight: float = 0.0

    slice_stride: int = 2
    slice_max_slices: int = 0

    early_patience: int = 30
    early_min_delta: float = 0.0


def _infer_dataset_name(batch: dict) -> str:
    for k in ("source_dataset", "dataset"):
        if k in batch:
            v = batch[k]
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return str(v[0])
            return str(v)
    return "unknown"


@torch.no_grad()
def _validate_supervised(
    *,
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    forward_fn: Callable[[torch.Tensor, Any], torch.Tensor],
    loss_fn: torch.nn.Module,
    perceptual_loss_fn: Optional[torch.nn.Module],
    cfg: TrainConfig,
    show_pbar: bool,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    model.eval()
    sums = defaultdict(float)
    n = 0
    per_ds: dict[str, dict[str, Any]] = {}

    total = len(val_loader)
    if cfg.max_val_batches and cfg.max_val_batches > 0:
        total = min(total, cfg.max_val_batches)

    pbar = tqdm(total=total, desc="Val", leave=False, disable=(not show_pbar))

    for bi, batch in enumerate(val_loader):
        if cfg.max_val_batches and cfg.max_val_batches > 0 and bi >= cfg.max_val_batches:
            break

        ds = _infer_dataset_name(batch)
        per_ds.setdefault(ds, {"n": 0, "sums": defaultdict(float)})

        images = torch.cat([batch["t1n"], batch["t2w"]], dim=1).to(device, non_blocking=True)
        labels = torch.cat([batch["t1c"], batch["t2f"]], dim=1).to(device, non_blocking=True)

        with autocast(enabled=torch.cuda.is_available()):
            pred = forward_fn(images, batch)
            l1 = loss_fn(pred, labels)

            lp_t1c = torch.tensor(0.0, device=device)
            lp_t2f = torch.tensor(0.0, device=device)
            if perceptual_loss_fn is not None and cfg.perceptual_weight > 0:
                lp_t1c = perceptual_loss_fn(pred[:, 0:1].float(), labels[:, 0:1].float())
                lp_t2f = perceptual_loss_fn(pred[:, 1:2].float(), labels[:, 1:2].float())

            total_loss = l1 + cfg.perceptual_weight * (lp_t1c + lp_t2f)

        # 3D metrics
        mae_t1c = mae3d(pred[:, 0:1], labels[:, 0:1])
        mae_t2f = mae3d(pred[:, 1:2], labels[:, 1:2])
        mae_c = 0.5 * (mae_t1c + mae_t2f)

        psnr_t1c = psnr_from_mse(mse3d(pred[:, 0:1], labels[:, 0:1]), max_val=cfg.data_range)
        psnr_t2f = psnr_from_mse(mse3d(pred[:, 1:2], labels[:, 1:2]), max_val=cfg.data_range)
        psnr_c = 0.5 * (psnr_t1c + psnr_t2f)

        # 2D slice-wise metrics
        sm_t1c = slice_metrics_2d(pred[:, 0:1], labels[:, 0:1], data_range=cfg.data_range, stride=cfg.slice_stride, max_slices=cfg.slice_max_slices)
        sm_t2f = slice_metrics_2d(pred[:, 1:2], labels[:, 1:2], data_range=cfg.data_range, stride=cfg.slice_stride, max_slices=cfg.slice_max_slices)

        ssim2d_c = 0.5 * (sm_t1c["ssim_2d"] + sm_t2f["ssim_2d"])
        psnr2d_c = 0.5 * (sm_t1c["psnr_2d"] + sm_t2f["psnr_2d"])
        mae2d_c = 0.5 * (sm_t1c["mae_2d"] + sm_t2f["mae_2d"])

        lp_c = 0.5 * (lp_t1c + lp_t2f)

        batch_metrics = {
            "val/loss": float(total_loss.item()),
            "val/l1": float(l1.item()),
            "val/lpips3d_proxy_avg": float(lp_c.item()),
            "val3d/mae_avg": float(mae_c.item()),
            "val3d/psnr_avg": float(psnr_c.item()),
            "val2d/ssim_avg": float(ssim2d_c.item()),
            "val2d/psnr_avg": float(psnr2d_c.item()),
            "val2d/mae_avg": float(mae2d_c.item()),
        }

        for k, v in batch_metrics.items():
            sums[k] += v
            per_ds[ds]["sums"][k] += v

        n += 1
        per_ds[ds]["n"] += 1

        if show_pbar:
            pbar.set_postfix(loss=batch_metrics["val/loss"], psnr3d=batch_metrics["val3d/psnr_avg"], ssim2d=batch_metrics["val2d/ssim_avg"])
            pbar.update(1)

    if show_pbar:
        pbar.close()

    overall = {k: (v / n if n else float("inf")) for k, v in sums.items()}
    return overall, per_ds


def run_training_supervised(
    *,
    model: torch.nn.Module,
    forward_fn: Callable[[torch.Tensor, Any], torch.Tensor],
    train_loader,
    val_loader,
    device: torch.device,
    cfg: TrainConfig,
    logger,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    perceptual_loss_fn: Optional[torch.nn.Module] = None,
) -> None:
    os.makedirs(cfg.logdir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(cfg.logdir) if is_main_process() else None

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), eps=1e-6)

    loss_fn = torch.nn.L1Loss(reduction="mean")
    scaler = GradScaler(enabled=torch.cuda.is_available(), init_scale=2.0 ** 8, growth_factor=1.5)

    early = EarlyStopping(patience=cfg.early_patience, min_delta=cfg.early_min_delta, mode="min")
    best_val = float("inf")
    global_step = 0

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        nb = 0

        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", disable=not is_main_process())
        for batch in train_loader:
            images = torch.cat([batch["t1n"], batch["t2w"]], dim=1).to(device, non_blocking=True)
            labels = torch.cat([batch["t1c"], batch["t2f"]], dim=1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                pred = forward_fn(images, batch)
                l1 = loss_fn(pred, labels)

                lp = torch.tensor(0.0, device=device)
                if perceptual_loss_fn is not None and cfg.perceptual_weight > 0:
                    lp = (perceptual_loss_fn(pred[:, 0:1].float(), labels[:, 0:1].float())
                          + perceptual_loss_fn(pred[:, 1:2].float(), labels[:, 1:2].float()))
                loss = l1 + cfg.perceptual_weight * lp

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            epoch_loss += float(loss.item())
            nb += 1

            if is_main_process():
                lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else float(optimizer.param_groups[0]["lr"])
                pbar.set_postfix(loss=float(loss.item()), lr=lr)
                pbar.update(1)

                if writer is not None and (global_step % cfg.log_every_steps == 0):
                    writer.add_scalar("train/loss", float(loss.item()), global_step)
                    writer.add_scalar("train/l1", float(l1.item()), global_step)
                    writer.add_scalar("train/perceptual", float(lp.item()), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.flush()

        if is_main_process():
            pbar.close()

        if scheduler is not None:
            scheduler.step()

        epoch_loss = (epoch_loss / nb) if nb else float("inf")
        if writer is not None and is_main_process():
            writer.add_scalar("train_epoch/loss", epoch_loss, epoch)
            writer.flush()

        # checkpoints
        if is_main_process():
            save_checkpoint(cfg.ckpt_dir, "latest", model, optimizer, scheduler, scaler, epoch, global_step)
            if cfg.save_every_epochs > 0 and ((epoch + 1) % cfg.save_every_epochs == 0):
                save_checkpoint(cfg.ckpt_dir, f"epoch{epoch:04d}", model, optimizer, scheduler, scaler, epoch, global_step)

        # validation
        if cfg.val_every_epochs > 0 and ((epoch + 1) % cfg.val_every_epochs == 0):
            metrics, per_ds = _validate_supervised(
                model=model,
                val_loader=val_loader,
                device=device,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
                perceptual_loss_fn=perceptual_loss_fn,
                cfg=cfg,
                show_pbar=is_main_process(),
            )

            metrics = ddp_reduce_mean(metrics, device=device)

            if is_main_process():
                logger.info(
                    f"[VAL] epoch={epoch} "
                    f"loss={metrics['val/loss']:.6f} "
                    f"psnr3d={metrics['val3d/psnr_avg']:.3f} "
                    f"mae3d={metrics['val3d/mae_avg']:.6f} "
                    f"ssim2d={metrics['val2d/ssim_avg']:.4f} "
                    f"psnr2d={metrics['val2d/psnr_avg']:.3f} "
                    f"mae2d={metrics['val2d/mae_avg']:.6f} "
                    f"lpips3d_proxy={metrics['val/lpips3d_proxy_avg']:.4f}"
                )

                if writer is not None:
                    for k, v in metrics.items():
                        writer.add_scalar(k, v, epoch)
                    writer.flush()

                # per-dataset summaries
                ds_names = sorted([d for d in per_ds.keys() if d not in ("unknown", "ALL")])
                for d in ds_names:
                    n_ = int(per_ds[d]["n"])
                    if n_ <= 0:
                        continue
                    avg = {k: float(per_ds[d]["sums"][k]) / float(n_) for k in per_ds[d]["sums"].keys()}
                    logger.info(f"  [VAL/{d}] n={n_} {avg}")

                if metrics["val/loss"] < best_val:
                    best_val = metrics["val/loss"]
                    p = save_checkpoint(cfg.ckpt_dir, "best", model, optimizer, scheduler, scaler, epoch, global_step, extra={"best_val_loss": best_val, **metrics})
                    logger.info(f"[CKPT] new best: {p} (best_val_loss={best_val:.6f})")

                if early.step(metrics["val/loss"]):
                    logger.info(f"[EARLY STOP] epoch={epoch} best={early.best:.6f} patience={early.patience}")
                    break

    if writer is not None and is_main_process():
        writer.close()
