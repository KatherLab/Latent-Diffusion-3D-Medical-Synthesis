from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from .metrics import mae3d, mse3d, psnr_from_mse, slice_metrics_2d


@torch.no_grad()
def run_inference_and_metrics(
    *,
    model: torch.nn.Module,
    forward_fn: Callable[[torch.Tensor, dict], torch.Tensor],
    dataloader,
    device: torch.device,
    data_range: float = 1.0,
    slice_stride: int = 2,
    slice_max_slices: int = 0,
    show_pbar: bool = True,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    model.eval()

    sums = defaultdict(float)
    n = 0
    per_ds: dict[str, dict[str, Any]] = {}

    pbar = tqdm(total=len(dataloader), desc="Infer", disable=(not show_pbar))
    for batch in dataloader:
        ds = "unknown"
        if "source_dataset" in batch:
            v = batch["source_dataset"]
            ds = str(v[0]) if isinstance(v, (list, tuple)) else str(v)

        per_ds.setdefault(ds, {"n": 0, "sums": defaultdict(float)})

        images = torch.cat([batch["t1n"], batch["t2w"]], dim=1).to(device, non_blocking=True)
        labels = torch.cat([batch["t1c"], batch["t2f"]], dim=1).to(device, non_blocking=True)

        with autocast(enabled=torch.cuda.is_available()):
            pred = forward_fn(images, batch)

        mae_c = 0.5 * (mae3d(pred[:, 0:1], labels[:, 0:1]) + mae3d(pred[:, 1:2], labels[:, 1:2]))
        psnr_c = 0.5 * (
            psnr_from_mse(mse3d(pred[:, 0:1], labels[:, 0:1]), max_val=data_range)
            + psnr_from_mse(mse3d(pred[:, 1:2], labels[:, 1:2]), max_val=data_range)
        )

        sm1 = slice_metrics_2d(pred[:, 0:1], labels[:, 0:1], data_range=data_range, stride=slice_stride, max_slices=slice_max_slices)
        sm2 = slice_metrics_2d(pred[:, 1:2], labels[:, 1:2], data_range=data_range, stride=slice_stride, max_slices=slice_max_slices)
        ssim2d = 0.5 * (sm1["ssim_2d"] + sm2["ssim_2d"])
        psnr2d = 0.5 * (sm1["psnr_2d"] + sm2["psnr_2d"])
        mae2d = 0.5 * (sm1["mae_2d"] + sm2["mae_2d"])

        bm = {
            "mae3d_avg": float(mae_c.item()),
            "psnr3d_avg": float(psnr_c.item()),
            "ssim2d_avg": float(ssim2d.item()),
            "psnr2d_avg": float(psnr2d.item()),
            "mae2d_avg": float(mae2d.item()),
        }

        for k, v in bm.items():
            sums[k] += v
            per_ds[ds]["sums"][k] += v
        n += 1
        per_ds[ds]["n"] += 1

        if show_pbar:
            pbar.set_postfix(psnr3d=bm["psnr3d_avg"], ssim2d=bm["ssim2d_avg"])
            pbar.update(1)

    if show_pbar:
        pbar.close()

    overall = {k: (v / n if n else float("nan")) for k, v in sums.items()}
    return overall, per_ds
