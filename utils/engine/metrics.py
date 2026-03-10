from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def mae3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


@torch.no_grad()
def mse3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    mse = torch.clamp(mse, min=eps)
    max_t = torch.tensor(float(max_val), device=mse.device, dtype=mse.dtype)
    return 20.0 * torch.log10(max_t) - 10.0 * torch.log10(mse)


def _gaussian_kernel2d(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / torch.sum(g)
    k2d = torch.outer(g, g)
    return k2d


def ssim2d(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Vectorized SSIM for 2D images. Inputs: [N,1,H,W]."""
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected [N,1,H,W], got {x.shape} and {y.shape}")
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}")
    # Force same dtype/device for SSIM convs (avoid AMP fp16/fp32 mismatch)
    if hasattr(x, "as_tensor"):
        x = x.as_tensor()
    if hasattr(y, "as_tensor"):
        y = y.as_tensor()

    x = x.float()
    y = y.float()
    device, dtype = x.device, x.dtype
    k = _gaussian_kernel2d(window_size, sigma, device=device, dtype=dtype)
    k = k.view(1, 1, window_size, window_size)

    pad = window_size // 2
    mu_x = F.conv2d(x, k, padding=pad)
    mu_y = F.conv2d(y, k, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, k, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, k, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, k, padding=pad) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + 1e-12)
    return torch.mean(ssim_map)


@torch.no_grad()
def slice_metrics_2d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    stride: int = 1,
    max_slices: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute slice-wise 2D metrics on tensors [B,1,D,H,W], averaged over sampled slices."""
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(f"Expected [B,1,D,H,W], got {pred.shape}")
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    B, C, D, H, W = pred.shape
    if C != 1:
        raise ValueError("slice_metrics_2d expects C=1; call per modality channel.")

    idx = torch.arange(0, D, step=max(1, int(stride)), device=pred.device)
    if max_slices and max_slices > 0 and idx.numel() > max_slices:
        lin = torch.linspace(0, idx.numel() - 1, steps=max_slices, device=pred.device)
        idx = idx[lin.round().long()]

    pred_s = pred[:, :, idx, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W)
    tgt_s = target[:, :, idx, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W)

    mae2 = torch.mean(torch.abs(pred_s - tgt_s))
    mse2 = torch.mean((pred_s - tgt_s) ** 2)
    psnr2 = psnr_from_mse(mse2, max_val=float(data_range))
    ssim2 = ssim2d(pred_s, tgt_s, data_range=float(data_range))

    return {"mae_2d": mae2, "psnr_2d": psnr2, "ssim_2d": ssim2}
