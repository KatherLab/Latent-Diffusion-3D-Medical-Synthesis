#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Updated inference script with optional NIfTI dumping and stable metrics.

Put this in your project as e.g. 4_inference_medsyn_updated.py and run.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# local imports (assume your repo structure)
from utils.engine.collate import dict_collate
from utils.engine.logger import make_logger
from utils.brain_data_utils import get_dataset
from guided_diffusion.unet_raw_3d import UNetModel

# reuse your metrics helpers
from utils.engine.metrics import mae3d, mse3d, psnr_from_mse, slice_metrics_2d


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config_medsyn.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-range", type=float, default=1.0)
    p.add_argument("--slice-stride", type=int, default=2)
    p.add_argument("--slice-max-slices", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="", help="If set, save per-case NIfTI outputs here")
    p.add_argument("--save-limit", type=int, default=0, help="If >0, only save first N cases")
    p.add_argument("--save-every", type=int, default=1, help="Save every k-th case")
    return p.parse_args()


# -----------------------
# Helpers (I/O)
# -----------------------
def save_nii(vol_3d: torch.Tensor | np.ndarray, out_path: str, spacing=None):
    """Save 3D volume (D,H,W) as NIfTI via SimpleITK."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if torch.is_tensor(vol_3d):
        arr = vol_3d.detach().cpu().numpy()
    else:
        arr = np.asarray(vol_3d)
    # Ensure float32
    arr = arr.astype(np.float32)

    img = sitk.GetImageFromArray(arr)
    if spacing is not None:
        try:
            img.SetSpacing(tuple(float(x) for x in spacing))
        except Exception:
            pass
    sitk.WriteImage(img, out_path)


def load_state_dict_fuzzy(model: torch.nn.Module, weight_path: str, strict: bool = True):
    sd = torch.load(weight_path, map_location="cpu")
    # unwrap possible "module" container
    if isinstance(sd, dict) and ("module" in sd or any(k.startswith("module.") for k in sd.keys())):
        if "module" in sd and isinstance(sd["module"], dict):
            sd = sd["module"]
        else:
            # patch keys
            new = {}
            for k, v in sd.items():
                k = str(k)
                new[k[7:] if k.startswith("module.") else k] = v
            sd = new
    # final fuzzy load
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError:
        # attempt fuzzy key strip for "module." prefix if any remain
        new = {}
        for k, v in sd.items():
            sk = k[7:] if k.startswith("module.") else k
            new[sk] = v
        model.load_state_dict(new, strict=strict)
    return model


# -----------------------
# Stable inference+metrics (local copy, cast to float32)
# -----------------------
@torch.no_grad()
def run_inference_and_metrics_safe(
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
    """
    Slightly modified copy of your infer_common.run_inference_and_metrics:
      - disables autocast for the forward pass (avoid mixed dtype issues inside ssim kernels)
      - ensures pred and labels are converted to float32 before metric ops
    """
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

        # build inputs / labels (keep channels in dim=1)
        images = torch.cat([batch["t1n"], batch["t2w"]], dim=1).to(device, non_blocking=True)
        labels = torch.cat([batch["t1c"], batch["t2f"]], dim=1).to(device, non_blocking=True)

        # IMPORTANT: do NOT use autocast here to avoid dtype inconsistencies inside metric kernels.
        with autocast(enabled=False):
            pred = forward_fn(images, batch)

        # Force metrics to compute in float32 to avoid "weight/input dtype mismatch" errors
        pred = pred.float()
        labels = labels.float()

        # per-channel 3D metrics
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


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # make sure val transform is float32 (defensive)
    # get_dataset returns (train_ds, val_ds) where dataset_val.transform should be accessible
    _, dataset_val = get_dataset(cfg)
    # if dataset_val is a wrapper with attribute transform, try to set it
    try:
        if hasattr(dataset_val, "transform") and hasattr(dataset_val.transform, "output_dtype"):
            dataset_val.transform.output_dtype = torch.float32
    except Exception:
        pass

    logger = make_logger(str(cfg.logdir), name="infer_updated") if "logdir" in cfg else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    val_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=dict_collate,
    )

    # build model and load checkpoint
    model = UNetModel(dims=3, image_size=96, in_channels=2, model_channels=96, out_channels=2, num_res_blocks=1, attention_resolutions=[32,16,8], channel_mult=[1,2,2,2]).to(device)
    model = load_state_dict_fuzzy(model, args.ckpt, strict=False)
    model.eval()

    def forward_fn(images, batch):
        # images expected shape [B,2,D,H,W]
        return model(images)

    # Optionally: dump a few cases to disk before metrics
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        n_saved = 0
        model.eval()
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if args.save_limit and n_saved >= args.save_limit:
                    break
                if (bi % max(1, int(args.save_every))) != 0:
                    continue

                # assemble images and labels as in training loop
                # note: batch values may be tensors or lists of tensors depending on collate
                try:
                    t1n = batch["t1n"].to(device, non_blocking=True)
                    t2w = batch["t2w"].to(device, non_blocking=True)
                    t1c = batch["t1c"].to(device, non_blocking=True)
                    t2f = batch["t2f"].to(device, non_blocking=True)
                except Exception as e:
                    print(f"[WARN] skipping save for bi={bi} due to missing keys: {e}", file=sys.stderr)
                    continue

                images = torch.cat([t1n, t2w], dim=1)  # [B,2,D,H,W]
                labels = torch.cat([t1c, t2f], dim=1)  # [B,2,D,H,W]

                # forward (no autocast to keep dtypes stable)
                with autocast(enabled=False):
                    pred = forward_fn(images, batch)
                pred = pred.float()

                # case identifiers
                case_id = "unknown"
                source_dataset = "unknown"
                subsource = "unknown"
                if "meta" in batch:
                    m = batch["meta"][0] if isinstance(batch["meta"], (list, tuple)) else batch["meta"]
                    if isinstance(m, dict):
                        case_id = str(m.get("case_id", case_id))
                        source_dataset = str(m.get("source_dataset", source_dataset))
                        subsource = str(m.get("subsource", subsource))
                else:
                    if "case_id" in batch:
                        case_id = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else batch["case_id"]
                    if "source_dataset" in batch:
                        source_dataset = batch["source_dataset"][0] if isinstance(batch["source_dataset"], (list, tuple)) else batch["source_dataset"]
                    if "subsource" in batch:
                        subsource = batch["subsource"][0] if isinstance(batch["subsource"], (list, tuple)) else batch["subsource"]

                case_id = str(case_id)
                source_dataset = str(source_dataset)
                subsource = str(subsource)

                out_dir = os.path.join(args.save_dir, f"{source_dataset}_{subsource}", case_id)
                os.makedirs(out_dir, exist_ok=True)

                # save channels: index 0 -> t1c / index 1 -> t2f for labels/preds? (training uses order t1c, t2f)
                # We saved images as [t1n,t2w] and labels as [t1c,t2f], preds in same order.
                B = images.shape[0]
                # use first sample in batch
                save_nii(images[0, 0].cpu(), os.path.join(out_dir, "t1n.nii.gz"))
                save_nii(images[0, 1].cpu(), os.path.join(out_dir, "t2w.nii.gz"))
                save_nii(labels[0, 0].cpu(), os.path.join(out_dir, "t1c.nii.gz"))
                save_nii(labels[0, 1].cpu(), os.path.join(out_dir, "t2f.nii.gz"))
                save_nii(pred[0, 0].cpu(), os.path.join(out_dir, "pred_t1c.nii.gz"))
                save_nii(pred[0, 1].cpu(), os.path.join(out_dir, "pred_t2f.nii.gz"))
                save_nii((pred[0, 0] - labels[0, 0]).abs().cpu(), os.path.join(out_dir, "abs_err_t1c.nii.gz"))
                save_nii((pred[0, 1] - labels[0, 1]).abs().cpu(), os.path.join(out_dir, "abs_err_t2f.nii.gz"))

                with open(os.path.join(out_dir, "info.txt"), "w") as f:
                    f.write(f"case_id: {case_id}\n")
                    f.write(f"source_dataset: {source_dataset}\n")
                    f.write(f"subsource: {subsource}\n")
                    f.write(f"bi: {bi}\n")
                    f.write(f"images_shape: {tuple(images.shape)}\n")
                    f.write(f"labels_shape: {tuple(labels.shape)}\n")
                    f.write(f"pred_shape: {tuple(pred.shape)}\n")

                n_saved += 1

        if logger:
            logger.info(f"[DUMP] Saved {n_saved} cases to {args.save_dir}")
        else:
            print(f"[DUMP] Saved {n_saved} cases to {args.save_dir}")

    # run stable inference + metrics using the safe function
    overall, per_ds = run_inference_and_metrics_safe(
        model=model,
        forward_fn=forward_fn,
        dataloader=val_loader,
        device=device,
        data_range=float(args.data_range),
        slice_stride=int(args.slice_stride),
        slice_max_slices=int(args.slice_max_slices),
        show_pbar=True,
    )

    if logger:
        logger.info(f"[OVERALL] {overall}")
        for ds, payload in sorted(per_ds.items()):
            n = payload["n"]
            if n <= 0:
                continue
            avg = {k: float(payload["sums"][k]) / float(n) for k in payload["sums"].keys()}
            logger.info(f"[{ds}] n={n} {avg}")
    else:
        print("[OVERALL]", overall)
        for ds, payload in sorted(per_ds.items()):
            n = payload["n"]
            if n <= 0:
                continue
            avg = {k: float(payload["sums"][k]) / float(n) for k in payload["sums"].keys()}
            print(f"[{ds}] n={n} {avg}")


if __name__ == "__main__":
    main()