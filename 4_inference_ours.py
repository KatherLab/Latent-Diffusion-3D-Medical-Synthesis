#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UNEST inference on validation split + per-dataset metrics."""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.engine.collate import dict_collate
from utils.engine.infer_common import run_inference_and_metrics
from utils.engine.logger import make_logger
from utils.brain_data_utils import get_dataset


def load_state_dict_fuzzy(model, weight_path: str, strict: bool = True):
    sd = torch.load(weight_path, map_location="cpu")
    if isinstance(sd, dict) and "module" in sd:
        sd = sd["module"]
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_sd[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new_sd, strict=strict)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config_unet3d_rcg.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-range", type=float, default=1.0)
    p.add_argument("--slice-stride", type=int, default=2)
    p.add_argument("--slice-max-slices", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    logger = make_logger(str(cfg.logdir), name="infer_ours")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, dataset_val = get_dataset(cfg)
    val_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=dict_collate,
    )

    from guided_diffusion.unet_raw_3d import UNetModel
    model = UNetModel(dims=3, image_size=96, in_channels=2, model_channels=96, out_channels=2, num_res_blocks=1, attention_resolutions=[32,16,8], channel_mult=[1,2,2,2]).to(device)
    model = load_state_dict_fuzzy(model, args.ckpt, strict=False)
    model.eval()

    def forward_fn(images, batch):
        return model(images)

    overall, per_ds = run_inference_and_metrics(
        model=model,
        forward_fn=forward_fn,
        dataloader=val_loader,
        device=device,
        data_range=float(args.data_range),
        slice_stride=int(args.slice_stride),
        slice_max_slices=int(args.slice_max_slices),
        show_pbar=True,
    )

    logger.info(f"[OVERALL] {overall}")
    for ds, payload in sorted(per_ds.items()):
        n = payload["n"]
        if n <= 0:
            continue
        avg = {k: float(payload["sums"][k]) / float(n) for k in payload["sums"].keys()}
        logger.info(f"[{ds}] n={n} {avg}")


if __name__ == "__main__":
    main()
