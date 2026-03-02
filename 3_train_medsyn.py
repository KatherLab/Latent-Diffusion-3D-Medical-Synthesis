#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_train_medsyn.py (refactored)

MedSyn Unet3D supervised training with shared engine:
- single GPU by default (no torch.distributed.launch needed)
- optional DDP via torchrun
- early stopping, periodic checkpoints, rich TensorBoard logging
- per-dataset validation metrics (3D + slice-wise 2D)

Smoke test (fast):
  python 3_train_medsyn.py --config configs/config_medsyn.yaml --max-train-batches 10 --max-val-batches 10
"""

from __future__ import annotations

import argparse
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.engine.collate import dict_collate
from utils.engine.ddp import setup_ddp_from_env, cleanup_ddp, is_main_process
from utils.engine.logger import make_logger
from utils.engine.train_supervised import TrainConfig, run_training_supervised
from utils.brain_data_utils import get_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config_medsyn.yaml")

    # dataloader / speed
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)

    # logging / ckpt / early stop
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--save-every-epochs", type=int, default=5)
    p.add_argument("--early-patience", type=int, default=30)
    p.add_argument("--early-min-delta", type=float, default=0.0)

    # metrics
    p.add_argument("--data-range", type=float, default=1.0)
    p.add_argument("--slice-stride", type=int, default=2)
    p.add_argument("--slice-max-slices", type=int, default=0)

    # smoke test (limits batches per epoch; does NOT change dataset size)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)

    return p.parse_args()


class _LimitedLoader:
    """Cap the number of batches yielded from an existing DataLoader (fast smoke tests)."""
    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = int(max_batches)

    def __len__(self):
        if self.max_batches and self.max_batches > 0:
            return min(len(self.loader), self.max_batches)
        return len(self.loader)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.max_batches and self.max_batches > 0 and i >= self.max_batches:
                break
            yield batch


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    distributed, local_rank = setup_ddp_from_env()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger = make_logger(str(cfg.logdir), name="train_medsyn") if is_main_process() else None

    dataset_train, dataset_val = get_dataset(cfg)

    train_loader = DataLoader(
        dataset_train,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        drop_last=True,
        collate_fn=dict_collate,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(int(args.num_workers), 2)),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        drop_last=False,
        collate_fn=dict_collate,
    )

    # Optional: cap batches for fast smoke tests
    if args.max_train_batches and args.max_train_batches > 0:
        train_loader = _LimitedLoader(train_loader, args.max_train_batches)
    if args.max_val_batches and args.max_val_batches > 0:
        val_loader = _LimitedLoader(val_loader, args.max_val_batches)

    # FIX: correct MedSyn model instantiation
    from models.medsyn.train_low_res import Unet3D
    model = Unet3D(dim=16, channels=2).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), eps=1e-6)

    train_cfg = TrainConfig(
        max_epochs=int(cfg.n_epochs),
        logdir=str(cfg.logdir),
        ckpt_dir=os.path.join(str(cfg.logdir), "checkpoints"),
        lr=float(cfg.lr),
        batch_size=int(cfg.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        log_every_steps=int(args.log_every_steps),
        save_every_epochs=int(args.save_every_epochs),
        val_every_epochs=1,
        max_val_batches=int(args.max_val_batches),
        data_range=float(args.data_range),
        perceptual_weight=0.0,
        slice_stride=int(args.slice_stride),
        slice_max_slices=int(args.slice_max_slices),
        early_patience=int(args.early_patience),
        early_min_delta=float(args.early_min_delta),
    )

    def forward_fn(images, batch):
        return model(images)

    run_training_supervised(
        model=model,
        forward_fn=forward_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=train_cfg,
        logger=logger,
        optimizer=optimizer,
        scheduler=None,
        perceptual_loss_fn=None,
    )

    cleanup_ddp()


if __name__ == "__main__":
    main()
