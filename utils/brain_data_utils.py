from __future__ import annotations

"""
utils/brain_data_utils.py (project-patched)

Changes vs your current file:
- Validation transform output_dtype changed to float32 for metric correctness (avoid fp16 metric bias).
- Adds optional "debug_force_val_patch_size" via env var DEBUG_VAL_PATCH_SIZE="128,128,128" to
  force val_patch_size without changing config (quick padding sanity check).
"""

from pathlib import Path
import os

import torch
from torch.utils.data import Dataset

from monai.data import CacheDataset, PersistentDataset

from .transforms import VAETransformMRI
from .local_path import get_data_path


class MultiModalDataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self) -> int:
        return len(self.data)


def dict_collate(batch):
    """Collate for dict batches that may include strings (dataset name, case_id)."""
    if len(batch) == 0:
        return {}
    if not isinstance(batch[0], dict):
        raise TypeError("Expected batch of dicts.")
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def _maybe_force_val_patch_size(args):
    """
    Allow forcing val patch size via env var for quick "are metrics fooled by padding?" check.
    Example:
      export DEBUG_VAL_PATCH_SIZE=128,128,128
    """
    s = os.environ.get("DEBUG_VAL_PATCH_SIZE", "").strip()
    if not s:
        return
    try:
        parts = [int(x) for x in s.replace(" ", "").split(",") if x]
        if len(parts) == 3:
            args.val_patch_size = parts
            print(f"[DEBUG] Forced args.val_patch_size={args.val_patch_size} via DEBUG_VAL_PATCH_SIZE")
    except Exception:
        print(f"[DEBUG] Failed to parse DEBUG_VAL_PATCH_SIZE='{s}', ignoring.")


def get_transforms(args):
    _maybe_force_val_patch_size(args)

    # Preserve these keys through the transform pipeline.
    additional_keys = ["case_id", "source_dataset", "subsource"]

    train_transform = VAETransformMRI(
        is_train=True,
        random_aug=bool(getattr(args, "random_aug", True)),
        k=4,
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float32,
        spacing_type=getattr(args, "spacing_type", "rand_zoom"),
        spacing=getattr(args, "spacing", None),
        image_keys=["t1c", "t1n", "t2f", "t2w"],
        label_keys=[],
        additional_keys=additional_keys,
        select_channel=0,
    )

    # IMPORTANT: output_dtype float32 for accurate validation metrics.
    val_transform = VAETransformMRI(
        is_train=False,
        random_aug=False,
        k=4,
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float32,
        spacing_type=getattr(args, "spacing_type", "rand_zoom"),
        spacing=getattr(args, "spacing", None),
        image_keys=["t1c", "t1n", "t2f", "t2w"],
        label_keys=[],
        additional_keys=additional_keys,
        select_channel=0,
    )
    return train_transform, val_transform


def get_dataset(args):
    train_files, val_files = get_data_path(
        train_ratio=float(getattr(args, "train_ratio", 0.8)),
        seed=int(getattr(args, "split_seed", 2026)),
        stratify_key=str(getattr(args, "stratify_key", "source_dataset")),
        num_workers=int(getattr(args, "index_num_workers", 16)),
    )

    train_transform, val_transform = get_transforms(args)

    print(f"Total number of training cases is {len(train_files)}.")
    print(f"Total number of validation cases is {len(val_files)}.")

    cache_rate = float(getattr(args, "cache_rate", 0.0))
    persistent_cache_dir = str(getattr(args, "persistent_cache_dir", "")).strip()

    if persistent_cache_dir:
        Path(persistent_cache_dir).mkdir(parents=True, exist_ok=True)
        dataset_train = PersistentDataset(data=train_files, transform=train_transform, cache_dir=persistent_cache_dir)
        dataset_val = PersistentDataset(data=val_files, transform=val_transform, cache_dir=persistent_cache_dir)
        print(f"[DATA] Using PersistentDataset cache_dir={persistent_cache_dir}")
        return dataset_train, dataset_val

    if cache_rate and cache_rate > 0:
        dataset_train = CacheDataset(data=train_files, transform=train_transform, cache_rate=cache_rate, num_workers=0)
        dataset_val = CacheDataset(data=val_files, transform=val_transform, cache_rate=min(1.0, cache_rate), num_workers=0)
        print(f"[DATA] Using CacheDataset cache_rate={cache_rate}")
        return dataset_train, dataset_val

    dataset_train = MultiModalDataset(data=train_files, transform=train_transform)
    dataset_val = MultiModalDataset(data=val_files, transform=val_transform)
    return dataset_train, dataset_val