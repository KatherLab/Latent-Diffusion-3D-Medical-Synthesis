from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _safe_get_affine(meta: Optional[dict[str, Any]]) -> np.ndarray:
    if isinstance(meta, dict):
        aff = meta.get("affine", None)
        if aff is not None:
            a = np.asarray(aff)
            if a.shape == (4, 4):
                return a
    return np.eye(4, dtype=np.float32)


def _to_nii_array(x: torch.Tensor) -> np.ndarray:
    """
    Convert tensor [B,C,D,H,W] or [C,D,H,W] or [D,H,W] -> numpy [D,H,W] float32.
    We save per-channel separately, so caller passes a single channel.
    """
    if x.ndim == 5:
        x = x[0]  # C,D,H,W
    if x.ndim == 4:
        x = x[0]  # D,H,W
    x = x.detach().float().cpu().contiguous()
    return x.numpy().astype(np.float32)


def save_nifti_patch(
    *,
    out_path: str,
    vol: torch.Tensor,
    affine: np.ndarray,
) -> None:
    """
    Saves a 3D patch to .nii.gz using nibabel if installed.
    If nibabel is missing, saves .npy next to the target path.
    """
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    arr = _to_nii_array(vol)

    try:
        import nibabel as nib  # type: ignore
        img = nib.Nifti1Image(arr, affine)
        nib.save(img, out_path)
    except Exception as e:
        # Fallback: .npy
        npy_path = out_path.replace(".nii.gz", ".npy")
        np.save(npy_path, arr)
        # Write a tiny note so you know why it's .npy
        txt_path = out_path.replace(".nii.gz", ".txt")
        with open(txt_path, "w") as f:
            f.write(f"Failed to save NIfTI, fell back to NPY. Error: {repr(e)}\n")
            f.write(f"Saved array shape: {arr.shape}\n")


@torch.no_grad()
def dump_batch_niftis(
    *,
    out_dir: str,
    tag: str,
    epoch: int,
    bi: int,
    batch: dict[str, Any],
    pred: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> None:
    """
    Dumps a single batch worth of patches.
    Expected:
      batch has keys t1n,t2w,t1c,t2f tensors shaped [B,1,D,H,W]
      and maybe t1n_meta_dict, ... with affine/spatial_shape.
    pred/labels are [B,2,D,H,W] (t1c,t2f order as in your code).
    """
    # best-effort identifiers (may be missing)
    case_id = "unknown"
    ds = "unknown"
    if "case_id" in batch and isinstance(batch["case_id"], (list, tuple)) and len(batch["case_id"]) > 0:
        case_id = str(batch["case_id"][0])
    if "source_dataset" in batch and isinstance(batch["source_dataset"], (list, tuple)) and len(batch["source_dataset"]) > 0:
        ds = str(batch["source_dataset"][0])

    # pick an affine from any modality meta
    affine = _safe_get_affine(batch.get("t1n_meta_dict", None))

    base = os.path.join(out_dir, f"{tag}_ep{epoch:04d}_bi{bi:04d}_{ds}_{case_id}")
    Path(base).mkdir(parents=True, exist_ok=True)

    # Save inputs/targets
    for k in ("t1n", "t2w", "t1c", "t2f"):
        if k in batch and torch.is_tensor(batch[k]):
            save_nifti_patch(out_path=os.path.join(base, f"{k}.nii.gz"), vol=batch[k], affine=affine)

    # Save model outputs and error maps (if provided)
    if pred is not None:
        save_nifti_patch(out_path=os.path.join(base, "pred_t1c.nii.gz"), vol=pred[:, 0:1], affine=affine)
        save_nifti_patch(out_path=os.path.join(base, "pred_t2f.nii.gz"), vol=pred[:, 1:2], affine=affine)

    if labels is not None and pred is not None:
        err_t1c = torch.abs(pred[:, 0:1] - labels[:, 0:1])
        err_t2f = torch.abs(pred[:, 1:2] - labels[:, 1:2])
        save_nifti_patch(out_path=os.path.join(base, "abs_err_t1c.nii.gz"), vol=err_t1c, affine=affine)
        save_nifti_patch(out_path=os.path.join(base, "abs_err_t2f.nii.gz"), vol=err_t2f, affine=affine)

    # Save a small text summary (meta info)
    with open(os.path.join(base, "info.txt"), "w") as f:
        f.write(f"tag={tag}\n")
        f.write(f"epoch={epoch}\n")
        f.write(f"batch_index={bi}\n")
        f.write(f"source_dataset={ds}\n")
        f.write(f"case_id={case_id}\n")
        for mk in ("t1n_meta_dict", "t2w_meta_dict", "t1c_meta_dict", "t2f_meta_dict"):
            md = batch.get(mk, None)
            if isinstance(md, dict):
                f.write(f"\n[{mk}]\n")
                if "spatial_shape" in md:
                    f.write(f"spatial_shape={md['spatial_shape']}\n")
                if "original_affine" in md:
                    f.write("original_affine present\n")
                if "affine" in md:
                    f.write("affine present\n")