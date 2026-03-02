import os
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib
import numpy as np


def check_nifti_corruption(
    items: List[Dict[str, str]],
    keys: Tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f"),
    max_workers: int = 8,
    force_full_read: bool = True,
    check_finite: bool = False,
    sample_slices: Optional[int] = None,
    verbose_every: int = 500,
) -> Dict[str, object]:
    """
    Try to read NIfTI files for each item and detect corruption.

    Args:
        items: list of modality dicts (each containing paths for `keys`)
        keys: modalities to test
        max_workers: thread parallelism (I/O bound)
        force_full_read: if True, forces reading image data (catches truncated/corrupt payload)
        check_finite: if True, checks for NaN/Inf (slower; requires reading data)
        sample_slices: if set (e.g., 8), reads only a few z-slices to reduce IO.
                      If None, reads entire volume when force_full_read=True.
        verbose_every: print progress every N files

    Returns:
        dict with summary and per-file failures.
    """

    def _read_one(path: str) -> Tuple[bool, str]:
        # returns (ok, reason)
        if not path or not os.path.isfile(path):
            return False, "missing_file"

        try:
            img = nib.load(path)  # parse header + prepare proxy
            _ = img.header.get_data_shape()  # basic header access

            if force_full_read or check_finite or (sample_slices is not None):
                dataobj = img.dataobj  # array proxy (lazy)

                if sample_slices is not None:
                    # Read a subset: center slice + a few around it (works for 3D; for 4D reads first volume)
                    shape = img.shape
                    if len(shape) < 3:
                        # still try full read; uncommon for your data
                        arr = np.asanyarray(dataobj)
                    else:
                        z = shape[2]
                        if z == 0:
                            return False, "zero_z_dim"
                        # choose indices around center
                        center = z // 2
                        half = max(1, sample_slices // 2)
                        z0 = max(0, center - half)
                        z1 = min(z, center + half)
                        slc = (slice(None), slice(None), slice(z0, z1))
                        if len(shape) == 4:
                            slc = slc + (0,)  # first timepoint/channel if 4D
                        arr = np.asanyarray(dataobj[slc])
                else:
                    # Full read
                    arr = np.asanyarray(dataobj)

                if check_finite:
                    # convert to float32 for fast finite check if needed
                    arrf = arr.astype(np.float32, copy=False)
                    if not np.isfinite(arrf).all():
                        return False, "non_finite_values"

            return True, "ok"

        except Exception as e:
            # Keep it concise but informative
            return False, f"{type(e).__name__}: {e}"

    # Flatten all paths to check, but keep provenance
    file_jobs: List[Tuple[int, str, str]] = []
    for i, it in enumerate(items):
        for k in keys:
            p = it.get(k)
            file_jobs.append((i, k, p))

    failures: List[Dict[str, str]] = []
    ok_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_read_one, p): (idx, key, p) for (idx, key, p) in file_jobs}
        for n, fut in enumerate(as_completed(futs), start=1):
            idx, key, p = futs[fut]
            ok, reason = fut.result()
            if ok:
                ok_count += 1
            else:
                failures.append({
                    "item_index": str(idx),
                    "modality": key,
                    "path": p or "",
                    "reason": reason,
                    "case_id": str(items[idx].get("case_id", "")),
                    "source_dataset": str(items[idx].get("source_dataset", "")),
                    "subsource": str(items[idx].get("subsource", "")),
                })

            if verbose_every and (n % verbose_every == 0):
                print(f"[check_nifti_corruption] checked {n}/{len(file_jobs)} files... failures={len(failures)}")

    total = len(file_jobs)
    summary = {
        "total_files": total,
        "ok_files": ok_count,
        "failed_files": len(failures),
        "failure_rate": (len(failures) / total) if total else 0.0,
        "failures": failures,  # list of dicts (can be large)
    }
    return summary