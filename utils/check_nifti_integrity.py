from __future__ import annotations

"""Quick NIfTI integrity scan.

Usage:
  python -m utils.check_nifti_integrity --max 0 --workers 16

It loads all nifti paths from utils/local_path.py get_data_path(), then attempts to read
header + data via nibabel. Any failures are reported.

Tip: start with --max 200 for a fast smoke test.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import nibabel as nib

from utils.local_path import get_data_path


def _try_load(path: str) -> tuple[bool, str, str]:
    try:
        img = nib.load(path)
        _ = img.header
        # Touch data to detect truncated/corrupt payload
        _ = img.get_fdata(dtype=None, caching='unchanged')
        return True, path, ""
    except Exception as e:
        return False, path, f"{type(e).__name__}: {e}"


def scan_all_niftis(max_files: int = 0, workers: int = 16) -> dict:
    train, val = get_data_path()
    items = list(train) + list(val)

    paths = []
    for it in items:
        for k in ("t1n", "t1c", "t2w", "t2f"):
            if k in it and it[k]:
                paths.append(it[k])

    # de-dup
    paths = sorted(set(paths))
    if max_files and max_files > 0:
        paths = paths[:max_files]

    ok = 0
    bad = []

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_try_load, p) for p in paths]
        for fut in as_completed(futs):
            good, p, err = fut.result()
            if good:
                ok += 1
            else:
                bad.append((p, err))

    return {"total": len(paths), "ok": ok, "bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="0=all files; otherwise cap")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    res = scan_all_niftis(max_files=args.max, workers=args.workers)
    print(f"Scanned: {res['total']}  OK: {res['ok']}  BAD: {len(res['bad'])}")
    for p, err in res["bad"][:50]:
        print(f"[BAD] {p} :: {err}")
    if len(res["bad"]) > 50:
        print(f"... and {len(res['bad'])-50} more")


if __name__ == "__main__":
    main()
