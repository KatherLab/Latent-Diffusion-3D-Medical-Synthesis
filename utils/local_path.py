#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified MRI dataset loader (multi-root capable) with:
- per-dataset loader specs
- multi-root merge inside a single dataset (e.g., BraTS2024 = GLI + MET + PED)
- automatic integrity validation
- faster IO via ThreadPoolExecutor
- deterministic split with optional stratification

IMPORTANT (fix for your current MONAI crash):
- We keep ONLY image paths at top-level keys: t1n,t1c,t2w,t2f
- All metadata goes under a nested dict: item["meta"] = {...}
  This prevents MONAI LoadImaged from trying to load metadata strings
  (e.g., "UPENN-GBM-00450_11") as if they were files.

This script indexes files and returns train/val dicts:
  {
    "t1n": "...nii(.gz)",
    "t1c": "...nii(.gz)",
    "t2w": "...nii(.gz)",
    "t2f": "...nii(.gz)",
    "meta": {"case_id":..., "source_dataset":..., "subsource":...}
  }

Usage:
  python -m utils.local_path
"""

from __future__ import annotations

import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed


# -----------------------------
# Common types / constants
# -----------------------------

ModalityMap = Dict[str, str]  # t1n/t1c/t2w/t2f are file paths; meta is nested dict
REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")

# Explicit blacklist for known corrupted cases (exclude whole case)
BLACKLIST_CASE_IDS = {
    "BraTS-PED-00255-000",  # corrupted t1c: not a gzip file
}


# -----------------------------
# Configuration (UPDATED)
# -----------------------------

DATA_ROOT = os.environ.get("MRI_DATA_ROOT", "/mnt/swarm_beta/xuewei/data")

# BraTS2021
BRATS21_TRAIN_ROOT = os.path.join(
    DATA_ROOT, "BraTS2021", "RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
)
BRATS21_VAL_ROOT = os.path.join(
    DATA_ROOT, "BraTS2021", "RSNA_ASNR_MICCAI_BraTS2021_ValidationData"
)

# BraTS2024 GLI
BRATS24_GLI_TRAIN_ROOT = os.path.join(DATA_ROOT, "BraTS2024", "training_data1")
BRATS24_GLI_VAL_ROOT = os.path.join(DATA_ROOT, "BraTS2024", "validation_data")

# BraTS2024 MET
BRATS24_MET_ROOT_1 = os.path.join(DATA_ROOT, "BraTS24_MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_1")
BRATS24_MET_ROOT_2 = os.path.join(DATA_ROOT, "BraTS24_MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_2")
BRATS24_MET_VAL_ROOT = os.path.join(DATA_ROOT, "BraTS24_MET", "MICCAI-BraTS2024-MET-Challenge-ValidationData")

# BraTS2024 PED
BRATS24_PED_TRAIN_ROOT = os.path.join(
    DATA_ROOT, "BraTS-PEDs2024_Training", "BraTS2024-PED-Challenge-TrainingData"
)
BRATS24_PED_VAL_ROOT = os.path.join(
    DATA_ROOT, "BraTS-PEDs2024_Training", "BraTS2024-PED-Challenge-ValidationData"
)

# EGD / UPENN / UCSF
EGD_ROOT = os.path.join(DATA_ROOT, "EGD")
UPENN_GBM_ROOT = os.path.join(DATA_ROOT, "UPENN-GBM")
UCSF_PDGM_ROOT = os.path.join(DATA_ROOT, "UCSF-PDGM")


# -----------------------------
# Exceptions
# -----------------------------

class DatasetConfigError(RuntimeError):
    pass

class ModalityIntegrityError(RuntimeError):
    pass


# -----------------------------
# Utility / validation
# -----------------------------

def _is_file(p: str) -> bool:
    return isinstance(p, str) and bool(p) and os.path.isfile(p)

def _pick_first(pattern: str) -> Optional[str]:
    hits = glob.glob(pattern)
    return hits[0] if hits else None

def _pick_first_any(patterns: Iterable[str]) -> Optional[str]:
    for pat in patterns:
        p = _pick_first(pat)
        if p:
            return p
    return None

def _assert_case_integrity(case_id: str, paths: ModalityMap) -> None:
    missing = [m for m in REQUIRED_MODALITIES if m not in paths or not paths[m]]
    if missing:
        raise ModalityIntegrityError(f"[{case_id}] missing modalities: {missing}")

    for m in REQUIRED_MODALITIES:
        if not _is_file(paths[m]):
            raise ModalityIntegrityError(f"[{case_id}] file not found for {m}: {paths[m]}")

    values = [paths[m] for m in REQUIRED_MODALITIES]
    if len(set(values)) != len(values):
        raise ModalityIntegrityError(f"[{case_id}] duplicate modality paths: {values}")

def _existing_dirs(paths: Iterable[str]) -> List[str]:
    out = []
    for p in paths:
        if p and os.path.isdir(p):
            out.append(p)
    return out

def _dedupe_by_modalities(items: List[dict]) -> List[dict]:
    """
    Deduplicate by the 4 modality absolute paths.
    """
    seen = set()
    out = []
    for it in items:
        key = tuple(os.path.abspath(it[m]) for m in REQUIRED_MODALITIES)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def _get_stratify_value(item: dict, key: str) -> str:
    """
    Supports:
      - "source_dataset"  -> item["meta"]["source_dataset"]
      - "subsource"       -> item["meta"]["subsource"]
      - "meta.case_id"    -> item["meta"]["case_id"]
    """
    if key.startswith("meta."):
        subk = key.split(".", 1)[1]
        return str(item.get("meta", {}).get(subk, "unknown"))
    # Backwards-compat alias: allow "source_dataset"/"subsource"/"case_id"
    if key in ("source_dataset", "subsource", "case_id"):
        return str(item.get("meta", {}).get(key, "unknown"))
    return str(item.get(key, "unknown"))


# -----------------------------
# Splitting (supports stratify)
# -----------------------------

def split_train_val(
    items: List[dict],
    train_ratio: float = 0.8,
    seed: int = 0,
    shuffle: bool = True,
    stratify_key: Optional[str] = "source_dataset",
) -> Tuple[List[dict], List[dict]]:
    """
    Deterministic split.
    stratify_key supports:
      - "source_dataset" (default; from item["meta"]["source_dataset"])
      - "subsource"      (from item["meta"]["subsource"])
      - "meta.case_id"
      - None for pure random
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")

    items = list(items)
    if not items:
        return [], []

    rnd = random.Random(seed)

    if not stratify_key:
        if shuffle:
            rnd.shuffle(items)
        n_train = int(train_ratio * len(items))
        return items[:n_train], items[n_train:]

    buckets: Dict[str, List[dict]] = {}
    for it in items:
        k = _get_stratify_value(it, stratify_key)
        buckets.setdefault(k, []).append(it)

    train: List[dict] = []
    val: List[dict] = []
    for _, group in buckets.items():
        group = list(group)
        if shuffle:
            rnd.shuffle(group)
        n_train = int(train_ratio * len(group))
        train.extend(group[:n_train])
        val.extend(group[n_train:])

    if shuffle:
        rnd.shuffle(train)
        rnd.shuffle(val)

    return train, val


# -----------------------------
# Reporting
# -----------------------------

@dataclass
class LoadReport:
    dataset: str
    roots: List[str]
    scanned_cases: int
    loaded_cases: int
    skipped_cases: int
    skipped_reasons: Dict[str, int]

    def pretty(self) -> str:
        roots_str = "\n    - " + "\n    - ".join(self.roots) if self.roots else " (none)"
        reasons = ", ".join([f"{k}={v}" for k, v in sorted(self.skipped_reasons.items(), key=lambda x: -x[1])])
        if not reasons:
            reasons = "none"
        return (
            f"Dataset: {self.dataset}\n"
            f"  Roots:{roots_str}\n"
            f"  Scanned cases: {self.scanned_cases}\n"
            f"  Loaded cases:  {self.loaded_cases}\n"
            f"  Skipped cases: {self.skipped_cases}\n"
            f"  Skip reasons:  {reasons}\n"
        )


# -----------------------------
# Framework
# -----------------------------

class BaseDatasetLoader:
    name: str = "BaseDataset"

    def __init__(self, roots: List[str]):
        self.roots = _existing_dirs(roots)

    def validate_roots(self) -> None:
        if not self.roots:
            raise DatasetConfigError(f"[{self.name}] no valid roots found. Check DATA_ROOT and folder names.")
        for r in self.roots:
            if not os.path.isdir(r):
                raise DatasetConfigError(f"[{self.name}] root is not a directory: {r}")

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        raise NotImplementedError

    def _process_one(self, case_dir: str, subsource: str) -> Tuple[bool, Optional[dict], Optional[str]]:
        case_id = os.path.basename(case_dir.rstrip("/"))

        # Explicit blacklist
        if case_id in BLACKLIST_CASE_IDS:
            return False, None, "blacklisted_case"

        try:
            paths = self.match_modalities(case_dir)
            if paths is None:
                return False, None, "no_match"

            _assert_case_integrity(f"{self.name}/{case_id}", paths)

            # Keep only image keys at top-level; metadata nested under "meta"
            item = dict(paths)
            item["meta"] = {
                "case_id": case_id,
                "source_dataset": self.name,
                "subsource": subsource,
            }
            return True, item, None

        except ModalityIntegrityError as e:
            msg = str(e)
            if "missing modalities" in msg:
                return False, None, "missing_modalities"
            if "file not found" in msg:
                return False, None, "file_not_found"
            if "duplicate modality paths" in msg:
                return False, None, "duplicate_paths"
            return False, None, "integrity_error"
        except Exception:
            return False, None, "exception"

    def load(self, strict: bool = False, num_workers: int = 16) -> Tuple[List[dict], LoadReport]:
        self.validate_roots()
        case_dirs = self.iter_case_dirs()

        data: List[dict] = []
        skipped_reasons: Dict[str, int] = {}

        if num_workers <= 1:
            for (cdir, subsource) in case_dirs:
                ok, item, reason = self._process_one(cdir, subsource)
                if ok and item:
                    data.append(item)
                else:
                    if strict:
                        raise RuntimeError(f"[{self.name}] strict mode failed on {cdir} reason={reason}")
                    skipped_reasons[reason or "unknown"] = skipped_reasons.get(reason or "unknown", 0) + 1
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futs = [ex.submit(self._process_one, cdir, sub) for (cdir, sub) in case_dirs]
                for fut in as_completed(futs):
                    ok, item, reason = fut.result()
                    if ok and item:
                        data.append(item)
                    else:
                        if strict:
                            raise RuntimeError(f"[{self.name}] strict mode failed reason={reason}")
                        skipped_reasons[reason or "unknown"] = skipped_reasons.get(reason or "unknown", 0) + 1

        report = LoadReport(
            dataset=self.name,
            roots=self.roots,
            scanned_cases=len(case_dirs),
            loaded_cases=len(data),
            skipped_cases=len(case_dirs) - len(data),
            skipped_reasons=skipped_reasons,
        )
        return data, report


# -----------------------------
# Dataset loaders
# -----------------------------

class BraTS2021TrainValLoader(BaseDatasetLoader):
    name = "BraTS2021"

    def __init__(self, train_root: str = BRATS21_TRAIN_ROOT, val_root: str = BRATS21_VAL_ROOT):
        super().__init__([train_root, val_root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for r in self.roots:
            sub = os.path.basename(r.rstrip("/"))
            try:
                dirs = [os.path.join(r, d) for d in sorted(os.listdir(r))]
            except FileNotFoundError:
                continue
            out.extend([(d, sub) for d in dirs if os.path.isdir(d)])
        return out

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        t1n = _pick_first(os.path.join(case_dir, "*_t1.nii.gz"))
        t1c = _pick_first(os.path.join(case_dir, "*_t1ce.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "*_t2.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "*_flair.nii.gz"))
        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class EGDLoader(BaseDatasetLoader):
    name = "EGD"

    def __init__(self, root: str = EGD_ROOT):
        super().__init__([root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        root = self.roots[0]
        dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
        return [(d, "EGD") for d in dirs if os.path.isdir(d)]

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        t1n = _pick_first(os.path.join(case_dir, "1_T1", "NIFTI", "*.nii.gz"))
        t1c = _pick_first(os.path.join(case_dir, "2_T1GD", "NIFTI", "*.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "3_T2", "NIFTI", "*.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "4_FLAIR", "NIFTI", "*.nii.gz"))
        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class UPENNGbmLoader(BaseDatasetLoader):
    """
    Fix: support .nii.gz AND .nii (some UPENN drops are uncompressed)
    """
    name = "UPENN-GBM"

    def __init__(self, root: str = UPENN_GBM_ROOT):
        super().__init__([root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        root = self.roots[0]
        dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
        return [(d, "UPENN-GBM") for d in dirs if os.path.isdir(d)]

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        t1n = _pick_first_any([
            os.path.join(case_dir, "*_T1_unstripped.nii.gz"),
            os.path.join(case_dir, "*_T1_unstripped.nii"),
        ])
        t1c = _pick_first_any([
            os.path.join(case_dir, "*_T1GD_unstripped.nii.gz"),
            os.path.join(case_dir, "*_T1GD_unstripped.nii"),
        ])
        t2w = _pick_first_any([
            os.path.join(case_dir, "*_T2_unstripped.nii.gz"),
            os.path.join(case_dir, "*_T2_unstripped.nii"),
        ])
        t2f = _pick_first_any([
            os.path.join(case_dir, "*_FLAIR_unstripped.nii.gz"),
            os.path.join(case_dir, "*_FLAIR_unstripped.nii"),
        ])

        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class UCSFPDGMLoader(BaseDatasetLoader):
    name = "UCSF-PDGM"

    def __init__(self, root: str = UCSF_PDGM_ROOT):
        super().__init__([root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        root = self.roots[0]
        dirs = sorted(glob.glob(os.path.join(root, "*_nifti")))
        return [(d, "nifti") for d in dirs if os.path.isdir(d)]

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        t1n = _pick_first(os.path.join(case_dir, "*_T1_bias.nii.gz"))
        t1c = _pick_first(os.path.join(case_dir, "*_T1gad_bias.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "*_T2_bias.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "*_FLAIR_bias.nii.gz"))
        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class BraTS2024MultiPackLoader(BaseDatasetLoader):
    name = "BraTS2024"

    def __init__(self, roots: Optional[List[Tuple[str, str]]] = None):
        if roots is None:
            roots = [
                (BRATS24_GLI_TRAIN_ROOT, "GLI_train"),
                (BRATS24_GLI_VAL_ROOT, "GLI_val"),
                (BRATS24_MET_ROOT_1, "MET_train_1"),
                (BRATS24_MET_ROOT_2, "MET_train_2"),
                (BRATS24_MET_VAL_ROOT, "MET_val"),
                (BRATS24_PED_TRAIN_ROOT, "PED_train"),
                (BRATS24_PED_VAL_ROOT, "PED_val"),
            ]
        self._root_labels: List[Tuple[str, str]] = [(p, lab) for (p, lab) in roots if p and os.path.isdir(p)]
        super().__init__([p for (p, _) in self._root_labels])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for (r, label) in self._root_labels:
            try:
                dirs = [os.path.join(r, d) for d in sorted(os.listdir(r))]
            except FileNotFoundError:
                continue
            for d in dirs:
                if os.path.isdir(d):
                    out.append((d, label))
        return out

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        t1c = _pick_first(os.path.join(case_dir, "*-t1c.nii.gz"))
        t1n = _pick_first(os.path.join(case_dir, "*-t1n.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "*-t2f.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "*-t2w.nii.gz"))

        if None in (t1n, t1c, t2w, t2f):
            # fallback to classic naming
            t1n = t1n or _pick_first(os.path.join(case_dir, "*_t1.nii.gz"))
            t1c = t1c or _pick_first(os.path.join(case_dir, "*_t1ce.nii.gz"))
            t2w = t2w or _pick_first(os.path.join(case_dir, "*_t2.nii.gz"))
            t2f = t2f or _pick_first(os.path.join(case_dir, "*_flair.nii.gz"))

        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


# -----------------------------
# Harness / entry
# -----------------------------

def _print_sample(items: List[dict], k: int = 1) -> None:
    if not items:
        return
    k = min(k, len(items))
    print("  Sample cases:")
    for i in range(k):
        d = items[i]
        meta = d.get("meta", {})
        print(f"    [{i}] case_id={meta.get('case_id')} source={meta.get('source_dataset')} sub={meta.get('subsource')}")
        for m in REQUIRED_MODALITIES:
            print(f"      {m}: {d[m]}")

def test_load_datasets(
    loaders: List[BaseDatasetLoader],
    train_ratio: float = 0.8,
    seed: int = 2026,
    strict: bool = False,
    stratify_key: Optional[str] = "source_dataset",
    num_workers: int = 16,
) -> Tuple[List[dict], List[dict]]:
    print("=" * 90)
    print("DATASET LOADING TEST")
    print("=" * 90)

    all_items: List[dict] = []
    for loader in loaders:
        print("-" * 90)
        try:
            data, report = loader.load(strict=strict, num_workers=num_workers)
            print(report.pretty().rstrip())
            all_items.extend(data)
        except DatasetConfigError as e:
            print(f"[CONFIG ERROR] {e}")
        except Exception as e:
            print(f"[UNEXPECTED ERROR] {loader.name}: {type(e).__name__}: {e}")

    before = len(all_items)
    all_items = _dedupe_by_modalities(all_items)
    after = len(all_items)

    print("-" * 90)
    print(f"Total loaded cases across datasets: {before} (deduped -> {after})")
    print("-" * 90)

    train, val = split_train_val(
        all_items,
        train_ratio=train_ratio,
        seed=seed,
        shuffle=True,
        stratify_key=stratify_key,
    )
    print(f"Final split: train={len(train)} val={len(val)} (train_ratio={train_ratio}, seed={seed})")

    if stratify_key:
        def _dist(x: List[dict]) -> Dict[str, int]:
            d: Dict[str, int] = {}
            for it in x:
                k = _get_stratify_value(it, stratify_key)
                d[k] = d.get(k, 0) + 1
            return dict(sorted(d.items(), key=lambda kv: -kv[1]))

        print(f"Train {stratify_key} distribution: {_dist(train)}")
        print(f"Val   {stratify_key} distribution: {_dist(val)}")

    _print_sample(train, k=2)
    return train, val


def get_data_path(
    train_ratio: float = 0.8,
    seed: int = 2026,
    stratify_key: Optional[str] = "source_dataset",
    num_workers: int = 16,
) -> Tuple[List[dict], List[dict]]:
    """
    Returns:
      train_items, val_items

    stratify_key:
      - "source_dataset" (default)
      - "subsource" (recommended if you want BraTS2024 GLI/MET/PED mixture stable)
      - None
    """
    loaders: List[BaseDatasetLoader] = [
        BraTS2021TrainValLoader(),
        BraTS2024MultiPackLoader(),
        EGDLoader(),
        UCSFPDGMLoader(),
        UPENNGbmLoader(),
    ]
    return test_load_datasets(
        loaders=loaders,
        train_ratio=train_ratio,
        seed=seed,
        strict=False,
        stratify_key=stratify_key,
        num_workers=num_workers,
    )


def main():
    print(f"DATA_ROOT = {DATA_ROOT}")
    train, val = get_data_path(
        train_ratio=0.8,
        seed=2026,
        stratify_key="source_dataset",
        num_workers=24,
    )
    # light sanity check
    for idx, item in enumerate(train[:10]):
        _assert_case_integrity(f"train[{idx}]", item)


if __name__ == "__main__":
    main()