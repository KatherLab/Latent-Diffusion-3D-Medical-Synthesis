#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified MRI dataset loader (multi-root capable) with:
- per-dataset loader specs
- multi-root merge inside a single dataset (e.g., BraTS2024 = GLI + MET + PED)
- automatic integrity validation
- faster IO via ThreadPoolExecutor
- deterministic split with optional stratification (default: by source_dataset)

IMPORTANT:
- The dataset roots below are updated to match your real paths used in local_path.py:
  /mnt/swarm_beta/xuewei/data/...
  (You can override via env var MRI_DATA_ROOT.)

This script is meant to *index* files and return train/val dicts:
  {"t1n":..., "t1c":..., "t2w":..., "t2f":..., "case_id":..., "source_dataset":..., "subsource":...}

Usage:
  python local_path_unified.py
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

ModalityMap = Dict[str, str]  # {"t1n": "...", "t1c": "...", "t2w": "...", "t2f": "...", ...metadata...}
REQUIRED_MODALITIES = ("t1n", "t1c", "t2w", "t2f")


# -----------------------------
# Configuration (UPDATED)
# -----------------------------
# You previously hardcoded /mnt/swarm_beta/xuewei/data in local_path.py.
# Keep that as default, but allow override.
DATA_ROOT = os.environ.get("MRI_DATA_ROOT", "/mnt/swarm_beta/xuewei/data")

# BraTS2021 (real paths from local_path.py)
BRATS21_TRAIN_ROOT = os.path.join(
    DATA_ROOT, "BraTS2021", "RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
)
BRATS21_VAL_ROOT = os.path.join(
    DATA_ROOT, "BraTS2021", "RSNA_ASNR_MICCAI_BraTS2021_ValidationData"
)

# BraTS2024 GLI (real paths from local_path.py)
BRATS24_GLI_TRAIN_ROOT = os.path.join(DATA_ROOT, "BraTS2024", "training_data1")
BRATS24_GLI_VAL_ROOT = os.path.join(DATA_ROOT, "BraTS2024", "validation_data")

# BraTS2024 MET (real paths from local_path.py)
BRATS24_MET_ROOT_1 = os.path.join(DATA_ROOT, "BraTS24_MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_1")
BRATS24_MET_ROOT_2 = os.path.join(DATA_ROOT, "BraTS24_MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_2")  # fixed cases already
BRATS24_MET_VAL_ROOT = os.path.join(DATA_ROOT, "BraTS24_MET", "MICCAI-BraTS2024-MET-Challenge-ValidationData")

# BraTS2024 PED (real paths from local_path.py)
BRATS24_PED_TRAIN_ROOT = os.path.join(
    DATA_ROOT, "BraTS-PEDs2024_Training", "BraTS2024-PED-Challenge-TrainingData"
)
BRATS24_PED_VAL_ROOT = os.path.join(
    DATA_ROOT, "BraTS-PEDs2024_Training", "BraTS2024-PED-Challenge-ValidationData"
)

# EGD / UPENN / UCSF (real paths from local_path.py)
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

def _dedupe_by_modalities(items: List[ModalityMap]) -> List[ModalityMap]:
    """
    Deduplicate across merged roots (especially BraTS2024 packs) by the 4 modality absolute paths.
    This replaces the brittle 'del train_files_brats24[1245]' logic from your old script.
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


# -----------------------------
# Splitting (supports stratify)
# -----------------------------

def split_train_val(
    items: List[ModalityMap],
    train_ratio: float = 0.8,
    seed: int = 0,
    shuffle: bool = True,
    stratify_key: Optional[str] = "source_dataset",
) -> Tuple[List[ModalityMap], List[ModalityMap]]:
    """
    Deterministic split.
    If stratify_key is provided, we do a per-group split so train/val preserve group proportions.

    Recommended options:
      - stratify_key="source_dataset"  (keeps dataset mixture stable)
      - stratify_key="subsource"       (keeps BraTS2024 GLI/MET/PED mixture stable)
      - stratify_key=None              (pure random split)
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

    # group by key
    buckets: Dict[str, List[ModalityMap]] = {}
    for it in items:
        k = str(it.get(stratify_key, "unknown"))
        buckets.setdefault(k, []).append(it)

    train: List[ModalityMap] = []
    val: List[ModalityMap] = []
    for _, group in buckets.items():
        group = list(group)
        if shuffle:
            rnd.shuffle(group)
        n_train = int(train_ratio * len(group))
        train.extend(group[:n_train])
        val.extend(group[n_train:])

    # final shuffle to mix groups (still deterministic)
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
        """
        Return list of (case_dir, subsource_label).
        subsource_label is useful when merging multiple roots inside one dataset.
        """
        raise NotImplementedError

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        """Return modality dict or None to skip."""
        raise NotImplementedError

    def _process_one(self, case_dir: str, subsource: str) -> Tuple[bool, Optional[ModalityMap], Optional[str]]:
        case_id = os.path.basename(case_dir.rstrip("/"))
        try:
            paths = self.match_modalities(case_dir)
            if paths is None:
                return False, None, "no_match"

            # attach metadata (helps debugging + stratification)
            paths = dict(paths)
            paths["case_id"] = case_id
            paths["source_dataset"] = self.name
            paths["subsource"] = subsource

            _assert_case_integrity(f"{self.name}/{case_id}", paths)
            return True, paths, None

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

    def load(
        self,
        strict: bool = False,
        num_workers: int = 16,
    ) -> Tuple[List[ModalityMap], LoadReport]:
        """
        strict=False: skip bad cases, count reasons
        strict=True: raise on first error
        """
        self.validate_roots()
        case_dirs = self.iter_case_dirs()

        data: List[ModalityMap] = []
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
    """
    BraTS2021 in your real script uses:
      .../RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/<case>/*.nii.gz
      .../RSNA_ASNR_MICCAI_BraTS2021_ValidationData/<case>/*.nii.gz

    tree.txt indicates each case typically has:
      *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz (+ *_seg.nii.gz)
    """
    name = "BraTS2021"

    def __init__(self, train_root: str = BRATS21_TRAIN_ROOT, val_root: str = BRATS21_VAL_ROOT):
        super().__init__([train_root, val_root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for r in self.roots:
            sub = os.path.basename(r.rstrip("/"))  # TrainingData_... or ValidationData
            try:
                dirs = [os.path.join(r, d) for d in sorted(os.listdir(r))]
            except FileNotFoundError:
                continue
            out.extend([(d, sub) for d in dirs if os.path.isdir(d)])
        return out

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        # Use explicit suffix matching (robust to file ordering).
        t1n = _pick_first(os.path.join(case_dir, "*_t1.nii.gz"))
        t1c = _pick_first(os.path.join(case_dir, "*_t1ce.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "*_t2.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "*_flair.nii.gz"))
        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class EGDLoader(BaseDatasetLoader):
    """
    EGD path matches your old script:
      /mnt/swarm_beta/xuewei/data/EGD/<case>/[1-4]_* /NIFTI/*.nii.gz
    """
    name = "EGD"

    def __init__(self, root: str = EGD_ROOT):
        super().__init__([root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        root = self.roots[0]
        dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
        return [(d, "EGD") for d in dirs if os.path.isdir(d)]

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        # Keep your old glob style, but ensure mapping is by folder number not by sort order.
        t1n = _pick_first(os.path.join(case_dir, "1_T1", "NIFTI", "*.nii.gz"))
        t1c = _pick_first(os.path.join(case_dir, "2_T1GD", "NIFTI", "*.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "3_T2", "NIFTI", "*.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "4_FLAIR", "NIFTI", "*.nii.gz"))
        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class UPENNGbmLoader(BaseDatasetLoader):
    """
    UPENN-GBM path matches your old script:
      /mnt/swarm_beta/xuewei/data/UPENN-GBM/<case>/*.nii.gz
    tree.txt indicates names like *_T1_unstripped, *_T1GD_unstripped, *_T2_unstripped, *_FLAIR_unstripped.
    """
    name = "UPENN-GBM"

    def __init__(self, root: str = UPENN_GBM_ROOT):
        super().__init__([root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        root = self.roots[0]
        dirs = [os.path.join(root, d) for d in sorted(os.listdir(root))]
        return [(d, "UPENN-GBM") for d in dirs if os.path.isdir(d)]

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        # Use suffix matching (more robust than relying on sorted order).
        t2f = _pick_first(os.path.join(case_dir, "*_FLAIR_unstripped.nii.gz"))
        t1c = _pick_first(os.path.join(case_dir, "*_T1GD_unstripped.nii.gz"))
        t1n = _pick_first(os.path.join(case_dir, "*_T1_unstripped.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "*_T2_unstripped.nii.gz"))
        if None in (t1n, t1c, t2w, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class UCSFPDGMLoader(BaseDatasetLoader):
    """
    UCSF-PDGM:
    - Your earlier 'better script' assumed *_nifti directories.
    - Your old local_path.py used /mnt/swarm_beta/xuewei/data/UCSF-PDGM/* (direct case dirs).
    We support BOTH layouts:
      - <root>/*_nifti/...
      - <root>/*/...
    """
    name = "UCSF-PDGM"

    def __init__(self, root: str = UCSF_PDGM_ROOT):
        super().__init__([root])

    def iter_case_dirs(self) -> List[Tuple[str, str]]:
        root = self.roots[0]

        nifti_dirs = sorted(glob.glob(os.path.join(root, "*_nifti")))
        if nifti_dirs:
            return [(d, "nifti") for d in nifti_dirs if os.path.isdir(d)]

        # fallback: any subdir that contains nii.gz files
        out: List[Tuple[str, str]] = []
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                if glob.glob(os.path.join(d, "*.nii.gz")):
                    out.append((d, "case_dir"))
        return out

    def match_modalities(self, case_dir: str) -> Optional[Dict[str, str]]:
        t1n = _pick_first(os.path.join(case_dir, "*_T1_bias.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "*_T2_bias.nii.gz"))
        t1c = _pick_first(os.path.join(case_dir, "*_T1gad_bias.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "*_FLAIR_bias.nii.gz"))
        if None in (t1n, t2w, t1c, t2f):
            return None
        return {"t1n": t1n, "t1c": t1c, "t2w": t2w, "t2f": t2f}


class BraTS2024MultiPackLoader(BaseDatasetLoader):
    """
    ONE dataset loader that merges BraTS2024-related packs used in your old local_path.py:
      - GLI: BraTS2024/training_data1 (+ optionally BraTS2024/validation_data)
      - MET: BraTS24_MET/MICCAI-BraTS2024-MET-Challenge-TrainingData_1
             BraTS24_MET/MICCAI-BraTS2024-MET-Challenge-TrainingData_2
             BraTS24_MET/MICCAI-BraTS2024-MET-Challenge-ValidationData
      - PED: BraTS-PEDs2024_Training/BraTS2024-PED-Challenge-TrainingData
             BraTS-PEDs2024_Training/BraTS2024-PED-Challenge-ValidationData (if present)
    """
    name = "BraTS2024"

    def __init__(
        self,
        roots: Optional[List[Tuple[str, str]]] = None,
    ):
        # roots: list of (path, label). label becomes "subsource".
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
        # Your BraTS2024 packs use "*-t1n.nii.gz" style (per local_path.py).
        t1c = _pick_first(os.path.join(case_dir, "*-t1c.nii.gz"))
        t1n = _pick_first(os.path.join(case_dir, "*-t1n.nii.gz"))
        t2f = _pick_first(os.path.join(case_dir, "*-t2f.nii.gz"))
        t2w = _pick_first(os.path.join(case_dir, "*-t2w.nii.gz"))

        # Fallback if a pack uses classic BraTS naming.
        if None in (t1n, t1c, t2w, t2f):
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

def _print_sample(items: List[ModalityMap], k: int = 1) -> None:
    if not items:
        return
    k = min(k, len(items))
    print("  Sample cases:")
    for i in range(k):
        d = items[i]
        meta = f"case_id={d.get('case_id')} source={d.get('source_dataset')} sub={d.get('subsource')}"
        print(f"    [{i}] {meta}")
        for m in REQUIRED_MODALITIES:
            print(f"      {m}: {d[m]}")

def test_load_datasets(
    loaders: List[BaseDatasetLoader],
    train_ratio: float = 0.8,
    seed: int = 2026,
    strict: bool = False,
    stratify_key: Optional[str] = "source_dataset",
    num_workers: int = 16,
) -> Tuple[List[ModalityMap], List[ModalityMap]]:
    print("=" * 90)
    print("DATASET LOADING TEST")
    print("=" * 90)

    all_items: List[ModalityMap] = []
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

    # Deduplicate after merging all datasets (safe + cheap).
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
        def _dist(x: List[ModalityMap]) -> Dict[str, int]:
            d: Dict[str, int] = {}
            for it in x:
                k = str(it.get(stratify_key, "unknown"))
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
) -> Tuple[List[ModalityMap], List[ModalityMap]]:
    """
    Replacement for your original get_data_path().
    Returns:
      train_items, val_items
    Each item includes:
      t1n,t1c,t2w,t2f + case_id, source_dataset, subsource

    NOTE:
    - If you want to keep BraTS2024 pack proportions stable, use stratify_key="subsource".
    """
    loaders: List[BaseDatasetLoader] = [
        BraTS2021TrainValLoader(),
        BraTS2024MultiPackLoader(),
        EGDLoader(),
        UCSFPDGMLoader(),
        UPENNGbmLoader(),
        # InhouseLoader()  # add once you confirm the root on this machine
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
    # You can set MRI_DATA_ROOT to override DATA_ROOT:
    #   export MRI_DATA_ROOT=/path/to/your/data_root
    print(f"DATA_ROOT = {DATA_ROOT}")

    train, val = get_data_path(
        train_ratio=0.8,
        seed=2026,
        stratify_key="source_dataset",   # or "subsource" for BraTS2024 pack-level stratification
        num_workers=24,                  # bump if your filesystem can handle it
    )

    # sanity check a few (light)
    for idx, item in enumerate(train[:10]):
        _assert_case_integrity(f"train[{idx}]", item)


if __name__ == "__main__":
    main()


# -----------------------------
# Optional: augmentation stub (keep outside path loader)
# -----------------------------
# If you are using MONAI, you typically define transforms in your training code, e.g.:
#
# from monai.transforms import (
#     Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
#     Orientationd, Spacingd, ScaleIntensityRanged,
#     RandFlipd, RandAffined, RandGaussianNoised, RandScaleIntensityd,
# )
#
# def get_train_transforms():
#     return Compose([
#         LoadImaged(keys=["t1n","t1c","t2w","t2f"]),
#         EnsureChannelFirstd(keys=["t1n","t1c","t2w","t2f"]),
#         Orientationd(keys=["t1n","t1c","t2w","t2f"], axcodes="RAS"),
#         # Spacingd(...)  # only if you want to resample to common spacing
#         # ScaleIntensityRanged(...) # if consistent intensity mapping makes sense for your data
#         RandFlipd(keys=["t1n","t1c","t2w","t2f"], prob=0.5, spatial_axis=0),
#         RandAffined(keys=["t1n","t1c","t2w","t2f"], prob=0.2, rotate_range=(0.1,0.1,0.1)),
#         RandGaussianNoised(keys=["t1n","t1c","t2w","t2f"], prob=0.15, mean=0.0, std=0.01),
#         EnsureTyped(keys=["t1n","t1c","t2w","t2f"]),
#     ])