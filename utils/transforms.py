# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
utils/transforms.py (project-patched)

What changed vs your current file:
1) Percentile intensity scaling performance:
   - Adds Lambdad(..., x.contiguous()) before ScaleIntensityRangePercentilesd (MRI).
2) Validation legitimacy / padding debug:
   - For validation pipelines, we keep MONAI metadata (track_meta=True) so you can inspect
     original spatial shape vs post-pad/crop shape.
   - We store orig spatial shape in meta under key "orig_spatial_shape" right after EnsureChannelFirstd
     (before any pad/crop), so you can later compare.
3) Training stability:
   - Training still uses track_meta=False by default (plain torch.Tensor) to avoid MetaTensor
     propagation into pure torch ops.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import torch
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

SUPPORT_MODALITIES = ["ct", "mri"]

# Optional: set to "1" to keep meta in TRAIN too (usually not needed).
_KEEP_META_TRAIN = os.environ.get("KEEP_META_TRAIN", "0").strip() == "1"


def _mark_orig_shape(x):
    """
    Save the pre-pad/crop spatial shape into MetaTensor.meta['orig_spatial_shape'] when possible.
    x is expected to be channel-first: [C, D, H, W] (3D) or [C, H, W] (2D).
    """
    try:
        if hasattr(x, "meta") and isinstance(x.meta, dict):
            # spatial dims are everything after channel
            x.meta["orig_spatial_shape"] = tuple(int(s) for s in x.shape[1:])
    except Exception:
        pass
    return x


def define_fixed_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """
    Define fixed intensity transform based on the modality.

    Args:
        modality (str): The imaging modality, either 'ct' or 'mri'.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].

    Returns:
        List: A list of intensity transforms.
    """
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
            f"Will not do any intensity transform and will use original intensities."
        )

    modality = modality.lower()

    intensity_transforms = {
        "mri": [
            # Avoid non-contiguous warning / extra copy inside torch.searchsorted.
            Lambdad(keys=image_keys, func=lambda x: x.contiguous()),
            ScaleIntensityRangePercentilesd(
                keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=False
            ),
        ],
        "ct": [
            ScaleIntensityRanged(keys=image_keys, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
        ],
    }

    return intensity_transforms.get(modality, [])


def define_random_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """
    Define random intensity transform based on the modality.
    """
    modality = modality.lower()
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
            f"Will not do any intensity transform and will use original intensities."
        )

    if modality == "ct":
        return []
    if modality == "mri":
        return [
            RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)),
            RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.5, 1.0)),
            RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)),
            RandHistogramShiftd(keys=image_keys, prob=0.05, num_control_points=10),
        ]
    return []


def define_vae_transform(
    is_train: bool,
    modality: str,
    random_aug: bool,
    k: int = 4,
    patch_size: List[int] = [128, 128, 128],
    val_patch_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> Compose:
    modality = modality.lower()
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
            f"Will not do any intensity transform and will use original intensities."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type must be in ['original','fixed','rand_zoom'], got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]

    # Mark original spatial shape BEFORE any padding/cropping.
    # Only meaningful if meta is kept (validation by default).
    common_transform.append(Lambdad(keys=image_keys, func=_mark_orig_shape, allow_missing_keys=True))

    common_transform.extend(define_fixed_intensity_transform(modality, image_keys=image_keys))

    if spacing_type == "fixed":
        common_transform.append(
            Spacingd(keys=image_keys + label_keys, allow_missing_keys=True, pixdim=spacing, mode=interp_mode)
        )

    random_transform = []
    if is_train and random_aug:
        random_transform.extend(define_random_intensity_transform(modality, image_keys=image_keys))
        random_transform.extend(
            [RandFlipd(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axis=axis) for axis in range(3)]
            + [
                RandRotate90d(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axes=axes)
                for axes in [(0, 1), (1, 2), (0, 2)]
            ]
            + [
                RandScaleIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, factors=(0.9, 1.1)),
                RandShiftIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, offsets=0.05),
            ]
        )
        if spacing_type == "rand_zoom":
            random_transform.extend(
                [
                    RandZoomd(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        min_zoom=0.5,
                        max_zoom=1.5,
                        keep_size=False,
                        mode=interp_mode,
                    ),
                    RandRotated(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        keep_size=True,
                        mode=interp_mode,
                    ),
                ]
            )

    if is_train:
        crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True),
        ]
    else:
        effective_k = max(int(k), 16)
        crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=effective_k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    # Train: default to plain Tensor. Val: keep meta so you can inspect orig vs padded shapes.
    track_meta = (not is_train) or _KEEP_META_TRAIN
    final = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True, track_meta=track_meta)]

    return Compose(common_transform + (random_transform if (is_train and random_aug) else []) + crop + final)


def define_vae_transform_mri3d(
    is_train: bool,
    modality: str,
    random_aug: bool,
    k: int = 4,
    patch_size: List[int] = [128, 128, 128],
    val_patch_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> Compose:
    # This path is the one you actually use (VAETransformMRI), so keep it aligned with define_vae_transform().
    modality = modality.lower()
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
            f"Will not do any intensity transform and will use original intensities."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type must be in ['original','fixed','rand_zoom'], got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        # Orientation intentionally off here (project choice)
    ]

    # Mark original shape (for validation debug)
    common_transform.append(Lambdad(keys=image_keys, func=_mark_orig_shape, allow_missing_keys=True))

    # Intensity (MRI fixed)
    common_transform.append(Lambdad(keys=image_keys, func=lambda x: x.contiguous(), allow_missing_keys=True))
    common_transform.append(
        ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=False)
    )

    random_transform = []
    if is_train and random_aug:
        random_transform.extend(define_random_intensity_transform(modality, image_keys=image_keys))
        random_transform.extend(
            [RandFlipd(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axis=axis) for axis in range(3)]
            + [
                RandRotate90d(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axes=axes)
                for axes in [(0, 1), (1, 2), (0, 2)]
            ]
            + [
                RandScaleIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, factors=(0.9, 1.1)),
                RandShiftIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, offsets=0.05),
            ]
        )
        if spacing_type == "rand_zoom":
            random_transform.extend(
                [
                    RandZoomd(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        min_zoom=0.5,
                        max_zoom=1.5,
                        keep_size=False,
                        mode=interp_mode,
                    ),
                    RandRotated(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        keep_size=True,
                        mode=interp_mode,
                    ),
                ]
            )

    if is_train:
        crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True),
        ]
    else:
        effective_k = max(int(k), 16)
        crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=effective_k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    track_meta = (not is_train) or _KEEP_META_TRAIN
    final = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True, track_meta=track_meta)]

    return Compose(common_transform + (random_transform if (is_train and random_aug) else []) + crop + final)


def define_vae_transform2d(
    is_train: bool,
    modality: str,
    random_aug: bool,
    k: int = 4,
    patch_size: List[int] = [128, 128, 128],
    val_patch_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> Compose:
    # Keep consistent with define_vae_transform, just without random 3D cropping in train (as your original).
    modality = modality.lower()
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
            f"Will not do any intensity transform and will use original intensities."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type must be in ['original','fixed','rand_zoom'], got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]

    common_transform.append(Lambdad(keys=image_keys, func=_mark_orig_shape, allow_missing_keys=True))
    common_transform.extend(define_fixed_intensity_transform(modality, image_keys=image_keys))

    if spacing_type == "fixed":
        common_transform.append(
            Spacingd(keys=image_keys + label_keys, allow_missing_keys=True, pixdim=spacing, mode=interp_mode)
        )

    random_transform = []
    if is_train and random_aug:
        random_transform.extend(define_random_intensity_transform(modality, image_keys=image_keys))
        random_transform.extend(
            [RandFlipd(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axis=axis) for axis in range(3)]
            + [
                RandRotate90d(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axes=axes)
                for axes in [(0, 1), (1, 2), (0, 2)]
            ]
            + [
                RandScaleIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, factors=(0.9, 1.1)),
                RandShiftIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, offsets=0.05),
            ]
        )
        if spacing_type == "rand_zoom":
            random_transform.extend(
                [
                    RandZoomd(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        min_zoom=0.5,
                        max_zoom=1.5,
                        keep_size=False,
                        mode=interp_mode,
                    ),
                    RandRotated(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        keep_size=True,
                        mode=interp_mode,
                    ),
                ]
            )

    if is_train:
        crop = [SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True)]
    else:
        effective_k = max(int(k), 16)
        crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=effective_k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    track_meta = (not is_train) or _KEEP_META_TRAIN
    final = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True, track_meta=track_meta)]

    return Compose(common_transform + (random_transform if (is_train and random_aug) else []) + crop + final)


class VAE_Transform:
    """
    A class to handle MAISI VAE transformations for different modalities.
    """
    def __init__(
        self,
        is_train: bool,
        random_aug: bool,
        k: int = 4,
        patch_size: List[int] = [128, 128, 128],
        val_patch_size: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        spacing_type: str = "original",
        spacing: Optional[List[float]] = None,
        image_keys: List[str] = ["image"],
        label_keys: List[str] = [],
        additional_keys: List[str] = [],
        select_channel: int = 0,
    ):
        if spacing_type not in ["original", "fixed", "rand_zoom"]:
            raise ValueError(f"spacing_type must be in ['original','fixed','rand_zoom'], got {spacing_type}.")
        self.is_train = is_train
        self.transform_dict = {}
        for modality in ["ct", "mri"]:
            self.transform_dict[modality] = define_vae_transform(
                is_train=is_train,
                modality=modality,
                random_aug=random_aug,
                k=k,
                patch_size=patch_size,
                val_patch_size=val_patch_size,
                output_dtype=output_dtype,
                spacing_type=spacing_type,
                spacing=spacing,
                image_keys=image_keys,
                label_keys=label_keys,
                additional_keys=additional_keys,
                select_channel=select_channel,
            )

    def __call__(self, img: dict, fixed_modality: Optional[str] = None) -> dict:
        modality = fixed_modality or img.get("class", "mri")
        modality = modality.lower()
        if modality not in ["ct", "mri"]:
            warnings.warn(
                f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. "
                f"Will not do any intensity transform and will use original intensities."
            )
        return self.transform_dict[modality](img)


class VAETransformMRI:
    """MRI 3D transform wrapper (used by your brain_data_utils.py)."""
    def __init__(
        self,
        is_train: bool,
        random_aug: bool,
        k: int = 4,
        patch_size: List[int] = [128, 128, 128],
        val_patch_size: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        spacing_type: str = "original",
        spacing: Optional[List[float]] = None,
        image_keys: List[str] = ["image"],
        label_keys: List[str] = [],
        additional_keys: List[str] = [],
        select_channel: int = 0,
    ):
        if spacing_type not in ["original", "fixed", "rand_zoom"]:
            raise ValueError(f"spacing_type must be in ['original','fixed','rand_zoom'], got {spacing_type}.")
        self.transform = define_vae_transform_mri3d(
            is_train=is_train,
            modality="mri",
            random_aug=random_aug,
            k=k,
            patch_size=patch_size,
            val_patch_size=val_patch_size,
            output_dtype=output_dtype,
            spacing_type=spacing_type,
            spacing=spacing,
            image_keys=image_keys,
            label_keys=label_keys,
            additional_keys=additional_keys,
            select_channel=select_channel,
        )

    def __call__(self, img: dict, fixed_modality: Optional[str] = None) -> dict:
        return self.transform(img)


class VAETransformMRI2D:
    """MRI 2D transform wrapper."""
    def __init__(
        self,
        is_train: bool,
        random_aug: bool,
        k: int = 4,
        patch_size: List[int] = [128, 128, 128],
        val_patch_size: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        spacing_type: str = "original",
        spacing: Optional[List[float]] = None,
        image_keys: List[str] = ["image"],
        label_keys: List[str] = [],
        additional_keys: List[str] = [],
        select_channel: int = 0,
    ):
        if spacing_type not in ["original", "fixed", "rand_zoom"]:
            raise ValueError(f"spacing_type must be in ['original','fixed','rand_zoom'], got {spacing_type}.")
        self.transform = define_vae_transform2d(
            is_train=is_train,
            modality="mri",
            random_aug=random_aug,
            k=k,
            patch_size=patch_size,
            val_patch_size=val_patch_size,
            output_dtype=output_dtype,
            spacing_type=spacing_type,
            spacing=spacing,
            image_keys=image_keys,
            label_keys=label_keys,
            additional_keys=additional_keys,
            select_channel=select_channel,
        )

    def __call__(self, img: dict, fixed_modality: Optional[str] = None) -> dict:
        return self.transform(img)