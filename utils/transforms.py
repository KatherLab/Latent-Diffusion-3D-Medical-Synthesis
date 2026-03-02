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
# from monai.transforms.intensity.dictionary import ScaleIntensityMRI3D


SUPPORT_MODALITIES = ["ct", "mri"]


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
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    modality = modality.lower()  # Normalize modality to lowercase

    intensity_transforms = {
        "mri": [
            Lambdad(keys=image_keys, func=lambda x: x.contiguous()),
            ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False),
        ],
        "ct": [ScaleIntensityRanged(keys=image_keys, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True)],
    }

    if modality not in intensity_transforms:
        return []

    return intensity_transforms[modality]


def define_random_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """
    Define random intensity transform based on the modality.

    Args:
        modality (str): The imaging modality, either 'ct' or 'mri'.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].

    Returns:
        List: A list of random intensity transforms.
    """
    modality = modality.lower()  # Normalize modality to lowercase
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    if modality == "ct":
        return []  # CT HU intensity is stable across different datasets
    elif modality == "mri":
        return [
            RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)),
            RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.5, 1.0)),
            RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)),
            RandHistogramShiftd(keys=image_keys, prob=0.05, num_control_points=10),
        ]
    else:
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
) -> tuple:
    """
    Define the MAISI VAE transform pipeline for training or validation.

    Notes (project-specific):
    - We set EnsureTyped(..., track_meta=False) to convert MONAI MetaTensor -> plain torch.Tensor.
      This avoids rare shape/metadata propagation issues when using pure torch ops inside networks.
    - We enforce a minimum divisible padding factor in validation (effective_k >= 16) because many 3D UNets
      downsample 4 times (2^4 = 16). If your UNet downsamples 5 times, you may prefer k=32.
    """
    modality = modality.lower()  # Normalize modality to lowercase
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]

    # if modality == "mri":
    #     common_transform.append(Lambdad(keys=image_keys, func=lambda x: x[select_channel : select_channel + 1, ...]))

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
        train_crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(
                keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True
            ),
        ]
    else:
        # Ensure val spatial dims are compatible with deep UNets (avoid 40 vs 39 skip mismatch).
        effective_k = max(int(k), 16)
        val_crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=effective_k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    # NOTE: track_meta=False is intentional (see docstring notes above).
    final_transform = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True, track_meta=False)]

    if is_train:
        train_transforms = Compose(
            common_transform + random_transform + train_crop + final_transform
            if random_aug
            else common_transform + train_crop + final_transform
        )
        return train_transforms
    else:
        val_transforms = Compose(common_transform + val_crop + final_transform)
        return val_transforms


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
) -> tuple:
    """
    Define the MAISI VAE transform pipeline for training or validation.

    IMPORTANT FIX:
    - Previously, validation returned Compose(common_transform + final_transform) with NO crop/pad,
      which can cause UNet skip-connection size mismatch (e.g., 40 vs 39) on odd-sized volumes.
    - We restore validation padding/cropping using DivisiblePadd (effective_k >= 16 by default).
    - We set EnsureTyped(track_meta=False) to avoid MetaTensor participating in torch ops.
    """
    modality = modality.lower()  # Normalize modality to lowercase
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        # Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]
    common_transform.append(Lambdad(keys=image_keys, func=lambda x: x.contiguous()))
    common_transform.append(
        ScaleIntensityRangePercentilesd(
            keys=image_keys,
            lower=0.0,
            upper=99.5,
            b_min=0.0,
            b_max=1,
            clip=False,
        )
    )
    # common_transform.append(ScaleIntensityMRI3D(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False))
    # common_transform.append(EnsureChannelFirstd(keys=keys, allow_missing_keys=True))
    # common_transform.append(Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True))

    # if spacing_type == "fixed":
    #     common_transform.append(
    #         Spacingd(keys=image_keys + label_keys, allow_missing_keys=True, pixdim=spacing, mode=interp_mode)
    #     )

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
        train_crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(
                keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True
            ),
        ]
    else:
        # Validation: either pad to divisible shape (default) or crop/pad to fixed patch size.
        effective_k = max(int(k), 16)
        val_crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=effective_k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    # NOTE: track_meta=False is intentional.
    final_transform = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True, track_meta=False)]

    if is_train:
        train_transforms = Compose(
            common_transform + random_transform + train_crop + final_transform
            if random_aug
            else common_transform + train_crop + final_transform
        )
        return train_transforms
    else:
        val_transforms = Compose(common_transform + val_crop + final_transform)
        return val_transforms


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
) -> tuple:
    """
    Define the MAISI VAE transform pipeline for training or validation.

    Notes:
    - track_meta=False for EnsureTyped to return plain torch.Tensor.
    - effective_k >= 16 for divisible padding (if val_patch_size is None).
    """
    modality = modality.lower()  # Normalize modality to lowercase
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]

    # if modality == "mri":
    #     common_transform.append(Lambdad(keys=image_keys, func=lambda x: x[select_channel : select_channel + 1, ...]))

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
        train_crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            # RandSpatialCropd(
            #     keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True
            # ),
        ]
    else:
        effective_k = max(int(k), 16)
        val_crop = (
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=effective_k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    final_transform = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True, track_meta=False)]

    if is_train:
        train_transforms = Compose(
            common_transform + random_transform + train_crop + final_transform
            if random_aug
            else common_transform + train_crop + final_transform
        )
        return train_transforms
    else:
        val_transforms = Compose(common_transform + val_crop + final_transform)
        return val_transforms


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
        """
        Initialize the VAE_Transform.
        """
        if spacing_type not in ["original", "fixed", "rand_zoom"]:
            raise ValueError(
                f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}."
            )

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
        """
        Apply the appropriate transform to the input image.
        """
        modality = fixed_modality or img["class"]
        modality = modality.lower()  # Normalize modality to lowercase
        if modality not in ["ct", "mri"]:
            warnings.warn(
                f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
            )

        transform = self.transform_dict[modality]
        return transform(img)


class VAETransformMRI:
    """
    A class to handle MAISI VAE transformations for MRI 3D.
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
            raise ValueError(
                f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}."
            )

        self.is_train = is_train

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
    """
    A class to handle MAISI VAE transformations for MRI 2D.
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
            raise ValueError(
                f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}."
            )

        self.is_train = is_train

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