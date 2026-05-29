"""Data augmentation pipelines and reproducibility helpers."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Reproducibility ----------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Set seeds for ``torch`` (CPU + CUDA), ``numpy``, and ``random``."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


# -- Vectorised batched augmentation -----------------------------------------


class BatchedAugmentation(nn.Module):
    """Per-sample randomised augmentation, vectorised across the batch.

    Each sample in a ``(B, C, H, W)`` tensor independently samples its own
    augmentation parameters (flips, lossless 90-degree rotations, optional
    brightness/contrast jitter, optional Gaussian noise, optional blur, optional
    sharpness adjustment) without Python-level per-sample iteration. A
    ``(C, H, W)`` input is also accepted.
    """

    def __init__(
        self,
        hflip: bool = True,
        vflip: bool = True,
        rot90: bool = True,
        brightness_delta: float = 0.0,
        brightness_p: float = 0.0,
        contrast_delta: float = 0.0,
        contrast_p: float = 0.0,
        noise_std: float = 0.0,
        noise_p: float = 0.0,
        sharpness_factor: float = 0.5,
        sharpness_min: Optional[float] = None,
        sharpness_max: Optional[float] = None,
        sharpness_p: float = 0.0,
        blur_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.hflip = hflip
        self.vflip = vflip
        self.rot90 = rot90
        self.brightness_delta = float(brightness_delta)
        self.brightness_p = float(brightness_p)
        self.contrast_delta = float(contrast_delta)
        self.contrast_p = float(contrast_p)
        self.noise_std = float(noise_std)
        self.noise_p = float(noise_p)
        self.sharpness_factor = float(sharpness_factor)
        self.sharpness_min = None if sharpness_min is None else float(sharpness_min)
        self.sharpness_max = None if sharpness_max is None else float(sharpness_max)
        self.sharpness_p = float(sharpness_p)
        self.blur_p = float(blur_p)
        if (self.sharpness_min is None) != (self.sharpness_max is None):
            raise ValueError("sharpness_min and sharpness_max must be set together.")
        if (
            self.sharpness_min is not None
            and self.sharpness_max is not None
            and self.sharpness_min > self.sharpness_max
        ):
            raise ValueError("sharpness_min must be <= sharpness_max.")

    @staticmethod
    def _sample_mask(
        batch_size: int,
        device: torch.device,
        p: float,
        apply_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        if apply_mask is not None:
            mask &= apply_mask.to(device=device, dtype=torch.bool)
        if p >= 1.0:
            return mask
        return mask & (torch.rand(batch_size, device=device) < p)

    def forward(self, x: torch.Tensor, apply_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        single = (x.ndim == 3)
        if single:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3-D or 4-D tensor, got {x.ndim}-D.")

        B = x.shape[0]
        device = x.device

        if self.hflip:
            mask = self._sample_mask(B, device, p=0.5, apply_mask=apply_mask).view(-1, 1, 1, 1)
            x = torch.where(mask, x.flip(-1), x)

        if self.vflip:
            mask = self._sample_mask(B, device, p=0.5, apply_mask=apply_mask).view(-1, 1, 1, 1)
            x = torch.where(mask, x.flip(-2), x)

        if self.rot90:
            rot_mask = self._sample_mask(B, device, p=0.5, apply_mask=apply_mask)
            ks = torch.randint(1, 4, (B,), device=device)
            out = x.clone()
            for k in (1, 2, 3):
                idx = (rot_mask & (ks == k)).nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    out[idx] = torch.rot90(x[idx], k=int(k), dims=(-2, -1))
            x = out

        if self.brightness_delta > 0.0 and self.brightness_p > 0.0:
            mask = self._sample_mask(B, device, p=self.brightness_p, apply_mask=apply_mask).view(-1, 1, 1, 1)
            factors = self._sample_factor(B, device, 1.0 - self.brightness_delta, 1.0 + self.brightness_delta)
            x = torch.where(mask, x * factors.view(-1, 1, 1, 1), x)

        if self.contrast_delta > 0.0 and self.contrast_p > 0.0:
            mask = self._sample_mask(B, device, p=self.contrast_p, apply_mask=apply_mask).view(-1, 1, 1, 1)
            factors = self._sample_factor(B, device, 1.0 - self.contrast_delta, 1.0 + self.contrast_delta)
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x_contrast = (x - mean) * factors.view(-1, 1, 1, 1) + mean
            x = torch.where(mask, x_contrast, x)

        if self.noise_std > 0.0 and self.noise_p > 0.0:
            mask = self._sample_mask(B, device, p=self.noise_p, apply_mask=apply_mask).view(-1, 1, 1, 1)
            x = torch.where(mask, x + torch.randn_like(x) * self.noise_std, x)

        if self.blur_p > 0.0:
            mask = self._sample_mask(B, device, p=self.blur_p, apply_mask=apply_mask).view(-1, 1, 1, 1)
            x = torch.where(mask, self._light_blur(x), x)

        if self.sharpness_p > 0.0:
            mask = self._sample_mask(B, device, p=self.sharpness_p, apply_mask=apply_mask).view(-1, 1, 1, 1)
            factor = self.sharpness_factor
            if self.sharpness_min is not None and self.sharpness_max is not None:
                factor = self._sample_factor(B, device, self.sharpness_min, self.sharpness_max).view(-1, 1, 1, 1)
            x = torch.where(mask, self._adjust_sharpness(x, factor), x)

        if self._uses_intensity_augmentation():
            # TODO: Clamp only intensity-augmented samples when apply_mask is used,
            # especially for processor outputs that are not guaranteed to be in [0, 1].
            x = x.clamp(0.0, 1.0)

        if single:
            x = x.squeeze(0)
        return x

    def _uses_intensity_augmentation(self) -> bool:
        return (
            (self.brightness_delta > 0.0 and self.brightness_p > 0.0)
            or (self.contrast_delta > 0.0 and self.contrast_p > 0.0)
            or (self.noise_std > 0.0 and self.noise_p > 0.0)
            or self.blur_p > 0.0
            or self.sharpness_p > 0.0
        )

    @staticmethod
    def _sample_factor(batch_size: int, device: torch.device, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(batch_size, device=device)

    @staticmethod
    def _light_blur(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-3]
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=x.dtype,
            device=x.device,
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).expand(C, 1, 3, 3)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel, groups=C)

    @staticmethod
    def _adjust_sharpness(x: torch.Tensor, factor) -> torch.Tensor:
        C = x.shape[-3]
        kernel = torch.ones(3, 3, dtype=x.dtype, device=x.device)
        kernel[1, 1] = 5.0
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).expand(C, 1, 3, 3)
        blurred = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel, groups=C)
        return factor * x + (1.0 - factor) * blurred


# -- Pre-built pipelines ------------------------------------------------------

_WEAK = BatchedAugmentation(
    hflip=True,
    vflip=True,
    rot90=True,
)

_STRONG = BatchedAugmentation(
    hflip=True,
    vflip=True,
    rot90=True,
    brightness_delta=0.05,
    brightness_p=0.35,
    contrast_delta=0.08,
    contrast_p=0.35,
    noise_std=0.003,
    noise_p=0.25,
    blur_p=0.2,
    sharpness_min=0.75,
    sharpness_max=1.25,
    sharpness_p=0.35,
)

_AUGMENTATIONS = {
    "none": None,
    "weak": _WEAK,
    "strong": _STRONG,
}


def get_augmentation(augment_type: str) -> Optional[BatchedAugmentation]:
    """Return the augmentation pipeline for the given intensity level."""
    if augment_type not in _AUGMENTATIONS:
        raise ValueError(
            f"Unknown augmentation type '{augment_type}'. "
            f"Choose from {list(_AUGMENTATIONS)}."
        )
    return _AUGMENTATIONS[augment_type]
