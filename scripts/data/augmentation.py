"""Data augmentation pipelines and reproducibility helpers."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T


# ── Reproducibility ──────────────────────────────────────────────────────


def set_global_seed(seed: int) -> None:
    """Set seeds for ``torch``, ``numpy``, and ``random``."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


# ── Custom transforms ────────────────────────────────────────────────────


class AddGaussianNoise:
    """Add zero-mean Gaussian noise to a tensor."""

    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


# ── Pre-built pipelines ──────────────────────────────────────────────────

_WEAK = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
])

_STRONG = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
    T.RandomApply([AddGaussianNoise(std=0.0005)], p=0.5),
    T.RandomAdjustSharpness(0.5, p=0.5),
])

_AUGMENTATIONS = {
    "none": None,
    "weak": _WEAK,
    "strong": _STRONG,
}


def get_augmentation(augment_type: str) -> Optional[T.Compose]:
    """Return the augmentation pipeline for the given intensity level."""
    if augment_type not in _AUGMENTATIONS:
        raise ValueError(
            f"Unknown augmentation type '{augment_type}'. "
            f"Choose from {list(_AUGMENTATIONS)}."
        )
    return _AUGMENTATIONS[augment_type]
