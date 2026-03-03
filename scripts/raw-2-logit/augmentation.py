import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T


def set_global_seed(seed: int) -> None:
    """Sets the seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))  # NOTE: 2**32 - 1 is the max seed for numpy
    random.seed(seed)


class AddGaussianNoise(object):
    """
    Add Gaussian noise to a tensor.
    """
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"

# Define Augmentation Pipelines

augmentation_weak = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
])

augmentation_strong = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
    T.RandomApply([AddGaussianNoise(std=0.0005)], p=0.5),
    T.RandomAdjustSharpness(0.5, p=0.5),
])


def get_augmentation(augment_type: str) -> Optional[T.Compose]:
    """Factory to get the augmentation pipeline."""
    if augment_type == 'none':
        return None
    elif augment_type == 'weak':
        return augmentation_weak
    elif augment_type == 'strong':
        return augmentation_strong
    else:
        raise ValueError(f"Unknown augmentation type: {augment_type}")
