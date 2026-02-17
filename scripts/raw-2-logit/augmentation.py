import random
from typing import List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torchvision.transforms as T


def set_global_seed(seed: int) -> None:
    """Sets the seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    # Note: 2**32 - 1 is the max seed for numpy
    np.random.seed(seed % (2**32 - 1))
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


class RandomRotate90(object):
    """
    Randomly rotate the image by 0, 90, 180, or 270 degrees.
    Note: This is different from T.RandomRotation(90) which rotates by ANY degree up to 90.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = random.randint(0, 3)
        return torch.rot90(x, k, dims=(-2, -1))

    def __repr__(self) -> str:
        return self.__class__.__name__


class ComposeState(T.Compose):
    """
    Composes several transforms together, with state retention support.
    Useful for segmentation where mask and image need identical transforms.
    """
    def __init__(self, transforms: List[Union[Any, Tuple[Any, bool]]]):
        self.transforms = []
        self.mask_transforms = []
        
        for t in transforms:
            apply_for_mask = True
            if isinstance(t, tuple):
                t, apply_for_mask = t
            self.transforms.append(t)
            if apply_for_mask:
                self.mask_transforms.append(t)
                
        self.seed: Optional[Any] = None

    def __call__(self, x: Any, retain_state: bool = False, mask_transform: bool = False) -> Any: # type: ignore
        """
        Args:
            retain_state (bool): If True, re-uses the random seed.
            mask_transform (bool): If True, applies only transforms suitable for masks.
        """
        if self.seed is not None:
            # Restore previous state
            set_global_seed(self.seed)
            
        if retain_state:
            # Generate and save a new seed, or keep existing
            self.seed = self.seed or torch.seed()
            set_global_seed(self.seed)
        else:
            # Reset state
            self.seed = None

        transforms = self.transforms if not mask_transform else self.mask_transforms
        
        for t in transforms:
            x = t(x)
            
        return x


# Define Augmentation Pipelines

augmentation_weak = ComposeState([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    RandomRotate90(),
])

augmentation_strong = ComposeState([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
    (T.RandomApply([AddGaussianNoise(std=0.0005)], p=0.5), False), # Noise not applied to mask
    (T.RandomAdjustSharpness(0.5, p=0.5), False),                  # Sharpness not applied to mask
])


def get_augmentation(augment_type: str) -> Optional[ComposeState]:
    """Factory to get the augmentation pipeline."""
    if augment_type == 'none':
        return None
    elif augment_type == 'weak':
        return augmentation_weak
    elif augment_type == 'strong':
        return augmentation_strong
    else:
        raise ValueError(f"Unknown augmentation type: {augment_type}")
