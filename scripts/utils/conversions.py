"""Tensor <-> NumPy conversion utilities."""

from __future__ import annotations

import numpy as np
import torch


def np2torch(arr: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor, re-ordering to channel-first.

    * 2-D -> returned as-is.
    * 3-D ``(H, W, C)`` -> ``(C, H, W)``   (detected by ``C <= 4, H > 4``).
    * 4-D ``(B, H, W, C)`` -> ``(B, C, H, W)``  (same heuristic).
    """
    tensor = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else torch.as_tensor(arr)
    if not tensor.is_floating_point():
        tensor = tensor.float()

    if tensor.ndim == 3:
        d0, _d1, d2 = tensor.shape
        if d2 <= 4 and d0 > 4:
            return tensor.permute(2, 0, 1)
    elif tensor.ndim == 4:
        _b, _d1, _d2, d3 = tensor.shape
        if d3 <= 4 and _d1 > 4:
            return tensor.permute(0, 3, 1, 2)
    return tensor


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a float32 NumPy array, channel-last.

    * 3-D ``(C, H, W)`` -> ``(H, W, C)``  (if ``C <= 3``).
    * 4-D ``(B, C, H, W)`` -> ``(B, H, W, C)``.
    """
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] <= 3:
        return arr.transpose(1, 2, 0)
    if arr.ndim == 4:
        return arr.transpose(0, 2, 3, 1)
    return arr
