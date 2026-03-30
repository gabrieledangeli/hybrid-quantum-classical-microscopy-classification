"""Visualisation / logging helpers."""

from __future__ import annotations

import os

import torch
import torchvision.utils as vutils


def log_tensor(batch: torch.Tensor, path: str, save_tensors: bool = True, nrow: int = 8) -> None:
    """Save a tensor as ``.pt`` and optionally as an image grid (``.png``)."""
    if save_tensors:
        torch.save(batch, path)

    is_image = batch.ndim == 4 or (batch.ndim == 3 and batch.shape[0] in (1, 3, 4))
    if not is_image:
        return

    img_path = path.replace(".pt", ".png")
    directory, filename = os.path.split(img_path)
    img_path = os.path.join(directory, f"img_{filename}")

    try:
        grid = vutils.make_grid(batch, nrow=nrow, normalize=True).squeeze()
        vutils.save_image(grid, img_path)
    except Exception as exc:
        print(f"[Warning] Could not save image for {path}: {exc}")
