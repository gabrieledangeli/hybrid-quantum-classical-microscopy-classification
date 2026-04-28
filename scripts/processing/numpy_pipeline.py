"""NumPy-based RAW image processing pipeline (non-differentiable)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
)
from scipy import fftpack, ndimage
from scipy.signal import convolve2d
from skimage.color import rgb2yuv, yuv2rgb
from skimage.filters import unsharp_mask

# -- Debayer dispatch ------------------------------------------------------

_DEBAYER_FN = {
    "bilinear": demosaicing_CFA_Bayer_bilinear,
    "malvar2004": demosaicing_CFA_Bayer_Malvar2004,
    "menon2007": demosaicing_CFA_Bayer_Menon2007,
}

# -- Public API ------------------------------------------------------------


class RawProcessingPipeline:
    """Transform a RAW Bayer image to a processed RGB tensor.

    Steps: black-level removal -> demosaic -> WB -> colour correction ->
    dynamic-range normalisation -> sharpening -> denoising -> gamma.

    Args:
        camera_parameters: ``(black_level, white_balance, colour_matrix)``.
        debayer / sharpening / denoising: Algorithm names.
    """

    def __init__(
        self,
        camera_parameters: Tuple,
        debayer: str = "bilinear",
        sharpening: str = "unsharp_masking",
        denoising: str = "gaussian_denoising",
    ) -> None:
        self.camera_parameters = camera_parameters
        self.debayer = debayer
        self.sharpening = sharpening
        self.denoising = denoising

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        black_level, white_balance, colour_matrix = self.camera_parameters
        out = process_raw(
            img, black_level, white_balance, colour_matrix,
            debayer=self.debayer,
            sharpening=self.sharpening,
            denoising=self.denoising,
        )
        return torch.from_numpy(out.transpose(2, 0, 1).copy()).float()


# -- Full pipeline ---------------------------------------------------------


def process_raw(
    img: np.ndarray,
    black_level,
    white_balance,
    colour_matrix,
    debayer: str = "bilinear",
    sharpening: str = "unsharp_masking",
    sharp_radius: float = 1.0,
    sharp_amount: float = 1.0,
    denoising: str = "gaussian_denoising",
    median_kernel_size: int = 3,
    gaussian_sigma: float = 0.5,
    fft_fraction: float = 0.3,
    gamma: float = 2.2,
) -> np.ndarray:
    """Apply the full RAW -> RGB processing pipeline."""
    # Black-level removal
    img = _remove_black_level(img, black_level)

    # Demosaic
    if debayer not in _DEBAYER_FN:
        raise ValueError(f"Unknown debayer: '{debayer}'. Choose from {list(_DEBAYER_FN)}.")
    img = _DEBAYER_FN[debayer](img)

    # White-balance + colour correction
    img = _wb_correction(img, white_balance)
    img = _colour_correction(img, colour_matrix)

    # Dynamic range normalisation (99th percentile)
    p99 = np.percentile(img, 99)
    if p99 > 0.0:
        img = img / max(float(p99), 1e-8)
    img = np.clip(img, 0, 1)

    # Sharpening
    img = _apply_sharpening(img, sharpening, sharp_radius, sharp_amount)

    # Denoising
    img = _apply_denoising(img, denoising, median_kernel_size, gaussian_sigma, fft_fraction)

    # Gamma correction
    img = np.clip(img, 0, 1)
    return _adjust_gamma(img, gamma)


# -- Primitive ops ---------------------------------------------------------


def _remove_black_level(img: np.ndarray, bl) -> np.ndarray:
    out = img.copy()
    out[0::2, 0::2] -= bl[0]
    out[0::2, 1::2] -= bl[1]
    out[1::2, 0::2] -= bl[2]
    out[1::2, 1::2] -= bl[3]
    return out


def _wb_correction(img: np.ndarray, wb) -> np.ndarray:
    return img * np.asarray(wb)


def _colour_correction(img: np.ndarray, cm) -> np.ndarray:
    return np.einsum("ijk,lk->ijl", img, np.asarray(cm).reshape(3, 3))


def _apply_sharpening(img, method, radius, amount):
    if method == "sharpening_filter":
        return _conv_sharpen(img)
    if method == "unsharp_masking":
        return _unsharp(img, radius, amount)
    if method != "none":
        raise ValueError(f"Unknown sharpening: '{method}'.")
    return img


def _unsharp(img, radius=1.0, amount=1.0):
    yuv = rgb2yuv(img)
    yuv[:, :, 0] = unsharp_mask(yuv[:, :, 0], radius=radius, amount=amount, channel_axis=None, preserve_range=True)
    return yuv2rgb(yuv)


_SHARP_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def _conv_sharpen(img, iterations=1):
    yuv = rgb2yuv(img)
    for _ in range(iterations):
        yuv[:, :, 0] = convolve2d(yuv[:, :, 0], _SHARP_KERNEL, "same", boundary="fill", fillvalue=0)
    return yuv2rgb(yuv)


def _apply_denoising(img, method, median_ks, gauss_sigma, fft_frac):
    if method == "gaussian_denoising":
        return _gaussian_denoise(img, gauss_sigma)
    if method == "median_denoising":
        return _median_denoise(img, median_ks)
    if method == "fft_denoising":
        return _fft_denoise(img, fft_frac)
    if method != "none":
        raise ValueError(f"Unknown denoising: '{method}'.")
    return img


def _median_denoise(img, size=3):
    yuv = rgb2yuv(img)
    yuv[:, :, 0] = ndimage.median_filter(yuv[:, :, 0], size)
    return yuv2rgb(yuv)


def _gaussian_denoise(img, sigma=0.5):
    yuv = rgb2yuv(img)
    yuv[:, :, 0] = ndimage.gaussian_filter(yuv[:, :, 0], sigma)
    return yuv2rgb(yuv)


def _fft_denoise(img, keep_fraction=0.3, row_cut=True, column_cut=True):
    yuv = rgb2yuv(img)
    y = yuv[:, :, 0]
    ft = fftpack.fft2(y).copy()
    r, c = ft.shape
    if row_cut:
        ft[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    if column_cut:
        ft[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    yuv[:, :, 0] = fftpack.ifft2(ft).real
    return yuv2rgb(yuv)


def _adjust_gamma(img, gamma=2.2):
    return img ** (1.0 / gamma)
