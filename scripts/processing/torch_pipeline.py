"""Differentiable (PyTorch) RAW-to-RGB processing modules."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

# -- Constant kernels / matrices -------------------------------------------

K_G = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=torch.float32) / 4
K_RB = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 4

M_RGB2YUV = torch.tensor([
    [0.299, 0.587, 0.114],
    [-0.14714119, -0.28886916, 0.43601035],
    [0.61497538, -0.51496512, -0.10001026],
], dtype=torch.float32)

M_YUV2RGB = torch.tensor([
    [1.0, -4.1827794561e-09, 1.1398830414],
    [1.0, -0.39464232326, -0.58062183857],
    [1.0, 2.0320618153, -1.2232658220e-09],
], dtype=torch.float32)

K_BLUR = torch.tensor([
    [6.9625e-08, 2.8089e-05, 2.0755e-04, 2.8089e-05, 6.9625e-08],
    [2.8089e-05, 1.1332e-02, 8.3731e-02, 1.1332e-02, 2.8089e-05],
    [2.0755e-04, 8.3731e-02, 6.1869e-01, 8.3731e-02, 2.0755e-04],
    [2.8089e-05, 1.1332e-02, 8.3731e-02, 1.1332e-02, 2.8089e-05],
    [6.9625e-08, 2.8089e-05, 2.0755e-04, 2.8089e-05, 6.9625e-08],
], dtype=torch.float32)

K_SHARP = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)

_DEFAULT_CAMERA = ([0., 0., 0., 0.], [1., 1., 1.], [1., 0., 0., 0., 1., 0., 0., 0., 1.])


# -- Bayer -> RGB helper ----------------------------------------------------


def raw2rgb(
    raw: torch.Tensor,
    black_level=None,
    reduce_size: bool = True,
    out_channels: int = 3,
) -> torch.Tensor:
    """Convert a ``(B, H, W)`` Bayer tensor to ``(B, C, H/2, W/2)`` or ``(B, C, H, W)``.

    Args:
        raw: ``(B, H, W)`` float tensor.
        black_level: Optional RGGB 4-element iterable to subtract.
        reduce_size: Halve spatial dims (True) or keep full size (False).
        out_channels: 3 (average greens) or 4.
    """
    assert out_channels in (3, 4)
    B, H, W = raw.shape
    assert H % 2 == 0 and W % 2 == 0, f"Dimensions must be even, got ({H}, {W})."

    if black_level is None:
        black_level = [0, 0, 0, 0]
    if not raw.is_floating_point():
        raw = raw.float()

    R = raw[:, 0::2, 0::2] - black_level[0]
    G1 = raw[:, 0::2, 1::2] - black_level[1]
    G2 = raw[:, 1::2, 0::2] - black_level[2]
    Bl = raw[:, 1::2, 1::2] - black_level[3]

    if reduce_size:
        rgb = torch.zeros(B, out_channels, H // 2, W // 2, device=raw.device, dtype=raw.dtype)
        rgb[:, 0] = R
        if out_channels == 3:
            rgb[:, 1] = (G1 + G2) / 2
            rgb[:, 2] = Bl
        else:
            rgb[:, 1], rgb[:, 2], rgb[:, 3] = G1, G2, Bl
    else:
        rgb = torch.zeros(B, out_channels, H, W, device=raw.device, dtype=raw.dtype)
        if out_channels == 3:
            rgb[:, 0, 0::2, 0::2] = R
            rgb[:, 1, 0::2, 1::2] = G1
            rgb[:, 1, 1::2, 0::2] = G2
            rgb[:, 2, 1::2, 1::2] = Bl
        else:
            rgb[:, 0, 0::2, 0::2] = R
            rgb[:, 1, 0::2, 1::2] = G1
            rgb[:, 2, 1::2, 0::2] = G2
            rgb[:, 3, 1::2, 1::2] = Bl
    return rgb


# -- Debayer conv layer ----------------------------------------------------


class _Debayer(nn.Conv2d):
    """Bilinear interpolation of Bayer mosaic channels."""

    def __init__(self) -> None:
        super().__init__(3, 3, kernel_size=3, padding=1, padding_mode="reflect", bias=False)
        self.weight.data.fill_(0)
        self.weight.data[0, 0] = K_RB.clone()
        self.weight.data[1, 1] = K_G.clone()
        self.weight.data[2, 2] = K_RB.clone()


# -- Simple mosaic -> RGB --------------------------------------------------


class RawToRGB(nn.Module):
    """Minimal: Bayer un-packing + optional mosaic normalisation.

    Args:
        reduce_size: Halve spatial dims.
        out_channels: 3 or 4.
        track_stages: Retain intermediate results for gradient analysis.
        normalize_mosaic: Optional normalisation applied after demosaic.
    """

    def __init__(
        self,
        reduce_size: bool = True,
        out_channels: int = 3,
        track_stages: bool = False,
        normalize_mosaic: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.reduce_size = reduce_size
        self.out_channels = out_channels
        self.track_stages = track_stages
        self.normalize_mosaic = normalize_mosaic
        self.stages: dict = {}
        self.buffer: dict = {}

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        self.stages = {}
        self.buffer = {}
        rgb = raw2rgb(raw, reduce_size=self.reduce_size, out_channels=self.out_channels)
        self.stages["demosaic"] = rgb
        if self.normalize_mosaic is not None:
            rgb = self.normalize_mosaic(rgb)
        if self.track_stages and raw.requires_grad:
            for v in self.stages.values():
                v.retain_grad()
        self.buffer["processed_rgb"] = rgb
        return rgb


# -- Neural-network processor ---------------------------------------------


class NNProcessing(nn.Module):
    """U-Net++ based RAW -> RGB processor.

    Requires ``segmentation-models-pytorch``.
    """

    def __init__(
        self,
        track_stages: bool = False,
        normalize_mosaic: Optional[nn.Module] = None,
        batch_norm_output: bool = True,
    ) -> None:
        super().__init__()
        if smp is None:
            raise ImportError(
                "segmentation_models_pytorch is required for NNProcessing. "
                "pip install segmentation-models-pytorch"
            )
        self.track_stages = track_stages
        self.normalize_mosaic = normalize_mosaic
        self.stages: dict = {}
        self.buffer: dict = {}
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34", encoder_depth=3,
            decoder_channels=[256, 128, 64], in_channels=3, classes=3,
        )
        self.batch_norm = nn.BatchNorm2d(3, affine=False) if batch_norm_output else None

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        self.stages = {}
        self.buffer = {}
        rgb = raw2rgb(raw)
        if self.normalize_mosaic is not None:
            rgb = self.normalize_mosaic(rgb)
        self.stages["demosaic"] = rgb
        rgb = self.model(rgb)
        if self.batch_norm is not None:
            rgb = self.batch_norm(rgb)
        self.stages["rgb"] = rgb
        if self.track_stages and raw.requires_grad:
            for v in self.stages.values():
                v.retain_grad()
        self.buffer["processed_rgb"] = rgb
        return rgb


# -- Parametrised differentiable pipeline ----------------------------------


class ParametrizedProcessing(nn.Module):
    """Fully differentiable processing pipeline with learnable camera params.

    Steps: mosaic -> debayer -> WB -> CC -> sharpen -> blur -> gamma [-> BN].
    """

    def __init__(
        self,
        camera_parameters=None,
        track_stages: bool = False,
        batch_norm_output: bool = True,
    ) -> None:
        super().__init__()
        self.track_stages = track_stages
        self.stages: dict = {}
        self.buffer: dict = {}

        if camera_parameters is None:
            camera_parameters = _DEFAULT_CAMERA
        bl, wb, cm = camera_parameters

        self.black_level = nn.Parameter(torch.as_tensor(bl, dtype=torch.float32))
        self.white_balance = nn.Parameter(torch.as_tensor(wb, dtype=torch.float32).reshape(1, 3))
        self.colour_correction = nn.Parameter(torch.as_tensor(cm, dtype=torch.float32).reshape(3, 3))
        self.gamma_correct = nn.Parameter(torch.tensor([2.2]))

        self.debayer = _Debayer()
        self.sharpening_filter = nn.Conv2d(1, 1, 3, padding=1, padding_mode="reflect", bias=False)
        self.sharpening_filter.weight.data[0, 0] = K_SHARP.clone()
        self.gaussian_blur = nn.Conv2d(1, 1, 5, padding=2, padding_mode="reflect", bias=False)
        self.gaussian_blur.weight.data[0, 0] = K_BLUR.clone()

        self.batch_norm = nn.BatchNorm2d(3, affine=False) if batch_norm_output else None

        self.register_buffer("M_RGB2YUV", M_RGB2YUV.clone())
        self.register_buffer("M_YUV2RGB", M_YUV2RGB.clone())

        self.additive_layer: Optional[nn.Parameter] = None

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        assert raw.ndim == 3, f"Expected (B, H, W), got {raw.shape}"
        self.stages = {}
        self.buffer = {}

        rgb = raw2rgb(raw, black_level=self.black_level, reduce_size=False).contiguous()
        self.stages["demosaic"] = rgb

        rgb = self.debayer(rgb)
        rgb = torch.einsum("bchw,kc->bchw", rgb, self.white_balance).contiguous()
        rgb = torch.einsum("bchw,kc->bkhw", rgb, self.colour_correction).contiguous()
        self.stages["color_correct"] = rgb

        yuv = torch.einsum("bchw,kc->bkhw", rgb, self.M_RGB2YUV).contiguous()

        # Sharpening (Y channel only, no in-place ops for autograd safety)
        y_sharp = self.sharpening_filter(yuv[:, [0]])
        yuv = torch.cat([y_sharp, yuv[:, 1:]], dim=1)

        if self.track_stages:
            rgb = torch.einsum("bchw,kc->bkhw", yuv.clone(), self.M_YUV2RGB).contiguous()
            self.stages["sharpening"] = rgb
            yuv = torch.einsum("bchw,kc->bkhw", rgb, self.M_RGB2YUV).contiguous()

        y_blur = self.gaussian_blur(yuv[:, [0]])
        yuv = torch.cat([y_blur, yuv[:, 1:]], dim=1)

        rgb = torch.einsum("bchw,kc->bkhw", yuv, self.M_YUV2RGB).contiguous()
        self.stages["gaussian"] = rgb

        rgb = torch.clip(rgb, 1e-5, 1)
        self.stages["clipped"] = rgb

        gamma = torch.nn.functional.softplus(self.gamma_correct) + 1e-6
        rgb = torch.exp((1 / gamma) * torch.log(rgb))
        self.stages["gamma_correct"] = rgb

        if self.additive_layer is not None:
            rgb = rgb + self.additive_layer
            self.stages["noise"] = rgb

        if self.batch_norm is not None:
            rgb = self.batch_norm(rgb)

        if self.track_stages and raw.requires_grad:
            for v in self.stages.values():
                v.retain_grad()

        self.buffer["processed_rgb"] = rgb
        return rgb
