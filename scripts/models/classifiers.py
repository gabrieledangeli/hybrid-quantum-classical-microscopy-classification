"""ResNet classifiers and End-to-End model wrapper."""

from __future__ import annotations

from typing import List, Optional, Set

import torch
import torch.nn as nn
from PIL import Image as _PILImage  # Preload Pillow before torchvision on Windows.
from torchvision.models import (
    ResNet,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

# -- ResNet factory --------------------------------------------------------

_RESNET_REGISTRY = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
}


def resnet_model(
    model_name: str = "resnet18",
    pretrained: bool = True,
    fc_out_features: int = 2,
    freeze_backbone: bool = False,
) -> ResNet:
    """Build a ResNet with a replaced final FC layer.

    When *freeze_backbone* is ``True``, only ``layer4`` (the last residual
    stage) and the freshly-initialised ``fc`` head are trainable; everything
    else (stem, layer1-3) is frozen for efficient fine-tuning.

    Note: this differs from the hybrid model's ``_ResNetFeatureExtractor``,
    which has no FC head (the head lives downstream as TN bridge + VQC), so
    ``freeze=True`` there leaves only ``layer4`` trainable.
    """
    key = model_name.lower().strip()
    if key not in _RESNET_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from {list(_RESNET_REGISTRY)}.")

    build_fn, default_weights = _RESNET_REGISTRY[key]
    model = build_fn(weights=default_weights if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, fc_out_features, bias=True)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    return model


# -- End-to-End model ------------------------------------------------------


class EndToEndModel(nn.Module):
    """Pipeline wrapper: *processor -> augmentation -> normalisation -> classifier*.

    When *conditional augmentation* is enabled only samples whose labels belong
    to ``augmentation_target_classes`` are augmented with probability 1, while
    non-target samples are augmented with ``non_target_augmentation_prob``.
    """

    def __init__(
        self,
        classifier: nn.Module,
        processor: Optional[nn.Module] = None,
        normalization: Optional[nn.Module] = None,
        augmentation: Optional[nn.Module] = None,
        augmentation_target_classes: Optional[List[int]] = None,
        non_target_augmentation_prob: float = 1.0,
        augmentation_on_eval: bool = False,
        freeze_processor: bool = False,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.processor = processor
        self.normalization = normalization
        self.augmentation = augmentation
        self.augmentation_target_classes: Optional[Set[int]] = (
            set(augmentation_target_classes) if augmentation_target_classes is not None else None
        )
        self.non_target_augmentation_prob = float(non_target_augmentation_prob)
        self.augmentation_on_eval = augmentation_on_eval
        self.freeze_processor = freeze_processor

        if freeze_processor and self.processor is not None:
            _freeze(self.processor)

    # -- Forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.processor is not None:
            x = self.processor(x)

        if self.augmentation is not None and (self.training or self.augmentation_on_eval):
            x = self._apply_augmentation(x, labels)

        if self.normalization is not None:
            x = self.normalization(x)

        return self.classifier(x)

    # -- Internals ---------------------------------------------------------

    def _apply_augmentation(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply augmentation, optionally conditioned on class membership."""
        if self.augmentation is None:
            return x
        if self.augmentation_target_classes is None or labels is None:
            return self.augmentation(x)

        target_classes = torch.as_tensor(
            sorted(self.augmentation_target_classes),
            device=labels.device,
            dtype=labels.dtype,
        )
        apply_mask = torch.isin(labels, target_classes)
        if self.non_target_augmentation_prob > 0.0:
            sampled_non_targets = (
                torch.rand(labels.shape[0], device=labels.device) < self.non_target_augmentation_prob
            )
            apply_mask = apply_mask | (~apply_mask & sampled_non_targets)

        if not apply_mask.any():
            return x
        return self.augmentation(x, apply_mask=apply_mask)


# -- Channel normalisation ------------------------------------------------


class TensorNormalize(nn.Module):
    """Channel-wise normalisation for ``(B, C, H, W)`` or ``(C, H, W)`` tensors."""

    def __init__(self, mean: List[float], std: List[float], eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.clamp(
            torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1), min=eps
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        if x.ndim == 4:
            return (x - self.mean) / self.std
        raise ValueError(f"Expected 3-D or 4-D tensor, got {x.ndim}-D.")


# -- Utilities -------------------------------------------------------------


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
