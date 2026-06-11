"""ResNet classifiers and End-to-End model wrapper."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Set

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

_RESNET_BACKBONE_ROOTS = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")


def _is_resnet_backbone_key(key: str) -> bool:
    """Return True for torchvision ResNet backbone keys, excluding ``fc``."""
    root = key.split(".", 1)[0]
    return root in _RESNET_BACKBONE_ROOTS


def load_resnet_backbone_weights(model: ResNet, path: str) -> Dict[str, int]:
    """Load a strict ResNet backbone-only state dict into ``model``."""
    backbone_state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(backbone_state, Mapping):
        raise ValueError(f"Backbone checkpoint at {path} is not a state dict.")

    invalid_keys = [
        str(key) for key, value in backbone_state.items()
        if not torch.is_tensor(value) or not _is_resnet_backbone_key(str(key))
    ]
    if invalid_keys:
        raise ValueError(
            "Backbone checkpoint must contain only ResNet backbone tensor keys "
            f"(conv1, bn1, layer1-4). Invalid keys: {invalid_keys[:10]}"
        )

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    missing_backbone = [
        key for key in missing
        if key.split(".", 1)[0] in _RESNET_BACKBONE_ROOTS
    ]
    if missing_backbone:
        raise ValueError(f"Backbone checkpoint is missing ResNet keys: {missing_backbone[:10]}")
    if unexpected:
        raise ValueError(f"Backbone checkpoint has unexpected keys: {unexpected[:10]}")

    return {
        "loaded_keys": len(backbone_state),
    }


def extract_resnet_backbone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only torchvision-style ResNet backbone tensors from ``model``."""
    backbone_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if _is_resnet_backbone_key(key)
    }
    if not backbone_state:
        raise ValueError("No ResNet backbone tensors were found in the model.")
    return backbone_state


def resnet_model(
    fc_out_features: int,
    model_name: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    backbone_weights: Optional[str] = None,
) -> ResNet:
    """Build a ResNet with a replaced final FC layer.

    ``pretrained`` controls whether the backbone is initialised from pretrained
    weights. If ``backbone_weights`` is provided, it is used as the pretrained
    source; otherwise torchvision ImageNet weights are used. The final ``fc``
    layer is always freshly initialised for the requested label space.

    When *freeze_backbone* is ``True``, only ``layer4`` (the last residual
    stage) and the freshly-initialised ``fc`` head are trainable; everything
    else (stem, layer1-3) is frozen for efficient fine-tuning.

    Note: this differs from the hybrid model's ``_ResNetFeatureExtractor``,
    which has no FC head (the head lives downstream as TN bridge + VQC), so
    ``freeze=True`` there leaves only ``layer4`` trainable.
    """
    model_name = model_name.lower().strip()
    if model_name not in _RESNET_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from {list(_RESNET_REGISTRY)}.")
    if backbone_weights and not pretrained:
        raise ValueError("backbone_weights requires pretrained=True.")
    if freeze_backbone and not pretrained:
        raise ValueError(
            "freeze_backbone=True requires pretrained=True; "
            "otherwise stem/layer1-3 would remain randomly initialized and frozen."
        )

    build_fn, default_weights = _RESNET_REGISTRY[model_name]
    model = build_fn(weights=None if backbone_weights or not pretrained else default_weights)
    if pretrained and backbone_weights:
        info = load_resnet_backbone_weights(model, backbone_weights)
        print(
            f"[Backbone] Loaded custom {model_name} backbone from {backbone_weights} "
            f"({info['loaded_keys']} tensors)"
        )
    elif pretrained:
        print(f"[Backbone] ImageNet-pretrained {model_name}")
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

    When *conditional augmentation* is enabled, samples whose labels belong to
    ``augmentation_target_classes`` are eligible for the augmentation pipeline,
    while non-target samples are eligible with ``non_target_augmentation_prob``.
    Each transform inside the augmentation module still uses its own sampling
    probability.
    """

    def __init__(
        self,
        classifier: nn.Module,
        processor: Optional[nn.Module] = None,
        normalization: Optional[nn.Module] = None,
        augmentation: Optional[nn.Module] = None,
        augmentation_target_classes: Optional[List[int]] = None,
        non_target_augmentation_prob: float = 1.0,
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
        self.freeze_processor = freeze_processor

        if freeze_processor and self.processor is not None:
            _freeze(self.processor)

    # -- Forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.processor is not None:
            x = self.processor(x)

        if self.augmentation is not None and self.training:
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
