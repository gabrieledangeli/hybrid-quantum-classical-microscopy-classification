"""Focal Loss for imbalanced classification.

Reference: Lin *et al.*, "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""Focal Loss: :math:`\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log p_t`.

    Compared with standard CE the :math:`(1-p_t)^\gamma` factor down-weights
    well-classified (easy) examples so the model can focus on hard negatives.

    Args:
        weight: Per-class weight tensor (same semantics as ``nn.CrossEntropyLoss``).
        gamma: Focusing parameter (0 -> standard CE; 2 -> default).
        label_smoothing: Smoothing applied to the target distribution.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight: Optional[torch.Tensor] = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()
        n_classes = inputs.shape[-1]

        # Target distribution (optionally smoothed)
        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing / n_classes
            target_dist = torch.full_like(probs, smooth)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
        else:
            target_dist = F.one_hot(targets, n_classes).float()

        focal_weight = (1.0 - probs).pow(self.gamma)
        loss = -(focal_weight * target_dist * log_probs).sum(dim=-1)  # [B]

        if self.weight is not None:
            sample_w = self.weight[targets]
            loss = loss * sample_w

        if self.reduction == "mean":
            if self.weight is not None:
                return loss.sum() / sample_w.sum().clamp(min=1e-8)  # type: ignore[possibly-undefined]
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
