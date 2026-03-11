from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Compared to standard CE, the (1 - p_t)^gamma factor down-weights
    well-classified (easy) examples and lets the model focus on hard,
    misclassified samples.

    Args:
        weight: Per-class weights tensor (same semantics as nn.CrossEntropyLoss).
        gamma: Focusing parameter. 0 = standard CE; 2 = default; higher = more focus.
        label_smoothing: Smoothing factor applied to target distribution before loss.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute log-softmax for numerical stability.
        log_probs = F.log_softmax(inputs, dim=-1)  # [B, C]
        probs = log_probs.exp()                     # [B, C]

        n_classes = inputs.shape[-1]

        # Build target distribution (one-hot, optionally smoothed).
        if self.label_smoothing > 0.0:
            smooth = self.label_smoothing / n_classes
            target_dist = torch.full_like(probs, smooth)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
        else:
            target_dist = F.one_hot(targets, n_classes).float()

        # Focal modulation: (1 - p_t)^gamma per class dimension.
        focal_weight = (1.0 - probs).pow(self.gamma)

        # Focal cross-entropy: -alpha * (1-p)^gamma * log(p), summed over classes.
        loss = -(focal_weight * target_dist * log_probs).sum(dim=-1)  # [B]

        # Per-class weighting (same as nn.CrossEntropyLoss weight semantics).
        if self.weight is not None:
            sample_weight = self.weight[targets]  # [B]
            loss = loss * sample_weight

        if self.reduction == "mean":
            if self.weight is not None:
                return loss.sum() / sample_weight.sum().clamp(min=1e-8)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
