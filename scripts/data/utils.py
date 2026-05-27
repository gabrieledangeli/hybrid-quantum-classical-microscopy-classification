"""Dataset I/O utilities, cross-validation splits, and sampling helpers."""

from __future__ import annotations

import os
from collections import Counter
from typing import List, Tuple

import numpy as np
import tifffile
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# -- Constants ----------------------------------------------------------------

IMAGE_FILE_TYPES = {"png", "tif", "tiff"}

# -- Image I/O ----------------------------------------------------------------


def load_image(path: str) -> np.ndarray:
    """Load an image file while preserving its on-disk dtype."""
    return np.asarray(tifffile.imread(path))


def list_images_in_dir(directory: str) -> List[str]:
    """Return sorted list of absolute image paths in directory."""
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.rsplit(".", 1)[-1].lower() in IMAGE_FILE_TYPES
    ]


# -- Cross-validation ---------------------------------------------------------


def make_stratified_splits(
    dataset,
    n_splits: int,
    seed: int,
    train_size: float = 0.7,
    strategy: str = "kfold",
) -> List[Tuple[List[int], List[int]]]:
    """Create stratified cross-validation splits.

    Args:
        dataset: Must expose a ``.labels`` attribute (list/array of ints).
        n_splits: Number of folds or re-shuffles.
        seed: Random seed.
        train_size: Train proportion for ``shuffle_split`` only. With
            ``kfold``, the train/validation ratio is determined by
            ``n_splits`` and this argument is ignored.
        strategy: ``'kfold'`` or ``'shuffle_split'``.

    Returns:
        List of ``(train_indices, val_indices)`` tuples.
    """
    y = np.asarray(dataset.labels)

    if strategy not in ("kfold", "shuffle_split"):
        raise ValueError(f"Unknown CV strategy: '{strategy}'. Use 'kfold' or 'shuffle_split'.")

    class_counts = Counter(y.tolist())
    min_class_count = min(class_counts.values()) if class_counts else 0

    if strategy == "kfold":
        # In k-fold CV, train_size is implied by n_splits and is not used.
        if n_splits < 2:
            raise ValueError(
                f"StratifiedKFold requires n_splits >= 2, got {n_splits}. "
                "Use --cv_strategy shuffle_split for a single train/test split."
            )
        if n_splits > min_class_count:
            raise ValueError(
                f"n_splits={n_splits} exceeds min class count ({min_class_count}). "
                f"Use n_splits <= {min_class_count}."
            )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        if not 0.0 < train_size < 1.0:
            raise ValueError(f"train_size must be in (0, 1), got {train_size}")
        for cls_idx, cls_count in class_counts.items():
            if cls_count < 2:
                raise ValueError(
                    f"Class {cls_idx} has only {cls_count} sample(s). "
                    "StratifiedShuffleSplit requires >= 2."
                )
            cls_train = int(np.floor(cls_count * train_size))
            cls_val = cls_count - cls_train
            if cls_train < 1 or cls_val < 1:
                raise ValueError(
                    f"Class {cls_idx} cannot be split with train_size={train_size}: "
                    f"train={cls_train}, val={cls_val}."
                )
        cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)

    dummy_x = np.zeros(len(y))
    return [
        (idx_train.tolist(), idx_val.tolist())
        for idx_train, idx_val in cv.split(dummy_x, y)
    ]


# -- Sampling helpers ---------------------------------------------------------


def compute_soft_undersampling_weights(
    train_labels: List[int],
    class_counts: Counter,
    strength: float,
) -> Tuple[torch.DoubleTensor, dict]:
    """Per-sample sampling weights with soft undersampling.

    Args:
        train_labels: Label for each training sample.
        class_counts: ``Counter`` of labels.
        strength: 0 = natural distribution, 1 = class-balanced.

    Returns:
        ``(sample_weights_tensor, per_class_weight_dict)``
    """
    if not class_counts:
        raise ValueError("class_counts is empty.")

    strength = float(np.clip(strength, 0.0, 1.0))
    max_count = max(class_counts.values())
    class_weight = {
        cls: (max_count / max(1, cnt)) ** strength
        for cls, cnt in class_counts.items()
    }
    sample_weights = [class_weight[lbl] for lbl in train_labels]
    return torch.DoubleTensor(sample_weights), class_weight


def get_current_undersample_strength(
    epoch: int,
    max_strength: float,
    warmup_epochs: int,
) -> float:
    """Linear warm-up for soft-undersampling strength."""
    max_strength = float(np.clip(max_strength, 0.0, 1.0))
    warmup_epochs = max(1, int(warmup_epochs))
    if warmup_epochs == 1:
        return max_strength
    progress = min(1.0, max(0.0, (epoch - 1) / (warmup_epochs - 1)))
    return max_strength * progress


def format_class_counts(class_counts: Counter, class_names: List[str]) -> dict:
    """Return ``{class_name: count}`` for pretty-printing."""
    return {class_names[i]: int(class_counts.get(i, 0)) for i in range(len(class_names))}
