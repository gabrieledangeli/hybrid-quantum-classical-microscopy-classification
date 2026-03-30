"""Classification metrics computation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute classification metrics suited for imbalanced multi-class problems.

    Returns a dict with aggregate and per-class metrics plus confusion matrix.
    """
    metrics: Dict[str, Any] = {}

    metrics["accuracy"] = float((preds == labels).mean())
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
    metrics["macro_precision"] = float(precision_score(labels, preds, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(labels, preds, average="macro", zero_division=0))

    n_classes = len(class_names) if class_names else max(int(labels.max()), int(preds.max())) + 1
    labels_eval = list(range(n_classes))

    per_class_p = precision_score(labels, preds, labels=labels_eval, average=None, zero_division=0)
    per_class_r = recall_score(labels, preds, labels=labels_eval, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, preds, labels=labels_eval, average=None, zero_division=0)

    per_class = []
    for idx, cls_idx in enumerate(labels_eval):
        name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
        per_class.append({
            "class": name,
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1": float(per_class_f1[idx]),
        })
    metrics["per_class"] = per_class

    metrics["confusion_matrix"] = confusion_matrix(labels, preds, labels=labels_eval)
    metrics["cm_labels"] = [
        class_names[i] if class_names and i < len(class_names) else str(i)
        for i in labels_eval
    ]

    counts = np.bincount(labels.astype(int), minlength=n_classes)
    if class_names:
        metrics["label_counts"] = {class_names[i]: int(counts[i]) for i in range(n_classes)}
    else:
        metrics["label_counts"] = {str(i): int(v) for i, v in enumerate(counts)}

    return metrics
