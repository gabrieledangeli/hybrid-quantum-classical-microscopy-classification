"""Training entry-point for the Hybrid Quantum-Classical CNN pipeline.

Usage from the `scripts/` directory:

    python train.py --dataset Microscopy --data_dir ./data --epochs 30
"""

from __future__ import annotations

import os
import json
import argparse
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as _PILImage  # Preload Pillow before torchvision on Windows.
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import wandb

# -- Local imports ---------------------------------------------------------
from config import parse_args
from data.augmentation import get_augmentation, set_global_seed
from data.dataset import Subset, get_dataset
from data.utils import (
    compute_soft_undersampling_weights,
    format_class_counts,
    get_current_undersample_strength,
    make_stratified_splits,
)
from models.classifiers import (
    EndToEndModel,
    TensorNormalize,
    extract_resnet_backbone_state_dict,
    resnet_model,
)
from models.hybrid import hybrid_vqc_model
from models.losses import FocalLoss
from processing.numpy_pipeline import RawProcessingPipeline
from processing.torch_pipeline import (
    NNProcessing,
    ParametrizedProcessing,
    RawToRGB,
)
from utils.metrics import compute_metrics

# Keep every W&B run cache in a single <project_root>/wandb folder
os.environ.setdefault(
    "WANDB_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# -- Device ----------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ===========================================================================
#  Training / Validation loops
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Run one training epoch and return ``(avg_loss, metrics)``."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    if len(loader) == 0:
        raise RuntimeError("Training dataloader is empty.")

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs, labels=labels) # Pass labels for conditional augmentation if enabled.
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.append(outputs.argmax(1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels), class_names)
    return avg_loss, metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray]:
    """Run validation and return ``(avg_loss, metrics, labels, preds)``."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    if len(loader) == 0:
        raise RuntimeError("Validation dataloader is empty.")

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        running_loss += criterion(outputs, labels).item()
        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    metrics = compute_metrics(preds_np, labels_np, class_names)
    return avg_loss, metrics, labels_np, preds_np


# ===========================================================================
#  Building blocks helpers
# ===========================================================================


def _build_processor(args, dataset):
    """Return ``(processor, normalization)`` based on CLI flags."""
    is_rgb = args.dataset in ("MicroscopyRGB", "MLL23") or getattr(dataset, "is_rgb", False)

    if is_rgb:
        dataset.transform = T.ToTensor()
        dataset.clear_cache()
        return None, TensorNormalize(dataset.dataset_mean, dataset.dataset_std)

    if args.processing_mode == "static":
        dataset.transform = T.Compose([
            RawProcessingPipeline(
                camera_parameters=dataset.camera_parameters,
                debayer=args.sp_debayer,
                sharpening=args.sp_sharpening,
                denoising=args.sp_denoising,
            ),
        ])
        dataset.clear_cache()
        return None, TensorNormalize(dataset.dataset_mean, dataset.dataset_std)

    mosaic_norm = TensorNormalize([0.5663, 0.1401, 0.0731], [0.097, 0.0423, 0.008])
    # TODO: Study the implementation of these parametrized processing modules.
    if args.processing_mode == "parametrized":
        return ParametrizedProcessing(camera_parameters=dataset.camera_parameters, batch_norm_output=True), None
    if args.processing_mode == "neural_network":
        return NNProcessing(batch_norm_output=True), mosaic_norm
    # "none"
    # TODO: Study whether this minimal RawToRGB path is a useful baseline.
    return RawToRGB(reduce_size=True, out_channels=3), mosaic_norm


def _build_classifier(args, n_classes: int):
    """Instantiate classifier (pure ResNet or hybrid VQC)."""
    if args.classifier_network == "hybrid_vqc":
        return hybrid_vqc_model(
            fc_out_features=n_classes,
            n_qubits=args.n_qubits,
            vqc_layers=args.vqc_layers,
            vqc_readout=args.vqc_readout,
            resnet_freeze_mode=args.resnet_freeze_mode,
            backbone=args.backbone,
            backbone_weights=args.backbone_weights,
            pretrained=args.classifier_pretrained,
            angle_map_type=args.vqc_angle_map,
            angle_map_hidden_dim=args.vqc_angle_hidden_dim,
            angle_map_dropout=args.vqc_angle_dropout,
            vqc_entangler=args.vqc_entangler,
            vqc_final_rotation=args.vqc_final_rotation,
            vqc_blocks=args.vqc_blocks,
            incremental_reuploading=args.incremental_reuploading,
        )
    elif args.classifier_network in ("resnet18", "resnet34", "resnet50"):
        return resnet_model(
            fc_out_features=n_classes,
            model_name=args.classifier_network,
            pretrained=args.classifier_pretrained,
            backbone_weights=args.backbone_weights,
            resnet_freeze_mode=args.resnet_freeze_mode,
        )
    else:
        raise ValueError(f"Unsupported classifier network: {args.classifier_network}.")


def _build_criterion(args, class_counts: Counter, n_classes: int, device: torch.device):
    """Return the training criterion, optionally with class weights."""
    use_weighted = args.imbalance_handling in ("loss", "both")
    weight = None

    if use_weighted:
        beta = 0.999
        raw_w = [0.0] * n_classes
        present = [i for i in range(n_classes) if class_counts.get(i, 0) > 0]
        for i in present:
            eff = (1 - beta ** class_counts[i]) / (1 - beta)
            raw_w[i] = 1.0 / max(eff, 1e-12)
        s = sum(raw_w[i] for i in present)
        norm = len(present) / s if s > 0 else 1.0
        for i in present:
            raw_w[i] *= norm
        weight = torch.FloatTensor(raw_w).to(device)

    if args.loss_fn == "focal":
        return FocalLoss(weight=weight, gamma=args.focal_gamma, label_smoothing=args.label_smoothing).to(device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)


# Quantum-circuit parameters: rotation offsets and encoding scales live on
# angle manifolds where L2 decay toward zero drives the circuit toward the
# identity and shuts off the data encoding, so they are excluded from decay.
_NO_DECAY_PARAM_NAMES = {
    "thetas",
    "feature_weights",
    "entangle_weights",
    "final_rot",
    "angle_scale",
}


def _build_optimizer(args, model: nn.Module):
    """Differential LR optimizer: backbone (low lr) vs head (high lr).

    Quantum parameters train at the head LR but with weight_decay=0.
    """
    backbone_params, head_params, no_decay_params = [], [], []

    uses_pretrained_backbone = bool(args.classifier_pretrained)

    if args.classifier_network in ("resnet18", "resnet34", "resnet50") and uses_pretrained_backbone:
        for name, p in model.classifier.named_parameters():
            if not p.requires_grad:
                continue
            (head_params if name.startswith("fc.") else backbone_params).append(p)
        if model.processor is not None:
            head_params.extend(p for p in model.processor.parameters() if p.requires_grad)
    elif args.classifier_network in ("resnet18", "resnet34", "resnet50"):
        head_params = [p for p in model.parameters() if p.requires_grad]
    else:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.rsplit(".", 1)[-1] in _NO_DECAY_PARAM_NAMES:
                no_decay_params.append(p)
            elif "resnet" in name or "backbone" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": args.lr_head, "weight_decay": 0.0})
    if not groups:
        raise ValueError("No trainable parameters found.")

    return optim.AdamW(groups, weight_decay=args.weight_decay)


def _build_scheduler(args, optimizer):
    """Reduce LR when validation macro F1 stops improving."""
    min_lrs = [group["lr"] * 0.01 for group in optimizer.param_groups]
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=min_lrs,
    )


def _build_sampler(args, train_labels: List[int], class_counts: Counter) -> Tuple[Optional[WeightedRandomSampler], bool]:
    """Return ``(sampler, is_dynamic)`` for class-imbalance handling."""
    if args.imbalance_handling not in ("sampler", "both"):
        return None, False

    if args.sampler_mode == "soft_undersample":
        # Ramp soft undersampling over the first 10 epochs to avoid abrupt distribution shifts.
        init_strength = get_current_undersample_strength(1, args.undersample_strength, 10)
        weights, _ = compute_soft_undersampling_weights(train_labels, class_counts, init_strength)
        return WeightedRandomSampler(weights, len(train_labels), replacement=True), True

    weights = torch.DoubleTensor([
        (1.0 / class_counts[train_label]) if class_counts[train_label] > 0 else 0.0
        for train_label in train_labels
    ])
    return WeightedRandomSampler(weights, len(train_labels), replacement=True), False


def _make_base_run_name(args: argparse.Namespace, started_at: datetime) -> str:
    """Generate the shared base name for W&B and local results."""
    started_label = started_at.strftime("%d-%m-%Y_%Hh%M")
    if args.run_name:
        return f"{started_label}_{args.run_name}"

    model_tag = args.classifier_network
    if args.classifier_network == "hybrid_vqc":
        model_tag = (
            f"{args.classifier_network}-{args.backbone}-q{args.n_qubits}"
            f"-{args.vqc_angle_map}map"
        )
        if args.vqc_blocks > 1:
            model_tag += f"-b{args.vqc_blocks}"
        if args.vqc_entangler != "cnot":
            model_tag += f"-{args.vqc_entangler}"
        if args.vqc_final_rotation:
            model_tag += "-frot"
        if args.incremental_reuploading:
            model_tag += "-increup"

    return (
        f"{started_label}_{model_tag}"
        f"_loss-{args.loss_fn}"
        f"_aug-{args.augmentation}_imbalance_handling-{args.imbalance_handling}"
    )


def _make_fold_run_name(base_run_name: str, fold: int, n_splits: int) -> str:
    """Generate a fold-specific run name."""
    return f"{base_run_name}_fold{fold + 1}of{n_splits}"


def _make_run_dir(args: argparse.Namespace, base_run_name: str, fold: int) -> str:
    """Generate the local results directory for one fold."""
    return os.path.join(
        "results",
        args.dataset,
        args.processing_mode,
        base_run_name,
        f"fold_{fold + 1}",
    )


def _make_backbone_metadata(
    args: argparse.Namespace,
    display_class_names: List[str],
    fold: int,
    epoch: int,
    val_loss: float,
    val_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Metadata saved next to reusable backbone checkpoints."""
    return {
        "created_at": datetime.now().strftime("%d-%m-%Y_%H-%M"),
        "selection_metric": "val_macro_f1",
        "dataset": args.dataset,
        "processing_mode": args.processing_mode,
        "classifier_network": args.classifier_network,
        "fold": fold + 1,
        "epoch": epoch,
        "seed": args.seed,
        "classes": display_class_names,
        "val_loss": float(val_loss),
        "val_metrics": val_metrics,
    }


def _save_backbone_checkpoint(
    model: nn.Module,
    run_dir: str,
    metadata: Dict[str, Any],
) -> None:
    """Save only reusable ResNet backbone tensors plus a readable metadata file."""
    backbone_state = extract_resnet_backbone_state_dict(model)
    torch.save(backbone_state, os.path.join(run_dir, "backbone.pth"))
    with open(os.path.join(run_dir, "backbone_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _copy_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot the current model state without keeping GPU tensor references."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _log_wandb_confusion_matrix(
    key: str,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
) -> None:
    """Log a single confusion matrix plot to W&B."""
    try:
        wandb.log({
            key: wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels.tolist(),
                preds=preds.tolist(),
                class_names=class_names,
            )
        })
    except Exception as exc:
        print(f"[Warning] W&B confusion matrix log failed for '{key}': {exc}")


def _log_wandb_per_class_table(key: str, per_class: List[Dict[str, Any]]) -> None:
    """Log best-epoch per-class precision, recall, and F1 to W&B."""
    try:
        table = wandb.Table(columns=["Class", "Best Precision", "Best Recall", "Best F1"])
        for row in per_class:
            table.add_data(row["class"], row["precision"], row["recall"], row["f1"])
        wandb.log({key: table})
    except Exception as exc:
        print(f"[Warning] W&B per-class table log failed for '{key}': {exc}")


def _make_fold_summary(
    best_val_snapshot: Dict[str, Any],
    test_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract the best validation metrics that should be aggregated across folds."""
    metrics = best_val_snapshot["metrics"]
    summary = {
        "fold": best_val_snapshot["fold"],
        "best_epoch": best_val_snapshot["epoch"],
        "val_loss": best_val_snapshot["loss"],
        "val_accuracy": metrics["accuracy"],
        "val_balanced_accuracy": metrics["balanced_accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_weighted_f1": metrics["weighted_f1"],
        "val_macro_precision": metrics["macro_precision"],
        "val_macro_recall": metrics["macro_recall"],
    }
    if test_snapshot is not None:
        test_metrics = test_snapshot["metrics"]
        summary.update({
            "test_loss": test_snapshot["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
        })
    return summary


def _log_wandb_cv_summary(
    args: argparse.Namespace,
    base_run_name: str,
    fold_summaries: List[Dict[str, Any]],
    oof_labels: Optional[List[np.ndarray]] = None,
    oof_preds: Optional[List[np.ndarray]] = None,
    oof_class_names: Optional[List[str]] = None,
) -> None:
    """Log a cross-validation summary run.

    For ``kfold`` CV (disjoint folds) validation metrics are reported as a single
    *pooled out-of-fold* score: the best-epoch predictions of every fold are
    concatenated (each sample is validated exactly once) and one metric is
    computed over the whole dataset.
    
    For other strategies the per-fold mean/std is kept.
    """
    if not fold_summaries:
        return

    # Pooled out-of-fold metrics are only possible with disjoint folds (kfold)
    pooled = (
        getattr(args, "cv_strategy", None) == "kfold"
        and oof_labels is not None
        and len(oof_labels) > 0
    )

    config = vars(args).copy()
    config["base_run_name"] = base_run_name
    config["folds_completed"] = len(fold_summaries)
    # If pooled, the summary metrics are computed on the concatenated best-epoch
    # predictions of every fold. Otherwise, the summary metrics are the mean/std
    # of the best-epoch metrics of every fold.
    config["summary_scope"] = (
        "pooled_out_of_fold" if pooled else "best_validation_epoch_per_fold"
    )

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{base_run_name}_cv_summary",
        group=base_run_name,
        job_type="cv_summary",
        config=config,
    )

    has_test = any("test_accuracy" in row for row in fold_summaries)

    val_metric_keys = [
        ("val_loss", "best_val_loss"),
        ("val_accuracy", "best_val_accuracy"),
        ("val_balanced_accuracy", "best_val_balanced_accuracy"),
        ("val_macro_f1", "best_val_macro_f1"),
        ("val_weighted_f1", "best_val_weighted_f1"),
        ("val_macro_precision", "best_val_macro_precision"),
        ("val_macro_recall", "best_val_macro_recall"),
    ]
    test_metric_keys = [
        ("test_loss", "test_loss"),
        ("test_accuracy", "test_accuracy"),
        ("test_balanced_accuracy", "test_balanced_accuracy"),
        ("test_macro_f1", "test_macro_f1"),
        ("test_weighted_f1", "test_weighted_f1"),
        ("test_macro_precision", "test_macro_precision"),
        ("test_macro_recall", "test_macro_recall"),
    ]

    aggregate = {"cv/n_folds": len(fold_summaries)}

    if pooled:
        # Disjoint folds means one prediction per sample, therfore a single pooled score
        # Concatenate the per-fold best-epoch predictions.
        labels = np.concatenate(oof_labels)
        preds = np.concatenate(oof_preds)
        pm = compute_metrics(preds, labels, oof_class_names)
        aggregate.update({
            "cv/pooled_accuracy": float(pm["accuracy"]),
            "cv/pooled_balanced_accuracy": float(pm["balanced_accuracy"]),
            "cv/pooled_macro_f1": float(pm["macro_f1"]),
            "cv/pooled_weighted_f1": float(pm["weighted_f1"]),
            "cv/pooled_macro_precision": float(pm["macro_precision"]),
            "cv/pooled_macro_recall": float(pm["macro_recall"]),
            "cv/n_pooled_samples": int(len(labels)),
        })
        _log_wandb_confusion_matrix(
            "cv/pooled_confusion_matrix", labels, preds, pm["cm_labels"],
        )
        _log_wandb_per_class_table("cv/pooled_per_class_metrics", pm["per_class"])
    else:
        std_ddof = 1 if len(fold_summaries) > 1 else 0
        aggregate["cv/std_ddof"] = std_ddof
        for key, summary_key in val_metric_keys:
            values = np.asarray([row[key] for row in fold_summaries], dtype=float)
            aggregate[f"cv/{summary_key}_mean"] = float(values.mean())
            aggregate[f"cv/{summary_key}_std"] = float(values.std(ddof=std_ddof))

    # A fixed test holdout is shared across folds and cannot be pooled, so it is
    # always summarised as mean/std when present.
    if has_test:
        std_ddof = 1 if len(fold_summaries) > 1 else 0
        for key, summary_key in test_metric_keys:
            values = np.asarray(
                [row[key] for row in fold_summaries if key in row], dtype=float
            )
            if values.size:
                aggregate[f"cv/{summary_key}_mean"] = float(values.mean())
                aggregate[f"cv/{summary_key}_std"] = float(values.std(ddof=std_ddof))

    # Aggregates are scalars, not a time-series: store them only in the run
    # summary instead of also logging single-point metrics.
    for key, value in aggregate.items():
        wandb.run.summary[key] = value

    wandb.finish()


# ===========================================================================
#  Main
# ===========================================================================


def run_train(args: argparse.Namespace) -> None:
    # -- Dataset ------------------------------------------------------------
    dataset = get_dataset(
        args.dataset, root_dir=args.data_dir,
        min_samples_per_class=args.min_samples_per_class,
        exclude_classes=args.exclude_classes,
        below_threshold_strategy=args.below_threshold_strategy,
        rare_class_name=args.rare_class_name,
        missing_image_policy=args.missing_image_policy,
    )
    print("*" * 60)
    print(f"Dataset: {type(dataset).__name__} [{len(dataset)} samples]")

    if hasattr(dataset, "excluded_classes") and dataset.excluded_classes:
        print(f"  Excluded classes: {dataset.excluded_classes}")
    if hasattr(dataset, "dropped_classes") and dataset.dropped_classes:
        print(f"  Dropped classes: {dataset.dropped_classes}")
    if hasattr(dataset, "merged_classes") and dataset.merged_classes:
        print(f"  Merged into '{args.rare_class_name}': {dataset.merged_classes}")
    if (
        hasattr(dataset, "missing_stem_count")
        and dataset.missing_stem_count
        and getattr(dataset, "missing_image_policy", "error") == "drop"
    ):
        print(f"  Dropped missing images: {dataset.missing_stem_count}")
    if hasattr(dataset, "classes"):
        final_classes = list(dataset.classes)
        display_classes = getattr(dataset, "display_classes", None)
        print("  Final classes:")
        if display_classes is not None:
            for cls, display_cls in zip(final_classes, display_classes):
                print(f"    {cls} [{display_cls}]")
        else:
            for cls in final_classes:
                print(f"    {cls}")
    print("*" * 60)

    # -- 2. Reproducibility ------------------------------------------------
    set_global_seed(args.seed)

    # Validate ranges
    if not 0.0 <= args.non_rare_aug_prob <= 1.0:
        raise ValueError(f"--non_rare_aug_prob must be in [0, 1], got {args.non_rare_aug_prob}")
    if not 0.0 <= args.undersample_strength <= 1.0:
        raise ValueError(f"--undersample_strength must be in [0, 1], got {args.undersample_strength}")
    if not 0.0 <= args.vqc_angle_dropout < 1.0:
        raise ValueError(f"--vqc_angle_dropout must be in [0, 1), got {args.vqc_angle_dropout}")
    if not 0.0 <= args.test_size < 1.0:
        raise ValueError(f"--test_size must be in [0, 1), got {args.test_size}")

    base_run_name = _make_base_run_name(args, datetime.now())

    # -- 3. CV splits ------------------------------------------------------
    train_val_pool_idx = None
    test_idx = []
    if args.test_size > 0.0:
        train_val_pool_idx, test_idx = make_stratified_splits(
            dataset,
            n_splits=1,
            seed=args.seed,
            train_size=1.0 - args.test_size,
            strategy="shuffle_split",
        )[0]

    if test_idx:
        print(
            f"Fixed stratified test holdout: {len(test_idx)} samples "
            f"({args.test_size:.1%}); train/val pool: {len(train_val_pool_idx)} samples"
        )

    splits = make_stratified_splits(
        dataset,
        args.n_splits,
        args.seed,
        args.train_size,
        args.cv_strategy,
        indices=train_val_pool_idx,
    )
    fold_summaries = []
    oof_labels = []
    oof_preds = []
    oof_class_names = None

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}\n  Fold {fold + 1}/{args.n_splits}\n{'='*60}")

        # W&B init
        fold_run_name = _make_fold_run_name(base_run_name, fold, args.n_splits)
        config = vars(args).copy()
        config["base_run_name"] = base_run_name
        config["fold"] = fold + 1
        config["fold_index"] = fold
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=fold_run_name,
            group=base_run_name,
            job_type="fold",
            config=config,
        )

        class_names = list(getattr(dataset, "classes", sorted(set(dataset.labels))))
        display_class_names = list(getattr(dataset, "display_classes", class_names))
        n_classes = len(class_names)
        train_labels = [dataset.labels[i] for i in train_idx]
        class_counts = Counter(train_labels)

        # Print fold distribution
        val_counts = Counter(dataset.labels[i] for i in val_idx)
        tc = format_class_counts(class_counts, display_class_names)
        vc = format_class_counts(val_counts, display_class_names)
        name_w = max(5, max((len(cn) for cn in display_class_names), default=0))
        count_w = max(5, len(str(max(len(train_idx), len(val_idx)))))
        sep = f"  +{'-' * (name_w + 2)}+{'-' * (count_w + 2)}+{'-' * (count_w + 2)}+"
        print("  Class distribution:")
        print(sep)
        print(f"  | {'Class':<{name_w}} | {'Train':>{count_w}} | {'Val':>{count_w}} |")
        print(sep)
        for cn in display_class_names:
            print(f"  | {cn:<{name_w}} | {tc[cn]:>{count_w}d} | {vc[cn]:>{count_w}d} |")
        print(sep)
        print(f"  | {'Total':<{name_w}} | {len(train_idx):>{count_w}d} | {len(val_idx):>{count_w}d} |")
        print(sep)

        # -- Build pipeline components -------------------------------------
        processor, normalization = _build_processor(args, dataset)
        classifier = _build_classifier(args, n_classes)

        augmentation = get_augmentation(args.augmentation)
        aug_targets = None
        non_target_prob = 1.0
        if args.conditional_augmentation and augmentation is not None:
            aug_targets = sorted(i for i in range(n_classes) if class_counts.get(i, 0) <= args.rare_class_threshold)
            non_target_prob = args.non_rare_aug_prob

        model = EndToEndModel(
            classifier=classifier, 
            processor=processor,
            normalization=normalization, 
            augmentation=augmentation,
            augmentation_target_classes=aug_targets,
            non_target_augmentation_prob=non_target_prob,
            freeze_processor=args.freeze_processor,
        ).to(DEVICE)

        # -- Loss / Optimizer / Scheduler / Sampler ----------------------------------
        criterion = _build_criterion(args, class_counts, n_classes, DEVICE)
        optimizer = _build_optimizer(args, model)
        scheduler = _build_scheduler(args, optimizer)
        sampler, dynamic_soft = _build_sampler(args, train_labels, class_counts)

        # -- DataLoaders ---------------------------------------------------
        nw = max(0, args.num_workers)
        ldr_kw: dict = {
            "batch_size": args.batch_size,
            "num_workers": nw,
            "pin_memory": DEVICE.type == "cuda",
        }
        if nw > 0:
            ldr_kw["persistent_workers"] = True

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            sampler=sampler,
            shuffle=(sampler is None),
            **ldr_kw,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            shuffle=False,
            **ldr_kw,
        )
        test_loader = None
        if test_idx:
            test_loader = DataLoader(
                Subset(dataset, test_idx),
                shuffle=False,
                **ldr_kw,
            )

        # -- Save dir ------------------------------------------------------
        run_dir = _make_run_dir(args, base_run_name, fold)
        if args.save_model or args.save_backbone:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run_config.txt"), "w", encoding="utf-8") as f:
                run_config = {
                    "base_run": base_run_name,
                    "fold": fold + 1,
                    **vars(args),
                }
                for k, v in run_config.items():
                    f.write(f"{k}={v}\n")

        # -- Training loop -------------------------------------------------
        best_f1 = float("-inf")
        best_epoch: Optional[int] = None
        best_val_snapshot: Optional[Dict[str, Any]] = None
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        if args.wandb_watch:
            wandb.watch(model, log="gradients", log_freq=100)

        pbar = tqdm(range(1, args.epochs + 1), desc="Epochs")
        for epoch in pbar:
            # Update dynamic sampler if needed
            if dynamic_soft and sampler is not None:
                strength = get_current_undersample_strength(epoch, args.undersample_strength, 10)
                sw, _ = compute_soft_undersampling_weights(train_labels, class_counts, strength)
                sampler.weights = sw

            train_loss, train_m = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
                display_class_names
            )

            val_loss, val_m, val_labels, val_preds = validate(
                model,
                val_loader,
                criterion,
                DEVICE,
                display_class_names
            )

            pbar.set_postfix(loss=f"{val_loss:.4f}", mF1=f"{val_m['macro_f1']:.4f}", bAcc=f"{val_m['balanced_accuracy']:.4f}")

            log_dict = {
                "train/loss": train_loss,
                "train/acc": train_m["accuracy"],
                "train/macro_f1": train_m["macro_f1"],
                "val/loss": val_loss,
                "val/acc": val_m["accuracy"],
                "val/balanced_acc": val_m["balanced_accuracy"],
                "val/macro_f1": val_m["macro_f1"],
                "val/weighted_f1": val_m["weighted_f1"],
                "val/macro_precision": val_m["macro_precision"],
                "val/macro_recall": val_m["macro_recall"],
            }

            wandb.log(log_dict)
            scheduler.step(val_m["macro_f1"])

            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                best_epoch = epoch
                best_val_snapshot = {
                    "fold": fold + 1,
                    "epoch": epoch,
                    "loss": val_loss,
                    "metrics": val_m,
                    "labels": val_labels.copy(),
                    "preds": val_preds.copy(),
                }
                if args.save_model:
                    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
                if args.save_backbone:
                    metadata = _make_backbone_metadata(
                        args,
                        display_class_names,
                        fold,
                        epoch,
                        val_loss,
                        val_m,
                    )
                    _save_backbone_checkpoint(model.classifier, run_dir, metadata)
                if test_loader is not None:
                    best_state_dict = _copy_state_dict_to_cpu(model)

        test_snapshot: Optional[Dict[str, Any]] = None
        if best_val_snapshot is not None:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_val_macro_f1"] = best_val_snapshot["metrics"]["macro_f1"]
            wandb.run.summary["best_val_balanced_accuracy"] = best_val_snapshot["metrics"]["balanced_accuracy"]
            wandb.run.summary["best_val_weighted_f1"] = best_val_snapshot["metrics"]["weighted_f1"]

            if test_loader is not None and best_state_dict is not None:
                model.load_state_dict(best_state_dict)
                test_loss, test_m, test_labels, test_preds = validate(
                    model,
                    test_loader,
                    criterion,
                    DEVICE,
                    display_class_names,
                )
                test_snapshot = {
                    "loss": test_loss,
                    "metrics": test_m,
                    "labels": test_labels.copy(),
                    "preds": test_preds.copy(),
                }
                wandb.log({
                    "test/loss": test_loss,
                    "test/acc": test_m["accuracy"],
                    "test/balanced_acc": test_m["balanced_accuracy"],
                    "test/macro_f1": test_m["macro_f1"],
                    "test/weighted_f1": test_m["weighted_f1"],
                    "test/macro_precision": test_m["macro_precision"],
                    "test/macro_recall": test_m["macro_recall"],
                })
                wandb.run.summary["test_macro_f1"] = test_m["macro_f1"]
                wandb.run.summary["test_balanced_accuracy"] = test_m["balanced_accuracy"]
                wandb.run.summary["test_weighted_f1"] = test_m["weighted_f1"]

            fold_summaries.append(_make_fold_summary(best_val_snapshot, test_snapshot))
            
            # Accumulate best-epoch predictions for pooled out-of-fold metrics.
            oof_labels.append(best_val_snapshot["labels"])
            oof_preds.append(best_val_snapshot["preds"])
            oof_class_names = best_val_snapshot["metrics"]["cm_labels"]

        wandb.finish()

    _log_wandb_cv_summary(
        args, base_run_name, fold_summaries, oof_labels, oof_preds, oof_class_names
    )

    print("\nAll folds completed.")


# ===========================================================================

if __name__ == "__main__":
    run_train(parse_args())
