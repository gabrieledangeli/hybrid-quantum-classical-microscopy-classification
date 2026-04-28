"""Training entry-point for the Hybrid Quantum-Classical CNN pipeline.

Usage:

    python -m scripts.train --dataset Microscopy --data_dir ./data --epochs 30

Or from the ``scripts/`` directory:

    python train.py --dataset Microscopy --data_dir ./data --epochs 30
"""

from __future__ import annotations

import os
import sys
import argparse
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import wandb

# -- Local imports ---------------------------------------------------------
# Support both `python -m scripts.train` and direct `python train.py`
try:
    from scripts.config import parse_args
    from scripts.data.augmentation import get_augmentation, set_global_seed
    from scripts.data.dataset import Subset, get_dataset
    from scripts.data.utils import (
        compute_soft_undersampling_weights,
        format_class_counts,
        get_current_undersample_strength,
        k_fold,
    )
    from scripts.models.classifiers import EndToEndModel, TensorNormalize, resnet_model
    from scripts.models.hybrid import hybrid_tn_model
    from scripts.models.losses import FocalLoss
    from scripts.processing.numpy_pipeline import RawProcessingPipeline
    from scripts.processing.torch_pipeline import (
        NNProcessing,
        ParametrizedProcessing,
        RawToRGB,
    )
    from scripts.utils.metrics import compute_metrics
except ImportError:
    from config import parse_args
    from data.augmentation import get_augmentation, set_global_seed
    from data.dataset import Subset, get_dataset
    from data.utils import (
        compute_soft_undersampling_weights,
        format_class_counts,
        get_current_undersample_strength,
        k_fold,
    )
    from models.classifiers import EndToEndModel, TensorNormalize, resnet_model
    from models.hybrid import hybrid_tn_model
    from models.losses import FocalLoss
    from processing.numpy_pipeline import RawProcessingPipeline
    from processing.torch_pipeline import (
        NNProcessing,
        ParametrizedProcessing,
        RawToRGB,
    )
    from utils.metrics import compute_metrics

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

        outputs = model(inputs, labels=labels) if isinstance(model, EndToEndModel) else model(inputs)
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
    is_rgb = args.dataset == "MicroscopyRGB"

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

    if args.processing_mode == "parametrized":
        return ParametrizedProcessing(camera_parameters=dataset.camera_parameters, batch_norm_output=True), None
    if args.processing_mode == "neural_network":
        return NNProcessing(batch_norm_output=True), mosaic_norm
    # "none"
    return RawToRGB(reduce_size=True, out_channels=3), mosaic_norm


def _build_classifier(args, n_classes: int):
    """Instantiate classifier (pure ResNet or hybrid TN)."""
    if args.classifier_network == "hybrid_tn":
        return hybrid_tn_model(
            fc_out_features=n_classes,
            n_qubits=args.n_qubits,
            tn_bond_dim=args.tn_bond_dim,
            tn_phys_dim=args.tn_phys_dim,
            tn_hidden_dim=args.tn_hidden_dim,
            tn_bridge=args.tn_bridge,
            vqc_layers=args.vqc_layers,
            freeze_resnet=args.freeze_resnet_backbone,
            backbone=args.backbone,
            backbone_weights=args.backbone_weights,
            pretrained=args.classifier_pretrained,
        )
    return resnet_model(
        model_name=args.classifier_network,
        pretrained=args.classifier_pretrained,
        fc_out_features=n_classes,
        freeze_backbone=args.freeze_resnet_backbone,
    )


def _build_criterion(args, class_counts: Counter, n_classes: int, device: torch.device):
    """Return ``(criterion, weight_tensor_or_none)``."""
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


def _build_optimizer(args, model: nn.Module):
    """Differential LR optimizer: backbone (low lr) vs head (high lr)."""
    backbone_params, head_params = [], []

    if args.classifier_network in ("resnet18", "resnet34", "resnet50") and args.classifier_pretrained:
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
            if "resnet" in name or "backbone" in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
    if not groups:
        raise ValueError("No trainable parameters found.")

    return optim.AdamW(groups, weight_decay=args.weight_decay)


def _build_scheduler(args, optimizer):
    """Cosine annealing, optionally preceded by linear warmup."""
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.lr_warmup_epochs), eta_min=1e-6
    )
    if args.lr_warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.lr_warmup_epochs
        )
        return optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[args.lr_warmup_epochs])
    return cosine


def _make_wandb_run_name(args: argparse.Namespace, fold: int) -> str:
    """Generate a fold-specific W&B run name."""
    fold_suffix = f"fold{fold + 1}of{args.n_splits}"
    if args.run_name:
        return f"{args.run_name}_{fold_suffix}"

    model_tag = args.classifier_network
    if args.classifier_network == "hybrid_tn":
        model_tag = f"{args.classifier_network}-{args.backbone}-{args.tn_bridge}"

    return (
        f"{datetime.now():%d-%m}_{model_tag}"
        f"_aug-{args.augmentation}_bal-{args.imbalance_handling}"
        f"_{fold_suffix}"
    )


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


# ===========================================================================
#  Main training routine
# ===========================================================================


def run_train(args: argparse.Namespace) -> None:
    """Execute the full cross-validated training loop."""

    # -- 1. Dataset --------------------------------------------------------
    dataset = get_dataset(
        args.dataset, root_dir=args.data_dir,
        min_samples_per_class=args.min_samples_per_class,
        below_threshold_strategy=args.below_threshold_strategy,
        rare_class_name=args.rare_class_name,
    )
    print(f"Dataset: {type(dataset).__name__} [{len(dataset)} samples]")
    if hasattr(dataset, "dropped_classes") and dataset.dropped_classes:
        print(f"  Dropped classes: {dataset.dropped_classes}")
    if hasattr(dataset, "merged_classes") and dataset.merged_classes:
        print(f"  Merged into '{args.rare_class_name}': {dataset.merged_classes}")
    if hasattr(dataset, "classes"):
        print(f"  Final classes: {dataset.classes}")

    # -- 2. Reproducibility ------------------------------------------------
    set_global_seed(args.seed)

    # Validate ranges
    if not 0.0 <= args.non_rare_aug_prob <= 1.0:
        raise ValueError(f"--non_rare_aug_prob must be in [0, 1], got {args.non_rare_aug_prob}")
    if not 0.0 <= args.undersample_strength <= 1.0:
        raise ValueError(f"--undersample_strength must be in [0, 1], got {args.undersample_strength}")

    # -- 3. CV splits ------------------------------------------------------
    splits = k_fold(dataset, args.n_splits, args.seed, args.train_size, args.cv_strategy)

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}\n  Fold {fold + 1}/{args.n_splits}\n{'='*60}")

        # W&B init
        config = vars(args).copy()
        config["fold"] = fold
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=_make_wandb_run_name(args, fold),
            config=config,
        )

        class_names = list(getattr(dataset, "classes", sorted(set(dataset.labels))))
        n_classes = len(class_names)
        train_labels = [dataset.labels[i] for i in train_idx]
        class_counts = Counter(train_labels)

        # Print fold distribution
        val_counts = Counter(dataset.labels[i] for i in val_idx)
        tc = format_class_counts(class_counts, class_names)
        vc = format_class_counts(val_counts, class_names)
        for cn in class_names:
            print(f"  {cn:>6s}: train={tc[cn]:4d} | val={vc[cn]:4d}")

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
            classifier=classifier, processor=processor,
            normalization=normalization, augmentation=augmentation,
            augmentation_target_classes=aug_targets,
            non_target_augmentation_prob=non_target_prob,
            freeze_processor=args.freeze_processor,
        ).to(DEVICE)

        # -- Loss / Optimizer / Scheduler ----------------------------------
        criterion = _build_criterion(args, class_counts, n_classes, DEVICE)
        optimizer = _build_optimizer(args, model)
        scheduler = _build_scheduler(args, optimizer)

        # -- Sampler -------------------------------------------------------
        use_sampler = args.imbalance_handling in ("sampler", "both")
        dynamic_soft = False
        sampler = None
        dyn_sampler = None

        if use_sampler:
            if args.sampler_mode == "soft_undersample":
                dynamic_soft = True
                init_strength = get_current_undersample_strength(1, args.undersample_strength, args.undersample_warmup_epochs)
                sw, _ = compute_soft_undersampling_weights(train_labels, class_counts, init_strength)
                dyn_sampler = WeightedRandomSampler(sw, len(train_idx), replacement=True)
                sampler = dyn_sampler
            else:
                sw = torch.DoubleTensor([
                    (1.0 / class_counts[l]) if class_counts[l] > 0 else 0.0
                    for l in train_labels
                ])
                sampler = WeightedRandomSampler(sw, len(train_idx), replacement=True)

        # -- DataLoaders ---------------------------------------------------
        nw = max(0, args.num_workers)
        ldr_kw: dict = {"batch_size": args.batch_size, "num_workers": nw, "pin_memory": DEVICE.type == "cuda"}
        if nw > 0:
            ldr_kw["persistent_workers"] = True

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            sampler=sampler, shuffle=(sampler is None), **ldr_kw,
        )
        val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **ldr_kw)

        # -- Save dir ------------------------------------------------------
        run_dir = os.path.join("results", args.dataset, args.processing_mode, f"fold_{fold}")
        if args.save_model:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "state_dict.txt"), "w") as f:
                f.write(f"Command: python {' '.join(sys.argv)}\n")
                for k, v in {**vars(args), "fold": fold}.items():
                    f.write(f"{k}={v}\n")

        # -- Training loop -------------------------------------------------
        best_f1 = float("-inf")
        best_epoch: Optional[int] = None
        best_val_snapshot: Optional[Dict[str, Any]] = None
        last_val_snapshot: Optional[Dict[str, Any]] = None
        if args.wandb_watch:
            wandb.watch(model, log="gradients", log_freq=100)

        pbar = tqdm(range(1, args.epochs + 1), desc="Epochs")
        for epoch in pbar:
            # Update dynamic sampler if needed
            if dynamic_soft and dyn_sampler is not None:
                strength = get_current_undersample_strength(epoch, args.undersample_strength, args.undersample_warmup_epochs)
                sw, _ = compute_soft_undersampling_weights(train_labels, class_counts, strength)
                dyn_sampler.weights = sw

            train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, class_names)
            scheduler.step()

            if epoch % args.check_val_every_n_epoch == 0:
                val_loss, val_m, val_labels, val_preds = validate(
                    model, val_loader, criterion, DEVICE, class_names
                )
                last_val_snapshot = {
                    "epoch": epoch,
                    "metrics": val_m,
                    "labels": val_labels.copy(),
                    "preds": val_preds.copy(),
                }

                pbar.set_postfix(loss=f"{val_loss:.4f}", mF1=f"{val_m['macro_f1']:.4f}", bAcc=f"{val_m['balanced_accuracy']:.4f}")

                log_dict = {
                    "train/loss": train_loss, "train/acc": train_m["accuracy"], "train/macro_f1": train_m["macro_f1"],
                    "val/loss": val_loss, "val/acc": val_m["accuracy"], "val/balanced_acc": val_m["balanced_accuracy"],
                    "val/macro_f1": val_m["macro_f1"], "val/weighted_f1": val_m["weighted_f1"],
                    "val/macro_precision": val_m["macro_precision"], "val/macro_recall": val_m["macro_recall"],
                }
                for row in val_m.get("per_class", []):
                    cn = row["class"]
                    log_dict[f"val/per_class_precision/{cn}"] = row["precision"]
                    log_dict[f"val/per_class_recall/{cn}"] = row["recall"]
                wandb.log(log_dict)

                if val_m["macro_f1"] > best_f1:
                    best_f1 = val_m["macro_f1"]
                    best_epoch = epoch
                    best_val_snapshot = {
                        "epoch": epoch,
                        "metrics": val_m,
                        "labels": val_labels.copy(),
                        "preds": val_preds.copy(),
                    }
                    if args.save_model:
                        torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        if best_val_snapshot is not None:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_val_macro_f1"] = best_val_snapshot["metrics"]["macro_f1"]
            wandb.run.summary["best_val_balanced_accuracy"] = best_val_snapshot["metrics"]["balanced_accuracy"]
            wandb.run.summary["best_val_weighted_f1"] = best_val_snapshot["metrics"]["weighted_f1"]
            _log_wandb_confusion_matrix(
                "val/best_confusion_matrix",
                best_val_snapshot["labels"],
                best_val_snapshot["preds"],
                best_val_snapshot["metrics"]["cm_labels"],
            )

        if last_val_snapshot is not None:
            wandb.run.summary["last_val_epoch"] = last_val_snapshot["epoch"]
            wandb.run.summary["last_val_macro_f1"] = last_val_snapshot["metrics"]["macro_f1"]
            wandb.run.summary["last_val_balanced_accuracy"] = last_val_snapshot["metrics"]["balanced_accuracy"]
            wandb.run.summary["last_val_weighted_f1"] = last_val_snapshot["metrics"]["weighted_f1"]
            _log_wandb_confusion_matrix(
                "val/last_confusion_matrix",
                last_val_snapshot["labels"],
                last_val_snapshot["preds"],
                last_val_snapshot["metrics"]["cm_labels"],
            )

        wandb.finish()

    print("\nAll folds completed.")


# ===========================================================================

if __name__ == "__main__":
    run_train(parse_args())
