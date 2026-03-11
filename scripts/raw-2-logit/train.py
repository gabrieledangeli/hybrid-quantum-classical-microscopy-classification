import sys
import os
import argparse
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter

from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix,
)

import wandb


# Local imports
from args import parse_args
from losses import FocalLoss
from dataset_utils import k_fold
from augmentation import get_augmentation, set_global_seed
from dataset import Subset, get_dataset
from pipeline_numpy import RawProcessingPipeline
from pipeline_torch import RawToRGB, ParametrizedProcessing, NNProcessing
from model import resnet_model, hybrid_tn_model, EndToEndModel, TensorNormalize

# Global Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def compute_metrics(all_preds: np.ndarray, 
                    all_labels: np.ndarray,
                    class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute classification metrics suited for imbalanced multiclass problems."""
    metrics = {}

    metrics['accuracy'] = (all_preds == all_labels).mean()
    metrics['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
    metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    metrics['macro_precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    labels_present = sorted(set(all_labels) | set(all_preds))
    labels_eval = list(range(len(class_names))) if class_names else labels_present

    # Per-class metrics
    per_class_p = precision_score(
        all_labels, all_preds, labels=labels_eval, average=None, zero_division=0
    )
    per_class_r = recall_score(
        all_labels, all_preds, labels=labels_eval, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        all_labels, all_preds, labels=labels_eval, average=None, zero_division=0
    )

    per_class = []
    for idx, cls_idx in enumerate(labels_eval):
        name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
        per_class.append({
            'class': name,
            'precision': float(per_class_p[idx]),
            'recall': float(per_class_r[idx]),
            'f1': float(per_class_f1[idx]),
        })
    metrics['per_class'] = per_class

    # Confusion matrix (raw numpy)
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds, labels=labels_eval)
    metrics['cm_labels'] = [class_names[i] if class_names and i < len(class_names) else str(i)
                            for i in labels_eval]
    if class_names:
        counts = np.bincount(all_labels.astype(int), minlength=len(class_names))
        metrics['label_counts'] = {
            class_names[i]: int(counts[i]) for i in range(len(class_names))
        }
    else:
        counts = np.bincount(all_labels.astype(int))
        metrics['label_counts'] = {str(i): int(v) for i, v in enumerate(counts)}

    return metrics


def compute_soft_undersampling_weights(train_labels: List[int],
                                       class_counts: Counter,
                                       strength: float) -> Tuple[torch.DoubleTensor, Dict[int, float]]:
    """Build per-sample sampling weights with soft undersampling controlled by strength in [0, 1]."""
    if len(class_counts) == 0:
        raise ValueError("class_counts is empty; cannot build sampler weights.")

    strength = min(1.0, max(0.0, float(strength)))
    max_count = max(class_counts.values())
    class_weight = {
        cls_idx: (max_count / max(1, cnt)) ** strength
        for cls_idx, cnt in class_counts.items()
    }
    sample_weights = [class_weight[label] for label in train_labels]
    return torch.DoubleTensor(sample_weights), class_weight


def get_current_undersample_strength(epoch: int,
                                     max_strength: float,
                                     warmup_epochs: int) -> float:
    """Linear warmup for soft undersampling strength."""
    max_strength = min(1.0, max(0.0, float(max_strength)))
    warmup_epochs = max(1, int(warmup_epochs))
    if warmup_epochs == 1:
        return max_strength
    progress = min(1.0, max(0.0, (epoch - 1) / (warmup_epochs - 1)))
    return max_strength * progress


def format_class_counts(class_counts: Counter, class_names: List[str]) -> Dict[str, int]:
    return {class_names[i]: int(class_counts.get(i, 0)) for i in range(len(class_names))}


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    class_names: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
    """Running training for one epoch. Returns (avg_loss, metrics_dict)."""
    model.train()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    total_steps = len(dataloader)
    if total_steps == 0:
        raise RuntimeError("Training dataloader is empty. Check dataset and split configuration.")

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if isinstance(model, EndToEndModel):
            outputs = model(inputs, labels=labels)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    avg_loss = running_loss / total_steps
    metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_labels), class_names)

    return avg_loss, metrics


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             class_names: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
    """Running validation. Returns (avg_loss, metrics_dict)."""
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    total_steps = len(dataloader)
    if total_steps == 0:
        raise RuntimeError("Validation dataloader is empty. Check dataset and split configuration.")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / total_steps

    all_preds_np = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels)
    metrics = compute_metrics(all_preds_np, all_labels_np, class_names)

    # Always log confusion matrix for multiclass monitoring.
    try:
        cm_labels = metrics['cm_labels']
        wandb.log({
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels_np.tolist(),
                preds=all_preds_np.tolist(),
                class_names=cm_labels,
            )
        }, commit=False)
    except Exception as e:
        print(f"[Warning] Could not log confusion matrix to WandB: {e}")

    return avg_loss, metrics


def run_train(args: argparse.Namespace) -> None:
    """Main training execution function."""
    
    # 1. Initialize Dataset
    dataset = get_dataset(
        args.dataset,
        root_dir=args.data_dir,
        min_samples_per_class=args.min_samples_per_class,
        below_threshold_strategy=args.below_threshold_strategy,
        rare_class_name=args.rare_class_name,
    )
    print(f'Dataset: {type(dataset).__name__} [{len(dataset)} samples]')   
    print(f'Task: {dataset.task}')
    print(
        f"Class thresholding: min_samples_per_class={args.min_samples_per_class}, "
        f"strategy={args.below_threshold_strategy}"
    )
    if hasattr(dataset, 'dropped_classes') and dataset.dropped_classes:
        print(f"[Info] Dropped classes: {dataset.dropped_classes}")
    if hasattr(dataset, 'merged_classes') and dataset.merged_classes:
        print(f"[Info] Merged into '{args.rare_class_name}': {dataset.merged_classes}")
    if hasattr(dataset, 'classes'):
        print(f"[Info] Final classes: {dataset.classes}")
    print(f'Cross-validation splits: {args.n_splits}')

    # 2. Set Seed used for reproducibility
    set_global_seed(args.seed)
    if not 0.0 <= args.non_rare_aug_prob <= 1.0:
        raise ValueError(f"--non_rare_aug_prob must be in [0, 1], got {args.non_rare_aug_prob}")
    if not 0.0 <= args.undersample_strength <= 1.0:
        raise ValueError(f"--undersample_strength must be in [0, 1], got {args.undersample_strength}")
    if args.undersample_warmup_epochs < 1:
        raise ValueError(f"--undersample_warmup_epochs must be >= 1, got {args.undersample_warmup_epochs}")
    # 3. Create K-Fold splits
    idxs_kfold = k_fold(dataset, n_splits=args.n_splits, seed=args.seed, train_size=args.train_size, strategy=args.cv_strategy)

    for k_iter, (train_indices, valid_indices) in enumerate(idxs_kfold):
        print(f'\n--- Starting Fold {k_iter + 1}/{args.n_splits} ---')
        
        # Initialize WandB per fold
        config = vars(args).copy()
        config['fold'] = k_iter
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name or (
                f"{datetime.now():%d-%m}"
                f"_{args.classifier_network}"
                f"_aug-{args.augmentation}"
                f"_bal-{args.imbalance_handling}"
                f"_fold{k_iter + 1}of{args.n_splits}"
            ),
            config=config
        )

        class_names = list(dataset.classes) if hasattr(dataset, 'classes') else [str(x) for x in sorted(set(dataset.labels))]
        n_classes = len(class_names)
        train_labels = [dataset.labels[i] for i in train_indices]
        valid_labels = [dataset.labels[i] for i in valid_indices]
        class_counts = Counter(train_labels)
        valid_class_counts = Counter(valid_labels)

        train_counts_named = format_class_counts(class_counts, class_names)
        valid_counts_named = format_class_counts(valid_class_counts, class_names)
        print("[Info] Fold class distribution (train | val):")
        for cls_name in class_names:
            print(f"  {cls_name:>6s}: {train_counts_named[cls_name]:4d} | {valid_counts_named[cls_name]:4d}")
        missing_in_train = [class_names[i] for i in range(n_classes) if class_counts.get(i, 0) == 0]
        if missing_in_train:
            print(
                f"[Warning] Classes absent in train fold: {missing_in_train}. "
                "Loss/sampler weighting for these classes will be set to 0."
            )
        
        print(f'Processing Mode: {args.processing_mode}')

        # 4. Configure Processing Pipeline
        processor = None
        normalization = None
        is_rgb_dataset = args.dataset == 'MicroscopyRGB'

        if is_rgb_dataset:
            # Keep dataset transform minimal and apply normalization after augmentation in-model.
            dataset.transform = T.ToTensor()
            normalization = TensorNormalize(dataset.dataset_mean, dataset.dataset_std)
            # Ensure processor is None for RGB dataset as we don't need raw processing.
            if args.processing_mode != 'static':
                print(f"Warning: Processing mode '{args.processing_mode}' ignored for RGB dataset.")

        elif args.processing_mode == 'static':
            # Static pipeline on raw data. Normalize in-model after augmentation.
            dataset.transform = T.Compose([
                RawProcessingPipeline(
                    camera_parameters=dataset.camera_parameters,
                    debayer=args.sp_debayer,
                    sharpening=args.sp_sharpening,
                    denoising=args.sp_denoising,
                ),
            ])
            normalization = TensorNormalize(dataset.dataset_mean, dataset.dataset_std)
        else:
            # Neural/Parametrized pipelines
            mosaic_mean = [0.5663, 0.1401, 0.0731]
            mosaic_std = [0.097, 0.0423, 0.008]
            post_mosaic_norm = TensorNormalize(mosaic_mean, mosaic_std)

            if args.processing_mode == 'parametrized':
                processor = ParametrizedProcessing(
                    camera_parameters=dataset.camera_parameters, 
                    batch_norm_output=True
                )
            elif args.processing_mode == 'neural_network':
                processor = NNProcessing(
                    normalize_mosaic=None,
                    batch_norm_output=True
                )
                normalization = post_mosaic_norm
            elif args.processing_mode == 'none':
                processor = RawToRGB(
                    reduce_size=True, 
                    out_channels=3,
                    normalize_mosaic=None
                )
                normalization = post_mosaic_norm

        # 5. Initialize Classifier
        if args.classifier_network == 'hybrid_tn':
            classifier = hybrid_tn_model(
                fc_out_features=len(dataset.classes),
                n_qubits=args.n_qubits,
                mps_bond_dim=args.mps_bond_dim,
                mps_phys_dim=args.mps_phys_dim,
                mps_hidden_dim=args.mps_hidden_dim,
                tn_bridge=args.tn_bridge,
                peps_rows=args.peps_rows,
                peps_cols=args.peps_cols,
                peps_message_passing_steps=args.peps_message_passing_steps,
                vqc_layers=args.vqc_layers,
                freeze_resnet=args.freeze_resnet_backbone,
                use_pennylane=args.use_pennylane,
                backbone=args.backbone,
                backbone_weights=args.backbone_weights,
            )
            if args.tn_bridge == 'peps':
                grid_desc = "auto"
                if args.peps_rows is not None or args.peps_cols is not None:
                    grid_desc = f"{args.peps_rows}x{args.peps_cols}"
                print(f"[Info] Hybrid TN bridge: PEPS (grid={grid_desc}, message_passing_steps={args.peps_message_passing_steps}).")
            else:
                print("[Info] Hybrid TN bridge: MPS.")
        else:
            classifier = resnet_model(
                model_name=args.classifier_network,
                pretrained=args.classifier_pretrained,
                fc_out_features=len(dataset.classes),
                freeze_backbone=args.freeze_resnet_backbone,
            )
            if args.freeze_resnet_backbone:
                print("[Info] Freezing classic ResNet stem/layer1-3; training only layer4 + fc.")

        if processor and args.freeze_processor:
             # Check if processor has parameters
            if len(list(processor.parameters())) == 0:
                print('Note: Freezing processor requested but it has no learnable parameters.')
        
        # 6. Initialize End-to-End Model
        augmentation = get_augmentation(args.augmentation)
        augmentation_target_classes = None
        non_target_augmentation_prob = 1.0
        if args.conditional_augmentation:
            if augmentation is None:
                print("[Warning] Conditional augmentation requested, but --augmentation is 'none'.")
            else:
                augmentation_target_classes = sorted([
                    cls_idx for cls_idx in range(n_classes)
                    if class_counts.get(cls_idx, 0) <= args.rare_class_threshold
                ])
                non_target_augmentation_prob = args.non_rare_aug_prob
                if augmentation_target_classes:
                    rare_names = [class_names[i] for i in augmentation_target_classes]
                    print(f"[Info] Conditional augmentation on rare classes (<= {args.rare_class_threshold}): {rare_names}")
                    print(f"[Info] Non-rare augmentation probability: {non_target_augmentation_prob:.2f}")
                else:
                    print("[Info] Conditional augmentation enabled but no classes match rare threshold.")

        model = EndToEndModel(
            classifier=classifier,
            processor=processor,
            normalization=normalization,
            augmentation=augmentation,
            augmentation_target_classes=augmentation_target_classes,
            non_target_augmentation_prob=non_target_augmentation_prob,
            freeze_processor=args.freeze_processor,
        )
        model = model.to(DEVICE)
        
        print(f"Model moved to {DEVICE}")
        # 7. Setup Training State

        # Prepare Imbalance Handling Strategy
        static_sampler = None
        dynamic_sampler = None
        dynamic_soft_sampler = False
        criterion_weight = None
        use_weighted_loss = args.imbalance_handling in ('loss', 'both')
        use_sampler = args.imbalance_handling in ('sampler', 'both')
        
        if use_weighted_loss:
            # Effective Number of Samples weighting (Cui et al., 2019)
            # Better than naive 1/N for extreme imbalance.
            # Classes absent in current train fold get zero weight.
            beta = 0.999  # closer to 1 = more aggressive re-weighting
            class_weights = [0.0 for _ in range(n_classes)]
            present_class_indices = [i for i in range(n_classes) if class_counts.get(i, 0) > 0]

            for i in present_class_indices:
                n_i = class_counts[i]
                effective_num = (1.0 - beta**n_i) / (1.0 - beta)
                class_weights[i] = 1.0 / max(effective_num, 1e-12)

            sum_present = sum(class_weights[i] for i in present_class_indices)
            if sum_present <= 0:
                raise RuntimeError("Invalid class weights: no positive weights for present classes.")

            # Normalize only over present classes to keep average scale stable.
            norm_factor = len(present_class_indices) / sum_present
            for i in present_class_indices:
                class_weights[i] *= norm_factor

            criterion_weight = torch.FloatTensor(class_weights).to(DEVICE)
            print(f"[Info] Using Effective Number of Samples weighted loss (beta={beta}).")
            for cls_idx in range(n_classes):
                cls_name = dataset.classes[cls_idx] if hasattr(dataset, 'classes') else str(cls_idx)
                if class_counts.get(cls_idx, 0) == 0:
                    print(f"  {cls_name:>6s}: weight=0.000 (absent in train fold)")
                else:
                    print(f"  {cls_name:>6s}: weight={class_weights[cls_idx]:.3f}")
            
        if use_sampler:
            if args.sampler_mode == 'soft_undersample':
                dynamic_soft_sampler = True
                init_strength = get_current_undersample_strength(
                    epoch=1,
                    max_strength=args.undersample_strength,
                    warmup_epochs=args.undersample_warmup_epochs,
                )
                init_sample_weights, _ = compute_soft_undersampling_weights(
                    train_labels=train_labels,
                    class_counts=class_counts,
                    strength=init_strength,
                )
                dynamic_sampler = WeightedRandomSampler(
                    weights=init_sample_weights,
                    num_samples=len(train_indices),
                    replacement=True,
                )
                static_sampler = dynamic_sampler
                print(
                    f"[Info] Using dynamic soft undersampling sampler "
                    f"(max_strength={args.undersample_strength}, warmup_epochs={args.undersample_warmup_epochs})."
                )
            else:
                # Standard inverse-frequency balanced sampling.
                sample_weights = [
                    (1.0 / class_counts[label]) if class_counts[label] > 0 else 0.0
                    for label in train_labels
                ]
                sample_weights = torch.DoubleTensor(sample_weights)
                static_sampler = WeightedRandomSampler(
                    weights=sample_weights, num_samples=len(train_indices), replacement=True
                )
                print("[Info] Using WeightedRandomSampler to balance batch composition.")
        
        if use_weighted_loss and use_sampler:
            print("[Info] Combined strategy: sampler balances batches + loss weights gradients.")

        # Initialize Criterion with or without weights (+ label smoothing)
        if args.loss_fn == 'focal':
            criterion = FocalLoss(
                weight=criterion_weight,
                gamma=args.focal_gamma,
                label_smoothing=args.label_smoothing,
            ).to(DEVICE)
            print(f"[Info] Using Focal Loss (gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}).")
        else:
            criterion = nn.CrossEntropyLoss(
                weight=criterion_weight,
                label_smoothing=args.label_smoothing,
            )
        
        # Differential learning rates: backbone (pretrained) gets lower lr,
        # head (randomly initialized) gets higher lr for faster convergence.
        backbone_params = []
        head_params = []

        if args.classifier_network in ('resnet18', 'resnet34', 'resnet50'):
            if args.classifier_pretrained:
                # For pretrained torchvision ResNet, everything except final FC is backbone.
                for name, param in model.classifier.named_parameters():
                    if not param.requires_grad:
                        continue
                    if name.startswith('fc.'):
                        head_params.append(param)
                    else:
                        backbone_params.append(param)

                # Processor parameters are treated as head (task-specific).
                if model.processor is not None:
                    for param in model.processor.parameters():
                        if param.requires_grad:
                            head_params.append(param)
            else:
                # When training ResNet from scratch, use a single LR for all trainable params.
                for param in model.parameters():
                    if param.requires_grad:
                        head_params.append(param)
                print("[Info] ResNet is not pretrained: using single LR for all trainable parameters.")
        else:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'resnet' in name or 'backbone' in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
        
        if backbone_params and head_params:
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': args.lr},
                {'params': head_params, 'lr': args.lr_head},
            ], weight_decay=args.weight_decay)
            print(f"[Info] Differential LR: backbone={args.lr}, head={args.lr_head}")
        elif head_params:
            optimizer = optim.AdamW(head_params, lr=args.lr_head, weight_decay=args.weight_decay)
            print(f"[Info] Single LR (head): {args.lr_head}")
        elif backbone_params:
            optimizer = optim.AdamW(backbone_params, lr=args.lr, weight_decay=args.weight_decay)
            print(f"[Info] Single LR (backbone): {args.lr}")
        else:
            raise ValueError("No trainable parameters found. Check freeze settings.")

        # Warmup + Cosine annealing scheduler
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.lr_warmup_epochs), eta_min=1e-6
        )
        if args.lr_warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=args.lr_warmup_epochs,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[args.lr_warmup_epochs],
            )
            print(f"[Info] LR warmup: {args.lr_warmup_epochs} epochs (1% -> 100%), then cosine decay.")
        else:
            scheduler = cosine_scheduler

        # Prepare state dict for saving
        experiment_dir = os.path.join('results', args.dataset, args.processing_mode)
        run_dir = os.path.join(experiment_dir, f"fold_{k_iter}")
        
        if args.save_model:
            os.makedirs(run_dir, exist_ok=True)
            
            state_dict_log = vars(args).copy()
            state_dict_log.update({
                'classes': dataset.classes,
                'device': str(DEVICE),
                'train_indices': train_indices,
                'valid_indices': valid_indices,
                'train_set_size': len(train_indices),
                'test_set_size': len(valid_indices)
            })

            hparams = {
                'dataset': args.dataset,
                'processing_mode': args.processing_mode,
                'lr': args.lr,
                'batch_size': args.batch_size
            }

            # Save configuration logs
            with open(os.path.join(run_dir, 'state_dict.txt'), 'w') as f:
                f.write(f"Command: python {' '.join(sys.argv)}\n")
                f.write('\n'.join([f'{k}={v}' for k, v in state_dict_log.items()]))
            
            with open(os.path.join(run_dir, 'hparams.txt'), 'w') as f:
                f.write('\n'.join([f'{k}={v}' for k, v in hparams.items()]))

        train_set = Subset(dataset, indices=train_indices)
        valid_set = Subset(dataset, indices=valid_indices)

        # DataLoader setup
        num_workers = max(0, args.num_workers)
        loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": num_workers,
            "pin_memory": (DEVICE.type == 'cuda'),
        }
        # Keep workers alive across epochs to avoid repeated spawn/cache warmup overhead.
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True

        if use_sampler:
            train_loader = DataLoader(
                train_set,
                sampler=static_sampler,
                shuffle=False,
                **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_set,
                shuffle=True,
                **loader_kwargs,
            )

        valid_loader = DataLoader(
            valid_set,
            shuffle=False,
            **loader_kwargs,
        )

        def update_dynamic_sampler(epoch: int) -> None:
            if not dynamic_soft_sampler:
                return
            if dynamic_sampler is None:
                raise RuntimeError("dynamic_soft_sampler is enabled but dynamic_sampler is None.")

            current_strength = get_current_undersample_strength(
                epoch=epoch,
                max_strength=args.undersample_strength,
                warmup_epochs=args.undersample_warmup_epochs,
            )
            sample_weights, _ = compute_soft_undersampling_weights(
                train_labels=train_labels,
                class_counts=class_counts,
                strength=current_strength,
            )
            dynamic_sampler.weights = sample_weights

        # 9. Training Loop
        best_macro_f1 = 0.0
        print("Starting training loop...")

        # Optional gradient watching (can add overhead).
        if args.wandb_watch:
            wandb.watch(model, log='gradients', log_freq=100)

        epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training Epochs")
        for epoch in epoch_pbar:
            update_dynamic_sampler(epoch=epoch)

            train_loss, train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE,
                class_names,
            )

            # Step the LR scheduler
            scheduler.step()

            if epoch % args.check_val_every_n_epoch == 0:
                val_loss, val_metrics = validate(
                    model,
                    valid_loader,
                    criterion,
                    DEVICE,
                    class_names,
                )

                # Update progress bar description
                epoch_pbar.set_postfix({
                    'loss': f'{val_loss:.4f}',
                    'mF1': f'{val_metrics["macro_f1"]:.4f}',
                    'bAcc': f'{val_metrics["balanced_accuracy"]:.4f}',
                })

                per_class_val_log = {}
                for row in val_metrics.get('per_class', []):
                    cls_name = row['class']
                    per_class_val_log[f"val/per_class_precision/{cls_name}"] = float(row['precision'])
                    per_class_val_log[f"val/per_class_recall/{cls_name}"] = float(row['recall'])

                wandb.log({
                    "train/loss": train_loss,
                    "train/acc": train_metrics['accuracy'],
                    "train/macro_f1": train_metrics['macro_f1'],
                    "val/loss": val_loss,
                    "val/acc": val_metrics['accuracy'],
                    "val/balanced_acc": val_metrics['balanced_accuracy'],
                    "val/macro_f1": val_metrics['macro_f1'],
                    "val/weighted_f1": val_metrics['weighted_f1'],
                    "val/macro_precision": val_metrics['macro_precision'],
                    "val/macro_recall": val_metrics['macro_recall'],
                    **per_class_val_log,
                })

                if val_metrics['macro_f1'] > best_macro_f1:
                    best_macro_f1 = val_metrics['macro_f1']
                    # Save best model
                    if args.save_model:
                        torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))

        # Finish WandB run
        wandb.finish()

    print("Cross-validation completed.")


if __name__ == "__main__":
    args = parse_args()
    run_train(args)
