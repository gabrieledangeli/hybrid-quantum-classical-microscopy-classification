import sys
import os
import argparse
from typing import Tuple, Dict, Any, Optional, List

from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
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
from dataset_utils import k_fold
from augmentation import get_augmentation, set_global_seed
from dataset import Subset, get_dataset
from pipeline_numpy import RawProcessingPipeline
from pipeline_torch import RawToRGB, ParametrizedProcessing, NNProcessing
from model import resnet_model, hybrid_mps_vqc_model, EndToEndModel

# Global Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> Tuple[int, int]:
    """Computes the number of correct predictions and total samples."""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct, len(target)


def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray,
                    class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute classification metrics suited for imbalanced multiclass problems."""
    metrics: Dict[str, Any] = {}

    metrics['accuracy'] = (all_preds == all_labels).mean()
    metrics['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
    metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    metrics['macro_precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['macro_recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-class metrics
    per_class_p = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    labels_present = sorted(set(all_labels) | set(all_preds))

    per_class = []
    for idx, cls_idx in enumerate(labels_present):
        name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
        per_class.append({
            'class': name,
            'precision': float(per_class_p[idx]),
            'recall': float(per_class_r[idx]),
            'f1': float(per_class_f1[idx]),
        })
    metrics['per_class'] = per_class

    # Confusion matrix (raw numpy)
    metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds, labels=labels_present)
    metrics['cm_labels'] = [class_names[i] if class_names and i < len(class_names) else str(i)
                            for i in labels_present]

    return metrics


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    class_names: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
    """Running training for one epoch. Returns (avg_loss, metrics_dict)."""
    model.train()
    running_loss = 0.0

    all_preds = []
    all_labels = []

    total_steps = len(dataloader)
    if total_steps == 0:
        raise RuntimeError("Training dataloader is empty. Check dataset and split configuration.")

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        # Log batch metrics
        batch_correct = preds.eq(labels).sum().item()
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/batch_acc": batch_correct / len(labels),
            "epoch": epoch
        })

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

    # For ISP visualization (once per validation)
    vis_inputs = None
    vis_predictions = None
    vis_labels = None

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Save first batch for visualization
            if i == 0:
                vis_inputs = inputs
                vis_predictions = outputs
                vis_labels = labels

    avg_loss = running_loss / total_steps

    all_preds_np = np.concatenate(all_preds)
    all_labels_np = np.concatenate(all_labels)
    metrics = compute_metrics(all_preds_np, all_labels_np, class_names)

    # Log confusion matrix to WandB
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

    # Log per-class metrics as WandB table
    try:
        table = wandb.Table(columns=["class", "precision", "recall", "f1"])
        for row in metrics['per_class']:
            table.add_data(row['class'], row['precision'], row['recall'], row['f1'])
        wandb.log({"val/per_class_metrics": table}, commit=False)
    except Exception as e:
        print(f"[Warning] Could not log per-class table to WandB: {e}")

    # Log processed images if possible
    if vis_inputs is not None:
        try:
            if model.processor is not None:
                with torch.no_grad():
                    processed_imgs = model.processor(vis_inputs)

                n_vis = min(8, processed_imgs.size(0))
                images_to_log = []

                for idx in range(n_vis):
                    img = processed_imgs[idx].cpu().detach()
                    img = torch.clamp(img, 0, 1)

                    pred_cls = torch.argmax(vis_predictions[idx]).item()
                    true_cls = vis_labels[idx].item()

                    images_to_log.append(
                        wandb.Image(img, caption=f"Pred: {pred_cls} | True: {true_cls}")
                    )

                wandb.log({"val/processed_images": images_to_log}, commit=False)
        except Exception as e:
            print(f"[Warning] Could not log images to WandB: {e}")

    return avg_loss, metrics


def run_train(args: argparse.Namespace) -> None:
    """Main training execution function."""
    
    # 1. Initialize Dataset
    dataset = get_dataset(args.dataset, root_dir=args.data_dir)
    print(f'Dataset: {type(dataset).__name__} [{len(dataset)} samples]')   
    print(f'Task: {dataset.task}')
    print(f'Cross-validation splits: {args.n_splits}')

    # 2. Set Seed used for reproducibility
    set_global_seed(args.seed)

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
        
        print(f'Processing Mode: {args.processing_mode}')

        # 4. Configure Processing Pipeline
        processor = None
        is_rgb_dataset = args.dataset == 'MicroscopyRGB'

        if is_rgb_dataset:
             # Basic normalization for RGB images
             # Using ImageNet stats as default or dataset specific if available
             mean = dataset.dataset_mean or [0.485, 0.456, 0.406]
             std = dataset.dataset_std or [0.229, 0.224, 0.225]
             
             dataset.transform = T.Compose([
                 T.ToTensor(), # dataset.py returns numpy, need to convert to tensor first if not already
                 T.Normalize(mean, std)
             ])
             # Ensure processor is None for RGB dataset as we don't need raw processing
             if args.processing_mode != 'static':
                 print(f"Warning: Processing mode '{args.processing_mode}' ignored for RGB dataset.")

        elif args.processing_mode == 'static':
            # Mean and Std for normalization (from dataset class attributes)
            mean = torch.tensor(dataset.dataset_mean)
            std = torch.tensor(dataset.dataset_std)
            
            # Static pipeline using CPU-based transforms (Numpy)
            dataset.transform = T.Compose([
                RawProcessingPipeline(
                    camera_parameters=dataset.camera_parameters,
                    debayer=args.sp_debayer,
                    sharpening=args.sp_sharpening,
                    denoising=args.sp_denoising,
                ),
                T.Normalize(mean, std)
            ])
        else:
            # Neural/Parametrized pipelines
            mosaic_mean = [0.5663, 0.1401, 0.0731]
            mosaic_std = [0.097, 0.0423, 0.008]
            normalize_mosaic = T.Normalize(mosaic_mean, mosaic_std)

            if args.processing_mode == 'parametrized':
                processor = ParametrizedProcessing(
                    camera_parameters=dataset.camera_parameters, 
                    batch_norm_output=True
                )
            elif args.processing_mode == 'neural_network':
                processor = NNProcessing(
                    normalize_mosaic=normalize_mosaic, 
                    batch_norm_output=True
                )
            elif args.processing_mode == 'none':
                processor = RawToRGB(
                    reduce_size=True, 
                    out_channels=3,
                    normalize_mosaic=normalize_mosaic
                )

        # 5. Initialize Classifier
        if args.classifier_network == 'hybrid_mps_vqc':
            classifier = hybrid_mps_vqc_model(
                fc_out_features=len(dataset.classes),
                n_qubits=args.n_qubits,
                mps_bond_dim=args.mps_bond_dim,
                mps_phys_dim=args.mps_phys_dim,
                mps_hidden_dim=args.mps_hidden_dim,
                vqc_layers=args.vqc_layers,
                freeze_resnet=args.freeze_resnet_backbone,
                use_pennylane=args.use_pennylane,
            )
        else:
            classifier = resnet_model(
                model_name=args.classifier_network,
                pretrained=args.classifier_pretrained,
                fc_out_features=len(dataset.classes)
            )

        if processor and args.freeze_processor:
             # Check if processor has parameters
            if len(list(processor.parameters())) == 0:
                print('Note: Freezing processor requested but it has no learnable parameters.')
        
        if args.freeze_processor and args.freeze_classifier:
            raise ValueError("Both processor and classifier are frozen. Nothing to train.")

        # 6. Initialize End-to-End Model
        augmentation = get_augmentation(args.augmentation)
        model = EndToEndModel(
            classifier=classifier,
            processor=processor,
            augmentation=augmentation,
            freeze_classifier=args.freeze_classifier,
            freeze_processor=args.freeze_processor,
        )
        model = model.to(DEVICE)
        
        print(f"Model moved to {DEVICE}")
        # 7. Setup Training State
        
        # Calculate class counts from training subset
        targets = [dataset.labels[i] for i in train_indices]
        from collections import Counter
        class_counts = Counter(targets)
        
        # Prepare Imbalance Handling Strategy
        sampler = None
        criterion_weight = None
        
        if args.imbalance_handling == 'loss':
            class_weights = []
            for i in range(len(dataset.classes)):
                if class_counts[i] > 0:
                    class_weights.append(1.0 / class_counts[i])
                else:
                    class_weights.append(0.0)
            criterion_weight = torch.FloatTensor(class_weights).to(DEVICE)
            print("[Info] Using Weighted Cross Entropy Loss to handle class imbalance.")
            
        elif args.imbalance_handling == 'sampler':
            # Calculate weight for each sample for WeightedRandomSampler
            sample_weights = []
            for i in train_indices:
                label = dataset.labels[i]
                weight = 1.0 / class_counts[label] if class_counts[label] > 0 else 0.0
                sample_weights.append(weight)
            sample_weights = torch.DoubleTensor(sample_weights)
            
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_indices), replacement=True)
            print("[Info] Using WeightedRandomSampler to handle class imbalance.")

        # Initialize Criterion with or without weights
        criterion = nn.CrossEntropyLoss(weight=criterion_weight)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

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
        shuffle = True
        if sampler is not None:
            shuffle = False
            
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, 
            num_workers=args.num_workers, sampler=sampler, shuffle=shuffle
        )
        valid_loader = DataLoader(
            valid_set, batch_size=args.batch_size, 
            num_workers=args.num_workers, shuffle=False
        )

        # 9. Training Loop
        best_macro_f1 = 0.0
        class_names = dataset.classes if hasattr(dataset, 'classes') else None
        print("Starting training loop...")

        # Watch model gradients
        wandb.watch(model, log='gradients', log_freq=100)

        epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training Epochs")
        for epoch in epoch_pbar:
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE, epoch, class_names
            )

            if epoch % args.check_val_every_n_epoch == 0:
                val_loss, val_metrics = validate(
                    model, valid_loader, criterion, DEVICE, class_names
                )

                # Update progress bar description
                epoch_pbar.set_postfix({
                    'loss': f'{val_loss:.4f}',
                    'mF1': f'{val_metrics["macro_f1"]:.4f}',
                    'bAcc': f'{val_metrics["balanced_accuracy"]:.4f}',
                })

                # Log Epoch Metrics
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
                    "epoch": epoch
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
