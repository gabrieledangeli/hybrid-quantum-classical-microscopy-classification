import sys
import os
import argparse
from typing import Tuple, Dict, Any, Optional

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int) -> Tuple[float, float]:
    """Running training for one epoch."""
    model.train()
    running_loss = 0.0
    
    total_correct = 0
    total_samples = 0
    
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
        
        # Calculate batch accuracy for logging/accumulation
        batch_correct, batch_size = calculate_accuracy(outputs, labels)
        total_correct += batch_correct
        total_samples += batch_size
        
        batch_acc = batch_correct / batch_size
        
        # Log batch metrics
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/batch_acc": batch_acc,
            "epoch": epoch
        })
        
    # Average loss per batch
    avg_loss = running_loss / total_steps
    
    # Accuracy over the whole epoch
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_acc


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Running validation."""
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
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
            
            batch_correct, batch_size = calculate_accuracy(outputs, labels)
            total_correct += batch_correct
            total_samples += batch_size
            
            # Save first batch for visualization
            if i == 0:
                vis_inputs = inputs
                vis_predictions = outputs
                vis_labels = labels
            
    avg_loss = running_loss / total_steps
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Log visuals if possible
    if vis_inputs is not None:
        try:
             # If model has a processor, we can visualize the intermediate output
            if model.processor is not None:
                with torch.no_grad():
                    processed_imgs = model.processor(vis_inputs)
                
                # Take first 8 images
                n_vis = min(8, processed_imgs.size(0))
                images_to_log = []
                
                for idx in range(n_vis):
                    img = processed_imgs[idx].cpu().detach()
                    # Denormalize if needed or just clamp for visualization
                    img = torch.clamp(img, 0, 1)
                    
                    pred_cls = torch.argmax(vis_predictions[idx]).item()
                    true_cls = vis_labels[idx].item()
                    
                    images_to_log.append(
                        wandb.Image(img, caption=f"Pred: {pred_cls} | True: {true_cls}")
                    )
                
                wandb.log({"val/processed_images": images_to_log}, commit=False)
        except Exception as e:
            print(f"[Warning] Could not log images to WandB: {e}")
            
    return avg_loss, avg_acc


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
            name=args.run_name or f"{args.dataset}_{args.processing_mode}_fold_{k_iter}",
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
            # Mean and Std for normalization
            mean = torch.tensor([0.91, 0.84, 0.94])
            std = torch.tensor([0.08, 0.12, 0.05])
            
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
            is_segmentation_task=(dataset.task == 'segmentation'),
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
        best_acc = 0.0
        print("Starting training loop...")
        
        # Watch model gradients
        wandb.watch(model, log='gradients', log_freq=100)
        
        epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training Epochs")
        for epoch in epoch_pbar:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
            
            if epoch % args.check_val_every_n_epoch == 0:
                val_loss, val_acc = validate(model, valid_loader, criterion, DEVICE)
                
                # Update progress bar description
                epoch_pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}'})
                
                # Log Epoch Metrics
                wandb.log({
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "epoch": epoch
                })

                if val_acc > best_acc:
                    best_acc = val_acc
                    # Save best model
                    if args.save_model:
                        torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))

        # Finish WandB run
        wandb.finish()

    print("Cross-validation completed.")


if __name__ == "__main__":
    args = parse_args()
    run_train(args)
