import sys
import os
import argparse
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# Local imports
from dataset_utils import k_fold
from augmentation import get_augmentation, set_global_seed
from dataset import Subset, get_dataset
from pipeline_numpy import RawProcessingPipeline
from pipeline_torch import RawToRGB, ParametrizedProcessing, NNProcessing
from model import resnet_model, EndToEndModel

# Global Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description='SecQDevOps QML Raw-Microscopy Classification Training Script'
    )

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='Microscopy',
                        choices=['Microscopy', 'MicroscopyRGB'],
                        help='Dataset to use for training and evaluation.')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset root directory')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Cross-validation arguments
    parser.add_argument('--n_splits', type=int, default=1,
                        help='Number of splits (folds) used for cross-validation.')
    parser.add_argument('--seed', type=int, default=20,
                        help='Seed used for reproducibility.')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='Proportion of the dataset to include in the train split.')

    # Processing arguments
    parser.add_argument('--processing_mode', type=str, default='static',
                        choices=['parametrized', 'static', 'neural_network', 'none'],
                        help='Type of processing to apply to raw images.')
    parser.add_argument('--sp_debayer', type=str, default='bilinear',
                        choices=['bilinear', 'malvar2004', 'menon2007'],
                        help='Algorithm used for debayering.')
    parser.add_argument('--sp_sharpening', type=str, default='sharpening_filter',
                        choices=['sharpening_filter', 'unsharp_masking'],
                        help='Algorithm used for sharpening.')
    parser.add_argument('--sp_denoising', type=str, default='gaussian_denoising',
                        choices=['gaussian_denoising', 'median_denoising', 'fft_denoising'],
                        help='Algorithm used for denoising.')
    
    # Model freezing arguments
    parser.add_argument('--freeze_processor', action='store_true',
                        help='Freeze raw to rgb processing model weights.')
    parser.add_argument('--freeze_classifier', action='store_true',
                        help='Freeze classification model weights.')

    # Training arguments
    parser.add_argument('--augmentation', type=str, default='none',
                        choices=['none', 'weak', 'strong'],
                        help='Applies augmentation to training.')
    parser.add_argument('--test_run', action='store_true',
                        help='Run training with only one batch for testing.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate used for training.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Frequency of validation checks.')
    
    # Classifier arguments
    parser.add_argument('--classifier_network', type=str, default='ResNet18',
                        choices=['ResNet18', 'ResNet34', 'Resnet50'],
                        help='Type of pretrained network.')
    parser.add_argument('--classifier_pretrained', action='store_true',
                        help='Whether to use a pre-trained model or not.')
    
    # Adversarial arguments
    parser.add_argument('--adv_parameters',
                        choices=['all', 'black_level', 'white_balance', 'colour_correction',
                                 'gamma_correct', 'sharpening_filter', 'gaussian_blur', 'additive_layer'],
                        help='Target individual parameters for adversarial training.')

    return parser.parse_args()


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy of the model predictions."""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct / len(target)


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int) -> Tuple[float, float]:
    """Running training for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, labels)
        
    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    
    print(f"[Train] Epoch {epoch}: Loss {avg_loss:.4f} | Acc {avg_acc:.4f}")
    return avg_loss, avg_acc


def validate(model: nn.Module,
             dataloader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Running validation."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels)
            
    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    
    print(f"[Val]   Loss {avg_loss:.4f} | Acc {avg_acc:.4f}")
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
    idxs_kfold = k_fold(dataset, n_splits=args.n_splits, seed=args.seed, train_size=args.train_size)

    for k_iter, (train_indices, valid_indices) in enumerate(idxs_kfold):
        print(f'\n--- Starting Fold {k_iter + 1}/{args.n_splits} ---')
        print(f'Processing Mode: {args.processing_mode}')

        # 4. Configure Processing Pipeline
        processor = None
        if args.processing_mode == 'static':
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
        classifier = resnet_model(
            model_name=args.classifier_network,
            pretrained=args.classifier_pretrained,
            in_channels=3,
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
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Prepare state dict for saving
        experiment_dir = os.path.join('results', args.dataset, args.processing_mode)
        run_dir = os.path.join(experiment_dir, f"fold_{k_iter}")
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

        # 8. Prepare DataLoaders
        if args.test_run:
            print("Test Run: Limiting dataset to single batch.")
            train_indices = train_indices[:args.batch_size]
            valid_indices = valid_indices[:args.batch_size]

        train_set = Subset(dataset, indices=train_indices)
        valid_set = Subset(dataset, indices=valid_indices)

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, 
            num_workers=args.num_workers, shuffle=True
        )
        valid_loader = DataLoader(
            valid_set, batch_size=args.batch_size, 
            num_workers=args.num_workers, shuffle=False
        )

        # 9. Training Loop
        best_acc = 0.0
        print("Starting training loop...")
        
        for epoch in range(1, args.epochs + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
            
            if epoch % args.check_val_every_n_epoch == 0:
                val_loss, val_acc = validate(model, valid_loader, criterion, DEVICE)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(run_dir, 'model_best.pth'))
                    print(f" -> New best model saved (Acc: {best_acc:.4f})")
        
        # Save final model
        torch.save(model.state_dict(), os.path.join(run_dir, 'model_final.pth'))
        print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    run_train(args)
