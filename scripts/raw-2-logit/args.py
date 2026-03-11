
import argparse


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
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--min_samples_per_class', type=int, default=5,
                        help='Minimum number of samples required per class before rebalancing strategy is applied.')
    parser.add_argument('--below_threshold_strategy', type=str, default='drop',
                        choices=['drop', 'merge_to_rare'],
                        help='How to handle classes below min_samples_per_class.')
    parser.add_argument('--rare_class_name', type=str, default='RARE',
                        help='Name of the merged class when below_threshold_strategy=merge_to_rare.')
    
    # Cross-validation arguments
    parser.add_argument('--n_splits', type=int, default=1,
                        help='Number of splits (folds) used for cross-validation.')
    parser.add_argument('--cv_strategy', type=str, default='shuffle_split',
                        choices=['kfold', 'shuffle_split'],
                        help='Cross-validation strategy.')
    parser.add_argument('--seed', type=int, default=20,
                        help='Seed used for reproducibility.')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='Proportion of the dataset to include in the train split.')
                        
    # Imbalance handling arguments
    parser.add_argument('--imbalance_handling', type=str, default='both',
                        choices=['sampler', 'loss', 'both', 'none'],
                        help='Strategy to handle class imbalance. "both" uses sampler + weighted loss.')
    parser.add_argument('--sampler_mode', type=str, default='balanced',
                        choices=['balanced', 'soft_undersample'],
                        help='Sampler behavior when imbalance_handling uses sampler.')
    parser.add_argument('--undersample_strength', type=float, default=0.7,
                        help='Maximum soft-undersampling strength in [0, 1]. 0 keeps natural distribution, 1 approaches class-balanced sampling.')
    parser.add_argument('--undersample_warmup_epochs', type=int, default=10,
                        help='Epochs used to ramp undersample strength from 0 to undersample_strength.')

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

    # Training arguments
    parser.add_argument('--augmentation', type=str, default='weak',
                        choices=['none', 'weak', 'strong'],
                        help='Applies augmentation to training.')
    parser.add_argument('--conditional_augmentation', action='store_true',
                        help='Apply augmentation preferentially to rare classes only.')
    parser.add_argument('--rare_class_threshold', type=int, default=30,
                        help='A class is considered rare if train-fold count <= this value.')
    parser.add_argument('--non_rare_aug_prob', type=float, default=0.0,
                        help='When conditional augmentation is enabled, probability of augmenting non-rare classes.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for backbone (pretrained layers).')
    parser.add_argument('--lr_head', type=float, default=1e-3,
                        help='Learning rate for randomly-initialized head layers.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) for AdamW.')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor for cross-entropy loss (0=off).')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal'],
                        help='Loss function. "focal" uses Focal Loss (Lin et al., 2017) '
                             'which down-weights easy examples and focuses on hard ones '
                             'recommended for heavily imbalanced datasets with rare classes.')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focusing parameter gamma for Focal Loss. '
                             'gamma=0 reduces to CE; gamma=2 is the standard default; '
                             'gamma=3-5 for extreme imbalance. Only used when --loss_fn focal.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='Number of linear LR warmup epochs (0=off). '
                             'Ramps LR from 1%% to 100%% before cosine decay. '
                             'Recommended 5-10 for hybrid_tn to prevent early collapse.')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Frequency of validation checks.')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the best model checkpoint.')
    
    # Classifier arguments
    parser.add_argument('--classifier_network', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'hybrid_tn'],
                        help='Type of pretrained network.')
    parser.add_argument('--classifier_pretrained', action='store_true',
                        help='Whether to use a pre-trained model or not.')

    # Hybrid TN + VQC arguments (used when --classifier_network hybrid_tn)
    parser.add_argument('--n_qubits', type=int, default=8,
                        help='Number of qubits / tensor-network sites.')
    parser.add_argument('--mps_bond_dim', type=int, default=64,
                        help='Bond dimension for MPS tensor network.')
    parser.add_argument('--mps_phys_dim', type=int, default=2,
                        help='Physical dimension per MPS site (2 = qubits).')
    parser.add_argument('--mps_hidden_dim', type=int, default=128,
                        help='Hidden dimension in per-site MPS encoders.')
    parser.add_argument('--tn_bridge', type=str, default='mps',
                        choices=['mps', 'peps'],
                        help='Tensor-network bridge type used before VQC.')
    parser.add_argument('--peps_rows', type=int, default=None,
                        help='Optional number of PEPS rows. Must satisfy rows*cols=n_qubits.')
    parser.add_argument('--peps_cols', type=int, default=None,
                        help='Optional number of PEPS cols. Must satisfy rows*cols=n_qubits.')
    parser.add_argument('--peps_message_passing_steps', type=int, default=3,
                        help='Number of local message-passing iterations in PEPS bridge.')
    parser.add_argument('--vqc_layers', type=int, default=2,
                        help='Number of parametrized layers in the variational quantum circuit.')
    parser.add_argument('--use_pennylane', action='store_true', default=True,
                        help='Use PennyLane quantum simulation (default). Disable with --no_pennylane.')
    parser.add_argument('--no_pennylane', dest='use_pennylane', action='store_false',
                        help='Use classical MLP fallback instead of PennyLane VQC.')
    parser.add_argument('--freeze_resnet_backbone', action='store_true',
                        help='Freeze ResNet stem and layer1-3 so only layer4 and final head remain trainable (applies to classic ResNet and hybrid backbone).')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='ResNet backbone architecture for the hybrid model. '
                             'resnet50 outputs 2048-d features (vs 512-d for resnet18/34). '
                             'Use with --backbone_weights to load pathology-pretrained weights '
                             '(e.g. RetCCL).')
    parser.add_argument('--backbone_weights', type=str, default=None,
                        help='Path to a .pth state-dict file with custom backbone weights '
                             '(e.g. pathology-pretrained RetCCL for resnet50). '
                             'If None, uses default ImageNet-pretrained weights. '
                             'Key prefixes like module.*/backbone.*/encoder.* are stripped '
                             'automatically.')
    
    # Adversarial arguments
    parser.add_argument('--adv_parameters',
                        choices=['all', 'black_level', 'white_balance', 'colour_correction',
                                 'gamma_correct', 'sharpening_filter', 'gaussian_blur', 'additive_layer'],
                        help='Target individual parameters for adversarial training.')
    
    # WandB arguments
    parser.add_argument('--wandb_project', type=str, default='secqdevops-raw-microscopy',
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity (username)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional run name for WandB')
    parser.add_argument('--wandb_watch', action='store_true',
                        help='Enable wandb.watch() gradient tracking (can add overhead).')

    return parser.parse_args()
