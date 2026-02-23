
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
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Cross-validation arguments
    parser.add_argument('--n_splits', type=int, default=1,
                        help='Number of splits (folds) used for cross-validation.')
    parser.add_argument('--cv_strategy', type=str, default='kfold',
                        choices=['kfold', 'shuffle_split'],
                        help='Cross-validation strategy.')
    parser.add_argument('--seed', type=int, default=20,
                        help='Seed used for reproducibility.')
    parser.add_argument('--train_size', type=float, default=0.8,
                        help='Proportion of the dataset to include in the train split.')
                        
    # Imbalance handling arguments
    parser.add_argument('--imbalance_handling', type=str, default='sampler',
                        choices=['sampler', 'loss', 'none'],
                        help='Strategy to handle class imbalance.')

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
    parser.add_argument('--augmentation', type=str, default='weak',
                        choices=['none', 'weak', 'strong'],
                        help='Applies augmentation to training.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate used for training.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Frequency of validation checks.')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the best model checkpoint.')
    
    # Classifier arguments
    parser.add_argument('--classifier_network', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'hybrid_mps_vqc'],
                        help='Type of pretrained network.')
    parser.add_argument('--classifier_pretrained', action='store_true',
                        help='Whether to use a pre-trained model or not.')

    # Hybrid MPS + VQC arguments (used when --classifier_network hybrid_mps_vqc)
    parser.add_argument('--n_qubits', type=int, default=8,
                        help='Number of qubits / MPS sites.')
    parser.add_argument('--mps_bond_dim', type=int, default=64,
                        help='Bond dimension for MPS tensor network.')
    parser.add_argument('--mps_phys_dim', type=int, default=2,
                        help='Physical dimension per MPS site (2 = qubits).')
    parser.add_argument('--mps_hidden_dim', type=int, default=128,
                        help='Hidden dimension in per-site MPS encoders.')
    parser.add_argument('--vqc_layers', type=int, default=2,
                        help='Number of parametrized layers in the variational quantum circuit.')
    parser.add_argument('--use_pennylane', action='store_true', default=True,
                        help='Use PennyLane quantum simulation (default). Disable with --no_pennylane.')
    parser.add_argument('--no_pennylane', dest='use_pennylane', action='store_false',
                        help='Use classical MLP fallback instead of PennyLane VQC.')
    parser.add_argument('--freeze_resnet_backbone', action='store_true',
                        help='Freeze ResNet18 backbone inside the hybrid MPS+VQC classifier.')
    
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

    return parser.parse_args()