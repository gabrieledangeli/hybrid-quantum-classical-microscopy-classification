"""Command-line argument parsing for the training pipeline."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

DATA_DIR: str = os.environ.get("MICROSCOPY_ROOT", "./data")


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical CNN – Medical Image Classification"
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = parser.add_argument_group("Dataset")
    ds.add_argument("--dataset", type=str, default="Microscopy",
                    choices=["Microscopy", "MicroscopyRGB"],
                    help="Dataset variant.")
    ds.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader worker processes.")
    ds.add_argument("--min_samples_per_class", type=int, default=15,
                    help="Minimum total samples per class (≥15 ensures ≥10 train samples with train_size=0.7).")
    ds.add_argument("--below_threshold_strategy", type=str, default="drop",
                    choices=["drop", "merge_to_rare"],
                    help="How to handle classes below min_samples_per_class.")
    ds.add_argument("--rare_class_name", type=str, default="RARE",
                    help="Merged class name when strategy=merge_to_rare.")

    # ── Cross-validation ──────────────────────────────────────────────────
    cv = parser.add_argument_group("Cross-validation")
    cv.add_argument("--n_splits", type=int, default=1,
                    help="Number of CV splits/folds.")
    cv.add_argument("--cv_strategy", type=str, default="shuffle_split",
                    choices=["kfold", "shuffle_split"],
                    help="Cross-validation strategy.")
    cv.add_argument("--seed", type=int, default=20,
                    help="Random seed for reproducibility.")
    cv.add_argument("--train_size", type=float, default=0.7,
                    help="Proportion of the dataset in the train split.")

    # ── Imbalance handling ────────────────────────────────────────────────
    ib = parser.add_argument_group("Imbalance handling")
    ib.add_argument("--imbalance_handling", type=str, default="both",
                    choices=["sampler", "loss", "both", "none"],
                    help="Strategy to handle class imbalance.")
    ib.add_argument("--sampler_mode", type=str, default="balanced",
                    choices=["balanced", "soft_undersample"],
                    help="Sampler mode when imbalance_handling uses sampler.")
    ib.add_argument("--undersample_strength", type=float, default=0.7,
                    help="Maximum soft-undersampling strength [0, 1].")
    ib.add_argument("--undersample_warmup_epochs", type=int, default=10,
                    help="Epochs to ramp undersampling from 0 → max strength.")

    # ── Processing ────────────────────────────────────────────────────────
    pr = parser.add_argument_group("Image processing")
    pr.add_argument("--processing_mode", type=str, default="static",
                    choices=["parametrized", "static", "neural_network", "none"],
                    help="RAW-to-RGB processing mode.")
    pr.add_argument("--sp_debayer", type=str, default="bilinear",
                    choices=["bilinear", "malvar2004", "menon2007"],
                    help="Demosaicing algorithm.")
    pr.add_argument("--sp_sharpening", type=str, default="sharpening_filter",
                    choices=["sharpening_filter", "unsharp_masking"],
                    help="Sharpening algorithm.")
    pr.add_argument("--sp_denoising", type=str, default="gaussian_denoising",
                    choices=["gaussian_denoising", "median_denoising", "fft_denoising"],
                    help="Denoising algorithm.")
    pr.add_argument("--freeze_processor", action="store_true",
                    help="Freeze processor weights.")

    # ── Augmentation ──────────────────────────────────────────────────────
    ag = parser.add_argument_group("Augmentation")
    ag.add_argument("--augmentation", type=str, default="weak",
                    choices=["none", "weak", "strong"],
                    help="Augmentation intensity.")
    ag.add_argument("--conditional_augmentation", action="store_true",
                    help="Augment only rare classes.")
    ag.add_argument("--rare_class_threshold", type=int, default=30,
                    help="Class is rare if train count ≤ this value.")
    ag.add_argument("--non_rare_aug_prob", type=float, default=0.0,
                    help="Probability of augmenting non-rare classes (conditional mode).")

    # ── Training ──────────────────────────────────────────────────────────
    tr = parser.add_argument_group("Training")
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for pretrained backbone layers.")
    tr.add_argument("--lr_head", type=float, default=1e-3,
                    help="Learning rate for randomly-initialised head layers.")
    tr.add_argument("--weight_decay", type=float, default=1e-4,
                    help="AdamW weight decay.")
    tr.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing factor for CE loss (0 = off).")
    tr.add_argument("--loss_fn", type=str, default="cross_entropy",
                    choices=["cross_entropy", "focal"],
                    help="Loss function.")
    tr.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focusing parameter γ for Focal Loss.")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--lr_warmup_epochs", type=int, default=0,
                    help="Linear LR warmup epochs (0 = off).")
    tr.add_argument("--check_val_every_n_epoch", type=int, default=1)
    tr.add_argument("--save_model", action="store_true",
                    help="Save the best model checkpoint.")

    # ── Classifier ────────────────────────────────────────────────────────
    cl = parser.add_argument_group("Classifier")
    cl.add_argument("--classifier_network", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50", "hybrid_tn"],
                    help="Classifier architecture.")
    cl.add_argument("--classifier_pretrained", action="store_true",
                    help="Use ImageNet-pretrained weights.")
    cl.add_argument("--freeze_resnet_backbone", action="store_true",
                    help="Freeze ResNet stem + layer1-3.")
    cl.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"],
                    help="ResNet backbone for the hybrid model.")
    cl.add_argument("--backbone_weights", type=str, default=None,
                    help="Path to custom .pth backbone weights (e.g. RetCCL).")

    # ── Hybrid TN + VQC ──────────────────────────────────────────────────
    hy = parser.add_argument_group("Hybrid TN + VQC")
    hy.add_argument("--n_qubits", type=int, default=8,
                    help="Number of MPS sites/qubits. PEPS uses 16 fixed sites; TTN uses 16 leaves and 1/2/4/8 final states.")
    hy.add_argument("--mps_bond_dim", type=int, default=64,
                    help="Internal TN bond dimension. For exact PEPS this is clipped to 8.")
    hy.add_argument("--mps_phys_dim", type=int, default=2)
    hy.add_argument("--mps_hidden_dim", type=int, default=128)
    hy.add_argument("--tn_bridge", type=str, default="mps",
                    choices=["mps", "peps", "ttn"],
                    help="Tensor-network bridge type. PEPS uses a fixed exact 4x4 topology; TTN uses a fixed 16-leaf binary tree.")
    hy.add_argument("--vqc_layers", type=int, default=2,
                    help="Parametrised VQC layers.")
    hy.add_argument("--use_pennylane", action="store_true", default=True,
                    help="Use PennyLane quantum simulation.")
    hy.add_argument("--no_pennylane", dest="use_pennylane", action="store_false",
                    help="Classical MLP fallback instead of VQC.")

    # ── Adversarial ───────────────────────────────────────────────────────
    parser.add_argument("--adv_parameters",
                        choices=["all", "black_level", "white_balance",
                                 "colour_correction", "gamma_correct",
                                 "sharpening_filter", "gaussian_blur",
                                 "additive_layer"],
                        help="Target parameters for adversarial training.")

    # ── Weights & Biases ─────────────────────────────────────────────────
    wb = parser.add_argument_group("Weights & Biases")
    wb.add_argument("--wandb_project", type=str, default="hqcnn-medical",
                    help="W&B project name.")
    wb.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B entity (user/team).")
    wb.add_argument("--run_name", type=str, default=None,
                    help="Custom W&B run name.")
    wb.add_argument("--wandb_watch", action="store_true",
                    help="Enable wandb.watch() gradient tracking.")

    args = parser.parse_args()
    args.data_dir = DATA_DIR
    return args
