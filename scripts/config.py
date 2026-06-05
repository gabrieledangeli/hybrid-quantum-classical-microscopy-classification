"""Command-line argument parsing for the training pipeline."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

MICROSCOPY_DATA_DIR = os.environ.get("MICROSCOPY_ROOT", "./data")
MLL23_DATA_DIR = os.environ.get("MLL23_DATA_DIR")


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical CNN - Medical Image Classification"
    )

    # -- Dataset -----------------------------------------------------------
    ds = parser.add_argument_group("Dataset")
    ds.add_argument("--dataset", type=str, default="Microscopy",
                    choices=["Microscopy", "MicroscopyRGB", "MLL23"],
                    help="Dataset variant.")
    ds.add_argument("--data_dir", type=str, default=None,
                    help="Path to the dataset root directory. If omitted, Microscopy "
                         "datasets use MICROSCOPY_ROOT from .env or './data', while "
                         "MLL23 uses MLL23_DATA_DIR from .env.")
    ds.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader worker processes.")
    ds.add_argument("--min_samples_per_class", type=int, default=15,
                    help="Drop (or merge, see --below_threshold_strategy) any class "
                         "with fewer than this many total samples. Default 15 ensures "
                         ">= 10 train samples both with the default 5-fold CV and with "
                         "--cv_strategy shuffle_split --train_size 0.7; raise it to be "
                         "more aggressive about removing extreme rare classes.")
    ds.add_argument("--exclude_classes", type=str, nargs="*", default=[],
                    help="Optional class labels to exclude entirely before class-count "
                         "thresholding, e.g. --exclude_classes UNC.")
    ds.add_argument("--below_threshold_strategy", type=str, default="drop",
                    choices=["drop", "merge_to_rare"],
                    help="How to handle classes below min_samples_per_class.")
    ds.add_argument("--rare_class_name", type=str, default="RARE",
                    help="Merged class name when strategy=merge_to_rare.")
    ds.add_argument("--missing_image_policy", type=str, default="error",
                    choices=["error", "drop"],
                    help="How to handle annotations whose image file is missing from "
                         "the selected image directory.")

    # -- Cross-validation --------------------------------------------------
    cv = parser.add_argument_group("Cross-validation")
    cv.add_argument("--n_splits", type=int, default=5,
                    help="Number of CV splits/folds.")
    cv.add_argument("--cv_strategy", type=str, default="kfold",
                    choices=["kfold", "shuffle_split"],
                    help="Cross-validation strategy.")
    cv.add_argument("--seed", type=int, default=20,
                    help="Random seed for reproducibility.")
    cv.add_argument("--train_size", type=float, default=0.7,
                    help="Proportion of the dataset in the train split (Default: 0.7).")

    # -- Imbalance handling ------------------------------------------------
    ib = parser.add_argument_group("Imbalance handling")
    ib.add_argument("--imbalance_handling", type=str, default="loss",
                    choices=["sampler", "loss", "both", "none"],
                    help="Strategy to handle class imbalance.")
    ib.add_argument("--sampler_mode", type=str, default="balanced",
                    choices=["balanced", "soft_undersample"],
                    help="Sampler mode when imbalance_handling uses sampler.")
    ib.add_argument("--undersample_strength", type=float, default=0.7,
                    help="Maximum soft-undersampling strength [0, 1].")

    # -- Processing --------------------------------------------------------
    pr = parser.add_argument_group("Image processing")
    pr.add_argument("--processing_mode", type=str, default="static",
                    choices=["parametrized", "static", "neural_network", "none"],
                    help="RAW-to-RGB processing mode (Mandatory for Raw images).")
    pr.add_argument("--sp_debayer", type=str, default="bilinear",
                    choices=["bilinear", "malvar2004", "menon2007"],
                    help="Demosaicing algorithm.")
    pr.add_argument("--sp_sharpening", type=str, default="unsharp_masking",
                    choices=["sharpening_filter", "unsharp_masking"],
                    help="Sharpening algorithm.")
    pr.add_argument("--sp_denoising", type=str, default="gaussian_denoising",
                    choices=["gaussian_denoising", "median_denoising"],
                    help="Denoising algorithm.")
    pr.add_argument("--freeze_processor", action="store_true",
                    help="Freeze processor weights.")

    # -- Augmentation ------------------------------------------------------
    ag = parser.add_argument_group("Augmentation")
    ag.add_argument("--augmentation", type=str, default="weak",
                    choices=["none", "weak", "strong"],
                    help="Augmentation intensity.")
    ag.add_argument("--conditional_augmentation", action="store_true",
                    help="Augment only rare classes.")
    ag.add_argument("--rare_class_threshold", type=int, default=30,
                    help="Class is rare if train count <= this value.")
    ag.add_argument("--non_rare_aug_prob", type=float, default=0.0,
                    help="Probability of augmenting non-rare classes (conditional mode).")

    # -- Training ----------------------------------------------------------
    tr = parser.add_argument_group("Training")
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for pretrained backbone layers.")
    tr.add_argument("--lr_head", type=float, default=1e-3,
                    help="Learning rate for randomly-initialised head layers.")
    tr.add_argument("--weight_decay", type=float, default=1e-4,
                    help="AdamW weight decay.")
    tr.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Label smoothing factor (0 = off). For focal loss, "
                         "0.0 is recommended unless intentionally testing "
                         "smoothed focal loss.")
    tr.add_argument("--loss_fn", type=str, default="focal",
                    choices=["cross_entropy", "focal"],
                    help="Loss function.")
    tr.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focusing parameter gamma for Focal Loss.")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--save_model", action="store_true",
                    help="Save the best model checkpoint.")

    # -- Classifier --------------------------------------------------------
    cl = parser.add_argument_group("Classifier")
    cl.add_argument("--classifier_network", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50", "hybrid_tn", "hybrid_vqc"],
                    help="Classifier architecture.")
    cl.add_argument("--classifier_pretrained", action="store_true",
                    help="Use pretrained weights. For hybrid models this applies to the "
                         "ResNet backbone unless --backbone_weights is provided.")
    cl.add_argument("--freeze_resnet_backbone", action="store_true",
                    help="Freeze ResNet stem + layer1-3. For pure ResNet, layer4 + fc "
                         "stay trainable; for hybrid models, only layer4 stays trainable "
                         "because the classifier head lives downstream.")
    cl.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"],
                    help="ResNet backbone for the hybrid model.")
    cl.add_argument("--backbone_weights", type=str, default=None,
                    help="Path to custom .pth backbone weights (e.g. RetCCL).")

    # -- Hybrid models ----------------------------------------------------
    hy = parser.add_argument_group("Hybrid models")
    hy.add_argument("--n_qubits", type=int, default=8,
                    help="Number of VQC qubits. For hybrid_tn, this is also the "
                         "number of TN sites/final states. PEPS uses 16 fixed sites.")
    hy.add_argument("--ttn_leaves", type=int, default=64,
                    choices=[16, 32, 64, 128],
                    help="Number of input leaves for TTN.")
    hy.add_argument("--tn_bond_dim", type=int, default=64,
                    help="Internal TN bond dimension. For exact PEPS this is clipped to 8.")
    hy.add_argument("--tn_phys_dim", type=int, default=2,
                    help="Physical (site) dimension of the TN.")
    hy.add_argument("--tn_hidden_dim", type=int, default=128,
                    help="Hidden dimension of the per-site encoder MLP feeding the TN.")
    hy.add_argument("--tn_bridge", type=str, default="mps",
                    choices=["mps", "peps", "ttn"],
                    help="Tensor-network bridge type for hybrid_tn. PEPS uses a fixed "
                         "exact 4x4 topology; TTN uses --ttn_leaves leaves.")
    hy.add_argument("--vqc_layers", type=int, default=2,
                    help="Parametrised VQC layers (also the number of data re-uploads).")
    hy.add_argument("--vqc_readout", type=str, default="z+zz",
                    choices=["z", "z+zz", "z+zz_mean", "bloch", "full"],
                    help="VQC read-out observables: 'z' single-qubit <Z>; 'z+zz' adds "
                         "nearest-neighbour <ZZ> ring correlators (default); 'z+zz_mean' "
                         "adds one pooled mean of the <ZZ> ring correlators; 'bloch' the "
                         "full (<X>,<Y>,<Z>) vector; 'full' adds {XX,YY,ZZ} ring correlators.")
    hy.add_argument("--vqc_dropout", type=float, default=0.0,
                    help="Dropout probability applied to the VQC read-out before q_proj.")

    # -- Adversarial -------------------------------------------------------
    parser.add_argument("--adv_parameters",
                        choices=["all", "black_level", "white_balance",
                                 "colour_correction", "gamma_correct",
                                 "sharpening_filter", "gaussian_blur",
                                 "additive_layer"],
                        help="Target parameters for adversarial training.")

    # -- Weights & Biases -------------------------------------------------
    wb = parser.add_argument_group("Weights & Biases")
    wb.add_argument("--wandb_project", type=str, default="hqcnn-medical",
                    help="W&B project name.")
    wb.add_argument("--wandb_entity", type=str, default=None,
                    help="W&B entity (user).")
    wb.add_argument("--run_name", type=str, default=None,
                    help="Optional prefix for W&B and local run names. Timestamp "
                         "and fold suffix are appended automatically.")
    wb.add_argument("--wandb_watch", action="store_true",
                    help="Enable wandb.watch() gradient tracking.")

    args = parser.parse_args()
    if args.data_dir is None:
        if args.dataset == "MLL23":
            if not MLL23_DATA_DIR:
                parser.error("--data_dir is required for MLL23 unless MLL23_DATA_DIR is set in .env.")
            args.data_dir = MLL23_DATA_DIR
        else:
            args.data_dir = MICROSCOPY_DATA_DIR
    return args
