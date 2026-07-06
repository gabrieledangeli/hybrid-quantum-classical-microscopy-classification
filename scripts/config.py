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
    cv.add_argument("--test_size", type=float, default=0.0,
                    help="Optional fixed stratified test holdout fraction. The holdout "
                         "is created before CV and is used only for final "
                         "evaluation of the best validation checkpoint. Default 0 disables it.")

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
    tr.add_argument("--save_backbone", action="store_true",
                    help="For ResNet classifiers, save a backbone checkpoint "
                         "at the best validation epoch. Useful for MLL23 -> "
                         "Microscopy transfer.")

    # -- Classifier --------------------------------------------------------
    cl = parser.add_argument_group("Classifier")
    cl.add_argument("--classifier_network", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50", "hybrid_vqc"],
                    help="Classifier architecture.")
    cl.add_argument("--classifier_pretrained", action="store_true",
                    help="Use pretrained ResNet backbone weights. If --backbone_weights "
                         "is omitted, torchvision ImageNet weights are used.")
    cl.add_argument("--resnet_freeze_mode", type=str, default="none",
                    choices=["none", "partial", "layer34", "full"],
                    help="How much of a pretrained ResNet backbone to freeze. "
                         "'none' trains the full network, "
                         "'partial' freezes stem + layer1-3 while leaving layer4 trainable, "
                         "'layer34' freezes stem + layer1-2 while leaving layer3 + layer4 trainable, "
                         "'full' freezes the entire ResNet backbone.")
    cl.add_argument("--backbone", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"],
                    help="ResNet backbone for the hybrid model.")
    cl.add_argument("--backbone_weights", type=str, default=None,
                    help="Path to custom .pth ResNet backbone weights used when "
                         "--classifier_pretrained is enabled. If omitted, "
                         "--classifier_pretrained uses ImageNet weights.")

    # -- Hybrid models ----------------------------------------------------
    hy = parser.add_argument_group("Hybrid models")
    hy.add_argument("--n_qubits", type=int, default=8,
                    help="Number of VQC qubits (minimum 2).")
    hy.add_argument("--vqc_layers", type=int, default=2,
                    help="Parametrised VQC layers (minimum 1; also the number of data re-uploads).")
    hy.add_argument("--vqc_readout", type=str, default="z+zz",
                    choices=["z", "z+zz", "z+zz_mean", "bloch", "full"],
                    help="VQC read-out observables: 'z' single-qubit <Z>; 'z+zz' adds "
                         "nearest-neighbour <ZZ> ring correlators (default); 'z+zz_mean' "
                         "adds one pooled mean of the <ZZ> ring correlators; 'bloch' the "
                         "full (<X>,<Y>,<Z>) vector; 'full' adds {XX,YY,ZZ} ring correlators.")
    hy.add_argument("--vqc_head_type", type=str, default="vqc",
                    choices=["vqc", "classical_affine", "classical_lowrank", "classical_mlp"],
                    help="hybrid_vqc head block: the VQC (default) or a classical control "
                         "swapped in its place. 'classical_affine'/'classical_lowrank' match the "
                         "VQC parameter budget; 'classical_mlp' is an over-capacity control.")
    hy.add_argument("--vqc_angle_map", type=str, default="linear",
                    choices=["linear", "mlp"],
                    help="hybrid_vqc adapter from ResNet features to VQC angles: 'linear' "
                         "(default) is a single feat->3q map; 'mlp' adds a hidden layer "
                         "of size --vqc_angle_hidden_dim with GELU and dropout.")
    hy.add_argument("--vqc_angle_hidden_dim", type=int, default=64,
                    help="Hidden dimension of the MLP angle map.")
    hy.add_argument("--vqc_angle_dropout", type=float, default=0.2,
                    help="Dropout probability inside the MLP angle map.")
    hy.add_argument("--vqc_entangler", type=str, default="cnot",
                    choices=["cnot", "crz", "ising_zz"],
                    help="Two-qubit entangler on the ring: fixed CNOTs (default) or "
                         "trainable CRZ / IsingZZ gates (+n_layers*n_pairs params, "
                         "near-identity init). CRZ/IsingZZ are diagonal in Z, so with "
                         "a Z-only --vqc_readout combine them with --vqc_final_rotation, "
                         "otherwise the last entangling ring is invisible to the read-out.")
    hy.add_argument("--vqc_final_rotation", action="store_true",
                    help="Append a trainable Rot(a,b,g) per qubit after the last "
                         "entangling ring (trainable measurement basis; "
                         "+3*n_qubits params, near-identity init).")
    hy.add_argument("--vqc_blocks", type=int, default=1,
                    help="Split the VQC into this many independent parallel sub-circuits "
                         "of n_qubits/vqc_blocks qubits each, with concatenated "
                         "read-outs. Must divide --n_qubits and leave at least "
                         "2 qubits per block. Default 1 = single circuit.")

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
    if args.n_qubits < 2:
        parser.error(f"--n_qubits must be >= 2, got {args.n_qubits}.")
    if args.vqc_layers < 1:
        parser.error(f"--vqc_layers must be >= 1, got {args.vqc_layers}.")
    if args.vqc_blocks < 1:
        parser.error(f"--vqc_blocks must be >= 1, got {args.vqc_blocks}.")
    if args.n_qubits % args.vqc_blocks != 0:
        parser.error(
            f"--vqc_blocks must divide --n_qubits "
            f"(got {args.vqc_blocks} with {args.n_qubits} qubits)."
        )
    if args.n_qubits // args.vqc_blocks < 2:
        parser.error(
            f"Each VQC block must have >= 2 qubits; got --n_qubits "
            f"{args.n_qubits} and --vqc_blocks {args.vqc_blocks}."
        )
    if args.resnet_freeze_mode != "none" and not args.classifier_pretrained:
        parser.error("--resnet_freeze_mode partial or full requires --classifier_pretrained.")

    if args.backbone_weights and not args.classifier_pretrained:
        parser.error("--backbone_weights requires --classifier_pretrained.")
    if args.save_backbone and args.classifier_network not in (
        "resnet18", "resnet34", "resnet50", "hybrid_vqc"
    ):
        parser.error("--save_backbone requires a ResNet-backed classifier.")
    if args.data_dir is None:
        if args.dataset == "MLL23":
            if not MLL23_DATA_DIR:
                parser.error("--data_dir is required for MLL23 unless MLL23_DATA_DIR is set in .env.")
            args.data_dir = MLL23_DATA_DIR
        else:
            args.data_dir = MICROSCOPY_DATA_DIR
    return args
