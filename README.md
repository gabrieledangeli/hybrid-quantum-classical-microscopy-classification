# HQCNNxLargeScaleMedicalImages

**Hybrid Quantum-Classical Convolutional Neural Network for Large-Scale Medical Image Classification**

A research framework that combines classical deep learning (ResNet) with quantum-inspired tensor networks (MPS, PEPS, TTN) and variational quantum circuits (VQC) for the classification of microscopy blood-cell images.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Usage](#usage)
7. [Configuration Reference](#configuration-reference)
8. [Results & Logging](#results--logging)
9. [License](#license)

---

## Overview

This project investigates whether **tensor-network (TN) bridges** and **variational quantum circuits (VQC)** can improve classification performance on imbalanced, high-resolution medical microscopy images compared to standard ResNet baselines.

### Key features

| Feature | Description |
|---|---|
| **Multiple classifiers** | ResNet-18/34/50 (pure) and Hybrid TN (ResNet -> MPS/PEPS -> VQC -> head) |
| **RAW image processing** | Static (NumPy), parametrised (differentiable PyTorch), and neural-network (U-Net++) pipelines |
| **Class-imbalance handling** | Weighted loss (Effective Number of Samples), weighted random sampling, soft undersampling with warm-up, conditional augmentation for rare classes |
| **Cross-validation** | Stratified K-Fold and Stratified Shuffle-Split |
| **Quantum simulation** | PennyLane with automatic backend selection (`lightning.gpu` -> `lightning.qubit` -> `default.qubit`) and classical MLP fallback |
| **Experiment tracking** | Weights & Biases integration (metrics, confusion matrices, per-class stats, gradient tracking) |

---

## Architecture

```
                    +--------------+
   RAW image ------>|  Processor   |--> RGB
                    | (static /    |
                    |  param / NN) |
                    +------+-------+
                           |
                    +------v-------+
                    | Augmentation  |   (weak / strong / conditional)
                    +------+-------+
                           |
                    +------v-------+
                    | Normalisation |   (channel-wise mu/sigma)
                    +------+-------+
                           |
              +------------v------------+
              |       Classifier        |
              |                         |
              |  +-----------------+    |
              |  | ResNet Backbone |    |
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |   TN Bridge     |    |  (MPS or PEPS)
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |       VQC       |    |  (PennyLane or classical)
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |       MLP       |    |
              |  +-----------------+    |
              +-------------------------+
                           |
                        logits
```

---

## Project Structure

```
HQCNNxLargeScaleMedicalImages/
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- notebooks/
|   +-- microscopy_dataset_analysis.ipynb
|   +-- MML23_dataset_analysis.ipynb
+-- scripts/
    +-- __init__.py
    +-- config.py               # CLI argument definitions
    +-- train.py                # Training entry-point
    +-- data/
    |   +-- __init__.py
    |   +-- augmentation.py     # Augmentation pipelines & seeding
    |   +-- dataset.py          # Dataset classes (RAW & RGB)
    |   +-- export_raw_scale100.py  # New acquisition -> synthetic raw_scale100 TIFF
    |   +-- utils.py            # I/O, CV splits, sampling helpers
    +-- models/
    |   +-- __init__.py
    |   +-- classifiers.py      # ResNet factory, EndToEndModel, TensorNormalize
    |   +-- hybrid.py           # MPS/PEPS bridges, VQC, HybridTNClassifier
    |   +-- losses.py           # Focal Loss
    +-- processing/
    |   +-- __init__.py
    |   +-- numpy_pipeline.py   # Non-differentiable RAW->RGB (NumPy/SciPy)
    |   +-- torch_pipeline.py   # Differentiable RAW->RGB (PyTorch)
    +-- utils/
        +-- __init__.py
        +-- conversions.py      # np <-> torch tensor helpers
        +-- metrics.py          # Multi-class classification metrics
```

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU recommended (but CPU training is supported)

### Steps

```bash
# Clone the repository
git clone https://github.com/gabrieledangeli/HQCNNxLargeScaleMedicalImages.git
cd HQCNNxLargeScaleMedicalImages

# Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Faster quantum simulation

```bash
pip install pennylane-lightning          # CPU adjoint (5-20x faster)
pip install pennylane-lightning[gpu]     # GPU adjoint (requires CUDA)
```

---

## Dataset Setup

The framework supports the raw microscopy dataset and the MLL23
single-cell peripheral blood image dataset.

### Raw microscopy dataset

The raw microscopy dataset is organised as follows:

```
data/
+-- images/
|   +-- raw_scale100/      # 16-bit RAW Bayer TIFF images
|   |   +-- <filename>.tiff
|   |   +-- ...
|   +-- rgb_scale100/      # Pre-processed RGB images
|       +-- <filename>.tiff
|       +-- ...
+-- labels/
    +-- Ma190c_annotations.dat   # <stem> <CLASS> per line
```

Pass `--data_dir ./data` (or the actual path) when launching training. If omitted, the CLI falls back to `MICROSCOPY_ROOT` from `.env`, then to `./data`.

#### Supported classes (15 leukocyte subtypes)

`BAS`, `EBO`, `EOS`, `KSC`, `LYA`, `LYT`, `MMZ`, `MOB`, `MON`, `MYB`, `MYO`, `NGB`, `NGS`, `PMB`, `PMO`

Classes below `--min_samples_per_class` can be **dropped** or **merged** into a single `RARE` class. The default `15` is chosen so that each class still contributes at least 10 training samples both with the default 5-fold CV and with a 70/30 stratified shuffle split.

Use `--exclude_classes` to remove specific labels entirely before thresholding.

By default the loader raises an error when an annotation points to a missing image file. Set `--missing_image_policy drop` to skip those annotations instead.

### MLL23 image dataset

For MLL23, the expected root is the folder that directly contains one
subdirectory per class:

```
MML23/
+-- basophil/
|   +-- basophil_0001.TIF
|   +-- ...
+-- eosinophil/
|   +-- eosinophil_0001.TIF
|   +-- ...
+-- ...
```

Labels are inferred from the class directory names. The loader accepts `.tif`
and `.tiff` files and treats the images as pre-processed RGB microscopy images.

If `--data_dir` is omitted for `--dataset MLL23`, the CLI reads
`MML23_DATA_DIR` from `.env`.

## Usage

### Basic training (ResNet-18, static pipeline)

```bash
python -m scripts.train \
    --dataset Microscopy \
    --data_dir ./data \
    --classifier_network resnet18 \
    --classifier_pretrained \
    --processing_mode static \
    --epochs 30 \
    --batch_size 64
```

### Hybrid TN + VQC (MPS bridge)

```bash
python -m scripts.train \
    --dataset Microscopy \
    --data_dir ./data \
    --classifier_network hybrid_tn \
    --backbone resnet18 \
    --tn_bridge mps \
    --n_qubits 8 \
    --vqc_layers 2 \
    --freeze_resnet_backbone \
    --epochs 30
```

### Hybrid TN + VQC (PEPS bridge)

```bash
python -m scripts.train \
    --dataset Microscopy \
    --data_dir ./data \
    --classifier_network hybrid_tn \
    --tn_bridge peps \
    --n_qubits 8 \
    --epochs 30
```

### RGB dataset with strong augmentation

```bash
python -m scripts.train \
    --dataset MicroscopyRGB \
    --data_dir ./data \
    --exclude_classes UNC \
    --missing_image_policy drop \
    --classifier_network resnet50 \
    --classifier_pretrained \
    --augmentation strong \
    --conditional_augmentation \
    --rare_class_threshold 30 \
    --imbalance_handling both \
    --epochs 50
```

### MLL23 reference baseline (ResNet-18)

```bash
python -m scripts.train \
    --dataset MLL23 \
    --classifier_network resnet18 \
    --classifier_pretrained \
    --freeze_resnet_backbone \
    --processing_mode none \
    --augmentation weak \
    --imbalance_handling both \
    --cv_strategy shuffle_split \
    --n_splits 1 \
    --train_size 0.7 \
    --epochs 30 \
    --batch_size 64 \
    --run_name mll23_resnet18_reference
```

---

## Configuration Reference

Run `python -m scripts.train --help` for the full list. Key groups:

| Group | Key flags |
|---|---|
| **Dataset** | `--dataset`, `--data_dir`, `--min_samples_per_class`, `--exclude_classes`, `--below_threshold_strategy`, `--missing_image_policy` |
| **Cross-validation** | `--n_splits`, `--cv_strategy`, `--seed`, `--train_size` |
| **Imbalance** | `--imbalance_handling`, `--sampler_mode`, `--undersample_strength` |
| **Processing** | `--processing_mode`, `--sp_debayer`, `--sp_sharpening`, `--sp_denoising` |
| **Augmentation** | `--augmentation`, `--conditional_augmentation`, `--rare_class_threshold` |
| **Training** | `--batch_size`, `--lr`, `--lr_head`, `--epochs`, `--loss_fn`, `--focal_gamma` |
| **Classifier** | `--classifier_network`, `--classifier_pretrained`, `--freeze_resnet_backbone` |
| **Hybrid TN** | `--n_qubits`, `--tn_bond_dim`, `--tn_phys_dim`, `--tn_hidden_dim`, `--tn_bridge`, `--vqc_layers` |
| **W&B** | `--wandb_project`, `--wandb_entity`, `--run_name`, `--wandb_watch` |

---

## Results & Logging

- **Weights & Biases**: Each fold creates a separate W&B run with loss curves, accuracy, macro/weighted F1, balanced accuracy, per-class precision/recall, plus one `val/last_confusion_matrix` and one `val/best_confusion_matrix` instead of logging a new confusion matrix every validation step.
- **Checkpoints**: When `--save_model` is enabled, the best model (by val macro-F1) is saved to `results/<dataset>/<processing_mode>/fold_<k>/best_model.pth`.
- **Config snapshot**: A `state_dict.txt` file records the full CLI invocation and hyperparameters alongside each checkpoint.
- **Fine-tuning semantics**: `--freeze_resnet_backbone` keeps `layer4 + fc` trainable for plain ResNets. In `hybrid_tn`, the ResNet extractor has no FC head, so the same flag leaves only `layer4` trainable.

---

## License

This project is released under the terms specified in [LICENSE](LICENSE).
