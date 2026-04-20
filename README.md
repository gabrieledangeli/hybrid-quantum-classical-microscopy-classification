# HQCNNxLargeScaleMedicalImages

**Hybrid Quantum-Classical Convolutional Neural Network for Large-Scale Medical Image Classification**

A research framework that combines classical deep learning (ResNet) with quantum-inspired tensor networks (MPS, PEPS) and variational quantum circuits (VQC) for the classification of microscopy blood-cell images - specifically the Matek *et al.* leukocyte morphology dataset.

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
              |  |  ResNet Backbone |    |
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |   TN Bridge     |    |  (MPS or PEPS)
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |   VQC / MLP     |    |  (PennyLane or classical)
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |   BN -> Linear   |    |
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
+-- scripts/
    +-- __init__.py
    +-- config.py               # CLI argument definitions
    +-- train.py                # Training entry-point
    +-- data/
    |   +-- __init__.py
    |   +-- augmentation.py     # Augmentation pipelines & seeding
    |   +-- dataset.py          # Dataset classes (RAW & RGB)
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
        +-- visualization.py    # Tensor saving & image grid export
```

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU recommended (but CPU training is supported)

### Steps

```bash
# Clone the repository
git clone https://github.com/<your-org>/HQCNNxLargeScaleMedicalImages.git
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

### Optional: faster quantum simulation

```bash
pip install pennylane-lightning          # CPU adjoint (5-20x faster)
pip install pennylane-lightning[gpu]     # GPU adjoint (requires CUDA)
```

---

## Dataset Setup

The framework expects a **Matek *et al.* microscopy dataset** organised as follows:

```
data/
+-- images/
|   +-- raw_scale100/      # 16-bit RAW Bayer TIFF images
|   |   +-- <filename>.tiff
|   |   +-- ...
|   +-- rgb_scale100/      # Pre-processed 16-bit RGB TIFF images
|       +-- <filename>.tiff
|       +-- ...
+-- labels/
    +-- Ma190c_annotations.dat   # <stem> <CLASS> per line
```

Pass `--data_dir ./data` (or the actual path) when launching training.

### Supported classes (16 leukocyte subtypes)

`BAS`, `EBO`, `EOS`, `KSC`, `LYA`, `LYT`, `MMZ`, `MOB`, `MON`, `MYB`, `MYO`, `NGB`, `NGS`, `PMB`, `PMO`, `UNC`

Classes below `--min_samples_per_class` can be **dropped** or **merged** into a single `RARE` class.

---

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
    --peps_message_passing_steps 3 \
    --epochs 30
```

### RGB dataset with strong augmentation

```bash
python -m scripts.train \
    --dataset MicroscopyRGB \
    --data_dir ./data \
    --classifier_network resnet50 \
    --classifier_pretrained \
    --augmentation strong \
    --conditional_augmentation \
    --rare_class_threshold 30 \
    --imbalance_handling both \
    --epochs 50
```

---

## Configuration Reference

Run `python -m scripts.train --help` for the full list. Key groups:

| Group | Key flags |
|---|---|
| **Dataset** | `--dataset`, `--data_dir`, `--min_samples_per_class`, `--below_threshold_strategy` |
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

- **Weights & Biases**: Each fold creates a separate W&B run with loss curves, accuracy, macro/weighted F1, balanced accuracy, per-class precision/recall, and confusion matrices.
- **Checkpoints**: When `--save_model` is enabled, the best model (by val macro-F1) is saved to `results/<dataset>/<processing_mode>/fold_<k>/best_model.pth`.
- **Config snapshot**: A `state_dict.txt` file records the full CLI invocation and hyperparameters alongside each checkpoint.

---

## License

This project is released under the terms specified in [LICENSE](LICENSE).
