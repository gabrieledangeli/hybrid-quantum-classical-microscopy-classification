# HQCNNxLargeScaleMedicalImages

**Hybrid Quantum-Classical Convolutional Neural Network for Large-Scale Medical Image Classification**

A research framework that combines classical deep learning (ResNet) with variational quantum circuits (VQC) for the classification of microscopy blood-cell images.

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

This project investigates whether **variational quantum circuits (VQC)** can improve classification performance on imbalanced, high-resolution medical microscopy images compared to standard ResNet baselines.

### Key features

| Feature | Description |
|---|---|
| **Multiple classifiers** | ResNet-18/34/50 (pure) and Hybrid VQC (ResNet -> angle map -> VQC -> head) |
| **RAW image processing** | Static (NumPy), parametrised (differentiable PyTorch), and neural-network (U-Net++) pipelines |
| **Class-imbalance handling** | Weighted loss (Effective Number of Samples), weighted random sampling, soft undersampling with warm-up, conditional augmentation for rare classes |
| **Cross-validation** | Stratified K-Fold and Stratified Shuffle-Split |
| **Quantum simulation** | PennyLane VQC with data re-uploading, trainable encoding frequencies, a configurable read-out (`--vqc_readout`: single-qubit `<Z>`, individual `<ZZ>` ring correlators, pooled mean `<ZZ>` ring correlator, full Bloch vector, or `XX/YY/ZZ` correlators), a configurable entangling ring (`--vqc_entangler`: fixed CNOT or trainable CRZ/IsingZZ), an optional trainable measurement basis (`--vqc_final_rotation`), optional parallel sub-circuits (`--vqc_blocks`), and optional incremental data re-uploading loading fresh features at every layer (`--incremental_reuploading`); automatic backend selection (`lightning.gpu` -> `lightning.qubit` -> `default.qubit` backprop fallback) |
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
              |  |    Angle Map    |    |  (linear / MLP)
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |       VQC       |    |  (re-uploading VQC, PennyLane)
              |  +--------+--------+    |
              |           |             |
              |  +--------v--------+    |
              |  |   BN + Linear   |    |
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
|   +-- microscopy_rgb_qsvm_full.ipynb
|   +-- MLL23_dataset_analysis.ipynb
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
    |   +-- hybrid.py           # VariationalQuantumLayer, ParallelVQCLayer, SimpleHybridVQCClassifier
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
MLL23/
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
`MLL23_DATA_DIR` from `.env`.

## Usage

### Basic training (ResNet-18, static pipeline)

```bash
python scripts/train.py \
    --dataset Microscopy \
    --data_dir ./data \
    --classifier_network resnet18 \
    --classifier_pretrained \
    --processing_mode static \
    --epochs 30 \
    --batch_size 64
```

### Hybrid VQC

```bash
python scripts/train.py \
    --dataset Microscopy \
    --data_dir ./data \
    --classifier_network hybrid_vqc \
    --backbone resnet18 \
    --n_qubits 8 \
    --vqc_layers 1 \
    --vqc_readout bloch \
    --vqc_angle_map linear \
    --classifier_pretrained \
    --resnet_freeze_mode partial \
    --epochs 30
```

The angle map adapting ResNet features to the VQC input is `linear` by default
(a single `feat -> 3*n_qubits` map that leaves the VQC as the head's only
nonlinearity); switch to `--vqc_angle_map mlp` for a `feat -> hidden -> 3*n_qubits`
adapter sized by `--vqc_angle_hidden_dim` / `--vqc_angle_dropout`.

The circuit itself is configurable beyond depth/width:

* `--vqc_entangler {cnot,crz,ising_zz}` replaces the fixed CNOT ring with
  trainable `CRZ`/`IsingZZ` gates (one angle per layer and ring pair,
  near-identity init), letting the model learn how much entanglement to use.
  Both are diagonal in Z, so with a Z-only read-out pair them with
  `--vqc_final_rotation`, otherwise the last entangling ring is invisible to
  the measurement.
* `--vqc_final_rotation` appends a trainable `Rot(a,b,g)` per qubit after the
  last entangling ring — a trainable measurement basis at `+3*n_qubits`
  parameters and effectively zero extra depth.
* `--vqc_blocks K` splits the qubit budget into `K` independent parallel
  sub-circuits of `n_qubits/K` qubits with concatenated read-outs (block-major).
  For compatible block sizes this can keep the read-out width and quantum
  parameter count close to the single-circuit setup, while simulation cost drops
  from `2^n_qubits` to `K * 2^(n_qubits/K)` — a "wide, not deep" scaling path.
  `K` must divide `--n_qubits` and leave at least two qubits per block, so ring
  correlators and trainable entangler parameters remain well-defined.
* `--incremental_reuploading` switches from standard data re-uploading (every
  layer re-encodes the *same* `3*n_qubits` features) to incremental
  re-uploading: the angle map emits `vqc_layers * 3*n_qubits` angles and each
  layer encodes its own fresh feature slice, so depth widens the
  classical-to-quantum bottleneck instead of re-encoding old information. The
  circuit gains no parameters and no depth; only the angle map output grows by
  a factor of `--vqc_layers` (no effect at `--vqc_layers 1`). Composes with
  `--vqc_blocks` (block-major, then layer-major slices).

### RGB dataset with strong augmentation

```bash
python scripts/train.py \
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
python scripts/train.py \
    --dataset MLL23 \
    --classifier_network resnet18 \
    --classifier_pretrained \
    --resnet_freeze_mode partial \
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

Run `python scripts/train.py --help` for the full list. Key groups:

| Group | Key flags |
|---|---|
| **Dataset** | `--dataset`, `--data_dir`, `--min_samples_per_class`, `--exclude_classes`, `--below_threshold_strategy`, `--missing_image_policy` |
| **Cross-validation** | `--n_splits`, `--cv_strategy`, `--seed`, `--train_size` |
| **Imbalance** | `--imbalance_handling`, `--sampler_mode`, `--undersample_strength` |
| **Processing** | `--processing_mode`, `--sp_debayer`, `--sp_sharpening`, `--sp_denoising` |
| **Augmentation** | `--augmentation`, `--conditional_augmentation`, `--rare_class_threshold`, `--non_rare_aug_prob` |
| **Training** | `--batch_size`, `--lr`, `--lr_head`, `--epochs`, `--loss_fn`, `--focal_gamma` |
| **Classifier** | `--classifier_network`, `--classifier_pretrained`, `--resnet_freeze_mode` |
| **Hybrid VQC** | `--n_qubits`, `--vqc_layers`, `--vqc_readout`, `--vqc_entangler`, `--vqc_final_rotation`, `--vqc_blocks`, `--incremental_reuploading`, `--vqc_angle_map`, `--vqc_angle_hidden_dim`, `--vqc_angle_dropout`, `--backbone` |
| **W&B** | `--wandb_project`, `--wandb_entity`, `--run_name`, `--wandb_watch` |

---

## Results & Logging

- **Weights & Biases**: Each fold is a separate W&B run. Per epoch it logs train/val loss, accuracy, macro/weighted F1, balanced accuracy, and macro precision/recall. At the best macro-F1 epoch it stores best validation metrics in the run summary. After all folds complete, it creates a grouped `*_cv_summary` run; for disjoint `kfold` validation it reports pooled out-of-fold metrics plus a pooled confusion matrix and per-class table, while other CV strategies are summarised with mean/std across folds.
- **Checkpoints**: When `--save_model` is enabled, the best model (by val macro-F1) is saved to `results/<dataset>/<processing_mode>/<run_name>/fold_<k>/best_model.pth`.
- **Config snapshot**: A `run_config.txt` file recording the base run name, fold, and full resolved hyperparameters is written next to each checkpoint.
- **Fine-tuning semantics**: `--resnet_freeze_mode` controls pretrained ResNet freezing. Use `none` to train the full ResNet, `partial` to freeze stem + `layer1-3` while keeping `layer4` trainable, or `full` to freeze the entire ResNet backbone. For plain ResNets, `partial` and `full` both keep the final `fc` head trainable. For hybrid models, `full` freezes the ResNet extractor and trains only the downstream hybrid head.

---

## License

This project is released under the terms specified in [LICENSE](LICENSE).
