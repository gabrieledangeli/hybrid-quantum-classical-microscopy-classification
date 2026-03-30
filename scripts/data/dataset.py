"""Microscopy dataset classes for RAW and RGB images."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .utils import list_images_in_dir, load_image

# ── Constants ─────────────────────────────────────────────────────────────

VALID_CLASSES = [
    "BAS", "EBO", "EOS", "KSC", "LYA", "LYT", "MMZ", "MOB",
    "MON", "MYB", "MYO", "NGB", "NGS", "PMB", "PMO", "UNC",
]

# ── Factory ───────────────────────────────────────────────────────────────


def get_dataset(
    name: str,
    root_dir: str,
    intensity_ratio: float = 1.0,
    min_samples_per_class: int = 5,
    below_threshold_strategy: str = "drop",
    rare_class_name: str = "RARE",
) -> Dataset:
    """Instantiate the requested dataset variant."""
    constructors = {
        "Microscopy": MicroscopyDataset,
        "MicroscopyRGB": MicroscopyDatasetRGB,
    }
    if name not in constructors:
        raise ValueError(f"Unknown dataset: '{name}'. Choose from {list(constructors)}.")

    return constructors[name](
        root_dir=root_dir,
        intensity_ratio=intensity_ratio,
        min_samples_per_class=min_samples_per_class,
        below_threshold_strategy=below_threshold_strategy,
        rare_class_name=rare_class_name,
    )


# ── Base dataset ──────────────────────────────────────────────────────────


class ImageFolderDataset(Dataset):
    """Generic image-folder dataset with lazy in-memory caching.

    Args:
        img_dir: Path to the image directory.
        labels: Per-image label list.
        transform: Optional transform applied to each image.
        bits: Bit depth used for normalisation (divides by ``2**bits - 1``).
    """

    task = "classification"

    def __init__(
        self,
        img_dir: str,
        labels: List[Any],
        transform: Optional[Callable] = None,
        bits: int = 8,
    ) -> None:
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.bits = bits
        self._cache: dict[int, Any] = {}

        self.images = list_images_in_dir(img_dir)
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Image count ({len(self.images)}) ≠ label count ({len(self.labels)})."
            )

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}[{len(self.images)} images]"]
        for i, (img, lbl) in enumerate(zip(self.images, self.labels)):
            if i >= 10:
                lines.append("...")
                break
            lines.append(f"  {img}  →  {lbl}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        label = self.labels[idx]

        if idx in self._cache:
            img = self._cache[idx]
        else:
            img = load_image(self.images[idx])
            img = img / (2 ** self.bits - 1)
            self._cache[idx] = img

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ── Subset ────────────────────────────────────────────────────────────────


class Subset(Dataset):
    """Index-based subset wrapper with an optional extra transform."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))  # type: ignore[arg-type]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return f"Subset[{len(self)}] of {self.dataset}"

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


# ── Microscopy RAW dataset ───────────────────────────────────────────────


class MicroscopyDataset(ImageFolderDataset):
    """Microscopy dataset for **RAW** Bayer-pattern images.

    Camera parameters and normalisation statistics are dataset-level constants
    used by the image-processing pipeline and model normalisation layer.
    """

    # Camera intrinsics
    black_level = [9.834368023181512e-06] * 4
    white_balance = [-0.6567, 1.9673, 3.5304]
    colour_matrix = [
        -2.0338,  0.0933,  0.4157,
        -0.0286,  2.6464, -0.0574,
        -0.5516, -0.0947,  2.9308,
    ]
    camera_parameters = (black_level, white_balance, colour_matrix)

    dataset_mean = [0.91, 0.83, 0.94]
    dataset_std = [0.08, 0.13, 0.05]

    # Files known to be corrupted / mislabelled
    _INVALID_FILES = frozenset({"Ma190c_lame3_zone13_composite_Mcropped_2.tiff"})

    def __init__(
        self,
        root_dir: str,
        intensity_ratio: float = 1.0,
        transform: Optional[Callable] = None,
        bits: int = 16,
        min_samples_per_class: int = 5,
        below_threshold_strategy: str = "drop",
        rare_class_name: str = "RARE",
    ) -> None:
        if intensity_ratio != 1.0:
            raise ValueError("Only intensity_ratio=1.0 is supported for RAW images.")
        if min_samples_per_class < 1:
            raise ValueError(f"min_samples_per_class must be ≥ 1, got {min_samples_per_class}")
        if below_threshold_strategy not in ("drop", "merge_to_rare"):
            raise ValueError(f"Unknown strategy: '{below_threshold_strategy}'.")

        self.root_dir = root_dir
        self._cache: dict[int, Any] = {}
        self.img_dir = os.path.join(
            root_dir, "images", f"raw_scale{int(intensity_ratio * 100):03d}"
        )
        self.min_samples_per_class = min_samples_per_class
        self.below_threshold_strategy = below_threshold_strategy
        self.rare_class_name = rare_class_name

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        label_file = os.path.join(root_dir, "labels", "Ma190c_annotations.dat")
        if not os.path.isfile(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        # ── Parse manifest & apply class-filtering strategy ──────────────
        image_names, class_labels_raw = self._read_label_file(label_file)

        label_counts = Counter(class_labels_raw)
        self.valid_classes_list = [c for c in VALID_CLASSES if label_counts.get(c, 0) > 0]

        class_counts_raw = {c: label_counts.get(c, 0) for c in self.valid_classes_list}
        eligible = [c for c in self.valid_classes_list if class_counts_raw[c] >= min_samples_per_class]
        low_count = [c for c in self.valid_classes_list if class_counts_raw[c] < min_samples_per_class]

        if below_threshold_strategy == "merge_to_rare":
            class_mapping = {c: c for c in eligible}
            class_mapping.update({c: rare_class_name for c in low_count})
            selected_classes = set(eligible + low_count)
        else:  # "drop"
            class_mapping = {c: c for c in eligible}
            selected_classes = set(eligible)

        # Metadata for logging / debugging
        self.class_counts_raw = class_counts_raw
        self.dropped_classes = sorted(c for c in low_count if c not in selected_classes)
        self.merged_classes = sorted(low_count) if below_threshold_strategy == "merge_to_rare" else []
        self.original_to_mapped_class = dict(class_mapping)

        # Filter
        filtered = [
            (img, class_mapping[cls])
            for img, cls in zip(image_names, class_labels_raw)
            if cls in selected_classes and img not in self._INVALID_FILES
        ]
        if not filtered:
            raise RuntimeError("No valid images remaining after filtering.")

        final_images, final_labels = zip(*filtered)

        # Build class-index mapping (RARE always last if present)
        self.classes = sorted(set(final_labels))
        if rare_class_name in self.classes:
            self.classes = [c for c in self.classes if c != rare_class_name] + [rare_class_name]
        if len(self.classes) < 2:
            raise RuntimeError(
                f"Need ≥ 2 classes after filtering, got {self.classes}. "
                "Try lowering --min_samples_per_class or using merge_to_rare."
            )

        self.images = [os.path.join(self.img_dir, name) for name in final_images]
        self.labels = [self.classes.index(lbl) for lbl in final_labels]
        self.transform = transform
        self.bits = bits

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _read_label_file(path: str) -> Tuple[List[str], List[str]]:
        """Parse the annotation file → ``(image_filenames, class_labels)``."""
        images: list[str] = []
        labels: list[str] = []
        with open(path, "rb") as fh:
            for line in fh:
                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue
                stem, cls = decoded.split()
                images.append(f"{stem}.tiff")
                labels.append(cls)
        return images, labels


# ── Microscopy RGB dataset ───────────────────────────────────────────────


class MicroscopyDatasetRGB(MicroscopyDataset):
    """Microscopy dataset for pre-processed **RGB** images.

    Inherits RAW dataset logic but swaps image paths from ``raw`` → ``rgb``
    and uses RGB-specific normalisation statistics.
    """

    camera_parameters = None  # type: ignore[assignment]
    dataset_mean = [0.5566, 0.4607, 0.5923]
    dataset_std = [0.0983, 0.1417, 0.0667]

    def __init__(
        self,
        root_dir: str,
        intensity_ratio: float = 1.0,
        transform: Optional[Callable] = None,
        bits: int = 16,
        min_samples_per_class: int = 5,
        below_threshold_strategy: str = "drop",
        rare_class_name: str = "RARE",
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            intensity_ratio=intensity_ratio,
            transform=transform,
            bits=bits,
            min_samples_per_class=min_samples_per_class,
            below_threshold_strategy=below_threshold_strategy,
            rare_class_name=rare_class_name,
        )
        # Point to the RGB image directory instead of RAW
        self.images = [p.replace("raw", "rgb") for p in self.images]
