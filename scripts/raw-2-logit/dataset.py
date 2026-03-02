import os
from typing import List, Tuple, Optional, Callable, Union, Any
from collections import Counter

from torch.utils.data import Dataset
import torch

# Local imports
from dataset_utils import list_images_in_dir, load_image

# Constants
IMAGE_FILE_TYPES = ['dng', 'png', 'tif', 'tiff']
VALID_CLASSES = [
    'BAS', 'EBO', 'EOS', 'KSC', 'LYA', 'LYT', 'MMZ', 'MOB',
    'MON', 'MYB', 'MYO', 'NGB', 'NGS', 'PMB', 'PMO', 'UNC'
]

def get_dataset(name: str,
                root_dir: str,
                intensity_ratio: float = 1.0,
                min_samples_per_class: int = 5,
                below_threshold_strategy: str = 'drop',
                rare_class_name: str = 'RARE') -> Dataset:
    """Factory function to get the dataset instance."""
    if name == 'Microscopy':
        return MicroscopyDataset(
            root_dir=root_dir,
            intensity_ratio=intensity_ratio,
            min_samples_per_class=min_samples_per_class,
            below_threshold_strategy=below_threshold_strategy,
            rare_class_name=rare_class_name,
        )
    elif name == 'MicroscopyRGB':
        return MicroscopyDatasetRGB(
            root_dir=root_dir,
            intensity_ratio=intensity_ratio,
            min_samples_per_class=min_samples_per_class,
            below_threshold_strategy=below_threshold_strategy,
            rare_class_name=rare_class_name,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")


class ImageFolderDataset(Dataset):
    """
    Creates a dataset of images in img_dir.

    Args:
        img_dir (str): Path to image folder.
        labels (List[Any]): List of labels for each image in img_dir.
        transform (Callable, optional): Transformation to apply to image.
        bits (int, optional): Bit depth for normalization (divides by 2^bits - 1).
    """

    task = 'classification'

    def __init__(self, 
                 img_dir: str, 
                 labels: List[Any], 
                 transform: Optional[Callable] = None, 
                 bits: int = 8):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.bits = bits
        self._cache: dict = {}  # lazy in-memory cache for loaded images

        self.images = list_images_in_dir(img_dir)

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Number of images ({len(self.images)}) does not match "
                f"number of labels ({len(self.labels)})."
            )

    def __repr__(self) -> str:
        rep = f"{type(self).__name__}: ImageFolderDataset[{len(self.images)}]"
        for n, (img, label) in enumerate(zip(self.images, self.labels)):
            if n > 10:
                rep += '\n...'
                break
            rep += f'\nimage: {img}\tlabel: {label}'
        return rep

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        label = self.labels[idx]

        # Use in-memory cache to avoid repeated TIFF disk reads
        if idx in self._cache:
            img = self._cache[idx]
        else:
            img = load_image(self.images[idx])
            img = img / (2**self.bits - 1)
            self._cache[idx] = img
        
        if self.transform is not None:
            img = self.transform(img)

        # Basic shape check
        if len(img.shape) == 2:
            expected_shape = (256, 256)
            if img.shape != expected_shape:
                 raise ValueError(f"Invalid size for {self.images[idx]}: {img.shape}, expected {expected_shape}")
        else:
            expected_shape = (3, 256, 256)
            if img.shape != expected_shape:
                 raise ValueError(f"Invalid size for {self.images[idx]}: {img.shape}, expected {expected_shape}")

        return img, label


class Subset(Dataset):
    """
    Define a subset of a dataset by only selecting given indices.
    """

    def __init__(self, 
                 dataset: Dataset, 
                 indices: Optional[List[int]] = None, 
                 transform: Optional[Callable] = None):
        self.dataset = dataset
        self.indices = indices if indices is not None else range(len(dataset)) # type: ignore
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __repr__(self) -> str:
        return f"Subset [{len(self)}] of {self.dataset}"

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class MicroscopyDataset(ImageFolderDataset):
    """
    MicroscopyDataset for raw images.

    Args:
        root_dir (str): Root directory of the dataset.
        intensity_ratio (float): Original image rescaled by this factor.
        transform (Callable, optional): Transformation to apply to image.
        bits (int, optional): Bit depth for normalization.
    """

    # Camera parameters constants
    black_level = [9.834368023181512e-06] * 4
    white_balance = [-0.6567, 1.9673, 3.5304]
    colour_matrix = [-2.0338, 0.0933, 0.4157, 
                     -0.0286, 2.6464, -0.0574, 
                     -0.5516, -0.0947, 2.9308]

    camera_parameters = (black_level, white_balance, colour_matrix)

    dataset_mean = [0.91, 0.83, 0.94]
    dataset_std = [0.08, 0.13, 0.05]

    def __init__(self, 
                 root_dir: str, 
                 intensity_ratio: float = 1.0, 
                 transform: Optional[Callable] = None, 
                 bits: int = 16,
                 min_samples_per_class: int = 5,
                 below_threshold_strategy: str = 'drop',
                 rare_class_name: str = 'RARE'):

        if intensity_ratio != 1.0:
            raise ValueError(f"Invalid intensity_ratio: {intensity_ratio}, only 1.0 is currently supported for Raw.")
        if min_samples_per_class < 1:
            raise ValueError(f"min_samples_per_class must be >= 1, got {min_samples_per_class}")
        if below_threshold_strategy not in ('drop', 'merge_to_rare'):
            raise ValueError(
                f"Unknown below_threshold_strategy: {below_threshold_strategy}. "
                f"Expected 'drop' or 'merge_to_rare'."
            )

        self.root_dir = root_dir
        self._cache: dict = {}  # lazy in-memory cache for loaded images
        self.img_dir = os.path.join(root_dir, 'images', f'raw_scale{int(intensity_ratio * 100):03d}')
        self.min_samples_per_class = min_samples_per_class
        self.below_threshold_strategy = below_threshold_strategy
        self.rare_class_name = rare_class_name

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found at: {self.img_dir}")
        
        label_file = os.path.join(root_dir, 'labels', 'Ma190c_annotations.dat')
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found at: {label_file}")

        self.valid_classes_list = VALID_CLASSES
        self.invalid_files = {'Ma190c_lame3_zone13_composite_Mcropped_2.tiff'}

        # Read manifest
        images_filenames, class_labels_raw = self._read_label_file(label_file)

        # Keep only configured classes present in labels.
        label_counts = Counter(class_labels_raw)
        self.valid_classes_list = [
            cls for cls in self.valid_classes_list
            if label_counts.get(cls, 0) > 0
        ]
        class_counts_raw = {cls: int(label_counts.get(cls, 0)) for cls in self.valid_classes_list}
        eligible_classes = [cls for cls in self.valid_classes_list if class_counts_raw[cls] >= min_samples_per_class]
        low_count_classes = [cls for cls in self.valid_classes_list if class_counts_raw[cls] < min_samples_per_class]

        class_mapping = {}
        if below_threshold_strategy == 'merge_to_rare':
            for cls in eligible_classes:
                class_mapping[cls] = cls
            for cls in low_count_classes:
                class_mapping[cls] = rare_class_name
            selected_classes = set(eligible_classes + low_count_classes)
        else:
            for cls in eligible_classes:
                class_mapping[cls] = cls
            selected_classes = set(eligible_classes)

        # Metadata for reporting/debugging.
        self.class_counts_raw = class_counts_raw
        self.dropped_classes = sorted([cls for cls in low_count_classes if cls not in selected_classes])
        self.merged_classes = sorted(low_count_classes) if below_threshold_strategy == 'merge_to_rare' else []
        self.original_to_mapped_class = dict(class_mapping)

        # Filter invalid files and classes
        filtered_data = [
            (img, class_mapping[cls])
            for img, cls in zip(images_filenames, class_labels_raw)
            if cls in selected_classes and img not in self.invalid_files
        ]
        
        if not filtered_data:
            raise RuntimeError("No valid images found after filtering.")

        final_images, final_labels = zip(*filtered_data)

        # Create class index mapping
        self.classes = sorted(list(set(final_labels)))
        if rare_class_name in self.classes:
            self.classes = [cls for cls in self.classes if cls != rare_class_name] + [rare_class_name]
        if len(self.classes) < 2:
            raise RuntimeError(
                f"Need at least 2 classes for classification after filtering. Got classes={self.classes}. "
                f"Try lowering --min_samples_per_class or using --below_threshold_strategy merge_to_rare."
            )
        
        # Store full paths
        self.images = [os.path.join(self.img_dir, img) for img in final_images]
        
        # Convert string labels to indices
        self.labels = [self.classes.index(label) for label in final_labels]
        
        self.transform = transform
        self.bits = bits

    @staticmethod
    def _read_label_file(label_file_path: str) -> Tuple[List[str], List[str]]:
        """Parses the annotation label file."""
        images = []
        class_labels = []

        with open(label_file_path, "rb") as data:
            for line in data:
                line_decoded = line.decode("utf-8").strip()
                if not line_decoded:
                    continue
                file_name, class_label = line_decoded.split()
                image_name = f"{file_name}.tiff"
                images.append(image_name)
                class_labels.append(class_label)

        return images, class_labels


class MicroscopyDatasetRGB(MicroscopyDataset):
    """
    MicroscopyDataset for RGB images.
    """
    camera_parameters = None
    # Dataset-specific normalization statistics computed on the filtered RGB split.
    dataset_mean = [0.5566, 0.4607, 0.5923]
    dataset_std = [0.0983, 0.1417, 0.0667]

    def __init__(self, 
                 root_dir: str, 
                 intensity_ratio: float = 1.0, 
                 transform: Optional[Callable] = None, 
                 bits: int = 16,
                 min_samples_per_class: int = 5,
                 below_threshold_strategy: str = 'drop',
                 rare_class_name: str = 'RARE'):
        
        super().__init__(
            root_dir=root_dir, 
            intensity_ratio=intensity_ratio, 
            transform=transform, 
            bits=bits,
            min_samples_per_class=min_samples_per_class,
            below_threshold_strategy=below_threshold_strategy,
            rare_class_name=rare_class_name,
        )
        # Modify paths to point to RGB images instead of RAW
        self.images = [image.replace('raw', 'rgb') for image in self.images]

