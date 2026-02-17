import os
from typing import List, Tuple, Optional, Callable, Union, Any

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

def get_dataset(name: str, root_dir: str, intensity_ratio: float = 1.0) -> Dataset:
    """Factory function to get the dataset instance."""
    if name == 'Microscopy':
        return MicroscopyDataset(root_dir=root_dir, intensity_ratio=intensity_ratio)
    elif name == 'MicroscopyRGB':
        return MicroscopyDatasetRGB(root_dir=root_dir, intensity_ratio=intensity_ratio)
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
                 bits: int = 1):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.bits = bits

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

        # Load image (returns numpy array)
        img = load_image(self.images[idx])
        
        # Normalize to [0, 1] range based on bit depth
        img = img / (2**self.bits - 1)
        
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

    dataset_mean = [0.91, 0.84, 0.94]
    dataset_std = [0.08, 0.12, 0.05]

    def __init__(self, 
                 root_dir: str, 
                 intensity_ratio: float = 1.0, 
                 transform: Optional[Callable] = None, 
                 bits: int = 16):

        if intensity_ratio != 1.0:
            raise ValueError(f"Invalid intensity_ratio: {intensity_ratio}, only 1.0 is currently supported for Raw.")

        self.root_dir = root_dir
        
        # Note: Directory structure logic is hardcoded here. 
        # Ideally, this should be configurable.
        img_folder_name = f'images/raw_scale{int(intensity_ratio * 100):03d}'
        self.img_dir = os.path.join(root_dir, img_folder_name)
        
        label_file = os.path.join(root_dir, 'labels/Ma190c_annotations.dat')
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found at: {label_file}")
            
        if not os.path.exists(self.img_dir):
            print(f"Warning: Image directory not found at: {self.img_dir}")

        self.valid_classes_list = VALID_CLASSES
        self.invalid_files = {'Ma190c_lame3_zone13_composite_Mcropped_2.tiff'}

        # Read manifest
        images_filenames, class_labels_raw = self._read_label_file(label_file)

        # Filter classes with low appearance (>4 samples)
        self.valid_classes_list = [
            cls for cls in self.valid_classes_list 
            if class_labels_raw.count(cls) > 4
        ]

        # Filter invalid files and classes
        filtered_data = [
            (img, cls) 
            for img, cls in zip(images_filenames, class_labels_raw)
            if cls in self.valid_classes_list and img not in self.invalid_files
        ]
        
        if not filtered_data:
            raise RuntimeError("No valid images found after filtering.")

        final_images, final_labels = zip(*filtered_data)
        
        # DEBUG: Print class distribution
        from collections import Counter
        class_counts = Counter(final_labels)
        print("\n[DEBUG] Dataset Class Distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count} samples")
        print(f"  Total: {len(final_labels)} samples\n")

        # Create class index mapping
        self.classes = sorted(list(set(final_labels)))
        
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
    dataset_mean = None
    dataset_std = None

    def __init__(self, 
                 root_dir: str, 
                 intensity_ratio: float = 1.0, 
                 transform: Optional[Callable] = None, 
                 bits: int = 16, 
                 force_download: bool = False):
        
        super().__init__(
            root_dir=root_dir, 
            intensity_ratio=intensity_ratio, 
            transform=transform, 
            bits=bits
        )
        # Modify paths to point to RGB images instead of RAW
        # Assuming filename structure allows simple replacement
        self.images = [image.replace('raw', 'rgb') for image in self.images]

