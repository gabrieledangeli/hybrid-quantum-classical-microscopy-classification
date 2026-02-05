import os
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset
from PIL import Image

# Dictionary mapping abbreviations to full names
class_abbreviations = {
    'BAS': 'Basophil',
    'EOS': 'Eosinophil',
    'KSC': 'Smudge cell',
    'LYA': 'Lymphocyte (atypical)',
    'LYT': 'Lymphocyte (typical)',
    'MON': 'Monocyte',
    'NGB': 'Neutrophil (band)',
    'NGS': 'Neutrophil (segmented)',
    'UNC': 'Image that could not be assigned a class during',
}

# Create mappings (Class -> Number)
class_names = sorted(list(class_abbreviations.keys()))
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

def load_annotations(annotation_file):
    """Loads annotations from a file into a dictionary."""
    annotations = {}
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Map filename to label
                # The file lists names like 'Ma190c_lame1_zone10_composite_Mcropped_1'
                # The actual image files have .tiff extension.
                img_name = parts[0]
                label = parts[1]
                annotations[img_name] = label
    return annotations

class MedicalDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, preload=True):
        self.root_dir = root_dir
        self.transform = transform
        self.preload = preload
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        self.annotations = load_annotations(annotation_file)
        
        # Filter valid images only (those that have a label and label is known)
        self.valid_images = []
        for f in self.image_files:
            name_no_ext = os.path.splitext(f)[0]
            if name_no_ext in self.annotations:
                label_str = self.annotations[name_no_ext]
                if label_str in class_to_idx:
                    self.valid_images.append((f, label_str))
        
        self.imgs_cache = []
        if self.preload:
            print(f"Preloading {len(self.valid_images)} images into RAM...")
            for idx in range(len(self.valid_images)):
                self.imgs_cache.append(self._load_image_from_disk(idx))
        
        print(f"Index built: {len(self.valid_images)} valid images found out of {len(self.image_files)} files.")

    def _load_image_from_disk(self, idx):
        img_name, _ = self.valid_images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            print("Failed to open image with PIL, trying tifffile...")
            try:
                img_arr = tifffile.imread(img_path)
                if img_arr.dtype not in [np.uint8, np.bool_]:
                    img_arr = img_arr.astype(np.float32)
                    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8) * 255.0
                    img_arr = img_arr.astype(np.uint8)
                
                image = Image.fromarray(img_arr).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                # Return black image in case of complete failure
                image = Image.new('RGB', (256, 256))
        return image

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        if self.preload:
            image = self.imgs_cache[idx]
        else:
            image = self._load_image_from_disk(idx)

        _, label_str = self.valid_images[idx]
        label = class_to_idx[label_str]
        
        if self.transform:
            image = self.transform(image)
        return image, label
