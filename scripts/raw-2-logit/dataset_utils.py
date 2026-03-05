import os
from collections import Counter

import numpy as np

import tifffile
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

IMAGE_FILE_TYPES = ['dng', 'png', 'tif', 'tiff']

def load_image(path):
    return np.array(tifffile.imread(path), dtype=np.float32)


def list_images_in_dir(path):
    image_list = [os.path.join(path, img_name)
                  for img_name in sorted(os.listdir(path))
                  if img_name.split('.')[-1].lower() in IMAGE_FILE_TYPES]
    return image_list

def k_fold(dataset, n_splits: int, seed: int, train_size: float = 0.8, strategy: str = 'kfold'):
    """Split dataset in subsets for cross-validation 

       Args:
            dataset (class): dataset to split
            n_splits (int): Number of re-shuffling & splitting iterations.
            seed (int): seed for k_fold splitting
            train_size (float): should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
            strategy (str): 'kfold' or 'shuffle_split'
       Returns:
           idxs (list): indeces for splitting the dataset. The list contain n_split pair of train/test indeces.
    """
    # Ensure y is a numpy array for sklearn
    y = np.array(dataset.labels)

    if strategy not in ('kfold', 'shuffle_split'):
        raise ValueError(
            f"Unknown CV strategy: '{strategy}'. Must be 'kfold' or 'shuffle_split'."
        )

    if strategy == 'kfold' and n_splits < 2:
        raise ValueError(
            f"StratifiedKFold requires n_splits >= 2, got {n_splits}. "
            f"Use --cv_strategy shuffle_split for a single train/test split."
        )

    class_counts = Counter(y.tolist())
    min_class_count = min(class_counts.values()) if class_counts else 0

    if strategy == 'kfold':
        if n_splits > min_class_count:
            raise ValueError(
                f"n_splits={n_splits} is too high for StratifiedKFold: minimum class count is {min_class_count}. "
                f"Use n_splits <= {min_class_count}, lower --min_samples_per_class, or merge/drop rare classes."
            )
    else:
        if not (0.0 < train_size < 1.0):
            raise ValueError(f"train_size must be in (0, 1), got {train_size}")
        for cls_idx, cls_count in class_counts.items():
            if cls_count < 2:
                raise ValueError(
                    f"Class {cls_idx} has only {cls_count} sample(s). StratifiedShuffleSplit requires at least 2."
                )
            cls_train = int(np.floor(cls_count * train_size))
            cls_val = cls_count - cls_train
            if cls_train < 1 or cls_val < 1:
                raise ValueError(
                    f"Class {cls_idx} cannot be split with train_size={train_size}: "
                    f"train={cls_train}, val={cls_val}, total={cls_count}. "
                    f"Adjust train_size or dataset class filtering."
                )

    # --- Build splits ---
    idxs = []

    if strategy == 'shuffle_split':
        cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)
    else:  # 'kfold'
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Use a dummy X because we only need indices based on y
    dummy_x = np.zeros(len(y))

    for idxs_train, idxs_test in cv.split(dummy_x, y):
        idxs.append((idxs_train.tolist(), idxs_test.tolist()))

    return idxs
