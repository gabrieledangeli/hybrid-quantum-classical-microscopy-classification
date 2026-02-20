import os

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
    x = dataset.images
    y = dataset.labels
    
    # Ensure y is a numpy array for sklearn
    y = np.array(y)

    idxs = []

    if strategy == 'shuffle_split':
        cv = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=seed)
    elif strategy == 'kfold':
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    # Use a dummy X because we only need indices based on y
    # X needs to have same length as y
    dummy_x = np.zeros(len(y))

    for idxs_train, idxs_test in cv.split(dummy_x, y):
        idxs.append((idxs_train.tolist(), idxs_test.tolist()))

    return idxs