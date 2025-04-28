
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset

from config import N_IMAGES
from src.utils.helpers import vprint


def _get_train_test_masks(split_file_path, verbose=False):
    split_flags = torch.full((N_IMAGES,), -1, dtype=torch.int8) # Initialize with -1

    with open(split_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            image_id = int(parts[0])
            flag = int(float(parts[1]))

            idx = image_id - 1
            split_flags[idx] = flag

    train_mask = (split_flags == 1)
    test_mask = (split_flags == 0)

    num_train = train_mask.sum().item()
    num_test = test_mask.sum().item()

    if num_train + num_test != N_IMAGES:
        print(f"Warning: Mismatch in split counts. Train ({num_train}) + Test ({num_test}) != Total ({N_IMAGES}). Check split file flags.")

    vprint(f"Split complete: {num_train} train images, {num_test} test images.", verbose)
    return train_mask, test_mask

def split_datasets(split_file, concept_labels, image_labels, uncertainty_matrix, image_tensors):
    # CREATE TRAIN TEST SPLIT USING TXT FILE
    train_mask, test_mask = _get_train_test_masks(split_file, verbose=True)

    train_concepts = concept_labels[train_mask]
    test_concepts = concept_labels[test_mask]

    train_img_labels = image_labels[train_mask]
    test_img_labels = image_labels[test_mask]

    train_uncertainty = uncertainty_matrix[train_mask]

    # image_tensors is a list because lists are much more efficient than tensors
    #   storing every image tensor as a stacked tensor would require ~12GB of RAM
    train_tensors = [tensor for tensor, keep in zip(image_tensors, train_mask) if keep.item()]
    test_tensors = [tensor for tensor, keep in zip(image_tensors, test_mask) if keep.item()]

    assert len(train_tensors) == train_concepts.shape[0] == train_img_labels.shape[0], "Train set size mismatch!"
    assert len(test_tensors) == test_concepts.shape[0] == test_img_labels.shape[0], "Test set size mismatch!"

    return {
        'train_concepts': train_concepts,
        'test_concepts': test_concepts,
        'train_img_labels': train_img_labels,
        'test_img_labels': test_img_labels,
        'train_uncertainty': train_uncertainty,
        'train_tensors': train_tensors,
        'test_tensors': test_tensors
    }

def train_val_split(full_train_dataset, val_size):
    all_indices = list(range(len(full_train_dataset)))
    all_train_labels = full_train_dataset.get_labels()

    train_indices, val_indices, _, _ = train_test_split(
        all_indices,
        all_train_labels,
        test_size=val_size,
        random_state=42, # for reproducibility
        stratify=all_train_labels
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    return train_dataset, val_dataset