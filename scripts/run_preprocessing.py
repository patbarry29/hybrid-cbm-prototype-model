import numpy as np
import os
import time

from config import PROJECT_ROOT
from src.dataset import ImageConceptDataset
from src.preprocessing import *

from torch.utils.data import DataLoader


def preprocessing_main(trim_concepts=False, verbose=False):
    # LOAD AND TRANSFORM IMAGES
    input_dir = os.path.join(PROJECT_ROOT, 'images')
    resol = 299
    training = True
    mapping_file = os.path.join(PROJECT_ROOT, 'data', 'images.txt')

    image_tensors, _ = load_and_transform_images(input_dir, mapping_file, resol, training, batch_size=32, verbose=verbose)

    # CREATE CONCEPT LABELS MATRIX
    concept_labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_concept_labels.txt')

    concept_labels, uncertainty_matrix = encode_image_concepts(concept_labels_file, verbose=verbose)

    # CREATE IMAGE LABELS MATRIX
    labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_class_labels.txt')
    classes_file = os.path.join(PROJECT_ROOT, 'data', 'classes.txt')

    image_labels = one_hot_encode_labels(labels_file, classes_file, verbose=verbose)

    # CREATE TRAIN TEST SPLIT USING TXT FILE
    split_file = os.path.join(PROJECT_ROOT, 'data', 'train_test_split.txt')
    split_data = split_datasets(split_file, concept_labels, image_labels, uncertainty_matrix, image_tensors)

    train_concept_labels = split_data['train_concepts']
    test_concept_labels = split_data['test_concepts']

    train_img_labels = split_data['train_img_labels']
    test_img_labels = split_data['test_img_labels']

    train_uncertainty = split_data['train_uncertainty']

    train_tensors = split_data['train_tensors']
    test_tensors = split_data['test_tensors']
    class_level_concepts = compute_class_level_concepts(train_concept_labels, train_uncertainty, train_img_labels)
    if trim_concepts:
        # common_concept_indices = select_common_concepts(class_level_concepts, min_class_count=10)

        common_concept_indices = np.array([1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
        93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
        183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
        254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311])

        train_concept_labels = train_concept_labels[:, common_concept_indices]
        test_concept_labels = test_concept_labels[:, common_concept_indices]

    # CREATE DATASET TRAIN AND TEST
    full_train_dataset = ImageConceptDataset(
        image_tensors=train_tensors,
        concept_labels=train_concept_labels,
        image_labels=train_img_labels
    )

    test_dataset = ImageConceptDataset(
        image_tensors=test_tensors,
        concept_labels=test_concept_labels,
        image_labels=test_img_labels
    )

    # CREATE DATALOADERS FROM DATASETS
    batch_size = 64
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return concept_labels, train_loader, test_loader

if __name__ == '__main__':
    start_time = time.time()
    preprocessing_main()
    end_time = time.time()
    print('exec time:', end_time-start_time)