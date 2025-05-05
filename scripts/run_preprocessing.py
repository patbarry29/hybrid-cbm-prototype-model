import numpy as np
import os
import time

import torch

from config import PROJECT_ROOT
from src.dataset import ImageConceptDataset
from src.preprocessing import *

from torch.utils.data import DataLoader


def preprocessing_main(class_concepts=False, verbose=False):
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

    # concept processing
    class_level_concepts = compute_class_level_concepts(train_concept_labels, train_uncertainty, train_img_labels)

    # apply class-level concepts to each instance
    if class_concepts:
        train_concept_labels, test_concept_labels = apply_class_concepts_to_instances(train_img_labels, train_concept_labels, class_level_concepts, test_img_labels, test_concept_labels)

    common_concept_indices = select_common_concepts(class_level_concepts, min_class_count=10)
    train_concept_labels = train_concept_labels[:, common_concept_indices]
    test_concept_labels = test_concept_labels[:, common_concept_indices]

    # CREATE TRAIN AND TEST DATASET
    train_dataset = ImageConceptDataset(
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return concept_labels, train_loader, test_loader

if __name__ == '__main__':
    start_time = time.time()
    preprocessing_main()
    end_time = time.time()
    print('exec time:', end_time-start_time)
