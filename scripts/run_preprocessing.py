import os
import time

from config import PROJECT_ROOT
from src.dataset import ImageConceptDataset
from src.preprocessing import *

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def preprocessing_main(verbose=False):
    # LOAD AND TRANSFORM IMAGES
    input_dir = os.path.join(PROJECT_ROOT, 'images')
    resol = 299
    training = True
    mapping_file = os.path.join(PROJECT_ROOT, 'data', 'images.txt')

    image_tensors, image_paths = load_and_transform_images(input_dir, mapping_file, resol, training, batch_size=32, verbose=verbose, dev=False)

    # CREATE CONCEPT LABELS MATRIX
    concept_labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_concept_labels.txt')

    concept_labels = encode_image_concepts(concept_labels_file, verbose=verbose)

    # CREATE IMAGE LABELS MATRIX
    labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_class_labels.txt')
    classes_file = os.path.join(PROJECT_ROOT, 'data', 'classes.txt')

    image_labels = one_hot_encode_labels(labels_file, classes_file, verbose=verbose)

    # CREATE TRAIN TEST SPLIT USING TXT FILE
    split_file = os.path.join(PROJECT_ROOT, 'data', 'train_test_split.txt')
    split_data = split_datasets(split_file, concept_labels, image_labels, image_tensors)

    train_concepts = split_data['train_concepts']
    test_concepts = split_data['test_concepts']

    train_img_labels = split_data['train_img_labels']
    test_img_labels = split_data['test_img_labels']

    train_tensors = split_data['train_tensors']
    test_tensors = split_data['test_tensors']

    # CREATE DATASET TRAIN AND TEST
    full_train_dataset = ImageConceptDataset(
        image_tensors=train_tensors,
        concept_labels=train_concepts,
        image_labels=train_img_labels
    )

    test_dataset = ImageConceptDataset(
        image_tensors=test_tensors,
        concept_labels=test_concepts,
        image_labels=test_img_labels
    )

    # CREATE VALIDATION SET FROM TRAIN

    train_dataset, val_dataset = train_val_split(full_train_dataset, val_size=0.2)

    # CREATE DATALOADERS FROM DATASETS
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return concept_labels, train_loader, val_loader, test_loader

if __name__ == '__main__':
    start_time = time.time()
    preprocessing_main()
    end_time = time.time()
    print('exec time:', end_time-start_time)