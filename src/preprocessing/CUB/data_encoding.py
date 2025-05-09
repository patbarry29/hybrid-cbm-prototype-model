import numpy as np
import pandas as pd

from src.utils.helpers import load_concept_names, vprint

def one_hot_encode_labels(image_class_labels_path, classes_path, verbose=False):
    try:
        # 1. determine the number of classes
        classes_df = pd.read_csv(classes_path, sep=' ', header=None, names=['class_id', 'class_name'])
        num_classes = len(classes_df)
        vprint(f"Found {num_classes} classes.", verbose)

        # 2. get image labels
        labels_df = pd.read_csv(image_class_labels_path, sep=' ', header=None, names=['image_id', 'class_id'])
        num_images = len(labels_df)
        vprint(f"Found labels for {num_images} images.", verbose)

        # 3. initialise label matrix with zeros
        one_hot_matrix = np.zeros((num_images, num_classes), dtype=int)

        # 4. populate matrix
        class_ids = labels_df['class_id'].values - 1
        one_hot_matrix[np.arange(len(labels_df)), class_ids] = 1

        vprint(f"Generated one-hot matrix with shape: {one_hot_matrix.shape}", verbose)
        return one_hot_matrix

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please check paths.")
        return None


def _parse_file(concept_labels_file):
    data = []
    with open(concept_labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # We only need the first 3 columns (image_id, concept_id, is_present)
            image_id = int(parts[0])
            concept_id = int(parts[1])
            is_present = int(parts[2])
            uncertainty = int(parts[3])
            data.append([image_id, concept_id, is_present, uncertainty])

    return data


def encode_image_concepts(concept_labels_file, verbose=False):
    try:
        # 1. get image_id, concept_id and is_present values from file
        data = _parse_file(concept_labels_file)

        # 2. Create a DataFrame from the parsed data
        concept_df = pd.DataFrame(data, columns=['image_id', 'concept_id', 'is_present', 'uncertainty'])

        # 3. get the number of unique images and concepts
        unique_images = concept_df['image_id'].unique()
        num_images = len(unique_images)

        # -- find the max concept_id to determine matrix dimensions
        max_concept_id = concept_df['concept_id'].max()

        vprint(f"Found {num_images} unique images.", verbose)
        vprint(f"Found {max_concept_id} unique concepts.", verbose)

        # 4. create concepts matrix initialized with zeros
        concept_matrix = np.zeros((num_images, max_concept_id), dtype=int)
        uncertainty_matrix = np.zeros((num_images, max_concept_id), dtype=int)

        # 5. populate matrix (vectorised)
        img_ids = concept_df['image_id'].values - 1
        concept_ids = concept_df['concept_id'].values - 1
        concept_matrix[img_ids, concept_ids] = concept_df['is_present'].values
        uncertainty_matrix[img_ids, concept_ids] = concept_df['uncertainty'].values

        vprint(f"Generated concept matrix with shape: {concept_matrix.shape}", verbose)

        return concept_matrix, uncertainty_matrix

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please check paths.")
        return None

def get_concepts(concept_vector, concepts_path):
    concept_names = load_concept_names(concepts_path)

    true_concept_indices = np.where(concept_vector == 1)[0]

    true_concept_ids = true_concept_indices + 1

    active_concepts = [concept_names[concept_id] for concept_id in true_concept_ids if concept_id in concept_names]

    return active_concepts