import numpy as np
import torch


def vprint(message, is_verbose):
    if is_verbose:
        print(message)

def get_filename_to_id_mapping(filepath, reverse=False):
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            image_id, filename = line.strip().split()
            if not reverse:
                mapping[filename] = int(image_id)-1
            else:
                mapping[int(image_id)-1] = filename

    return mapping

def load_concept_names(concepts_path):
    concept_names = {}
    with open(concepts_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                concept_id = int(parts[0])
                concept_name = parts[1]
                concept_names[concept_id] = concept_name

    return concept_names

def find_class_imbalance(concept_labels):
    _, num_concepts = concept_labels.shape
    concept_ratios = []
    for i in range(num_concepts):
        attribute_column = concept_labels[:, i]

        # Count occurrences of 0 and 1
        counts = np.bincount(attribute_column.astype(int), minlength=2)
        num_neg = counts[0]
        num_pos = counts[1]

        # Calculate ratio (handle division by zero)
        ratio_neg_pos = num_neg / num_pos if num_pos > 0 else float('inf') # Negatives per Positive

        concept_ratios.append(ratio_neg_pos)

    # concept_ratios_tensor = torch.tensor(concept_ratios, device=device, dtype=torch.float)
    return concept_ratios