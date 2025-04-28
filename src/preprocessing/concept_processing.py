import numpy as np

def compute_class_level_concepts(concept_labels_matrix, uncertainty_matrix, image_labels_matrix):
    _, num_concepts = concept_labels_matrix.shape
    num_classes = image_labels_matrix.shape[1]

    # Get class index for each instance
    instance_class_indices = np.argmax(image_labels_matrix, axis=1)

    # Initialize results matrix
    class_level_concepts = np.zeros((num_classes, num_concepts), dtype=int)

    # Process each class
    for class_id in range(num_classes):
        # Get instances belonging to this class
        class_instances = np.where(instance_class_indices == class_id)[0]

        if len(class_instances) == 0:
            continue

        # Get concept labels and uncertainties for this class
        class_concepts = concept_labels_matrix[class_instances]
        class_uncertainty = uncertainty_matrix[class_instances]

        # Create visibility mask (concept is visible if NOT (concept=0 AND uncertainty=1))
        visible_mask = ~((class_concepts == 0) & (class_uncertainty == 1))

        # Count visible votes and visible "1" votes per concept
        visible_votes = np.sum(visible_mask, axis=0)
        visible_ones = np.sum(class_concepts * visible_mask, axis=0)

        # Default to 1 for concepts with no visible votes
        majority_vote = np.ones(num_concepts, dtype=int)

        # Apply majority voting rule where there are visible votes
        has_votes = visible_votes > 0
        majority_vote[has_votes] = (visible_ones[has_votes] >= visible_votes[has_votes]/2.0).astype(int)

        class_level_concepts[class_id] = majority_vote

    return class_level_concepts

def select_common_concepts(class_level_concepts, min_class_count):
    concept_counts = np.sum(class_level_concepts, axis=0)

    # Find indices where the count meets the threshold
    selected_concept_indices = np.where(concept_counts >= min_class_count)[0]

    return selected_concept_indices