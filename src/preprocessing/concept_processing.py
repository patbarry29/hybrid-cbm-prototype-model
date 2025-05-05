import numpy as np

from config import N_CLASSES

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
    return [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

    # concept_counts = np.sum(class_level_concepts, axis=0)

    # # Find indices where the count meets the threshold
    # selected_concept_indices = np.where(concept_counts >= min_class_count)[0]

    # return selected_concept_indices


def apply_class_concepts_to_instances(train_img_labels, train_concept_labels, class_level_concepts, test_img_labels, test_concept_labels):
    for y in range(N_CLASSES):
        choice = train_img_labels[:, y] == 1
        train_concept_labels[choice,:] = class_level_concepts[y,:]
        choice = test_img_labels[:, y] == 1
        test_concept_labels[choice,:] = class_level_concepts[y,:]

    return train_concept_labels, test_concept_labels
