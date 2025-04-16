import os
import time
import numpy as np

from config import PROJECT_ROOT
from src.preprocessing import encode_image_concepts
from src.preprocessing import one_hot_encode_labels

# --- ENCODE CONCEPTS ---
print('Encoding image concepts...')

# Define full paths
concept_labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_concept_labels.txt')

start_time = time.time()
concept_matrix = encode_image_concepts(concept_labels_file, verbose=True)
end_time = time.time()
print('exec time:', end_time-start_time)

if concept_matrix is not None:
    # Print for the first 3 images
    for i in range(min(3, len(concept_matrix))):
        print(f"Image ID {i+1} (Row {i}):")
        # Show which concepts are present (1) for this image
        present_concepts = np.where(concept_matrix[i] == 1)[0] + 1
        print(f"  Present concepts: {present_concepts}")

    print(f"Image ID {500} (Row {499}):")
    present_concepts = np.where(concept_matrix[499] == 1)[0] + 1
    print(f"  Present concepts: {present_concepts}")

#     # Verify shape
#     print(f"\nTotal shape of concept matrix: {concept_matrix.shape}")
#     # Get statistics on concepts per image
#     concepts_per_image = np.sum(concept_matrix, axis=1)
#     print(f"Average concepts per image: {np.mean(concepts_per_image):.2f}")
#     print(f"Min/Max concepts per image: {np.min(concepts_per_image)}/{np.max(concepts_per_image)}")


# --- ENCODE LABELS ---
print('\nEncoding image labels...')

# Define full paths
labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_class_labels.txt')
classes_file = os.path.join(PROJECT_ROOT, 'data', 'classes.txt')

start_time = time.time()
one_hot_labels = one_hot_encode_labels(labels_file, classes_file, verbose=True)
end_time = time.time()
print('exec time:', end_time-start_time)