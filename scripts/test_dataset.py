import os
import time

from config import PROJECT_ROOT
from src.preprocessing import *
from src.utils import get_filename_to_id_mapping

from src.dataset import ImageConceptDataset

if __name__ == '__main__':

    input_dir = os.path.join(PROJECT_ROOT, 'images')
    resol = 299
    training = True

    image_tensors, image_paths = load_and_transform_images(input_dir, resol, training, batch_size=32, verbose=True, dev=False)

    # CREATE CONCEPT LABELS MATRIX
    concept_labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_concept_labels.txt')

    concept_labels = encode_image_concepts(concept_labels_file, verbose=True)

    # CREATE IMAGE LABELS MATRIX
    labels_file = os.path.join(PROJECT_ROOT, 'data', 'image_class_labels.txt')
    classes_file = os.path.join(PROJECT_ROOT, 'data', 'classes.txt')

    image_labels = one_hot_encode_labels(labels_file, classes_file, verbose=True)

    # GET IMAGE ID TO IMAGE FILENAME MAPPING
    images_file = os.path.join(PROJECT_ROOT, 'data', 'images.txt')
    image_id_mapping = get_filename_to_id_mapping(images_file)

    dataset = ImageConceptDataset(
        image_tensors=image_tensors,
        concept_labels=concept_labels,
        image_labels=image_labels,
        image_id_mapping=image_id_mapping
    )
    print(f"Dataset length: {len(dataset)}")

    # --- Test __getitem__ ---
    print("\nTesting __getitem__:")
    item_index = 5
    if item_index < len(dataset):
        img_tensor, concepts, img_label, img_id = dataset[item_index]
        print(f"Item at index {item_index}:")
        print(f"  Image Tensor Shape: {img_tensor.shape}")
        print(f"  Concept Labels Shape: {concepts.shape}")
        print(f"  Image Label Shape: {img_label.shape}")
        print(f"  Image ID: {img_id}")
        print(f"  Filename (lookup): {dataset.image_id_mapping.get(img_id)}")
        print(f"  Concept vector (first 10): {concepts[:10].numpy()}")
        print(f"  Image Label vector: {img_label.numpy()}")
    else:
        print(f"Index {item_index} is out of bounds.")

    # --- Test get_item_by_filename ---
    print("\nTesting get_item_by_filename:")
    # Get a valid filename from the mapping
    test_filename = dataset.image_id_mapping.get(item_index + 1) # Get filename for ID=6
    if test_filename:
        print(f"Looking up filename: '{test_filename}'")
        result = dataset.get_item_by_filename(test_filename)
        if result:
            img_tensor_fn, concepts_fn, img_label_fn, img_id_fn = result
            print(f"  Found item:")
            print(f"    Image Tensor Shape: {img_tensor_fn.shape}")
            print(f"    Concept Labels Shape: {concepts_fn.shape}")
            print(f"    Image Label Shape: {img_label_fn.shape}")
            print(f"    Image ID: {img_id_fn}")
            print(f"    Concept vector (first 10): {concepts_fn[:10].numpy()}")
            print(f"    Image Label vector: {img_label_fn.numpy()}")
        else:
            print("  Item not found by filename.")
    else:
        print("Could not get a valid test filename.")

#     # --- Test DataLoader Integration (Optional) ---
#     print("\nTesting DataLoader integration:")
#     from torch.utils.data import DataLoader
#     try:
#         dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#         print(f"  DataLoader created successfully.")
#         # Get one batch
#         for batch_idx, (batch_tensors, batch_concepts, batch_ids) in enumerate(dataloader):
#             print(f"  Batch {batch_idx + 1}:")
#             print(f"    Tensor Batch Shape: {batch_tensors.shape}")
#             print(f"    Concepts Batch Shape: {batch_concepts.shape}")
#             print(f"    Image IDs in Batch: {batch_ids}")
#             break # Only show the first batch
#     except Exception as e:
#          print(f"  Error creating or iterating DataLoader: {e}")