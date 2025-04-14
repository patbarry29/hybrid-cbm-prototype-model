# import numpy as np
# import os
# import pandas as pd

# def encode_image_concepts(image_attribute_labels_path, attributes_path=None):
#     """
#     Creates a matrix where rows represent images and columns represent concepts,
#     with binary values indicating whether each concept is present for an image.
#     Handles inconsistent column counts in the input file.

#     Parameters:
#     image_attribute_labels_path (str): Path to the file containing image-attribute assignments
#     attributes_path (str, optional): Path to the file containing attribute definitions (if needed)

#     Returns:
#     numpy.ndarray: Matrix of image concepts with shape (num_images, num_concepts)
#     list: List of image IDs corresponding to the rows in the matrix
#     """
#     try:
#         # Read the file as raw text to handle inconsistent column counts
#         with open(image_attribute_labels_path, 'r') as f:
#             lines = f.readlines()

#         # Parse each line manually
#         data = []
#         for line in lines:
#             parts = line.strip().split()
#             # We only care about the first 3 columns (image_id, concept_id, is_present)
#             if len(parts) >= 3:
#                 image_id = int(parts[0])
#                 concept_id = int(parts[1])
#                 is_present = int(parts[2])
#                 data.append([image_id, concept_id, is_present])

#         # Create a DataFrame from the parsed data
#         attr_df = pd.DataFrame(data, columns=['image_id', 'concept_id', 'is_present'])

#         # 2. Determine the number of unique images and concepts
#         unique_images = attr_df['image_id'].unique()
#         num_images = len(unique_images)

#         # Find the maximum concept_id to determine matrix dimensions
#         max_concept_id = attr_df['concept_id'].max()

#         print(f"Found {num_images} unique images.")
#         print(f"Found {max_concept_id} unique concepts.")

#         # 3. Create mapping from image_id to row index
#         image_to_row = {img_id: idx for idx, img_id in enumerate(sorted(unique_images))}

#         # 4. Create the concepts matrix initialized with zeros
#         concept_matrix = np.zeros((num_images, max_concept_id), dtype=int)

#         # 5. Populate the matrix
#         for _, row in attr_df.iterrows():
#             image_id = row['image_id']
#             concept_id = row['concept_id']
#             is_present = row['is_present']

#             # Set the corresponding concept value (0 or 1)
#             row_idx = image_to_row[image_id]
#             concept_matrix[row_idx, concept_id-1] = is_present

#         print(f"Generated concept matrix with shape: {concept_matrix.shape}")
#         return concept_matrix, sorted(unique_images)  # Return image IDs for reference

#     except FileNotFoundError as e:
#         print(f"Error: File not found - {e}. Please check paths.")
#         return None, None
#     except Exception as e:
#         print(f"Error processing data: {e}")
#         return None, None

# def load_concept_names(concepts_path):
#     """
#     Load concept names from a file.

#     Parameters:
#     concepts_path (str): Path to the file containing concept definitions

#     Returns:
#     dict: Dictionary mapping concept IDs to concept names
#     """
#     try:
#         # Read concept names with proper handling of text after the concept ID
#         concept_names = {}
#         with open(concepts_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split(' ', 1)  # Split by first space only
#                 if len(parts) == 2:
#                     concept_id = int(parts[0])
#                     concept_name = parts[1]
#                     concept_names[concept_id] = concept_name

#         print(f"Loaded {len(concept_names)} concept names.")
#         return concept_names
#     except FileNotFoundError:
#         print(f"Error: Concepts file not found at {concepts_path}")
#         return {}
#     except Exception as e:
#         print(f"Error loading concept names: {e}")
#         return {}

# def get_image_concepts(image_id, concept_matrix, image_ids, concept_names):
#     """
#     Get the concepts present for a specific image, with their names.

#     Parameters:
#     image_id (int): The ID of the image
#     concept_matrix (numpy.ndarray): Matrix of image concepts
#     image_ids (list): List of image IDs corresponding to rows in concept_matrix
#     concept_names (dict): Dictionary mapping concept IDs to names

#     Returns:
#     list: List of tuples with (concept_id, concept_name) for present concepts
#     """
#     try:
#         # Find the row index for this image_id
#         if image_id in image_ids:
#             row_idx = image_ids.index(image_id)
#         else:
#             print(f"Image ID {image_id} not found in the dataset.")
#             return []

#         # Get the present concept IDs
#         present_concept_indices = np.where(concept_matrix[row_idx] == 1)[0]
#         present_concept_ids = [idx + 1 for idx in present_concept_indices]  # +1 because concept_ids start at 1

#         # Match with concept names
#         result = []
#         for concept_id in present_concept_ids:
#             name = concept_names.get(concept_id, f"Unknown concept {concept_id}")
#             result.append((concept_id, name))

#         return result
#     except Exception as e:
#         print(f"Error getting image concepts: {e}")
#         return []

# if __name__ == "__main__":
#     attribute_labels_file = 'image_concept_labels.txt'
#     concepts_file = 'concepts.txt'

#     # Generate the concept matrix
#     concept_matrix, image_ids = encode_image_concepts(attribute_labels_file)

#     # Load concept names
#     concept_names = load_concept_names(concepts_file)

#     if concept_matrix is not None and concept_names:
#         # Print for the first 3 images
#         for i in range(min(3, len(image_ids))):
#             img_id = image_ids[i]
#             print(f"\nImage ID {img_id} (Row {i}):")
#             present_concepts = get_image_concepts(img_id, concept_matrix, image_ids, concept_names)
#             if present_concepts:
#                 print("  Present concepts:")
#                 for concept_id, name in present_concepts:
#                     print(f"    {concept_id}: {name}")
#             else:
#                 print("  No concepts present for this image.")

#         # Allow user to query for a specific image
#         print("\nYou can look up concepts for a specific image:")
#         try:
#             query_id = int(input("Enter an image ID (or -1 to quit): "))
#             while query_id != -1:
#                 present_concepts = get_image_concepts(query_id, concept_matrix, image_ids, concept_names)
#                 if present_concepts:
#                     print(f"\nImage ID {query_id} has these concepts:")
#                     for concept_id, name in present_concepts:
#                         print(f"  {concept_id}: {name}")
#                 else:
#                     print(f"\nNo concepts found for Image ID {query_id} or image not in dataset.")
#                 query_id = int(input("\nEnter another image ID (or -1 to quit): "))
#         except ValueError:
#             print("Invalid input. Must be an integer.")
#         except KeyboardInterrupt:
#             print("\nExiting.")

#         # Verify shape
#         print(f"\nTotal shape of concept matrix: {concept_matrix.shape}")
#         # Get statistics on concepts per image
#         concepts_per_image = np.sum(concept_matrix, axis=1)
#         print(f"Average concepts per image: {np.mean(concepts_per_image):.2f}")
#         print(f"Min/Max concepts per image: {np.min(concepts_per_image)}/{np.max(concepts_per_image)}")