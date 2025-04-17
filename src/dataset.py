import torch
import numpy as np
from torch.utils.data import Dataset

class ImageConceptDataset(Dataset):
    """
    PyTorch Dataset combining pre-loaded image tensors, concept labels,
    and image ID/filename mappings.
    """

    def __init__(self, image_tensors, concept_labels, image_labels):
        super().__init__()
        self.image_tensors = image_tensors

        # Convert labels to float tensors if they aren't already
        self.concept_labels = torch.as_tensor(concept_labels, dtype=torch.float32)
        self.image_labels = torch.as_tensor(image_labels, dtype=torch.float32)

        # --- Validate Consistency ---
        num_images_tensor = len(self.image_tensors)
        num_images_concepts = self.concept_labels.shape[0]
        num_images_labels = self.image_labels.shape[0]

        if not (num_images_tensor == num_images_concepts == num_images_labels):
            raise ValueError(f"Inconsistent number of items found: "
                            f"{num_images_tensor} tensors, "
                            f"{num_images_concepts} concept rows, "
                            f"{num_images_labels} label rows. "
                            f"Inputs must be pre-sorted and have the same length.")

        # --- Store Metadata (Optional but good practice) ---
        self.num_samples = num_images_tensor
        # Check if concept_labels is 1D or 2D
        self.num_concepts = self.concept_labels.shape[1] if self.concept_labels.ndim > 1 else 1
        # Check if image_labels is 1D or 2D
        self.num_classes = self.image_labels.shape[1] if self.image_labels.ndim > 1 else 1


        print(f"Dataset initialized with {self.num_samples} pre-sorted items.")


    def __len__(self):
        """Returns the total number of samples (images) in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset with size {self.num_samples}")

        image_tensor = self.image_tensors[idx]
        concepts = self.concept_labels[idx]
        image_label = self.image_labels[idx]

        return image_tensor, concepts, image_label, idx+1
