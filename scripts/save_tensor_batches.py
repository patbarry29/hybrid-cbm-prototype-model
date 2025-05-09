import os
import time
from config import PROJECT_ROOT
from src.preprocessing.CUB import load_and_transform_images

# Define full paths
input_dir = os.path.join(PROJECT_ROOT, 'images')
# output_dir = os.path.join(PROJECT_ROOT, 'output', 'transformed_tensor_batches')

resol = 299
training = False
batch_size = 256

# transform_and_save_batches(input_dir, resol, training, batch_size, verbose=True)

start = time.time()
image_tensors, image_paths = load_and_transform_images(input_dir, resol, training, batch_size=32, verbose=True)
end = time.time()
print("exec time:", end-start)

if image_tensors:
    print(f"\nReturned {len(image_tensors)} tensors.")
    print(f"Shape of the first tensor: {image_tensors[0].shape}")
    print(f"Path of the first tensor: {image_paths[0]}")
else:
    print("\nNo tensors were returned. Check the input directory and logs.")
