import os
from config import PROJECT_ROOT
from src.preprocessing.CUB import resize_images

# Define full paths
input_dir = os.path.join(PROJECT_ROOT, 'images')
output_dir = os.path.join(PROJECT_ROOT, 'output', 'resized_images')
target_size = (299,299)

resize_images(input_dir, output_dir, target_size)