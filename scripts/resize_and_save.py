import os
from src.preprocessing import resize_images

# Determine project root relative to the script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Define full paths
input_dir = os.path.join(PROJECT_ROOT, 'images')
output_dir = os.path.join(PROJECT_ROOT, 'output', 'resized_images')
target_size = (299,299)

resize_images(input_dir, output_dir, target_size)