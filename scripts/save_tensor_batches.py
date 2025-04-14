import os
from src.preprocessing import transform_and_save_batches

# Determine project root relative to the script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Define full paths
input_dir = os.path.join(PROJECT_ROOT, 'images')
output_dir = os.path.join(PROJECT_ROOT, 'output', 'transformed_tensor_batches')

resol = 299
resized_resol = int(resol * 256/224)
use_training_transforms = False
batch_size_to_save = 256

transform_and_save_batches(input_dir, output_dir, resol, resized_resol, use_training_transforms, batch_size=batch_size_to_save)
