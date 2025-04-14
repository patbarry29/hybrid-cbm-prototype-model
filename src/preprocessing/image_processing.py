from PIL import Image
import os

import torch
import torchvision.transforms as transforms
import math

from src.utils.helpers import vprint


def resize_images(input_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)

    for dir in os.listdir(input_dir):
        curr_dir = os.path.join(input_dir, dir)

        if not os.path.isdir(curr_dir):
            continue

        os.makedirs(os.path.join(output_dir, dir), exist_ok=True)

        for filename in os.listdir(curr_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(curr_dir, filename)
                with Image.open(img_path) as img:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    img.save(os.path.join(output_dir, dir, filename))


def _get_transform_pipeline(use_training_transforms, resol, resized_resol):
    if use_training_transforms:
        print("Using TRAINING transformations:")
        return transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
    # Use LANCZOS resampling for better quality
    print("Using VALIDATION/TEST transformations:")
    return transforms.Compose([
        transforms.Resize(resized_resol, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])

def _get_all_filenames(input_dir):
    all_image_paths = []

    for subdir, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_image_paths.append(os.path.join(subdir, filename))

    return all_image_paths

def transform_and_save_batches(input_dir, output_dir, resol, resized_resol, use_training_transforms, batch_size=64, verbose=False):
    # get the transformation pipeline
    transform_pipeline = _get_transform_pipeline(use_training_transforms, resol, resized_resol)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    # get all image paths
    all_image_paths = _get_all_filenames(input_dir)

    vprint(f"Found {len(all_image_paths)} images.", verbose)
    num_batches = math.ceil(len(all_image_paths) / batch_size)
    vprint(f"Processing in {num_batches} batches of size {batch_size}...", verbose)

    processed_count = 0
    for i in range(num_batches):
        batch_paths = all_image_paths[i * batch_size : (i + 1) * batch_size]
        batch_tensors = []
        vprint(f"Processing Batch {i+1}/{num_batches}...", verbose)

        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            transformed_img_tensor = transform_pipeline(img)
            batch_tensors.append(transformed_img_tensor)
            processed_count += 1

        if batch_tensors:
            # Stack tensors in the batch along a new dimension (dim=0)
            batch_tensor_stacked = torch.stack(batch_tensors, dim=0)
            batch_filename = f"batch_{i:04d}.pt"
            output_path = os.path.join(output_dir, batch_filename)
            # save batch to file `batch_0001.pt`
            torch.save(batch_tensor_stacked, output_path)
            vprint(f"Saved {output_path} (Shape: {batch_tensor_stacked.shape})", verbose)

    vprint(f"\nFinished processing and saving {processed_count} images in {num_batches} batches.", verbose)
