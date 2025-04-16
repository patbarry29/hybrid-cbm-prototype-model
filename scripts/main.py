import os
import time

from config import PROJECT_ROOT
from src.preprocessing import load_and_transform_images, one_hot_encode_labels, encode_image_concepts, get_image_id_mapping

def main():
    # LOAD AND TRANSFORM IMAGES
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

    one_hot_labels = one_hot_encode_labels(labels_file, classes_file, verbose=True)

    # GET IMAGE ID TO IMAGE FILENAME MAPPING
    images_file = os.path.join(PROJECT_ROOT, 'data', 'images.txt')
    image_id_mapping = get_image_id_mapping(images_file)

    print(image_id_mapping[1])
    print(concept_labels[1])
    print(image_tensors[image_paths.index(image_id_mapping[1])])

    # dataset = dataset(images, concept_labels, image_labels)
    # dataloader = datadataloader(images, concept_labels, image_labels)

    # train_test_split()
    # batch_split()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('exec time:', end_time-start_time)