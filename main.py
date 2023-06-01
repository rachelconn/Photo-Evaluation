import argparse
from math import ceil
from pathlib import Path
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Use dynamic memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from data_loader import (
    load_blur_type_dataset,
    load_ebb_dataset,
    load_exposure_dataset,
    load_realblur_dataset,
    load_sidd_dataset,
)
from model import (
    create_exposure_model,
    create_blur_model,
    create_blur_type_model,
    create_noise_model,
    create_bokeh_model,
)


# Set up argparser
parser = argparse.ArgumentParser()
model_types = ['exposure', 'blur', 'blur_type', 'noise', 'bokeh']
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--type', choices=model_types, required=True)
parser.add_argument('--name', required=True)
args = parser.parse_args()


# Dataset directories
EXPOSURE_TRAINING_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'exposure', 'training', 'INPUT_IMAGES')
EXPOSURE_VALIDATION_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'exposure', 'validation', 'INPUT_IMAGES')
EXPOSURE_TESTING_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'exposure', 'testing', 'INPUT_IMAGES')
EXPOSURE_DATASET_FOLDERS = [
    EXPOSURE_TRAINING_DATASET_FOLDER,
    EXPOSURE_VALIDATION_DATASET_FOLDER,
    EXPOSURE_TESTING_DATASET_FOLDER,
]

BLUR_TRAINING_DATASET_FILE = Path('/mnt', 'a', 'Datasets', 'blur', 'RealBlur_J_train_list.txt')
BLUR_TESTING_DATASET_FILE = Path('/mnt', 'a', 'Datasets', 'blur', 'RealBlur_J_test_list.txt')

BLUR_TYPE_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'blurtype')

NOISE_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'noise', 'train')

BOKEH_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'EBB')

def divide_dataset(
    dataset,
    train_augmentations,
    valid_augmentations=keras.Sequential([]),
    *,
    num_images,
    proportion_training,
    proportion_validation,
    proportion_testing
):
    """ Helper function creating training, validation, and testing sets """
    assert proportion_training + proportion_validation + proportion_testing == 1, 'Dataset is not being completely used!'

    num_training_images = ceil(num_images * proportion_training)
    num_validation_images = round(num_images * proportion_validation)
    num_testing_images = num_images - num_training_images - num_validation_images
    print(f'Using dataset with {num_images} images...')
    print(f'    Training set size: {num_training_images}')
    print(f'    Validation set size: {num_validation_images}')
    print(f'    Testing set size: {num_testing_images}')

    training_dataset = dataset.take(num_training_images).map(lambda x, y: (train_augmentations(x), y))
    validation_dataset = dataset.skip(num_training_images).take(num_validation_images).map(lambda x, y: (valid_augmentations(x), y))
    testing_dataset = dataset.skip(num_training_images + num_validation_images).map(lambda x, y: (valid_augmentations(x), y))
    return training_dataset, validation_dataset, testing_dataset

# Main loops for model training and testing
def run_exposure_model(train=True):
    train_augmentations = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomRotation(0.25),
        keras.layers.RandomZoom(-0.2, 0.2),
    ])

    dataset = load_exposure_dataset(EXPOSURE_DATASET_FOLDERS)
    training_dataset, validation_dataset, testing_dataset = divide_dataset(
        dataset,
        train_augmentations,
        num_images=24_330,
        proportion_training=0.9,
        proportion_validation=0.05,
        proportion_testing=0.05,
    )

    model = create_exposure_model(args.name)

    if train:
        model.train(training_dataset, validation_dataset, 50)
    model.test(testing_dataset)

def run_blur_model(train=True):
    train_augmentations = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
    ])

    training_dataset = load_realblur_dataset(BLUR_TRAINING_DATASET_FILE)
    blur_training_dataset = training_dataset.map(lambda x, y: (augmentations(x), y))
    testing_dataset = load_realblur_dataset(BLUR_TESTING_DATASET_FILE)

    # TODO: don't resize image
    model = create_blur_model(args.name)
    if train:
        model.train(training_dataset, testing_dataset, 100)
    model.test(testing_dataset)

def run_blur_type_model(train=True):
    train_augmentations = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomBrightness(0.1),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomCrop(800, 800)
    ])

    dataset = load_blur_type_dataset(BLUR_TYPE_DATASET_FOLDER)
    training_dataset, validation_dataset, testing_dataset = divide_dataset(
        dataset,
        train_augmentations,
        num_images=1_050,
        proportion_training=0.9,
        proportion_validation=0.05,
        proportion_testing=0.05,
    )

    # TODO: don't resize image
    model = create_blur_type_model(args.name)
    if train:
        model.train(training_dataset, validation_dataset, 100)
    model.test(testing_dataset)

"""
Best results:
[1, 3, 3]: 0.12 @ 5, 0.10 @ 11 (97.5%)
"""
def run_noise_model(train=True):
    train_augmentations = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomBrightness(0.1),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomCrop(800, 800)
    ])

    dataset = load_sidd_dataset(NOISE_DATASET_FOLDER)
    # noise_dataset = noise_dataset.map(lambda x, y: (augmentations(x), y))
    # TODO: use full dataset
    training_dataset, validation_dataset, testing_dataset = divide_dataset(
        dataset,
        train_augmentations,
        num_images=640,
        proportion_training=0.9,
        proportion_validation=0.05,
        proportion_testing=0.05,
    )

    model = create_noise_model(args.name)
    if train:
        model.train(training_dataset, validation_dataset, 100)
    model.test(testing_dataset)

"""
Best results:
- [1, 3, 3]: 0.36 loss @ 10, 0.22 loss @ 21 (90%)
"""
def run_bokeh_model(train=True):
    train_augmentations = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomBrightness(0.1),
        keras.layers.RandomContrast(0.1),
    ])

    NUM_TRAINING_IMAGES = 9_000
    dataset = load_ebb_dataset(BOKEH_DATASET_FOLDER)
    training_dataset, validation_dataset, testing_dataset = divide_dataset(
        dataset,
        train_augmentations,
        num_images=9_388,
        proportion_training=0.9,
        proportion_validation=0.05,
        proportion_testing=0.05,
    )

    model = create_bokeh_model(args.name)
    if train:
        model.train(training_dataset, validation_dataset, 100)
    model.test(testing_dataset)

if __name__ == '__main__':
    train = args.mode == 'train'
    # TODO: need to make work with any resolution, not just 512x512 - maybe make multiple predictions per image and average?
    if args.type == 'exposure':
        run_exposure_model(train=train)
    elif args.type == 'blur':
        run_blur_model(train=train)
    elif args.type == 'blur_type':
        run_blur_type_model(train=train)
    elif args.type == 'noise':
        run_noise_model(train=train)
    elif args.type == 'bokeh':
        run_bokeh_model(train=train)
    else:
        print(f'Unsupported type {args.type}.\n  Supported types: {(", ".join(model_types))}.')
        exit(1)
