import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
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

# Use dynamic memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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

BLUR_TRAINING_DATASET_FILE = Path('/mnt', 'a', 'Datasets', 'blur', 'RealBlur_J_train_list.txt')
BLUR_TESTING_DATASET_FILE = Path('/mnt', 'a', 'Datasets', 'blur', 'RealBlur_J_test_list.txt')

BLUR_TYPE_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'blurtype')

NOISE_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'noise', 'train')

BOKEH_DATASET_FOLDER = Path('/mnt', 'a', 'Datasets', 'EBB')

# Main loops for model training and testing
def run_exposure_model(train=True):
    if train:
        augmentations = keras.Sequential([
            keras.layers.RandomFlip(),
            keras.layers.RandomRotation(0.25),
            keras.layers.RandomZoom(0.2, 0.2),
        ])
    else:
        augmentations = keras.Sequential([])

    exposure_training_dataset = load_exposure_dataset(EXPOSURE_TRAINING_DATASET_FOLDER)
    exposure_training_dataset = exposure_training_dataset.map(lambda x, y: (augmentations(x), y))
    exposure_validation_dataset = load_exposure_dataset(EXPOSURE_VALIDATION_DATASET_FOLDER)
    exposure_testing_dataset = load_exposure_dataset(EXPOSURE_TESTING_DATASET_FOLDER)

    model = create_exposure_model(args.name)

    if train:
        model.train(exposure_training_dataset, exposure_validation_dataset, 50)
    model.test(exposure_testing_dataset)

def run_blur_model(train=True):
    augmentations = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
    ])

    blur_training_dataset = load_realblur_dataset(BLUR_TRAINING_DATASET_FILE)
    blur_training_dataset = blur_training_dataset.map(lambda x, y: (augmentations(x), y))
    blur_testing_dataset = load_realblur_dataset(BLUR_TESTING_DATASET_FILE)

    # TODO: don't resize image
    model = create_blur_model(args.name)
    if train:
        model.train(blur_training_dataset, blur_testing_dataset, 100)
    model.test(blur_testing_dataset)

def run_blur_type_model(train=True):
    NUM_TRAINING_IMAGES = 1_000

    # TODO: don't apply augmentations on test dataset (map after take/skip)
    augmentations = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomBrightness(0.1, value_range=(0, 1)),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomCrop(800, 800)
    ])

    blur_type_dataset = load_blur_type_dataset(BLUR_TYPE_DATASET_FOLDER)
    blur_type_dataset = blur_type_dataset.map(lambda x, y: (augmentations(x), y))
    blur_type_training_dataset = blur_type_dataset.take(NUM_TRAINING_IMAGES)
    blur_type_testing_dataset = blur_type_dataset.skip(NUM_TRAINING_IMAGES)

    # TODO: don't resize image
    model = create_blur_type_model(args.name)
    if train:
        model.train(blur_type_training_dataset, blur_type_testing_dataset, 100)
    model.test(blur_type_testing_dataset)

"""
Best results:
[1, 3, 3]: 0.12 @ 5, 0.10 @ 11 (97.5%)
"""
def run_noise_model(train=True):
    NUM_TRAINING_IMAGES = 600

    augmentations = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomBrightness(0.1, value_range=(0, 1)),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomCrop(800, 800)
    ])

    noise_dataset = load_sidd_dataset(NOISE_DATASET_FOLDER)
    noise_dataset = noise_dataset.map(lambda x, y: (augmentations(x), y))
    noise_training_dataset = noise_dataset.take(NUM_TRAINING_IMAGES)
    noise_testing_dataset = noise_dataset.skip(NUM_TRAINING_IMAGES)

    model = create_noise_model(args.name)
    if train:
        model.train(noise_training_dataset, noise_testing_dataset, 100)
    model.test(noise_testing_dataset)

"""
Best results:
- [1, 3, 3]: 0.36 loss @ 10, 0.22 loss @ 21 (90%)
"""
def run_bokeh_model(train=True):
    if train:
        augmentations = keras.Sequential([
            keras.layers.RandomFlip(),
            keras.layers.RandomBrightness(0.1, value_range=(0, 1)),
            keras.layers.RandomContrast(0.1),
        ])
    else:
        augmentations = keras.Sequential([])

    NUM_TRAINING_IMAGES = 9_000
    bokeh_dataset = load_ebb_dataset(BOKEH_DATASET_FOLDER)
    bokeh_dataset = bokeh_dataset.map(lambda x, y: (augmentations(x), y))
    bokeh_training_dataset = bokeh_dataset.take(NUM_TRAINING_IMAGES)
    bokeh_testing_dataset = bokeh_dataset.skip(NUM_TRAINING_IMAGES)

    model = create_bokeh_model(args.name)
    if train:
        model.train(bokeh_training_dataset, bokeh_testing_dataset, 100)
    model.test(bokeh_testing_dataset)

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
