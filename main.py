import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import (
    load_blur_type_dataset,
    load_certh_training_dataset,
    load_certh_testing_dataset,
    load_ebb_dataset,
    load_exposure_dataset,
    load_realblur_dataset,
    load_sidd_dataset,
)
from model import (
    ImageBinaryClassification,
    ImageClassification,
    ImageRegression,
)

parser = argparse.ArgumentParser()
model_types = ['exposure', 'blur', 'blur_type', 'noise', 'bokeh']
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--type', choices=model_types, required=True)
parser.add_argument('--name', required=True)
args = parser.parse_args()

EXPOSURE_TRAINING_DATASET_FOLDER = Path('a:/', 'Datasets', 'exposure', 'training', 'INPUT_IMAGES')
EXPOSURE_VALIDATION_DATASET_FOLDER = Path('a:/', 'Datasets', 'exposure', 'validation', 'INPUT_IMAGES')
EXPOSURE_TESTING_DATASET_FOLDER = Path('a:/', 'Datasets', 'exposure', 'testing', 'INPUT_IMAGES')

BLUR_TRAINING_DATASET_FILE = Path('a:/', 'Datasets', 'blur', 'RealBlur_J_train_list.txt')
BLUR_TESTING_DATASET_FILE = Path('a:/', 'Datasets', 'blur', 'RealBlur_J_test_list.txt')

BLUR_TYPE_DATASET_FOLDER = Path('a:/', 'Datasets', 'blurtype')

NOISE_DATASET_FOLDER = Path('a:/', 'Datasets', 'noise', 'train')

BOKEH_DATASET_FOLDER = Path('a:/', 'Datasets', 'EBB')

def run_exposure_model(train=True):
    if train:
        augmentations = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomZoom(0.2, 0.2),
        ])
    else:
        augmentations = tf.keras.Sequential([])

    exposure_training_dataset = load_exposure_dataset(EXPOSURE_TRAINING_DATASET_FOLDER)
    exposure_training_dataset = exposure_training_dataset.map(lambda x, y: (augmentations(x), y))
    exposure_validation_dataset = load_exposure_dataset(EXPOSURE_VALIDATION_DATASET_FOLDER)
    exposure_testing_dataset = load_exposure_dataset(EXPOSURE_TESTING_DATASET_FOLDER)

    model = ImageRegression(4, args.name)
    if train:
        model.train(exposure_training_dataset, exposure_validation_dataset, 50)
    model.test(exposure_testing_dataset)

def run_blur_model(train=True):
    augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
    ])

    blur_training_dataset = load_realblur_dataset(BLUR_TRAINING_DATASET_FILE)
    blur_training_dataset = blur_training_dataset.map(lambda x, y: (augmentations(x), y))
    blur_testing_dataset = load_realblur_dataset(BLUR_TESTING_DATASET_FILE)

    # TODO: don't resize image
    model = ImageBinaryClassification(1, args.name)
    if train:
        model.train(blur_training_dataset, blur_testing_dataset, 100)
    model.test(blur_testing_dataset)

def run_blur_type_model(train=True):
    NUM_TRAINING_IMAGES = 1_000

    augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(),
        # tf.keras.layers.RandomBrightness(0.1, value_range=(0, 1)),
        # tf.keras.layers.RandomContrast(0.1),
        # tf.keras.layers.RandomCrop(800, 800)
    ])

    blur_type_dataset = load_blur_type_dataset(BLUR_TYPE_DATASET_FOLDER)
    blur_type_dataset = blur_type_dataset.map(lambda x, y: (augmentations(x), y))
    blur_type_training_dataset = blur_type_dataset.take(NUM_TRAINING_IMAGES)
    blur_type_testing_dataset = blur_type_dataset.skip(NUM_TRAINING_IMAGES)

    # TODO: don't resize image
    # TODO: see if performing augmentation here helps
    model = ImageClassification(3, 1, args.name)
    if train:
        model.train(blur_type_training_dataset, blur_type_testing_dataset, 100)
    model.test(blur_type_testing_dataset)

def run_noise_model(train=True):
    NUM_TRAINING_IMAGES = 600
    noise_dataset = load_sidd_dataset(NOISE_DATASET_FOLDER)
    noise_training_dataset = noise_dataset.take(NUM_TRAINING_IMAGES)
    noise_testing_dataset = noise_dataset.skip(NUM_TRAINING_IMAGES)

    model = ImageBinaryClassification(1, args.name)
    if train:
        model.train(noise_training_dataset, noise_testing_dataset, 100)
    model.test(noise_testing_dataset)

def run_bokeh_model(train=True):
    if train:
        augmentations = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(),
        ])
    else:
        augmentations = tf.keras.Sequential([])

    NUM_TRAINING_IMAGES = 9_000
    bokeh_dataset = load_ebb_dataset(BOKEH_DATASET_FOLDER)
    bokeh_dataset = bokeh_dataset.map(lambda x, y: (augmentations(x), y))
    bokeh_training_dataset = bokeh_dataset.take(NUM_TRAINING_IMAGES)
    bokeh_testing_dataset = bokeh_dataset.skip(NUM_TRAINING_IMAGES)

    model = ImageBinaryClassification(1, args.name)
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
