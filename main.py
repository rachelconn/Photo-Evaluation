import os
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_exposure_dataset, load_certh_training_dataset, load_certh_testing_dataset
from model import ImageRegression, ImageBinaryClassification

EXPOSURE_TRAINING_DATASET_FOLDER = r'E:\photography\exposure\training\INPUT_IMAGES'
EXPOSURE_VALIDATION_DATASET_FOLDER = r'E:\photography\exposure\validation\INPUT_IMAGES'
EXPOSURE_TESTING_DATASET_FOLDER = r'E:\photography\exposure\testing\INPUT_IMAGES'

BLUR_TRAINING_DATASET_FILE = r'E:\photography\blur\CERTH_ImageBlurDataset\TrainingSet'
BLUR_TESTING_DATASET_FILE = r'E:\photography\blur\CERTH_ImageBlurDataset\EvaluationSet\NaturalBlurSet'

def run_exposure_model(train=True):
    augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
    ])
    exposure_training_dataset = load_exposure_dataset(EXPOSURE_TRAINING_DATASET_FOLDER)
    exposure_training_dataset = exposure_training_dataset.map(lambda x, y: (augmentations(x), y))
    exposure_validation_dataset = load_exposure_dataset(EXPOSURE_VALIDATION_DATASET_FOLDER)
    exposure_testing_dataset = load_exposure_dataset(EXPOSURE_TESTING_DATASET_FOLDER)

    model = ImageRegression(augmentations)
    if train:
        model.train(exposure_training_dataset, exposure_validation_dataset, 50)
    model.test(exposure_testing_dataset)

def run_blur_model(train=True):
    augmentations = tf.keras.Sequential([
        tf.keras.layers.RandomCrop(512, 512),
        tf.keras.layers.RandomFlip('horizontal'),
    ])

    blur_training_dataset = load_certh_training_dataset(BLUR_TRAINING_DATASET_FILE)
    blur_training_dataset = blur_training_dataset.map(lambda x, y: (augmentations(x), y))
    blur_testing_dataset = load_certh_testing_dataset(BLUR_TESTING_DATASET_FILE)
    blur_testing_dataset = blur_testing_dataset.map(lambda x, y: (tf.image.crop_to_bounding_box(x, 0, 0, 512, 512), y))

    model = ImageBinaryClassification(1)
    if train:
        model.train(blur_training_dataset, blur_testing_dataset, 50)
    model.test(blur_testing_dataset)

if __name__ == '__main__':
    # run_exposure_model()
    # TODO: need to make work with any resolution, not just 512x512 - maybe make multiple predictions per image and average?
    run_blur_model()
