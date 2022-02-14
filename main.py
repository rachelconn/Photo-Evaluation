import os
import matplotlib.pyplot as plt
import tensorflow as tf
from data_loader import load_dataset
from model import ImageRegression

TRAINING_DATASET_FOLDER = r'E:\photography\exposure\training\INPUT_IMAGES'
VALIDATION_DATASET_FOLDER = r'E:\photography\exposure\validation\INPUT_IMAGES'

training_dataset = load_dataset(TRAINING_DATASET_FOLDER)
validation_dataset = load_dataset(VALIDATION_DATASET_FOLDER)

model = ImageRegression()
model.train(training_dataset, validation_dataset, 50)
