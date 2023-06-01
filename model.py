import os
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from network import (
    CNNImageRegressor,
    CNNImageBinaryClassifier,
    CNNImageClassifier,
)

class ImageRegression:
    def __init__(self, model_name, *, batch_size=1, blocks=[1, 3]):
        self.batch_size = batch_size
        self.model_name = model_name

        self.network = CNNImageRegressor(
            blocks=blocks,
        )
        self.optimizer = keras.optimizers.Adam(
            learning_rate=0.00005,
        )
        self.loss = keras.losses.MeanSquaredError()
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['mean_squared_error']
        )
        self.load()

    def train(self, training_dataset, validation_dataset, num_epochs):
        training_dataset = training_dataset.shuffle(10).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(self.batch_size).cache()

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            hist = self.network.fit(
                x=training_dataset,
                validation_data=validation_dataset,
                epochs=1,
            )

            val_loss = hist.history['val_loss'][-1]
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f'Achieved new best validation loss ({val_loss}), saving...')
                self.save(f'epoch_{epoch}')

        # Done training, save final model
        self.save('final')

    def test(self, testing_dataset, classification=False, multiclass=False):
        testing_dataset = testing_dataset.batch(1)

        total_loss = 0
        num_correct = 0
        samples = 0

        for image, label in testing_dataset:
            pred = self.network(image)
            # plt.imshow(tf.squeeze(image))
            # plt.title(f'Real: {exposure[0]}\nPredicted: {pred.numpy()[0][0]:3f}')
            # plt.show()

            samples += 1
            total_loss += self.loss(label, pred)
            if classification:
                predicted_class = tf.cast(tf.math.argmax(pred, axis=1), tf.float32) if multiclass else tf.math.round(pred)
                if predicted_class == label:
                    num_correct += 1
        print(f'Overall loss: {total_loss / samples}')
        if classification:
            print(f'Overall accuracy: {num_correct / samples}')

    def get_dataset_statistics(self, dataset):
        dataset = dataset.batch(1)

        # Iterate over dataset
        sum_mean = tf.constant([0., 0., 0.])
        sum_std = tf.constant([0., 0., 0.])
        samples = 0
        for image, _ in dataset:
            sum_mean += tf.math.reduce_mean(image, axis=(0, 1, 2))
            sum_std += tf.math.reduce_std(image, axis=(0, 1, 2))
            samples += 1
            if samples % 100 == 0:
                print(f'Completed: {samples}')

        # Average means and stdev values
        means = sum_mean / samples
        stds = sum_std / samples
        print(f'Channel means: {means}')
        print(f'Channel stds: {stds}')

    def save(self, name):
        path = os.path.join(os.path.dirname(__file__), 'trained', self.model_name, name, 'model')
        self.network.save_weights(path)
        print(f'Saved model to {path}.')

    def load(self):
        path = os.path.join(os.path.dirname(__file__), 'trained', self.model_name, 'final', 'model')
        try:
            self.network.load_weights(path).expect_partial()
            print(f'Loaded existing model from {path}')
        except:
            print(f'No model found at {os.path.abspath(path)}.\nCreating new model...')
            return

class ImageBinaryClassification(ImageRegression):
    """
    Blur
        0.00025 (0.58)
        0.0001  (0.61)
    RealBlur
        0.00005 (0.90)
    """

    def __init__(self, model_name, *, batch_size=1, max_size=(800, 800), blocks=[1, 3]):
        self.batch_size = batch_size
        self.model_name = model_name

        self.network = CNNImageBinaryClassifier(
            max_size=max_size,
            blocks=blocks,
        )
        self.optimizer = keras.optimizers.Adam(
            learning_rate=0.00005,
        )
        self.loss = keras.losses.BinaryCrossentropy(from_logits=False)
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['binary_accuracy'],
        )
        self.load()

    def test(self, testing_dataset):
        super().test(testing_dataset, True)

class ImageClassification(ImageRegression):
    def __init__(self, model_name, *, batch_size=1, num_classes, max_size=(800, 800), blocks=[1, 3]):
        self.batch_size = batch_size
        self.model_name = model_name

        self.network = CNNImageClassifier(
            num_classes=3,
            max_size=max_size,
            blocks=blocks,
        )
        self.optimizer = keras.optimizers.Adam(
            learning_rate=0.00005,
        )
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['sparse_categorical_accuracy'],
        )
        self.load()

    def test(self, testing_dataset):
        super().test(testing_dataset, True, True)

# Functions to generate specific model instances for tasks (ensures sync between training and )
def create_exposure_model(name):
    return ImageRegression(name, batch_size=4, blocks=[1, 3, 3])

def create_blur_model(name):
    return ImageBinaryClassification(name, blocks=[1, 3, 3])

def create_blur_type_model(name):
    return ImageClassification(name, num_classes=3, blocks=[1, 3, 3])

def create_noise_model(name):
    return ImageBinaryClassification(name, blocks=[1, 3, 3])

def create_bokeh_model(name):
    return ImageBinaryClassification(name, blocks=[1, 3, 3])
