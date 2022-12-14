import os
from matplotlib import pyplot as plt
import tensorflow as tf
from network import ImageRegressor, ImageBinaryClassifier, ResNetImageRegressor, CNNImageRegressor, CNNImageBinaryClassifier

class ImageRegression:
    def __init__(self, batch_size, model_name):
        self.batch_size = batch_size
        self.model_name = model_name

        self.network = CNNImageRegressor()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=3e-5,
        )
        self.loss = tf.keras.losses.MeanSquaredError()
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )
        self.load()

    def train(self, training_dataset, validation_dataset, num_epochs):
        training_dataset = training_dataset.shuffle(10).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(self.batch_size).cache()

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            # for x, y in training_dataset:
            #     y_p = self.network(x)
            #     print(f'y    : {y}')
            #     print(f'Preds: {y_p}')
            #     print(f'Loss: {self.network.loss(y, y_p)}')
            #     print(f'Vals: {tf.nn.sigmoid(y_p)}')
            hist = self.network.fit(
                x=training_dataset,
                validation_data=validation_dataset,
                epochs=1,
            )
            print(f'New LR: {self.network.optimizer._decayed_lr(tf.float32).numpy()}')

            val_loss = hist.history['val_loss'][-1]
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f'Achieved new best validation loss ({val_loss}), saving...')
                self.save(f'epoch_{epoch}')

        # Done training, save final model
        self.save('final')

    def test(self, testing_dataset, classification=False):
        testing_dataset = testing_dataset.batch(1)

        total_loss = 0
        correct = 0
        samples = 0

        for image, exposure in testing_dataset:
            pred = self.network(image)
            # plt.imshow(tf.squeeze(image))
            # plt.title(f'Real: {exposure[0]}\nPredicted: {pred.numpy()[0][0]:3f}')
            # plt.show()

            samples += 1
            total_loss += self.loss(exposure, pred)
            if tf.math.round(pred) == exposure:
                correct += 1
        print(f'Overall loss: {total_loss / samples}')
        if classification:
            print(f'Overall accuracy: {correct / samples}')

    def save(self, name):
        path = os.path.join('trained', self.model_name, name, 'model')
        self.network.save_weights(path)
        print(f'Saved model to {path}.')

    def load(self):
        path = os.path.join(os.path.dirname(__file__), 'trained', self.model_name, 'final', 'model')
        try:
            self.network.load_weights(path)
            print(f'Loaded existing model from {path}')
        except:
            print(f'No model found at {os.path.abspath(path)}.\nCreating new model...')
            return

class ImageBinaryClassification(ImageRegression):
    """
    Cats + Dogs
        LR for CNN:
            0.0005 (0.79 acc)
        LR for resnet:
            0.0005 (0.53 acc)
            0.00025 (0.57)
            0.00001 (each epoch: 0.56, 0.62, 0.68, 0.75)
    Blur
        CNN:
            0.00025 (0.58)
            0.0001  (0.61)
        Resnet:
            0.000001 (0.5, not just guessing one label)
            0.0000005 (same)
    RealBlur
        CNN:
            0.00005 (0.90)
    """

    def __init__(self, batch_size, model_name, max_size=(800, 800), use_resnet=True):
        self.batch_size = batch_size
        self.model_name = model_name

        self.network = CNNImageBinaryClassifier(max_size)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00005,
        )
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['binary_accuracy'],
        )
        self.load()

    def test(self, testing_dataset):
        super().test(testing_dataset, True)
