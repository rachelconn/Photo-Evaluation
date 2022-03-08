import os
from matplotlib import pyplot as plt
import tensorflow as tf
from network import ImageRegressor, ImageBinaryClassifier, ResNetImageRegressor

class ImageRegression:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size

        self.network = ImageRegressor(256)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=3e-5,
        )
        self.loss = tf.keras.losses.MeanSquaredError()
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )
        # self.network.summary()

    def train(self, training_dataset, validation_dataset, num_epochs):
        training_dataset = training_dataset.shuffle(100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
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
                class_weight={
                    0: 370 / 630,
                    1: 630 / 370,
                },
            )
            print(f'New LR: {self.network.optimizer._decayed_lr(tf.float32).numpy()}')

            val_loss = hist.history['val_loss'][-1]
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f'Achieved new best validation loss ({val_loss}), saving...')
                self.save(epoch)

    def test(self, testing_dataset):
        testing_dataset = testing_dataset.batch(1)
        for image, exposure in testing_dataset:
            pred = self.network(image)
            plt.imshow(tf.squeeze(image))
            plt.title(f'Real: {exposure[0]}\nPredicted: {pred:3f}')
            plt.show()

    def save(self, epoch):
        path = os.path.join('trained', f'model_{epoch}', 'model')
        self.network.save_weights(path)
        print(f'Saved model to {path}.')

    def load(self):
        path = os.path.join('trained', 'model', 'model')
        try:
            self.network.load_weights(path)
            print(f'Loaded existing model from {path}')
        except:
            print('Created new model')
            return

class ImageBinaryClassification(ImageRegression):
    def __init__(self, batch_size, use_resnet=True):
        self.batch_size = batch_size

        if use_resnet:
            self.network = ResNetImageRegressor()
        else:
            self.network = ImageBinaryClassifier(512)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(3e-2, 250, 0.9),
        )

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['binary_accuracy'],
        )
