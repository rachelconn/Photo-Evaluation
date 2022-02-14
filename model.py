import os
import tensorflow as tf
from network import ImageRegressor

class ImageRegression:
    def __init__(self):
        self.network = ImageRegressor()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=3e-4,
        )
        self.loss = tf.keras.losses.MeanSquaredError()
        self.network.compile(
            optimizer=self.optimizer,
            loss=self.loss,
        )
        # self.network.summary()

    def train(self, training_dataset, validation_dataset, num_epochs):
        training_dataset = training_dataset.shuffle(100).batch(4).prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.batch(4).cache()

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
                self.save(epoch)

    def save(self, epoch):
        path = os.path.join('trained', f'model_{epoch}', 'model')
        self.network.save_weights(path)
        print(f'Saved model to {path}.')
