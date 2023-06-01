import tensorflow as tf
import keras

def ResNetImageRegressor(activation=None):
    inputs = keras.Input(shape=(None, None, 3))
    encoder = keras.applications.ResNet50(
        include_top=False,
        input_tensor=inputs,
        weights=None,
        pooling='max',
    )
    encoded = encoder(inputs)
    outputs = keras.layers.Dense(1, activation=activation)(encoded)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def CNNImageRegressor(*, activation=None, num_classes=1, blocks=[1, 3]):
    # Dataset mean and stdev per channel - from bokeh dataset
    CHANNEL_MEANS = tf.constant([110.33158,  108.660904, 100.70173])
    CHANNEL_STDEVS = tf.constant([56.15182, 55.05964, 57.52548])
    CHANNEL_VARS = tf.math.square(CHANNEL_STDEVS)

    # Create input layer
    model = keras.Sequential()
    model.add(keras.layers.InputLayer((None, None, 3)))

    # Normalize input
    model.add(keras.layers.Normalization(mean=CHANNEL_MEANS, variance=CHANNEL_VARS))

    # Create convolutional layers
    channels = 16
    for layers_for_block in blocks:
        for _ in range(layers_for_block):
            model.add(keras.layers.Conv2D(channels, (3,3), activation='relu'))
        model.add(keras.layers.MaxPool2D(2,2))

        channels *= 2

    # Global pooling over each channel
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(512,activation='relu'))

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation=activation))

    return model

def CNNImageBinaryClassifier(*, max_size=None, blocks=[1, 3]):
    image = keras.Input(shape=(None, None, 3))

    if max_size:
        outputs = tf.image.resize(image, max_size, preserve_aspect_ratio=True)

    outputs = CNNImageRegressor(activation='sigmoid', blocks=blocks)(image)

    return keras.Model(image, outputs)

def CNNImageClassifier(*, num_classes, max_size=None, blocks=[1, 3]):
    image = keras.Input(shape=(None, None, 3))

    if max_size:
        outputs = tf.image.resize(image, max_size, preserve_aspect_ratio=True)

    outputs = CNNImageRegressor(activation='softmax', num_classes=num_classes, blocks=blocks)(image)

    return keras.Model(image, outputs)
