import tensorflow as tf

def ImageRegressor():
    inputs = tf.keras.Input(shape=(None, None, 3))
    preprocessed = tf.keras.applications.resnet50.preprocess_input(inputs)
    encoder = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        input_tensor=preprocessed,
        weights='imagenet',
    )
    encoder.trainable = False
    encoded = encoder(preprocessed)

    outputs = tf.keras.layers.BatchNormalization()(encoded)
    outputs = tf.keras.layers.Conv2D(1, 1)(outputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)
    outputs = tf.squeeze(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
