import tensorflow as tf
import keras

def ImageRegressor(input_size, output_activation=None):
    """
        Image regressor using a transformer, adapted from:
        https://keras.io/examples/vision/image_classification_with_vision_transformer/
    """
    # Note: shape should be (input_size, input_size, 3) if augmentation doesn't resize
    inputs = keras.Input(shape=(input_size, input_size, 3))
    patch_size = input_size // 8
    num_patches = (input_size // patch_size) ** 2
    projection_dim = 64
    num_transformer_layers = 6
    transformer_layer_sizes = [128, 64]
    num_heads = 4
    num_mlp_layers = 3

    # Encode into patches with positional embeddings
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
    )
    patches = tf.reshape(patches, (tf.shape(patches)[0], -1, tf.shape(patches)[-1]))
    positions = tf.range(start=0, limit=num_patches, delta=1)
    projection = keras.layers.Dense(units=projection_dim)
    positional_encoding = keras.layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim,
    )
    encoded = projection(patches) + positional_encoding(positions)

    # Create transformer layers
    for _ in range(num_transformer_layers):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=0.1,
        )(x1, x1)
        x2 = keras.layers.Add()([encoded, attention_output])
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        for _ in range(num_mlp_layers):
            for layer_size in transformer_layer_sizes:
                x3 = keras.layers.Dense(layer_size, activation=tf.nn.relu)(x3)
                x3 = keras.layers.Dropout(0.1)(x3)
        encoded = keras.layers.Add()([x2, x3])

    # Convert each prediction into size projection_dim
    encoded = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
    encoded = keras.layers.Flatten()(encoded)
    encoded = keras.layers.Dropout(0.5)(encoded)

    # Non-linear classifier over projection features
    outputs = keras.layers.Dense(2048, activation=tf.nn.relu)(encoded)
    outputs = keras.layers.Dropout(0.3)(outputs)
    outputs = keras.layers.Dense(1024, activation=tf.nn.relu)(outputs)
    outputs = keras.layers.Dropout(0.3)(outputs)
    outputs = keras.layers.Dense(1, activation=output_activation)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def ImageBinaryClassifier(input_size):
    return ImageRegressor(input_size, output_activation='sigmoid')

def ResNetImageRegressor():
    inputs = keras.Input(shape=(None, None, 3))
    encoder = keras.applications.resnet50.ResNet50(
        include_top=False,
        input_tensor=inputs,
        weights=None,
        pooling='max',
    )
    encoded = encoder(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(encoded)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def CNNImageRegressor(activation=None, num_classes=1):
    model = keras.Sequential()

    channels = 16
    blocks = [1, 3]

    model.add(keras.layers.InputLayer((None, None, 3)))

    for layers_for_block in blocks:
        for _ in range(layers_for_block):
            model.add(keras.layers.Conv2D(channels, (3,3), activation='relu'))
            model.add(keras.layers.MaxPool2D(2,2))

        channels *= 2

    # Global pooling over each channel
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(512,activation='relu'))

    # Output binary prediction in range [0, 1]
    model.add(keras.layers.Dense(num_classes, activation=activation))

    return model

def CNNImageBinaryClassifier(max_size=None):
    model = keras.Sequential()
    image = keras.Input(shape=(None, None, 3))

    if max_size:
        outputs = tf.image.resize(image, max_size, preserve_aspect_ratio=True)

    outputs = CNNImageRegressor(activation='sigmoid')(image)

    return keras.Model(image, outputs)

def CNNImageClassifier(num_classes, max_size=None):
    model = keras.Sequential()
    image = keras.Input(shape=(None, None, 3))

    if max_size:
        outputs = tf.image.resize(image, max_size, preserve_aspect_ratio=True)

    outputs = CNNImageRegressor(activation='sigmoid', num_classes=num_classes)(image)

    return keras.Model(image, outputs)
