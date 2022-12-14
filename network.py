import tensorflow as tf

def ImageRegressor(input_size, output_activation=None):
    """
        Image regressor using a transformer, adapted from:
        https://keras.io/examples/vision/image_classification_with_vision_transformer/
    """
    # Note: shape should be (input_size, input_size, 3) if augmentation doesn't resize
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
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
    projection = tf.keras.layers.Dense(units=projection_dim)
    positional_encoding = tf.keras.layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim,
    )
    encoded = projection(patches) + positional_encoding(positions)

    # Create transformer layers
    for _ in range(num_transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=0.1,
        )(x1, x1)
        x2 = tf.keras.layers.Add()([encoded, attention_output])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        for _ in range(num_mlp_layers):
            for layer_size in transformer_layer_sizes:
                x3 = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)(x3)
                x3 = tf.keras.layers.Dropout(0.1)(x3)
        encoded = tf.keras.layers.Add()([x2, x3])

    # Convert each prediction into size projection_dim
    encoded = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
    encoded = tf.keras.layers.Flatten()(encoded)
    encoded = tf.keras.layers.Dropout(0.5)(encoded)

    # Non-linear classifier over projection features
    outputs = tf.keras.layers.Dense(2048, activation=tf.nn.relu)(encoded)
    outputs = tf.keras.layers.Dropout(0.3)(outputs)
    outputs = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(outputs)
    outputs = tf.keras.layers.Dropout(0.3)(outputs)
    outputs = tf.keras.layers.Dense(1, activation=output_activation)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def ImageBinaryClassifier(input_size):
    return ImageRegressor(input_size, output_activation='sigmoid')

def ResNetImageRegressor():
    inputs = tf.keras.Input(shape=(None, None, 3))
    encoder = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        input_tensor=inputs,
        weights=None,
        pooling='max',
    )
    encoded = encoder(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(encoded)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def CNNImageRegressor(activation=None):
    model = tf.keras.Sequential()

    channels = 16
    blocks = [1, 3]

    model.add(tf.keras.layers.InputLayer((None, None, 3)))

    for layers_for_block in blocks:
        for _ in range(layers_for_block):
            model.add(tf.keras.layers.Conv2D(channels, (3,3), activation='relu'))
            model.add(tf.keras.layers.MaxPool2D(2,2))

        channels *= 2

    # Global pooling over each channel
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(512,activation='relu'))

    # Output binary prediction in range [0, 1]
    model.add(tf.keras.layers.Dense(1, activation=activation))

    return model

def CNNImageBinaryClassifier(max_size=None):
    model = tf.keras.Sequential()
    image = tf.keras.Input(shape=(None, None, 3))

    if max_size:
        outputs = tf.image.resize(image, [800, 800], preserve_aspect_ratio=True)

    outputs = CNNImageRegressor(activation='sigmoid')(image)

    return tf.keras.Model(image, outputs)
