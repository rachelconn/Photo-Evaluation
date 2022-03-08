import tensorflow as tf

def ImageRegressor(input_size):
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
    outputs = tf.keras.layers.Dense(1)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def ImageBinaryClassifier(input_size):
    return ImageRegressor(input_size)

def ResNetImageRegressor():
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
    outputs = tf.keras.layers.Conv2D(2048, 3, padding='same')(outputs)
    outputs = tf.keras.layers.ReLU()(outputs)
    outputs = tf.keras.layers.Conv2D(2048, 3, padding='same')(outputs)
    outputs = outputs + encoded
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)
    outputs = tf.nn.sigmoid(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
