import os
import numpy as np
import tensorflow as tf

def get_dataset_item(filename):
    # Load image
    image = tf.keras.utils.load_img(filename, target_size=(256, 256), interpolation='bilinear')
    image = np.array(image) / 255

    # Parse label into float - labels in filename are either N<number>, 0, or P<number>
    label_string = filename.split(b'_')[-1].rsplit(b'.', 1)[0]
    if label_string[0] == ord('P'):
        label = float(label_string[1:])
    elif label_string[0] == ord('N'):
        label = -float(label_string[1:])
    else:
        assert label_string == b'0', f'Unexpected label: {label_string}'
        label = 0.0

    return image, label

def generate_dataset(filenames):
    for filename in filenames:
        yield get_dataset_item(filename)


def load_dataset(folder):
    label_files = [os.path.join(folder, f) for f in os.listdir(folder)]
    dataset = tf.data.Dataset.from_generator(
        generate_dataset,
        args=(label_files,),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )

    return dataset
