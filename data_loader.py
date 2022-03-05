import os
import numpy as np
import tensorflow as tf

def get_exposure_dataset_item(filename):
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

def generate_exposure_dataset(filenames):
    for filename in filenames:
        yield get_exposure_dataset_item(filename)


def load_exposure_dataset(folder):
    label_files = [os.path.join(folder, f) for f in os.listdir(folder)]
    dataset = tf.data.Dataset.from_generator(
        generate_exposure_dataset,
        args=(label_files,),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )

    return dataset

def get_realblur_dataset_item(filename, label):
    # Load image
    image = tf.keras.utils.load_img(filename)
    image = np.array(image) / 255

    return image, label

def generate_realblur_dataset(list_file):
    base_dir = os.path.dirname(list_file)
    with open(list_file, 'r') as f:
        for line in f.readlines():
            gt_filename, blurred_filename, *_ = line.split(' ')
            yield get_blur_dataset_item(os.path.join(base_dir, blurred_filename.encode('utf-8')), 1)
            yield get_blur_dataset_item(os.path.join(base_dir, gt_filename.encode('utf-8')), 0)

def load_realblur_dataset(list_file):
        dataset = tf.data.Dataset.from_generator(
            generate_blur_dataset,
            args=(list_file,),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )

        return dataset
