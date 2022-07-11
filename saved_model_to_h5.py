import argparse
import os
import tensorflow.keras
import tensorflowjs as tfjs
from model import ImageRegression, ImageBinaryClassification

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('--type', choices=['regression', 'classification'], required=True)
args = parser.parse_args()

def main():
    if args.type == 'regression':
        model = ImageRegression(1, args.name)
    else:
        model = ImageBinaryClassification(1, args.name)

    model.load()
    tfjs.converters.save_keras_model(model.network, os.path.join('h5', args.name))

if __name__ == '__main__':
    main()
