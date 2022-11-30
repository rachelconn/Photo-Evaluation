import sys
from os import path

import numpy as np
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
import tensorflow as tf

sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', '..')))
from model import ImageRegression, ImageBinaryClassification

exposure_model = ImageRegression(1, 'exposure').network
blur_model = ImageBinaryClassification(1, 'blur').network
noise_model = ImageBinaryClassification(1, 'noise').network

def process_request(request):
    image = tf.constant(request.data['image'])
    image = tf.io.decode_base64(image)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, 'float32') / 255
    return np.array([image])

# Create your views here.
@api_view(['POST'])
def evaluate_photo(request):
    image = process_request(request)
    response = {
        'exposure': exposure_model.predict(image)[0][0],
        'blur': blur_model.predict(image)[0][0],
        'noise': noise_model.predict(image)[0][0],
    }

    return Response(response)
