import base64
from io import BytesIO
from os import path
import sys

import numpy as np
from PIL import Image
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
import tensorflow as tf
import torch
import torchvision.transforms.functional as F
from transformers import BeitImageProcessor

sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', '..')))
from model import ImageRegression, ImageBinaryClassification
from beit.utils import load_model
from beit.dataset import resize

# Create tf configuration with dynamic memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

exposure_model = ImageRegression(1, 'exposure').network
blur_model = ImageBinaryClassification(1, 'blur').network
noise_model = ImageBinaryClassification(1, 'noise').network
focus_model, *_ = load_model('eval')
focus_model.eval()
focus_processor = BeitImageProcessor()

def process_request(request):
    """ Returns (tf_tensor, torch_tensor) for a given request's image """
    # print(request.data['image'][:100])
    image = tf.constant(request.data['image'])
    image = tf.io.decode_base64(image)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, 'float32') / 255
    torch_image = torch.unsqueeze(torch.from_numpy(image.numpy()), 0)
    return np.array([image]), torch.permute(torch_image, (0, 3, 1, 2))

def get_focus_image(torch_image: torch.Tensor) -> str:
    """ Returns a base64-encoded image of the focus values given the torch tensor for an image """
    focus = focus_model(pixel_values=resize(torch_image))
    output_size = [[dim // 8 for dim in torch_image.size()[2:]]]
    focus = focus_processor.post_process_semantic_segmentation(focus, output_size)[0]
    focus_image = Image.fromarray(focus.cpu().numpy(), mode='P')
    focus_image.putpalette([
        255, 0, 0,   # Out of focus: red
        0, 212, 102, # In focus: green
    ])
    focus_file = BytesIO()
    focus_image.save(focus_file, format="PNG")
    return base64.b64encode(focus_file.getvalue())

# Create your views here.
@api_view(['POST'])
def evaluate_photo(request):
    tf_image, torch_image = process_request(request)
    focus_image = get_focus_image(torch_image)

    response = dict(
        exposure=exposure_model.predict(tf_image)[0][0],
        blur=blur_model.predict(tf_image)[0][0],
        noise=noise_model.predict(tf_image)[0][0],
        focus=focus_image,
    )

    return Response(response)
