import base64
from io import BytesIO
from os import path
import sys

import numpy as np
from PIL import Image, ImageOps
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
from emanet.network import EMANet
from emanet import settings as emanet_settings
from matplotlib import pyplot as plt

# Create tf configuration with dynamic memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

exposure_model = ImageRegression(1, 'exposure').network
blur_model = ImageBinaryClassification(1, 'blur').network
noise_model = ImageBinaryClassification(1, 'noise').network
bokeh_model = ImageBinaryClassification(1, 'bokeh').network

focus_model = EMANet(emanet_settings.N_CLASSES, emanet_settings.N_LAYERS).cuda()
state_dict = torch.load(emanet_settings.MODEL_DIR / 'final.pth', map_location=lambda storage, loc: storage.cuda())
focus_model.load_state_dict(state_dict['net'])
focus_model.eval()

def process_request(request):
    """ Returns (tf_tensor, torch_tensor) for a given request's image """
    image_data = base64.b64decode(request.data['image'])
    image_data = BytesIO(image_data)
    with Image.open(image_data, formats=['JPEG']) as image:
        image = ImageOps.exif_transpose(image)
        image = np.asarray(image)
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, 'float32') / 255
        torch_image = torch.unsqueeze(torch.from_numpy(image.numpy()), 0).cuda()
        return np.array([image]), torch.permute(torch_image, (0, 3, 1, 2))

def get_focus_image(torch_image: torch.Tensor) -> str:
    """ Returns a base64-encoded image of the focus values given the torch tensor for an image """
    torch_image = torch.nn.functional.interpolate(
        torch_image,
        scale_factor=0.5,
        mode='nearest-exact',
    )
    logits = focus_model(torch_image)
    # Get predicted classes
    focus = torch.squeeze(torch.argmax(logits, dim=1)).type(torch.ByteTensor)

    # Convert to image
    focus_image = Image.fromarray(focus.cpu().numpy(), mode='P')
    # TODO: calculate how much of image is in/out of focus and add to API response
    focus_image.putpalette([
        0, 212, 102, # In focus: green
        255, 0, 0,   # Blurred: red
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
        blur=blur_model.predict(tf_image)[0][0],
        bokeh=bokeh_model.predict(tf_image)[0][0],
        exposure=exposure_model.predict(tf_image)[0][0],
        focus=focus_image,
        noise=noise_model.predict(tf_image)[0][0],
    )

    return Response(response)
