from pathlib import Path
import logging
import numpy as np
from torch import Tensor

# TODO:
# Make sure mean and stdev are correct
# Add rotation, refine scale factors?
# Larger crop size, batch size

# Data settings
DATASET_FOLDER = Path('a:/', 'Datasets', 'blurdetection')
MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
STD = Tensor(np.array([0.229, 0.224, 0.225]))
SCALE_VALUES = (0.5, 0.6, 0.75, 1.0, 1.25, 1.5)
CROP_SIZE = 257
IGNORE_LABEL = 255
TRAIN_DATA_RATIO = 0.975

# Eval settings
SHOW_EVAL = True

# Model definition
MODEL_TYPE = 'isgeological'
MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'focus_16stride'

N_LAYERS = 50
STRIDE = 16
BN_MOM = 3e-4
EM_MOM = 0.9
STAGE_NUM = 3

# Training settings
BATCH_SIZE = 16
ITER_MAX = 40_000
ITER_SAVE = 2_000

LR_DECAY = 10
LR = 9e-3
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4

DEVICE = 0
DEVICES = [0]

N_CLASSES = 2
CLASS_WEIGHTS = Tensor(np.ones(N_CLASSES))

LOG_DIR = './logdir'
# NOTE: NUM_WORKERS has a huge effect on CPU memory usage when input images are large.
# Try lowering this value if you're running out of CPU memory.
NUM_WORKERS = 1

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
