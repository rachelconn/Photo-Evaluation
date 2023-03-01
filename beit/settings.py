import torch
from pathlib import Path

# Paths
DATASET_FOLDER = Path('a:/', 'Datasets', 'blurdetection')

# Configuration
MODEL_NAME = 'blur975'
NUM_WORKERS = 4
BATCH_SIZE = 16
BATCHES_PER_UPDATE = 4
MAX_ITER = 1_200
INITIAL_LEARNING_RATE = 1e-3
SAVE_EVERY = 100
LOG_EVERY = 10
VALIDATE_EVERY = 50
TRAIN_DATA_RATIO = 0.975
VISUALIZE_DURING_TESTING = True

# Static values (should require no configuration)
CROP_SIZE = 224
MODEL_INPUT_DIM = 224
IGNORE_INDEX = 255
SCALE_VALUES = [0.5, 0.6, 0.75, 1.0, 1.25, 1.5]
N_CLASSES = 2
CLASS_WEIGHTS = torch.ones(N_CLASSES).cuda()
