import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TVF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import emanet.settings as settings

def crop(image, label=None):
    # TODO: fix crop sizes being too small
    h, w = image.size()[-2:]
    crop_size = settings.CROP_SIZE
    s_h = np.random.randint(0, h - crop_size + 1) if h > crop_size else 0
    s_w = np.random.randint(0, w - crop_size + 1) if w > crop_size else 0
    e_h = min(s_h + crop_size, h)
    e_w = min(s_w + crop_size, w)
    image = image[:, :, s_h: e_h, s_w: e_w]
    label = label[:, :, s_h: e_h, s_w: e_w]
    return image, label

def flip(image, label=None):
    if np.random.rand() < 0.5:
        image = torch.flip(image, [3])
        if label is not None:
            label = torch.flip(label, [3])
    return image, label

def adjust_brightness(image, label=None):
    brightness = np.random.uniform(0.7, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    image = transforms.functional.adjust_brightness(image, brightness)
    image = transforms.functional.adjust_contrast(image, contrast)
    return image, label

def rotate(image, label=None):
    angle = np.random.uniform(-20, 20)
    image = transforms.functional.rotate(image, angle)
    label = transforms.functional.rotate(label, angle, fill=255)
    return image, label

def scale(image, label=None):
    scale_factor = random.choice(settings.SCALE_VALUES)
    image = torch.nn.functional.interpolate(
        image,
        scale_factor=scale_factor,
        mode='nearest-exact',
    )
    label = torch.nn.functional.interpolate(
        label,
        scale_factor=scale_factor,
        mode='nearest-exact',
    )
    return image, label

""" Dataset classes """
# TODO: make sure during validation, images are squished to 224x224 and that during training images with improper aspect ratios are used
class BaseDataset(Dataset):
    def __init__(self):
        image_folder = settings.DATASET_FOLDER / 'image'
        label_folder = settings.DATASET_FOLDER / 'gt'
        self.image_paths = [image_folder / filename for filename in image_folder.iterdir()]
        self.label_paths = [label_folder / f'{filename.stem}.png' for filename in image_folder.iterdir()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        return self._get_by_idx(idx)

    def _fetch(self, idx: int):
        # Since images don't always have the same file extension, determine what it is programatically
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        with Image.open(image_path).convert('RGB') as image:
            image = np.array(ImageOps.exif_transpose(image))
            image = torch.FloatTensor(image) / 255
            # TODO: normalize with mean and stdev
            # image = (image - settings.MEAN) / settings.STD
            image = image.permute(2, 0, 1).unsqueeze(dim=0)

        with Image.open(label_path).convert('P') as label:
            label = torch.FloatTensor(np.array(ImageOps.exif_transpose(label)))
            label = label.unsqueeze(dim=0).unsqueeze(dim=1)
            # Dataset uses black and white and has no masked out pixels, convert 255 values to the correct class index
            label[label == 255] = 1

        # Change tensors to correct dimensionality and type
        return image, label.to(torch.uint8)

class TrainDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        paths = list(zip(self.image_paths, self.label_paths))
        random.seed(0)
        random.shuffle(paths)
        self.image_paths, self.label_paths = zip(*paths)
        self.image_paths = self.image_paths[:round(len(paths) * settings.TRAIN_DATA_RATIO)]
        self.label_paths = self.label_paths[:round(len(paths) * settings.TRAIN_DATA_RATIO)]

    def __getitem__(self, idx: int):
        image, label = self._fetch(idx)

        # Perform augmentation
        image, label = crop(image, label)
        image, label = scale(image, label)
        image, label = flip(image, label)
        image, label = adjust_brightness(image, label)
        # image, label = rotate(image, label)

        return image.squeeze(dim=0), label.squeeze().to(torch.int64), self.image_paths[idx].name

class ValDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        paths = list(zip(self.image_paths, self.label_paths))
        random.seed(0)
        random.shuffle(paths)
        self.image_paths, self.label_paths = zip(*paths)
        self.image_paths = self.image_paths[round(len(paths) * settings.TRAIN_DATA_RATIO):]
        self.label_paths = self.label_paths[round(len(paths) * settings.TRAIN_DATA_RATIO):]

    def __getitem__(self, idx: int):
        image, label = self._fetch(idx)

        return image.squeeze(dim=0), label.squeeze().to(torch.int64), self.image_paths[idx].name

def _collate_fn(batch):
    """
    Collate function for dataloader to keep batches as List[torch.Tensor]
    instead of merging into a single tensor
    """
    images, labels, filenames = zip(*batch)
    return torch.stack(images), torch.stack(labels), filenames

def create_dataloader(dataset):
    training = isinstance(dataset, TrainDataset)
    return DataLoader(
        dataset,
        batch_size=1,
        # batch_size=settings.BATCH_SIZE if training else 1,
        num_workers=settings.NUM_WORKERS,
        shuffle=training,
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
