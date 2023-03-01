from pathlib import Path
import shutil

import numpy as np
from matplotlib import cm, colors, pyplot as plt
import torch
from torch.nn import functional as F
from transformers import BeitConfig, BeitForSemanticSegmentation

from beit import settings

def _batch_to_example(tensor):
    return np.squeeze(tensor.cpu().numpy(), axis=0)

def visualize(image, label, pred=None, filename='[unnamed image]'):
    NUM_SUBPLOTS = 2 if pred is None else 4
    plt.gcf().suptitle(f'Labels for {filename}')
    # Plot image
    ax1 = plt.subplot(1, NUM_SUBPLOTS, 1)
    ax1.set_title('Image')
    image_to_display = np.transpose(_batch_to_example(image), (1, 2, 0))
    ax1.imshow(image_to_display, vmin=0., vmax=1.)

    # Convert labels to image of predicted classes
    label_color_map = cm.get_cmap('gist_rainbow', settings.N_CLASSES)(np.linspace(0, 1, settings.N_CLASSES))
    # Colors for 5 class:
    # 0: red
    # 1: yellow
    # 2: seafoam
    # 3: blue
    # 4: magenta
    label_color_map = np.vstack((label_color_map, [0., 0., 0., 1.])) # For 255 labels

    def label_to_image(labels, cmap):
        labels = _batch_to_example(labels)
        labels[labels == 255] = settings.N_CLASSES
        output = cmap[labels.flatten()]
        r, c = labels.shape[:2]
        return np.reshape(output, (r, c, 4))

    label_image = label_to_image(label, label_color_map)

    # Plot ground truth labels
    ax2 = plt.subplot(1, NUM_SUBPLOTS, 2)
    ax2.set_title('Ground truth')
    ax2.imshow(label_image)

    if pred is not None:
        logit = pred
        pred = torch.argmax(pred, 1)
        pred_image = label_to_image(pred, label_color_map)

        # Convert labels to image of per-pixel confidence
        conf_color_map = cm.get_cmap('RdYlGn')
        # TODO: make sure batch size has been removed
        conf_image = conf_color_map(torch.squeeze(logit.max(dim=1)[0]).cpu().numpy())

        # Plot predicted labels
        ax3 = plt.subplot(1, NUM_SUBPLOTS, 3)
        ax3.set_title('Predicted')
        ax3.imshow(pred_image)

        # Plot confidence
        ax3 = plt.subplot(1, NUM_SUBPLOTS, NUM_SUBPLOTS)
        ax3.set_title('Confidence')
        ax3.imshow(conf_image)

    plt.show()

def save(*, model, optimizer, lr_scheduler, iteration, min_valid_loss):
    checkpoint = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
        lr_scheduler=lr_scheduler.state_dict(),
        iteration=iteration,
        min_valid_loss=min_valid_loss,
    )
    output_folder = Path(__file__).resolve().parent / 'trained_models' / settings.MODEL_NAME
    output_folder.mkdir(parents=True, exist_ok=True)
    save_file = output_folder / f'checkpoint_{iteration}.pt'
    torch.save(checkpoint, save_file)
    print(f'Saved current model to {save_file}.')

    shutil.copyfile(save_file, output_folder / 'latest.pt')
    # TODO: save best model separately

def load_model(mode: ['train', 'eval']):
    # TODO: configure if loading final or latest
    # Create model
    config = BeitConfig(
        num_labels=settings.N_CLASSES,
        semantic_loss_ignore_index=settings.IGNORE_INDEX,
    )
    model = BeitForSemanticSegmentation(config)
    model.to(torch.device('cuda'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.INITIAL_LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.MAX_ITER)
    iteration = 0
    min_valid_loss = float('inf')

    checkpoint_folder = Path(__file__).resolve().parent / 'trained_models' / settings.MODEL_NAME
    checkpoint_path = checkpoint_folder / ('final.pt' if mode == 'eval' else 'latest.pt')

    # Load from checkpoint file if it exists
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        iteration = checkpoint['iteration']
        min_valid_loss = checkpoint['min_valid_loss']
        print(f'Loaded existing model from {checkpoint_path}.')
    except Exception as e:
        print(f'No checkpoint found at {checkpoint_path}. Creating new model.')
        model = BeitForSemanticSegmentation.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k',
            config=config,
        )
        model.to(torch.device('cuda'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.INITIAL_LEARNING_RATE)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.MAX_ITER)

    # Freeze layers before the decoder
    # for submodule in ['embeddings', 'encoder', 'fpn1', 'fpn2', 'fpn3', 'fpn4']:
    model.get_submodule('beit').requires_grad_(False)

    return model, optimizer, lr_scheduler, iteration, min_valid_loss

def upscale(logits):
    return F.interpolate(logits, scale_factor=4, mode='nearest')
