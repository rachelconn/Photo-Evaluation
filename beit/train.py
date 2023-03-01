import torch
from transformers import (
    BeitConfig,
    BeitForSemanticSegmentation,
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset import TrainingBlurDetectionDataset, ValidationBlurDetectionDataset, create_dataloader, resize
from beit import settings
from test import validate
from utils import load_model, save, visualize

# TODO: add logging code
def train():
    # Use pre-trained BEiT model to fine-tune
    # TODO: ensure backbone isn't being trained
    model, optimizer, lr_scheduler, iteration, min_valid_loss = load_model('train')

    train_dataset = create_dataloader(TrainingBlurDetectionDataset())
    test_dataset = create_dataloader(ValidationBlurDetectionDataset())

    # Training loop
    model.train()
    total_loss = 0
    # Resume running iterations from where the previous model left off (or from the beginning)
    for iteration in range(iteration + 1, settings.MAX_ITER + 1):
        for _ in range(settings.BATCHES_PER_UPDATE):
            images, labels, filename = next(iter(train_dataset))
            images, labels = resize(images, labels)

            # Predict
            preds = model(pixel_values=images, labels=labels)
            loss = preds.loss
            total_loss += loss
            loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Log results from epoch
        if iteration % settings.LOG_EVERY == 0:
            # TODO: print LR as well
            # TODO: log to file
            mean_loss = total_loss / settings.LOG_EVERY / settings.BATCHES_PER_UPDATE
            print(f'Iteration {iteration} completed, average loss: {mean_loss} | current LR: {lr_scheduler.get_last_lr()[0]}')
            total_loss = 0

        # Save
        if iteration % settings.SAVE_EVERY == 0:
            save(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                iteration=iteration,
                min_valid_loss=min_valid_loss,
            )

        # Validate
        if iteration % settings.VALIDATE_EVERY == 0:
            # TODO: add metrics other than loss (mIoU, fIoU...)
            print(f'Running validation after {iteration} iterations...')
            valid_metrics = validate(model, test_dataset)
            min_valid_loss = min(min_valid_loss, valid_metrics['loss'])
            print(f"    Validation loss: {valid_metrics['loss']}, validation IoU: {valid_metrics['iou']}")

    # # Visualize as a test that datasets work
    # for image, label in dataset:
    #     visualize(image, label)

if __name__ == '__main__':
    train()
