from torchmetrics.classification import MulticlassJaccardIndex
import torch
from torch.nn import functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset import ValidationBlurDetectionDataset, create_dataloader, resize
from beit import settings
from utils import load_model, upscale, visualize

iou_metric = MulticlassJaccardIndex(
    num_classes=settings.N_CLASSES,
    average='micro',
    ignore_index=settings.IGNORE_INDEX,
).to('cuda:0')

def validate(model, dataset, do_visualize=False):
    with torch.no_grad():
        total_loss = 0
        total_iou = 0
        num_batches = 0
        for batch in dataset:
            images, labels, filenames = batch
            images, labels = resize(images, labels)
            preds = model(pixel_values=images, labels=labels)
            total_loss += preds.loss
            pred_labels = upscale(preds.logits)
            # TODO: use average='none' and calculate based on weights at the end
            total_iou += iou_metric(pred_labels, labels.cuda())
            num_batches += 1

            # Visualize each image in the batch if requested
            if do_visualize:
                for image, label, pred, filename in zip(images, labels, pred_labels, filenames):
                    visualize(image.unsqueeze(0), label.unsqueeze(0), pred.unsqueeze(0), filename=filename)
        return dict(
            loss=total_loss / num_batches,
            iou=total_iou / num_batches,
        )

def test():
    print('Loading model...')
    model, *_ = load_model('test')
    test_dataset = create_dataloader(ValidationBlurDetectionDataset())

    print('Running evaluation...')
    valid_metrics = validate(model, test_dataset, settings.VISUALIZE_DURING_TESTING)
    print(f"Evaluation complete.\n    Mean loss: {valid_metrics['loss']}\n    IoU: {valid_metrics['iou']}")

if __name__ == '__main__':
    test()
