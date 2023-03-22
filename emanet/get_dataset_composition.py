import numpy as np
from torch.utils.data import DataLoader
from dataset import ValDataset
import settings

def main():
    # Load dataset
    dataset = ValDataset(split='trainaug')
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    # Count occurrences of each label
    total_occs = np.zeros(settings.N_CLASSES + 1)
    image_counts = np.zeros(settings.N_CLASSES + 1)
    num_images = 0
    channel_means = np.zeros(3)
    channel_stdevs = np.zeros(3)

    for image, label in dataloader:
        label = label.numpy()
        label[label == 255] = settings.N_CLASSES

        # Count number of pixels that correspond to each label
        occs = np.bincount(label.flatten())
        occs = np.pad(occs, (0, settings.N_CLASSES - len(occs) + 1))

        # Calculate mean and stdev for each channel in image
        # TODO: dataloader currently already transforms with mean and stdev, need to undo
        image = image.numpy()
        channel_means += np.mean(image, axis=(0, 2, 3))
        channel_stdevs += np.std(image, axis=(0, 2, 3))

        # Update stats
        total_occs += occs
        num_images += 1
        image_counts += np.where(occs > 0, 1, 0)

    channel_means /= num_images
    sorted_occs = sorted((occs, i) for i, occs in enumerate(total_occs))
    total_labels = np.sum(total_occs[:-1])

    # Print stats
    for occs, i in sorted_occs:
        print(f'Label {i:>2}: {int(occs):>15} occurrences')
        print(f'    Frequency: {(occs / total_labels * 100):.4}%')
        print(f'    Images containing this label: {int(image_counts[i])}')
    print(f'Dataset means per channel: {channel_means}')
    print(f'Dataset stdev values per channel: {channel_stdevs / 255}')

    # Calculate class weights to balance dataset
    print(f'Balanced class weights: {total_labels / (settings.N_CLASSES * total_occs)}')

if __name__ == '__main__':
    main()
