import os
import os.path as osp

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from dataset import ValDataset, create_dataloader
from metric import fast_hist, cal_scores
from network import EMANet 
import settings

import matplotlib.pyplot as plt
from matplotlib import cm, colors

logger = settings.logger


class Session:
    def __init__(self, dt_split):
        torch.cuda.set_device(settings.DEVICE)

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR

        self.net = EMANet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        self.net = DataParallel(self.net, device_ids=[settings.DEVICE])
        dataset = ValDataset()
        self.dataloader = create_dataloader(dataset)
        self.hist = 0

    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path, 
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s.' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])

    def inf_batch(self, image, label):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]

        def batch_to_example(tensor):
            return np.squeeze(tensor.cpu().numpy(), axis=0)

        if settings.SHOW_EVAL:
            # Plot image
            ax1 = plt.subplot(1, 4, 1)
            ax1.set_title('Image')
            image_to_display = np.transpose(batch_to_example(image), (1, 2, 0)) / 2.64
            ax1.imshow(image_to_display)

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
                labels = batch_to_example(labels)
                labels[labels == 255] = settings.N_CLASSES
                output = cmap[labels.flatten()]
                r, c = labels.shape[:2]
                return np.reshape(output, (r, c, 4))

            label_image = label_to_image(label, label_color_map)
            pred_image = label_to_image(pred, label_color_map)

            # Convert labels to image of per-pixel confidence
            conf_color_map = cm.get_cmap('RdYlGn')
            # TODO: make sure batch size has been removed
            conf_image = conf_color_map(torch.squeeze(logit.max(dim=1)[0]).cpu().numpy())

            # Plot ground truth labels
            ax2 = plt.subplot(1, 4, 2)
            ax2.set_title('Ground truth')
            ax2.imshow(label_image)

            # Plot predicted labels
            ax3 = plt.subplot(1, 4, 3)
            ax3.set_title('Predicted')
            ax3.imshow(pred_image)

            # Plot confidence
            ax3 = plt.subplot(1, 4, 4)
            ax3.set_title('Confidence')
            ax3.imshow(conf_image)

            plt.show()

        self.hist += fast_hist(label, pred)


def main(ckp_name='final.pth'):
    sess = Session(dt_split='val')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()

    for i, [image, label, filename] in enumerate(dt_iter):
        sess.inf_batch(image, label)
        if i % 10 == 0:
            logger.info('num-%d' % i)
            scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
            for k, v in scores.items():
                logger.info('%s-%f' % (k, v))

    print('-' * 80)
    print('Summary statistics:')
    scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
    for k, v in scores.items():
        logger.info('%s-%f' % (k, v))
    logger.info('')
    for k, v in cls_iu.items():
        logger.info('%s-%f' % (k, v))


if __name__ == '__main__':
    main()
