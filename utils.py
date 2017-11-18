import glob

import numpy as np
import torch.nn as nn

GRAY_IMAGES = ['data/ILSVRC2014_DET_train/ILSVRC2014_train_0000/ILSVRC2014_train_00008823.JPEG',
          'data/ILSVRC2014_DET_train/ILSVRC2014_train_0000/ILSVRC2014_train_00000278.JPEG',
          'data/ILSVRC2014_DET_train/ILSVRC2014_train_0000/ILSVRC2014_train_00008774.JPEG',
          'data/ILSVRC2014_DET_train/ILSVRC2014_train_0000/ILSVRC2014_train_00002194.JPEG',
          'data/ILSVRC2014_DET_train/ILSVRC2014_train_0000/ILSVRC2014_train_00004122.JPEG',
          'data/ILSVRC2014_DET_train/ILSVRC2014_train_0000/ILSVRC2014_train_00001535.JPEG']


def get_images(n=None, random=False):
    image_list = glob.glob('data/ILSVRC2014_DET_train/ILSVRC2014_train_0000/*')
    image_list = [img for img in image_list if img not in GRAY_IMAGES]
    if n is not None:
        image_list = image_list[:n]
    
    return image_list


def gaussian_smooth(seq, sigma=5):
    x_vals = np.arange(len(seq))
    smoothed_seq = np.zeros(len(seq))
    for i in range(len(seq)):
        kernel = np.exp(-(x_vals - i) ** 2 / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        smoothed_seq[i] = np.sum(kernel * seq)
    return smoothed_seq

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_smoothed_label(label, nearest_neighbors, ab2cat, sigma=5, quantize_step=10):
    n_spaces, (k, m, n) = len(ab2cat), label.shape
    label = np.clip(label, -128, 128)
    quantized_label = np.zeros((n_spaces, m, n))
    for i in range(m):
        for j in range(n):
            nearest_5 = nearest_neighbors[(int(label[0, i, j]), int(label[1, i, j]))]
            quantized = np.zeros(n_spaces)
            for ab in nearest_5:
                quantized[ab] = 1 / (np.linalg.norm(ab - label[:, i, j]) + 1)
            quantized /= np.sum(quantized)
            quantized_label[:, i, j] = quantized
    return quantized_label