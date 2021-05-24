import os
import struct

import numpy as np
import torch
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, mode: str, net_type='cnn'):

        super(MNIST, self).__init__()

        current_path = os.path.abspath(__file__)
        dir_path = os.path.abspath(
            os.path.dirname(current_path) + os.path.sep + ".")
        dir_path += '\\..\\data\\mnist'

        if mode == 'train' or mode == 'valid':
            labels_path = os.path.join(dir_path, 'train-labels-idx1-ubyte')
            images_path = os.path.join(dir_path, 'train-images-idx3-ubyte')
        elif mode == 'test':
            labels_path = os.path.join(dir_path, 't10k-labels-idx1-ubyte')
            images_path = os.path.join(dir_path, 't10k-images-idx3-ubyte')

        with open(labels_path, 'rb') as lbpath:
            _, n = struct.unpack('>II', lbpath.read(8))
            self.y = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            _, __, ___, ____ = struct.unpack('>IIII', imgpath.read(16))
            self.x = np.fromfile(imgpath, dtype=np.uint8)
        if net_type == 'cnn':
            self.x = self.x.reshape([-1, 1, 28, 28])
        else:
            self.x = self.x.reshape([-1, 28*28])

        if mode == 'train':
            self.x = self.x[:-10000]
            self.y = self.y[:-10000]
        elif mode == 'valid':
            self.x = self.x[-10000:]
            self.y = self.y[-10000:]

        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y).long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
    pass


DATASETS = {
    "MNIST": MNIST,
}


def get_dataset(
        name: str = "MNIST",
        train: bool = False, valid: bool = False,
        test: bool = False,):
    dataset = DATASETS[name]
    res = []
    if train:
        res.append(dataset('train'))
    if valid:
        res.append(dataset('valid'))
    if test:
        res.append(dataset('test'))
    return tuple(res)
