import os
import sys
import struct
import pickle

import torchvision
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import model_class, DATA_PATH


class MNIST(Dataset):
    def __init__(self, mode: str, net_class='cnn'):

        super(MNIST, self).__init__()
        dir_path = DATA_PATH + '/mnist/'

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
        if net_class == 'mlp':
            self.x = self.x.reshape([-1, 28*28])
        else:
            self.x = self.x.reshape([-1, 1, 28, 28])

        if mode == 'train':
            self.x = self.x[:-10000]
            self.y = self.y[:-10000]
        elif mode == 'valid':
            self.x = self.x[-10000:]
            self.y = self.y[-10000:]

        self.x = self.x / 255.0
        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y).long()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]
    pass


class CIFAR10(Dataset):
    train_list = [
        'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'
    ]
    valid_list = [
        'data_batch_1',
    ]
    test_list = [
        'test_batch'
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, mode: str, net_class='cnn'):

        super(CIFAR10, self).__init__()
        if mode == 'train':
            downloaded_list = self.train_list
        elif mode == 'valid':
            downloaded_list = self.valid_list
        else:
            downloaded_list = self.test_list

        self.x = []
        self.y = []
        dir_path = DATA_PATH + 'cifar-10/'

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.x.append(entry['data'])
                if 'labels' in entry:
                    self.y.extend(entry['labels'])
                else:
                    self.y.extend(entry['fine_labels'])

        self.x = np.vstack(self.x)
        if net_class == 'mlp':
            self.x = self.x.reshape(-1, 3*32*32)
        else:
            self.x = self.x.reshape(-1, 3, 32, 32)

        self.x = self.x / 255.0
        self.x = torch.Tensor(self.x)
        self.y = torch.Tensor(self.y).long()

        # self._load_meta()
        pass

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


DATASETS = {
    "mnist": MNIST,
    "cifar-10": CIFAR10,
}


def get_dataset(
        name: str = "mnist", net_type="lenet",
        train: bool = False, valid: bool = False,
        test: bool = False,):
    dataset = DATASETS[name]
    res = []
    net_class = model_class(net_type)

    if train:
        res.append(dataset('train', net_class=net_class))
    if valid:
        res.append(dataset('valid', net_class=net_class))
    if test:
        res.append(dataset('test', net_class=net_class))
    return tuple(res)
