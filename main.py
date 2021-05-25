import os
import sys
import argparse

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

from src.train import Trainer
from src.test import Tester
from src.utils import MODEL_PATH, model_class
from src.config import config


RANDOM_SEED = 1
USE_CUDA = True


def train(model_type, dataset, model_name):
    net_type = model_type
    train_params = config[dataset][net_type]["train_params"]
    hyper_params = config[dataset][net_type]["hyper_params"]

    if(model_name is not None):
        train_params["model_path"] = "{0}/{1}/{2}.pth".format(
            MODEL_PATH, model_class(model_type), model_name
        )
    trainer = Trainer(
        dataset=dataset,
        net_type=net_type,
        train_params=train_params,
        hyper_params=hyper_params,
        use_cuda=USE_CUDA,
    )
    print("Model {0} ready on dataset {1}.".format(net_type, dataset))
    print(hyper_params)
    print(train_params)
    trainer.train()


def summary(model_type):
    train_params = {
        "batch_size": 64,
        "input_shape": [1, 28, 28]
    }
    batch_size = train_params["batch_size"]
    input_size = train_params["input_shape"]
    summary(self.net, (3, input_size), batch_size)
    pass


def test(model_type, dataset, model_name):
    if model_name is None:
        print("Test: model name not found.")
    test_params = {
        "batch_size": 64,
    }
    tester = Tester(
        dataset=dataset,
        net_type=model_type,
        model_name=model_name,
        test_params=test_params,
        use_cuda=True
    )
    tester.test()
    pass


def show(model_type, dataset, model_name):
    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i+1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i], cmap='gray', interpolation='none')
    #     plt.title("Ground Truth: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
    pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'train', 'test', 'summary' or 'show'")
    parser.add_argument("-d", "--dataset",
                        help="dataset name", default='mnist')
    parser.add_argument("-m", "--model-type",
                        help="model type", default='lenet')
    parser.add_argument("-n", "--model-name",
                        help="model name")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(RANDOM_SEED)
    if (args.command == "train"):
        train(args.model_type, args.dataset, args.model_name)
    elif(args.command == "show"):
        show(args.model_type, args.dataset, args.model_name)
    elif(args.command == "test"):
        test(args.model_type, args.dataset, args.model_name)
    elif(args.command == "summary"):
        summary(args.model_type)
