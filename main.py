import os
import sys

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

from src.train import Trainer
from src.test import Tester
from src.utils import MODEL_PATH

RANDOM_SEED = 1
USE_CUDA = True


def train(args):
    if (args.__len__() < 1):
        print("Train: Too few arguments.")
        return
    train_params = {
        "batch_size": 64,
        "epochs": 1,
        "epoch_lapse": 1,
        "epoch_save": 20,
        "use_saved": False,
    }
    hyper_params = {
        "learning_rate": 1e-3,
        "optimizer": "SGD",
        # "adam_betas": (0.9, 0.999),
        "momentum": 0.9,
    }
    if(train_params["use_saved"]):
        if len(args) < 2:
            print("Augment-main: no model path.")
            return
        else:
            train_params["model_path"] = MODEL_PATH + \
                "cnn/" + args[1] + ".pth"
    trainer = Trainer(
        net_type=args[0],
        train_params=train_params,
        hyper_params=hyper_params,
        use_cuda=USE_CUDA,
    )
    print("Model ready.")
    print(hyper_params)
    print(train_params)
    trainer.train()


def summary(args):
    train_params = {
        "batch_size": 64,
        "input_shape": [1, 28, 28]
    }
    batch_size = train_params["batch_size"]
    input_size = train_params["input_shape"]
    summary(self.net, (3, input_size), batch_size)
    pass


def test(args):
    if(args.__len__() < 2):
        print("Augment-main: too few arguments.")
        return
    test_params = {
        "batch_size": 64,
    }
    tester = Tester(
        net_type=args[0],
        model_path=MODEL_PATH + "cnn/",
        model_name=args[1],
        test_params=test_params,
        use_cuda=True
    )
    tester.test()
    pass


def show(args):
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


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    args = sys.argv
    if (args.__len__() < 2):
        print("Too few arguments.")
    else:
        if (args[1] == "train"):
            train(args[2:])
        elif(args[1] == "show"):
            show(args[2:])
        elif(args[1] == "test"):
            test(args[2:])
        elif(args[1] == "summary"):
            summary(args[2:])
    pass
