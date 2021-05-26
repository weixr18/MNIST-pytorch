import os
import sys
import argparse

import torch
import torchsummary
import matplotlib.pyplot as plt
import numpy as np

from src.train import Trainer
from src.test import Tester
from src.utils import MODEL_PATH, model_class
from src.config import config
from src.models.model import get_model
from src.ml_main import run_ml


RANDOM_SEED = 1
USE_CUDA = True


def summary(model_type, dataset):
    batch_size = config[dataset][model_type]["train_params"]["batch_size"]
    input_shape = config[dataset][model_type]["train_params"]["input_shape"]
    net = get_model(net_type=model_type, dataset=dataset)
    print("Summary of model {0} on dataset {1}:".format(model_type, dataset))
    torchsummary.summary(net, tuple(input_shape), batch_size, device="cpu")
    pass


def train(model_type, dataset, model_name):
    net_type = model_type
    train_params = config[dataset][net_type]["train_params"]
    hyper_params = config[dataset][net_type]["hyper_params"]

    if(model_name is not None):
        train_params["model_path"] = "{0}{1}/{2}/{3}.pth".format(
            MODEL_PATH, dataset, model_class(net_type), model_name
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
    if(args.model_type == "ml"):
        run_ml(args.dataset)
    elif (args.command == "train"):
        train(args.model_type, args.dataset, args.model_name)
    elif(args.command == "show"):
        show(args.model_type, args.dataset, args.model_name)
    elif(args.command == "test"):
        test(args.model_type, args.dataset, args.model_name)
    elif(args.command == "summary"):
        summary(args.model_type, args.dataset)
