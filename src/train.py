# train
import os
import gc
import time

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .models.model import get_model
from .data import get_dataset
from .validate import Validator
from .utils import MODEL_PATH

SHOW_NET = False


class Trainer():

    def __init__(self, net_type="LeNet", train_params=None, hyper_params=None,
                 use_cuda=True, model_path="", module_save_dir="",):
        """setup the module"""
        self.train_dataset, self.valid_dataset = get_dataset(
            name="MNIST", net_type=net_type, train=True, valid=True, )
        self.net_type = net_type

        self.hyper_params = hyper_params
        self.train_params = train_params
        self.train_data_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_params["batch_size"],
            shuffle=True
        )
        self.valid_data_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.train_params["batch_size"],
            shuffle=False
        )

        self.use_cuda = use_cuda
        self.net = get_model(net_type=self.net_type)
        if use_cuda:
            self.net = self.net.cuda()

        if(self.hyper_params["optimizer"] == "SGD"):
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.hyper_params["learning_rate"],
                momentum=self.hyper_params["momentum"])
        elif (self.hyper_params["optimizer"] == "Adam"):
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=self.hyper_params["learning_rate"],
            )

        self.criterion = torch.nn.NLLLoss()
        self.v = Validator(net=self.net,
                           hyper_params=hyper_params,
                           validate_params=train_params,
                           use_cuda=use_cuda,
                           data_loader=self.valid_data_loader)
        pass

    def train(self):
        """train the model"""
        epochs = self.train_params["epochs"]
        epoch_lapse = self.train_params["epoch_lapse"]
        batch_size = self.train_params["batch_size"]
        epoch_save = self.train_params["epoch_save"]

        val_acc = self.v.validate()
        print("Validation accuracy before train: {0}".format((val_acc)))

        for _ in range(1, epochs+1):
            total_loss = 0
            for data in tqdm(self.train_data_loader, ascii=True, ncols=120):
                batch_train_x, batch_train_y = data
                if self.use_cuda:
                    batch_train_x = batch_train_x.cuda()
                    batch_train_y = batch_train_y.cuda()

                batch_loss = self.train_step(
                    batch_train_x, batch_train_y,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    net=self.net
                )
                total_loss += batch_loss

            if (_) % epoch_lapse == 0:
                val_acc = self.v.validate()
                print("Total loss in epoch %d : %f and validation accuracy" %
                      (_, total_loss), val_acc)
            if (_) % epoch_save == 0:
                self.save_model(str(_))
            pass

        gc.collect()
        self.save_model(str(epochs))
        pass

    def train_step(self, inputs, labels, optimizer,
                   criterion, net):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(torch.log(outputs+1e-7), labels)
        loss.backward()
        optimizer.step()
        return loss

    def save_model(self, name_else=""):

        net_type = self.net_type
        if net_type == "LeNet" or net_type == "lenet":
            prefix = "cnn/LeNet"
        elif net_type[:3] == "MLP" or net_type[:3] == "mlp":
            prefix = "mlp/"+net_type
        else:
            prefix = "attention/"
        time_str = time.strftime(
            "%Y%m%d_%H%M%S", time.localtime())
        name = prefix + "_" + time_str + "_" + name_else + ".pth"
        torch.save(self.net.state_dict(), MODEL_PATH + name)
        print("model saved:", name)
        pass
