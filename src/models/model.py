from .mlp import MLP_1, MLP_2, MLP_3
from .cnn import LeNet
from ..config import config


def get_model(net_type: str = 'lenet', dataset: str = "mnist"):

    input_shape = config[dataset][net_type]["train_params"]["input_shape"]
    if(net_type == 'lenet'):
        return LeNet(input_shape=input_shape)
    elif(net_type == 'mlp1'):
        return MLP_1(input_shape=input_shape)
    elif(net_type == 'mlp2'):
        return MLP_2(input_shape=input_shape)
    elif(net_type == 'mlp3'):
        return MLP_3(input_shape=input_shape)
