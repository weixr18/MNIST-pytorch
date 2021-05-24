from .mlp import MLP_1, MLP_2, MLP_3
from .cnn import LeNet


def get_model(net_type: str = 'LeNet'):
    if(net_type == 'LeNet' or net_type == 'lenet'):
        return LeNet()
    elif(net_type == 'mlp1'):
        return MLP_1()
    elif(net_type == 'mlp2'):
        return MLP_2()
    elif(net_type == 'mlp3'):
        return MLP_3()
