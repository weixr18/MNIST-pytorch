
from .cnn import LeNet


def get_model(net_type: str = 'LeNet'):
    if(net_type == 'LeNet' or net_type == 'lenet'):
        return LeNet()
