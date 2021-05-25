from .mlp import MLP_1, MLP_2, MLP_3
from .cnn import LeNet
from .attention import VIT
from .fancy import MLPMixer, VFNet
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
    elif(net_type == 'vit'):
        P = config[dataset][net_type]["train_params"]["p_len"]
        n_patches = config[dataset][net_type]["train_params"]["n_patches"]
        input_channels = config[dataset][net_type]["train_params"]["input_shape"][0]
        return VIT(input_channels=input_channels,
                   num_patches=n_patches,
                   patch_size=P, num_layers=2)
    elif(net_type == 'vfnet'):
        return VFNet(input_shape=input_shape)
    elif(net_type == 'mlpmixer'):
        P = config[dataset][net_type]["train_params"]["p_len"]
        hidden_dim_1 = config[dataset][net_type]["train_params"]["hidden_dim_1"]
        hidden_dim_2 = config[dataset][net_type]["train_params"]["hidden_dim_2"]
        return MLPMixer(input_size=input_shape, patch_size=P,
                        hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                        num_classes=10, num_layers=2,)

    pass
