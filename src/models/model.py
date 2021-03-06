from .mlp import MLP_1, MLP_2, MLP_3
from .cnn import LeNet
from .attention import VIT
from .fancy import MLPMixer, VFNetA, VFNetB
from .local_attention import LANet
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
        input_channels = input_shape[0]
        return VIT(input_channels=input_channels,
                   num_patches=n_patches,
                   patch_size=P, num_layers=2)
    elif(net_type == 'vfneta'):
        return VFNetA(input_shape=input_shape)
    elif(net_type == 'vfnetb'):
        return VFNetB(input_shape=input_shape)
    elif(net_type == 'mlpmixer'):
        P = config[dataset][net_type]["train_params"]["p_len"]
        hidden_dim_1 = config[dataset][net_type]["train_params"]["hidden_dim_1"]
        hidden_dim_2 = config[dataset][net_type]["train_params"]["hidden_dim_2"]
        return MLPMixer(input_size=input_shape, patch_size=P,
                        hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                        num_classes=10, num_layers=2,)
    elif(net_type == 'lanet'):
        input_channels = input_shape[0]
        input_size = input_shape[1:]
        kernel_sizes = config[dataset][net_type]["train_params"]["kernel_sizes"]
        return LANet(input_channels=input_channels,
                    kernel_sizes=kernel_sizes,
                    input_size=input_size,
                    num_classes=10)

    pass
