
import torch
from torch import nn
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import numpy as np


#######################################################
# Google MLP-Mixer https://arxiv.org/pdf/2105.01601.pdf
# from https://github.com/rishikksh20/MLP-Mixer-pytorch
#######################################################


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, hidden_dim_1),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, hidden_dim_2),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):

    def __init__(self, input_size, patch_size,
                 hidden_dim_1, hidden_dim_2,
                 num_classes=10, num_layers=2,):
        super().__init__()

        in_channels = input_size[0]
        assert input_size[1] == input_size[2], 'Image H and W must be equal.'
        image_size = input_size[1]
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        dim = patch_size*patch_size
        self.chunking = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                      stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(num_layers):
            self.mixer_blocks.append(MixerBlock(
                dim, self.num_patch, hidden_dim_1, hidden_dim_2))
        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x = self.chunking(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

####################################################
# Google FNet https://arxiv.org/pdf/2105.03824.pdf
# Add fourier layer to normal CNN to classify images
####################################################


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, n, d = x.shape
        x = torch.cat((x, torch.zeros_like(x)), dim=2)
        x = x.reshape([b, c, 2, n, d]).permute(0, 1, 3, 4, 2)
        x = torch.fft(x, signal_ndim=2)
        x = x[:, :, :, :, 0]
        return x


class VFNetA(nn.Module):
    fc_input_size = {
        (1, 28, 28): 320,
        (3, 32, 32): 500,
    }

    def __init__(self, input_shape=[1, 28, 28]):
        super(VFNetA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(),
        )
        self.flayer = FNetBlock()
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(),
        )
        fc_size = self.fc_input_size[tuple(input_shape)]
        # self.fc1 = nn.Sequential(
        #     nn.Linear(fc_size, 50),
        #     nn.LeakyReLU(inplace=True),
        # )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_size, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flayer(x)
        x = x.flatten(start_dim=1)
        x = self.fc2(x)
        return x


class VFNetB(nn.Module):
    fc_input_size = {
        (1, 28, 28): 320,
        (3, 32, 32): 500,
    }

    def __init__(self, input_shape=[1, 28, 28]):
        super(VFNetB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(),
        )
        self.flayer1 = FNetBlock()
        #self.flayer2 = FNetBlock()
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(),
        )
        fc_size = self.fc_input_size[tuple(input_shape)]
        # self.fc1 = nn.Sequential(
        #     nn.Linear(fc_size, 50),
        #     nn.LeakyReLU(inplace=True),
        # )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_size, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.flayer1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=1)
        x = self.fc2(x)
        return x
