
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
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(
                dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):

        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.mlp_head(x)

##################################################
# Google FNet https://arxiv.org/pdf/2105.03824.pdf
# from https://github.com/rishikksh20/FNet-pytorch
##################################################


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, n, d = x.shape
        x = torch.cat((x, torch.zeros_like(x)), dim=1)
        x = x.reshape([b, 2, n, d]).permute(0, 2, 3, 1)
        x = torch.fft(x, signal_ndim=2)
        x = x[:, :, :, 0]
        return x


class VFNet(nn.Module):
    def __init__(self, input_channels=3, num_patches=16, patch_size=8, hidden_dim=64,
                 num_classes=10, num_layers=2,):
        super().__init__()
        dim = patch_size*patch_size

        self.chunking = nn.Conv2d(
            input_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, hidden_dim))
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        # input: [B, C, H, W]
        x = self.chunking(x)
        # now: [B, P*P, H/P, W/P]
        x = rearrange(x, 'b c h w -> b (h w) c')
        # now: [B, N=(H/P*W/P), D=(P*P),]

        # concatenate class tokens
        cls_tokens = repeat(self.cls, '() n d -> b n d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        # add position embedding
        x += self.pos_embedding[:, :]

        # f-net modified transformers
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        # output
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x
