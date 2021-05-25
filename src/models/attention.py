# vision models using attention

import torch
import torch.nn as nn
import math
from einops import repeat, rearrange


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, input):
        output = self.net(input)
        return output


class MSA(nn.Module):
    """
    multi-head self attention
    """

    def __init__(self, dim, heads=8, dim_head=64,):
        super(MSA, self).__init__()
        self.dim = dim
        self.heads = heads

        # 论文里面的Dh
        self.Dh = dim_head ** -0.5

        # self-attention里面的Wq，Wk和Wv矩阵
        inner_dim = dim_head * heads
        self.linear_q = nn.Linear(dim, inner_dim, bias=False)
        self.linear_k = nn.Linear(dim, inner_dim, bias=False)
        self.linear_v = nn.Linear(dim, inner_dim, bias=False)

        self.output = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, input):
        """
        input: [batch, N, D]
        """
        # calc Q/K/V: [batch, N, inner_dim]
        q = self.linear_q(input)
        k = self.linear_k(input)
        v = self.linear_v(input)

        # calc A: [batch, N, N]
        A = torch.bmm(q, k.permute(0, 2, 1)) * self.Dh
        A = torch.softmax(A.view(A.shape[0], -1), dim=-1)
        A = A.view(A.shape[0], int(math.sqrt(A.shape[1])),
                   int(math.sqrt(A.shape[1])))

        # [batch, N, inner_dim]
        SA = torch.bmm(A, v)
        # [batch, N, D]
        out = self.output(SA)
        return out


class TransformerEncoder(nn.Module):
    """
    Encoder block: self attention + norm&add + mlp + norm&add
    """

    def __init__(self, dim, hidden_dim=64):
        super(TransformerEncoder, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.msa = MSA(dim)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, input):
        output = self.norm(input)
        output = self.msa(output)
        output_s1 = output + input
        output = self.norm(output_s1)
        output = self.mlp(output)
        output_s2 = output + output_s1
        return output_s2


class VIT(nn.Module):
    def __init__(self, input_channels, num_patches, patch_size, hidden_dim=64,
                 num_classes=10, num_layers=10):
        super(VIT, self).__init__()

        dim = patch_size*patch_size
        self.chunking = nn.Conv2d(
            input_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim))
        self.cls = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerEncoder(dim, hidden_dim))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # input: [B, C, H, W]
        x = self.chunking(x)  # now: [B, P*P, H/P, W/P]
        x = rearrange(x, 'b c h w -> b (h w) c')  # now: [B, (H/P*W/P), (P*P),]

        # concatenate class tokens
        cls_tokens = repeat(self.cls, '() n d -> b n d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        # add position embedding
        x += self.pos_embedding[:, :]

        # transformers
        for layer in self.layers:
            x = layer(x)

        # output
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x
