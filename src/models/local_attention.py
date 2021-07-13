import torch
import torch.nn as nn
from torch.autograd import Function
import local_att

class LocalAttention(Function):
    @staticmethod
    def forward(ctx, x, weight):
        outputs = local_att.local_attention_forward_cuda(x, weight)
        ctx.save_for_backward(x, weight)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        x, weight=ctx.saved_variables
        d_x, d_weight = local_att.local_attention_backward_cuda(
            grad_output.contiguous(), x, weight)
        return d_x, d_weight


class LABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_cuda=True):
        super(LABlock, self).__init__()
        self.kernel_size = kernel_size
        self.w = nn.parameter.Parameter(
            torch.Tensor(5, in_channels, kernel_size, kernel_size),
            requires_grad=True)
        self.w.data.uniform_(-0.01, 0.01)
        self.register_parameter("weight", self.w)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        initial_x = x.shape[2]
        initial_y = x.shape[3]
        x_pad = self.kernel_size - (initial_x % self.kernel_size)
        y_pad = self.kernel_size - (initial_y % self.kernel_size)
        x = nn.ZeroPad2d((0, x_pad, 0, y_pad))(x)
        att = LocalAttention.apply(x, self.w)
        att = torch.softmax(att, dim=1)
        x1 = x * att  + x
        x = self.conv(x1)
        # x = LocalAttention.apply(x, self.w)
        # x = att + x
        # x = self.conv(x)
        # x = self.bn(x)
        x = x[:, :, :initial_x, :initial_y]
        return x



class LANet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, 
                hidden_channels=[64,64,50], kernel_sizes=[3,3],
                input_size=[28,28]):
        super(LANet, self).__init__()
        self.la1 = LABlock(input_channels, hidden_channels[0], kernel_sizes[0])
        self.la2 = LABlock(hidden_channels[0], hidden_channels[1], kernel_sizes[1])
        self.la3 = LABlock(hidden_channels[1], hidden_channels[1], kernel_sizes[1])
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)
        self.pool3 = nn.MaxPool2d(2, 2, 0)
        mlp_input_size = hidden_channels[1] *int(input_size[0]/2/2/2) * int(input_size[1]/2/2/2)
        self.head = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_channels[2]), 
            nn.Linear(hidden_channels[2], num_classes), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.la1(x)
        x = self.pool1(x)
        x = self.la2(x)
        x = self.pool2(x)
        x = self.la3(x)
        x = self.pool3(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        return x
