import torch
import torch.nn as nn

input_size = {
    (1, 28, 28): 1*28*28,
    (1, 28*28): 1*28*28,
    (3, 32, 32): 3*32*32,
    (1, 3*32*32): 3*32*32,
}


class MLP_1(nn.Module):
    """
    Note: 1 full connect layer is not "multy" layer perceptron.
    Usually, it is called "Softmax Regression".
    """

    def __init__(self, input_shape=[1, 28, 28]):
        super(MLP_1, self).__init__()
        fc_shape = input_size[tuple(input_shape)]
        self.fc = nn.Sequential(
            nn.Linear(fc_shape, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MLP_2(nn.Module):
    def __init__(self, input_shape=[1, 28, 28]):
        super(MLP_2, self).__init__()
        fc_shape = input_size[tuple(input_shape)]
        self.fc1 = nn.Sequential(
            nn.Linear(fc_shape, 50),
            nn.LeakyReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class MLP_3(nn.Module):
    def __init__(self, input_shape=[1, 28, 28]):
        super(MLP_3, self).__init__()
        fc_shape = input_size[tuple(input_shape)]
        self.fc1 = nn.Sequential(
            nn.Linear(fc_shape, 300),
            nn.LeakyReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(300, 50),
            nn.LeakyReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
