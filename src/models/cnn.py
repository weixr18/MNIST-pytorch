import torch
import torch.nn as nn


class LeNet(nn.Module):
    fc_input_size = {
        (1, 28, 28): 320,
        (3, 32, 32): 500,
    }

    def __init__(self, input_shape=[1, 28, 28]):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.LeakyReLU(),
        )
        fc_size = self.fc_input_size[tuple(input_shape)]
        self.fc1 = nn.Sequential(
            nn.Linear(fc_size, 50),
            nn.LeakyReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
