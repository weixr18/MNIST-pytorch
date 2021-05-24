import torch
import torch.nn as nn


class MLP_1(nn.Module):
    def __init__(self):
        super(MLP_1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MLP_2(nn.Module):
    def __init__(self):
        super(MLP_2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 50),
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
    def __init__(self):
        super(MLP_3, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 300),
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
