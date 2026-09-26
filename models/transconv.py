import torch.nn as nn

from .blocks import ResidualBlock2


class TransConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        self.resblock = ResidualBlock2(64)

        self.upconv = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.resblock(x)
        x = self.relu1(self.upconv(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x.clamp(0, 1)
