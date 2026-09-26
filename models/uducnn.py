import torch.nn as nn

from .blocks import ResidualBlock2


class UDUCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.res1 = ResidualBlock2(64)

        self.down = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

        self.res2 = ResidualBlock2(32)

        self.output_conv = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.up1(x))
        x = self.res1(x)
        x = self.relu(self.down(x))
        x = self.relu(self.up2(x))
        x = self.res2(x)
        x = self.output_conv(x)
        return x.clamp(0, 1)
