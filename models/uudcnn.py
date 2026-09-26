import torch.nn as nn

from .blocks import ResidualBlock


class UUDCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2, padding_mode='reflect')
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.upsample(x)
        x = self.decoder(x)
        return x.clamp(0, 1)
