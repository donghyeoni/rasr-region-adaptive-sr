import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


def patch_mask(importance_map, patch_size, grid_size, k, temperature, hard):
    b = importance_map.shape[0]
    flat = F.avg_pool2d(importance_map, kernel_size=patch_size, stride=patch_size).view(b, -1)
    if hard:
        small = torch.zeros_like(flat).scatter_(1, flat.detach().topk(k, dim=1).indices, 1.0)
    else:
        small = F.softmax(flat / temperature, dim=1) * k
    small = small.view(b, 1, grid_size, grid_size)
    return F.interpolate(small, scale_factor=patch_size, mode='nearest')
