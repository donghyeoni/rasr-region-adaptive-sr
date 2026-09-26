import torch.nn as nn
import torch.nn.functional as F

from .blocks import patch_mask


def _predictor():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 1, kernel_size=1)
    )


class MRIMCNN(nn.Module):
    def __init__(self, image_size, patch_size, temperature, use_hard_mask=False):
        super().__init__()
        self.patch_size = patch_size
        self.total_memory = 128 * 128
        self.K = self.total_memory // (patch_size ** 2)
        self.grid_size = image_size // patch_size
        self.temperature = temperature
        self.use_hard_mask = use_hard_mask

        self.imp256 = _predictor()
        self.imp128 = _predictor()
        self.imp64 = _predictor()

    def forward(self, x):
        H, W = x.shape[-2:]
        resample = dict(mode='bicubic', align_corners=True, antialias=True)

        x128 = F.interpolate(x, scale_factor=0.5, **resample)
        x64 = F.interpolate(x, scale_factor=0.25, **resample)

        importance_map = (self.imp256(x)
                          + F.interpolate(self.imp128(x128), size=(H, W), **resample)
                          + F.interpolate(self.imp64(x64), size=(H, W), **resample))
        return patch_mask(importance_map, self.patch_size, self.grid_size,
                          self.K, self.temperature, self.use_hard_mask)
