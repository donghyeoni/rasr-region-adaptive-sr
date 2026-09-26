import torch.nn as nn

from .blocks import patch_mask


class IMCNN(nn.Module):
    def __init__(self, image_size, patch_size, temperature, use_hard_mask=False):
        super().__init__()
        self.patch_size = patch_size
        self.total_memory = 128 * 128
        self.K = self.total_memory // (patch_size ** 2)
        self.grid_size = image_size // patch_size
        self.temperature = temperature
        self.use_hard_mask = use_hard_mask

        self.importance_predictor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        return patch_mask(self.importance_predictor(x), self.patch_size, self.grid_size,
                          self.K, self.temperature, self.use_hard_mask)
