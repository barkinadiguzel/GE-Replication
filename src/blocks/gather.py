import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgGather(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, 1)


class LargeKernelGather(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)


class IdentityGather(nn.Module):
    def forward(self, x):
        return x
