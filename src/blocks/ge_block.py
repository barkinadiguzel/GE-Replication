import torch.nn as nn

from .gather import GlobalAvgGather, LargeKernelGather, IdentityGather
from .excite import ChannelExcite


def build_gather(gather_type, channels):
    if gather_type == "avg":
        return GlobalAvgGather()

    elif gather_type == "conv7":
        return LargeKernelGather(channels, kernel_size=7)

    elif gather_type == "conv11":
        return LargeKernelGather(channels, kernel_size=11)

    elif gather_type == "identity":
        return IdentityGather()

    else:
        raise ValueError(f"Unknown gather type: {gather_type}")


class GEBlock(nn.Module):

    def __init__(self, channels, gather_type="avg", reduction=4):
        super().__init__()

        self.gather = build_gather(gather_type, channels)
        self.excite = ChannelExcite(channels, reduction=reduction)

    def forward(self, x):
        g = self.gather(x)
        scale = self.excite(x, g)
        return x * scale
