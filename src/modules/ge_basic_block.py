import torch.nn as nn
from ..blocks.ge_block import GEBlock


class GEBasicBlock(nn.Module):
    def __init__(self, channels, gather_type="avg"):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.ge = GEBlock(channels, gather_type=gather_type)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.ge(out)
        return out
