import torch.nn as nn
from ..blocks.ge_block import GEBlock


class GEResBlock(nn.Module):
    def __init__(self, channels, gather_type="avg"):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.ge = GEBlock(channels, gather_type=gather_type)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.ge(out)

        return self.relu(out + identity)
