import torch.nn as nn

from .classifier import Classifier
from ..modules.ge_resblock import GEResBlock


class ResNetGE(nn.Module):
    def __init__(self, channels=64, num_blocks=4, num_classes=10, gather_type="avg"):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            *[
                GEResBlock(
                    channels=channels,
                    gather_type=gather_type
                )
                for _ in range(num_blocks)
            ]
        )

        self.classifier = Classifier(
            in_channels=channels,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x
