import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.fc(x)
        return x
