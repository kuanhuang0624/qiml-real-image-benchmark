"""Frozen compact full-image reference architecture."""
from __future__ import annotations

import torch
from torch import nn


class CompactCNN(nn.Module):
    def __init__(self, input_channels: int, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image).flatten(1))
