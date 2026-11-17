import torch
from torch import nn


class SevenLayerCNN(nn.Module):
    """Six Conv1d layers and one linear head, with no running statistics."""

    def __init__(self, input_dim, classes, channels=(32, 32, 64, 64, 128, 128), kernel_size=3, normalization="group"):
        super().__init__()
        if len(channels) != 6 or kernel_size % 2 != 1:
            raise ValueError("Seven-layer model requires six convolutions and an odd kernel")
        self.input_dim, self.classes = input_dim, classes
        layers, previous = [], 1
        for j, channel in enumerate(channels):
            layers.append(nn.Conv1d(previous, channel, kernel_size, padding=kernel_size // 2))
            if normalization == "group":
                layers.append(nn.GroupNorm(1, channel))
            elif normalization != "none":
                raise ValueError("normalization must be group or none")
            layers.append(nn.ReLU())
            if j in (1, 3):
                layers.append(nn.AvgPool1d(2, ceil_mode=True))
            previous = channel
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], classes)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.shape[1:] != (1, self.input_dim):
            raise ValueError(f"Expected (N,1,{self.input_dim}), got {tuple(x.shape)}")
        return self.classifier(self.features(x))


def make_model(input_dim, classes, config):
    return SevenLayerCNN(input_dim, classes, **config)
