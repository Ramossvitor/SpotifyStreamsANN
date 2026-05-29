"""Feed-forward network used to classify track genres."""

import torch.nn as nn

from . import config


class GenreANN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_dim in config.HIDDEN_DIMS:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
