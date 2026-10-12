"""A token-wise residual adapter for frozen visual features."""

import torch.nn as nn


class ResidualVisualAdapter(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens):
        return tokens + self.fc2(self.activation(self.fc1(self.norm(tokens))))
