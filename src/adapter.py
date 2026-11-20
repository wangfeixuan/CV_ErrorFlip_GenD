from __future__ import annotations

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        dimension: int,
        bottleneck: int = 128,
        dropout: float = 0.1,
        scale_init: float = 0.0,
    ):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dimension)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.activation(self.down(features))))
        # GenD's published classifier expects L2-normalized frame features.
        # Re-normalization keeps the original decision function at scale=0.
        return torch.nn.functional.normalize(
            features + torch.tanh(self.scale) * delta, p=2, dim=-1
        )


class GenDAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 2,
        bottleneck: int = 128,
        dropout: float = 0.1,
        scale_init: float = 0.0,
    ):
        super().__init__()
        self.adapter = ResidualAdapter(
            feature_dim, bottleneck, dropout, scale_init=scale_init
        )
        self.head = nn.Linear(feature_dim, num_classes)

    @classmethod
    def from_pretrained_head(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor,
        bottleneck: int = 128,
        dropout: float = 0.1,
        scale_init: float = 0.0,
    ) -> "GenDAdapter":
        model = cls(
            feature_dim=weight.shape[1],
            num_classes=weight.shape[0],
            bottleneck=bottleneck,
            dropout=dropout,
            scale_init=scale_init,
        )
        with torch.no_grad():
            model.head.weight.copy_(weight)
            model.head.bias.copy_(bias)
        return model

    def forward(
        self, features: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adapted = self.adapter(features)
        logits = self.head(adapted)
        probabilities = logits.softmax(dim=-1)[..., 1]
        video_probability = probabilities.mean(dim=-1)
        if return_features:
            return logits, video_probability, adapted
        return logits, video_probability
