from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseModule
from .embedding import InputEmbedding
from .transformer import TransformerEncoder, TransformerDecoder


class TransformerAutoencoder(BaseModule):

    __slots__ = [
        "embedding",
        "encoder",
        "decoder",
        "numeric_head",
        "categorical_head",
        "noise_factor",
    ]

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        noise_factor: float = 0.1,
        numeric_dim: Optional[int] = None,
        categorical_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = InputEmbedding(input_dim, model_dim, dropout)
        self.encoder = TransformerEncoder(
            model_dim, num_heads, num_layers, ff_dim, dropout
        )
        self.decoder = TransformerDecoder(
            model_dim, num_heads, num_layers, ff_dim, dropout
        )

        self.numeric_head = nn.Linear(model_dim, numeric_dim) if numeric_dim else None
        self.categorical_head = (
            nn.Linear(model_dim, categorical_dim) if categorical_dim else None
        )

        self.noise_factor = noise_factor

    def forward(self, x: Tensor) -> tuple[Optional[Tensor], Optional[Tensor]]:
        x = self.embedding(x)
        x_noisy = x + self.noise_factor * torch.randn_like(x)

        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded, x)

        decoded_numeric = self.numeric_head(decoded) if self.numeric_head else None
        decoded_categorical = (
            self.categorical_head(decoded) if self.categorical_head else None
        )

        return decoded_numeric, decoded_categorical
