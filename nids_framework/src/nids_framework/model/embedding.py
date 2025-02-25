import torch.nn as nn
from torch import Tensor
from .base import BaseModule

class InputEmbedding(BaseModule):

    __slots__ = ["embedding", "dropout"]

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.embedding(x))
