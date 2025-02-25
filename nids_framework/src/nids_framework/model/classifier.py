from torch import Tensor
import torch.nn as nn

from .base import BaseModule
from .embedding import InputEmbedding
from .transformer import TransformerEncoder

# class ClassificationHead(BaseModule):
#
#     __slots__ = ["classifier", "dropout", "output_dim"]
#
#     def __init__(self, latent_dim: int, output_dim: int, dropout: float = 0.1) -> None:
#         super().__init__()
#         self.classifier = nn.Linear(latent_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.output_dim = output_dim
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.dropout(x[:, -1, :])
#         x = self.classifier(x)
#         return x.squeeze(-1) if self.output_dim == 1 else x


class ClassificationHead(BaseModule):

    __slots__ = ["classifier", "output_dim", "fc_sequence"]

    def __init__(self, latent_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        intermediate_dim = latent_dim // 2

        self.fc_sequence = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(intermediate_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, -1, :]
        x = self.fc_sequence(x)
        x = self.classifier(x)
        return x.squeeze(-1) if self.output_dim == 1 else x



class TransformerClassifier(BaseModule):

    __slots__ = ["embedding", "encoder", "classifier"]

    def __init__(
        self, output_dim: int, input_dim: int, model_dim: int = 128, num_heads: int = 2, num_layers: int = 4,
        ff_dim: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embedding = InputEmbedding(input_dim, model_dim, dropout)
        self.encoder = TransformerEncoder(model_dim, num_heads, num_layers, ff_dim, dropout)
        self.classifier = ClassificationHead(model_dim, output_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(self.embedding(x)))
