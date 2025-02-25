import torch.nn as nn
from torch import Tensor

from .base import BaseModule

class TransformerEncoder(BaseModule):

    __slots__ = ["encoder"]

    def __init__(self, model_dim: int, num_heads: int = 2, num_layers: int = 4, ff_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class TransformerDecoder(BaseModule):

    __slots__ = ["decoder"]

    def __init__(self, model_dim: int, num_heads: int = 2, num_layers: int = 4, ff_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        return self.decoder(tgt, memory)
