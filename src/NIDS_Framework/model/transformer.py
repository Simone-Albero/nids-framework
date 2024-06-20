from typing import Optional

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderOnly(nn.Module):

    __slots__ = [
        "transformer_encoder",
    ]

    def __init__(
        self,
        input_dim: int,
        num_heads: Optional[int] = 4,
        num_layers: Optional[int] = 2,
        dim_feedforward: Optional[int] = 128,
        dropout: Optional[float] = 0.1,
    ) -> None:
        super(TransformerEncoderOnly, self).__init__()
        encoder_layers = TransformerEncoderLayer(
            input_dim, num_heads, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: (batch, seq, features)
        x = self.transformer_encoder(x)
        return x
