import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderOnlyModel(nn.Module):

    __slots__ = [
        "transformer_encoder",
    ]

    def __init__(self, model_dim, num_heads, num_layers, dim_feedforward, dropout=0.1) -> None:
        super(TransformerEncoderOnlyModel, self).__init__()

        encoder_layers = TransformerEncoderLayer(model_dim, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x) -> torch.Tensor:
        x = x.transpose(0, 1)  # shape: (seq_length, batch_size, model_dim)
        x = self.transformer_encoder(x)
        return x.transpose(0, 1) # Transpose back to (batch_size, seq_length, model_dim)