import torch
import torch.nn as nn


class InputEncoder(nn.Module):

    __slots__ = [
        "model_dim",
        "embedding",
        "positional_encoding",
    ]

    def __init__(self, input_dim, model_dim, window_size) -> None:
        super(InputEncoder, self).__init__()

        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, window_size, model_dim))

    def forward(self, x) -> torch.Tensor:
        seq_length = x.size(1)
        x = self.embedding(x) * (self.model_dim ** 0.5)
        x = x + self.positional_encoding[:, :seq_length, :]
        return x
