import torch
import torch.nn as nn


class InputEncoder(nn.Module):

    __slots__ = [
        "positional_encoding",
    ]

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(InputEncoder, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        return x