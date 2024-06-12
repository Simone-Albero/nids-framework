import torch
import torch.nn as nn


class InputEncoder(nn.Module):

    __slots__ = [
        "positional_encoding",
    ]

    def __init__(self, input_dim) -> None:
        super(InputEncoder, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.linear_relu_stack(x)
        return x