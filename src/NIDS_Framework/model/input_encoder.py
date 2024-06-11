import torch
import torch.nn as nn


class InputEncoder(nn.Module):

    __slots__ = [
        "positional_encoding",
    ]

    def __init__(self, input_dim, hidden_dim=256, output_dim=80) -> None:
        super(InputEncoder, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x) -> torch.Tensor:
        batch_size, seq_length, feat = x.size()
        x = x.view(-1, feat)
        x = self.linear_relu_stack(x)
        x = x.view(batch_size, seq_length, -1)
        return x