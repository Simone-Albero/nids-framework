import torch
import torch.nn as nn


class ClassificationHead(nn.Module):

    __slots__ = [
        "fc_out",
    ]

    def __init__(self, input_dim, model_dim) -> None:
        super(ClassificationHead, self).__init__()

        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc_out(x)
