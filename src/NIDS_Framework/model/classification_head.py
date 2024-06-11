import torch
import torch.nn as nn


class ClassificationHead(nn.Module):

    __slots__ = [
        "classifier",
    ]

    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.classifier(x)

