import torch
import torch.nn as nn


class ClassificationHead(nn.Module):

    __slots__ = [
        "classifier",
    ]

    def __init__(self, input_dim, hidden_dim=256):
        super(ClassificationHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x[...,-1,:] # last token of the context window
        return self.classifier(x)

