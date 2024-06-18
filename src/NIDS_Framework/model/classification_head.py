import torch
import torch.nn as nn


class ClassificationHead(nn.Module):

    __slots__ = [
        "classifier",
    ]

    def __init__(self, input_dim):
        super(ClassificationHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x[...,-1,:] # last token of the context window
        x = self.classifier(x)
        x = torch.squeeze(x)
        return x

