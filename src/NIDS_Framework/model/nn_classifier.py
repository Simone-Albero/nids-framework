import torch
import torch.nn as nn


class NNClassifier(nn.Module):
    
    __slots__ = [
        "input_encoding",
        "transformer_block",
        "classification_head",
    ]

    def __init__(self, input_encoding: nn.Module, transformer_block: nn.Module, classification_head: nn.Module) -> None:
        super(NNClassifier, self).__init__()

        self.input_encoding: nn.Module = input_encoding
        self.transformer_block: nn.Module = transformer_block
        self.classification_head: nn.Module = classification_head
        
        
    def forward(self, x) -> torch.Tensor:
        x = self.input_encoding(x)
        x = self.transformer_block(x)
        x = self.classification_head(x)
        return x
