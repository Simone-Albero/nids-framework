from typing import List

import torch.nn as nn

from model.input_encoder import InputEncoder
from model.transformer import TransformerEncoderOnlyModel
from model.classification_head import ClassificationHead



class NNClassifier(nn.Module):
    
    __slots__ = [
        "input_encoding",
        "transformer_block",
        "classification_head",
    ]

    def __init__(self, input_dim, num_heads=8, num_layers=3, dim_feedforward=2048, output_dim=9, dropout=0.1):
        super(NNClassifier, self).__init__()

        self.input_encoding = InputEncoder(input_dim)
        self.transformer_block = TransformerEncoderOnlyModel(80, num_heads, num_layers, dim_feedforward, dropout)
        self.classification_head = ClassificationHead(80, output_dim)
        
        
    def forward(self, x):
        x = self.input_encoding(x)
        x = self.transformer_block(x)
        x = self.classification_head(x)
        return x
