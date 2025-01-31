import logging
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

    
class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def save_model_weights(self, f_path: str = "saves/model.pt") -> None:
        logging.info("Saving model weights...")
        os.makedirs(os.path.dirname(f_path), exist_ok=True)

        # curr_device = next(self.parameters()).device
        # self.to('cpu')
        torch.save(self.state_dict(), f_path)
        # self.to(curr_device)
        logging.info("Model weights saved successfully")

    def load_model_weights(self, f_path: str = "saves/model.pt", map_location: str = "cpu") -> None:
        logging.info("Loading model weights...")
        self.load_state_dict(torch.load(f_path, map_location=map_location, weights_only=True))
        logging.info("Model weights loaded successfully")


class InputEmbedding(BaseModule):

    __slots__ = [
        "embedding",
        "dropout",
    ]

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1) -> None:
        super(InputEmbedding, self).__init__()
        self.embedding: nn.Module = nn.Linear(input_dim, embed_dim)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class ClassificationHead(BaseModule):

    __slots__ = [
        "classifier",
        "dropout",
        "num_classes",
    ]

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super(ClassificationHead, self).__init__()
        self.classifier: nn.Module = nn.Linear(embed_dim, num_classes)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, -1, :]  # last token classification
        x = self.dropout(x)
        x = self.classifier(x)

        x = x.squeeze(-1) if self.num_classes == 1 else x
        # x = torch.sigmoid(x).squeeze(-1)
        
        # else:
        #     x = F.softmax(x, dim=-1)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x
    

class TransformerEncoder(BaseModule):

    __slots__ = [
        "encoder",
        "pos_encoder",
    ]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        window_size: int = 10,
    ) -> None:
        super(TransformerEncoder, self).__init__()
        
        # self.pos_encoder = PositionalEncoding(embed_dim, window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder: nn.Module = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        # x = self.pos_encoder(x)
        x = self.encoder(x)
        return x


class TransformerDecoder(BaseModule):

    __slots__ = [
        "decoder",
        "pos_encoder",
    ]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        window_size: int = 10,
    ) -> None:
        super(TransformerDecoder, self).__init__()
        
        # self.pos_encoder = PositionalEncoding(embed_dim, window_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        self.decoder: nn.Module = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    def forward(self, encoded: Tensor, x: Tensor) -> Tensor:
        # x = self.pos_encoder(x)
        decoded = self.decoder(x, encoded)
        return decoded



class TransformerClassifier(BaseModule):

    __slots__ = [
        "num_classes",
        "embedding",
        "encoder",
        "classifier",
    ]

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        window_size: int = 10,
    ) -> None:
        super(TransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.embedding = InputEmbedding(input_dim, embed_dim, dropout)
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers, ff_dim, dropout, window_size
        )

        self.classifier = ClassificationHead(embed_dim, num_classes, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.classifier(x)

        return x

class TransformerAutoencoder(BaseModule):

    __slots__ = [
        "embedding",
        "encoder",
        "decoder",
        "noise_factor",
    ]
        
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        window_size: int = 10,
        noise_factor: float = 0.1
    ):
        super(TransformerAutoencoder, self).__init__()

        self.embedding = InputEmbedding(input_dim, embed_dim, dropout)

        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            window_size=window_size,
        )

        self.decoder = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            window_size=window_size,
        )

        self.noise_factor = noise_factor

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)

        x_noisy = x.clone()
        # x_noisy[:, -1, :] += self.noise_factor * torch.randn_like(x[:, -1, :])
        x_noisy = x + self.noise_factor * torch.randn_like(x)

        
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded, x_noisy)
        return decoded[:, -1, :], x[:, -1, :]