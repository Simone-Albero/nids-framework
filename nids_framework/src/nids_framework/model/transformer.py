import logging
import os

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

        curr_device = next(self.parameters()).device
        self.to('cpu')
        torch.save(self.state_dict(), f_path)
        self.to(curr_device)
        logging.info("Model weights saved successfully")

    def load_model_weights(self, f_path: str = "saves/model.pt", map_location: str = "cpu") -> None:
        logging.info("Loading model weights...")
        self.load_state_dict(torch.load(f_path, map_location=map_location), strict=False)
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
        x = x[..., -1, :]  # last token classification
        x = self.dropout(x)
        x = self.classifier(x)

        if self.num_classes == 1:
            x = torch.sigmoid(x).squeeze(-1)
        else:
            x = F.softmax(x, dim=-1)

        return x


class TransformerEncoder(BaseModule):

    __slots__ = [
        "encoder",
    ]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerEncoder, self).__init__()
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
        x = self.encoder(x)
        return x


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
    ) -> None:
        super(TransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.embedding = InputEmbedding(input_dim, embed_dim, dropout)
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers, ff_dim, dropout
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
        "reconstructor",
        "dropout",
        "border",
    ]

    def __init__(
        self,
        input_dim: int,
        border: int,
        embed_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerAutoencoder, self).__init__()
        self.embedding = InputEmbedding(input_dim, embed_dim, dropout)
        self.encoder = TransformerEncoder(
            embed_dim, num_heads, num_layers, ff_dim, dropout
        )
        self.reconstructor = nn.Linear(embed_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.border = border

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.reconstructor(x)

        reconstructed_numeric = x[:, : self.border]
        reconstructed_categorical = x[:, self.border :]

        return reconstructed_numeric, reconstructed_categorical
