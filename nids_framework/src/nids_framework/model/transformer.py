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

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1) -> None:
        super(InputEmbedding, self).__init__()
        self.embedding: nn.Module = nn.Linear(input_dim, latent_dim)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class ClassificationHead(BaseModule):

    __slots__ = [
        "classifier",
        "dropout",
        "output_dim",
    ]

    def __init__(self, latent_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super(ClassificationHead, self).__init__()
        self.classifier: nn.Module = nn.Linear(latent_dim, output_dim)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, -1, :]  # last token classification
        x = self.dropout(x)
        x = self.classifier(x)

        x = x.squeeze(-1) if self.output_dim == 1 else x
        return x
    

class TransformerEncoder(BaseModule):

    __slots__ = [
        "encoder",
        "positional_encoding",
    ]

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        seq_length: int = 10,
    ) -> None:
        super(TransformerEncoder, self).__init__()
        
        #self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder: nn.Module = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: Tensor) -> Tensor:
        # x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        return x


class TransformerDecoder(BaseModule):

    __slots__ = [
        "decoder",
        "pos_encoder",
    ]

    def __init__(
        self,
        model_dim: int,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerDecoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        self.decoder: nn.Module = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        decoded = self.decoder(tgt, memory)
        return decoded



class TransformerClassifier(BaseModule):

    __slots__ = [
        "embedding",
        "encoder",
        "classifier",
    ]

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        seq_length: int = 10,
    ) -> None:
        super(TransformerClassifier, self).__init__()
        self.embedding = InputEmbedding(input_dim, model_dim, dropout)
        self.encoder = TransformerEncoder(
            model_dim, num_heads, num_layers, ff_dim, dropout, seq_length
        )

        self.classifier = ClassificationHead(model_dim, output_dim, dropout)

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
        "numeric_head",
        "categorical_head",
        "noise_factor",
    ]
        
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.1,
        seq_length: int = 10,
        noise_factor: float = 0.1,
        numeric_dim: int = None,
        categorical_dim: int = None,
    ):
        super(TransformerAutoencoder, self).__init__()

        self.embedding = InputEmbedding(input_dim, model_dim, dropout)

        self.encoder = TransformerEncoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            seq_length=seq_length,
        )

        self.decoder = TransformerDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.numeric_head = nn.Linear(model_dim, numeric_dim) if numeric_dim else None
        self.categorical_head = nn.Linear(model_dim, categorical_dim) if categorical_dim else None

        self.noise_factor = noise_factor

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.embedding(x)
        x_noisy = x + self.noise_factor * torch.randn_like(x)

        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded, x)

        decoded_numeric = self.numeric_head(decoded) if self.numeric_head else None
        decoded_categorical = self.categorical_head(decoded) if self.categorical_head else None

        return decoded_numeric, decoded_categorical
