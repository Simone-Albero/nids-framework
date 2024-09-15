import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):

    __slots__ = [
        "num_classes",
        "embedding",
        "encoder",
        "pooling",
        "classifier",
        "dropout",
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
        self.embedding: nn.Module = nn.Linear(input_dim, embed_dim)
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
        self.pooling: nn.Module = nn.AdaptiveAvgPool1d(1)
        self.classifier: nn.Module = nn.Linear(embed_dim, num_classes)
        self.dropout: nn.Module = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)

        # Pooling operation: (batch_size, seq_len, embed_dim) -> (batch_size, embed_dim)
        x = self.pooling(x.permute(0, 2, 1)).squeeze(-1)
        
        x = self.dropout(x)
        x = self.classifier(x)  # (batch_size, num_classes)

        if self.num_classes == 1:
            x = torch.sigmoid(x).squeeze(-1)
        else:
            x = F.softmax(x, dim=-1)

        return x


class TransformerAutoencoder(nn.Module):

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
        self.embedding: nn.Module = nn.Linear(input_dim, embed_dim)
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
        self.reconstructor: nn.Module = nn.Linear(embed_dim, input_dim)
        self.dropout: nn.Module = nn.Dropout(dropout)
        self.border: int = border

    def forward(self, x) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.reconstructor(x)

        reconstructed_numeric = x[:, : self.border]
        reconstructed_categorical = x[:, self.border :]

        return reconstructed_numeric, reconstructed_categorical
