import torch
import torch.nn as nn


class HybridReconstructionLoss(nn.Module):
    __slots__ = ["_mse_loss", "_bce_loss", "_border"]

    def __init__(self, border: int):
        super(HybridReconstructionLoss, self).__init__()
        self._mse_loss: nn.Module = nn.MSELoss()
        self._bce_loss: nn.Module = nn.BCEWithLogitsLoss()
        self._border = border

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
    ) -> float:
        reconstructed_numeric, reconstructed_categorical = reconstructed

        # Loss for numeric features (MSE)
        numeric_loss = self._mse_loss(
            reconstructed_numeric, original[:, : self._border]
        )
        # Loss for categorical features (BCE)
        categorical_loss = self._bce_loss(
            reconstructed_categorical, original[:, self._border :]
        )

        total_loss = numeric_loss + categorical_loss
        return total_loss