import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridReconstructionLoss(nn.Module):
    __slots__ = ["mse_loss", "bce_loss", "border"]

    def __init__(self, border: int):
        super(HybridReconstructionLoss, self).__init__()
        self.mse_loss: nn.Module = nn.MSELoss()
        self.bce_loss: nn.Module = nn.BCEWithLogitsLoss()
        self.border = border

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
    ):
        reconstructed_numeric, reconstructed_categorical = reconstructed
        
        # Loss for numeric features (MSE)
        numeric_loss = self.mse_loss(reconstructed_numeric, original[:, :self.border])
        # Loss for categorical features (BCE)
        categorical_loss = self.bce_loss(
            reconstructed_categorical, original[:, self.border:]
        )

        total_loss = numeric_loss + categorical_loss
        return total_loss
