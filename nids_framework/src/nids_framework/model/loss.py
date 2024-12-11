import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        original, reconstructed = inputs
        return self.mse_loss(reconstructed, original)
    
class HybridReconstructionLoss(nn.Module):
    __slots__ = ["_mse_loss", "_bce_loss", "_border"]

    def __init__(self, border: int):
        super(HybridReconstructionLoss, self).__init__()
        self._mse_loss: nn.MSELoss = nn.MSELoss()
        self._bce_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self._border: int = border

    def forward(
        self,
        reconstructed: tuple[torch.Tensor, torch.Tensor],
        original: torch.Tensor,
    ) -> torch.Tensor:
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
