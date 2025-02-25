import torch
import torch.nn as nn


class HybridReconstructionLoss(nn.Module):
    __slots__ = [
        "mse_loss",
        "ce_loss",
        "lambda_num",
        "lambda_cat",
        "categorical_levels",
    ]

    def __init__(
        self,
        lambda_num: float = 1.0,
        lambda_cat: float = 1.0,
        categorical_levels: int = 32,
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_num = lambda_num
        self.lambda_cat = lambda_cat
        self.categorical_levels = categorical_levels

    def forward(
        self,
        num_recon: torch.Tensor,
        num_target: torch.Tensor,
        cat_recon: torch.Tensor,
        cat_target: torch.Tensor,
    ) -> torch.Tensor:
        # Extract the last token
        num_recon, num_target = num_recon[:, -1, :], num_target[:, -1, :]
        cat_recon, cat_target = cat_recon[:, -1, :], cat_target[:, -1, :]

        # Reshape categorical recon for cross-entropy loss
        cat_recon = cat_recon.view(
            cat_target.shape[0], self.categorical_levels, cat_target.shape[1]
        )

        num_loss = self.mse_loss(num_recon, num_target)
        cat_loss = self.ce_loss(cat_recon, cat_target)

        return self.lambda_num * num_loss + self.lambda_cat * cat_loss
