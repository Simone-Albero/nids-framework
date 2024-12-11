import torch
import torch.nn as nn

class MSEReconstructionLoss(nn.Module):
    def __init__(self):
        super(MSEReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        original, reconstructed = inputs
        original, reconstructed = original[..., -1], reconstructed[..., -1]
        return self.mse_loss(reconstructed, original)


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        original, reconstructed = inputs
        original, reconstructed = original[..., -1], reconstructed[..., -1]
        similarity = self.cosine_similarity(original, reconstructed)
        loss = -torch.mean(similarity)
        return loss