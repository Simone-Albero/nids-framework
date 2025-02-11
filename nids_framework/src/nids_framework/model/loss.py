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
    
class HybridReconstructionLoss(nn.Module):
    def __init__(self, lambda_num=1.0, lambda_cat=1.0, categorical_levels=32):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_num = lambda_num
        self.lambda_cat = lambda_cat
        self.categorical_levels = categorical_levels

    def forward(self, num_recon, num_target, cat_recon, cat_target):
        num_recon = num_recon[:, -1, :]
        num_target = num_target[:, -1, :]

        cat_recon = cat_recon[:, -1, :]
        cat_target = cat_target[:, -1, :]
        cat_recon = cat_recon.view(cat_target.shape[0], self.categorical_levels, cat_target.shape[1])

        num_loss = self.mse_loss(num_recon, num_target)
        cat_loss = self.ce_loss(cat_recon, cat_target)
        
        return self.lambda_num * num_loss + self.lambda_cat * cat_loss