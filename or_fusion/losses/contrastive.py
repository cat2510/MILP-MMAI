# losses/contrastive.py
# CLIP-style symmetric InfoNCE (graph↔text). This is the lightweight “contrastive alignment” baseline.
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log())

    def forward(self, z_g: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        # Normalize
        z_g = F.normalize(z_g, dim=-1)
        z_t = F.normalize(z_t, dim=-1)

        scale = self.logit_scale.exp().clamp(1e-3, 1e3)
        # Similarity matrix [B,B]: einsum as required for homework
        logits = scale * torch.einsum("bd,td->bt", z_g, z_t)

        targets = torch.arange(logits.size(0), device=logits.device)
        loss_g2t = F.cross_entropy(logits, targets)
        loss_t2g = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_g2t + loss_t2g)