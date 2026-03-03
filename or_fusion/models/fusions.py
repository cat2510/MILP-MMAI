# models/fusions.py
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusion(nn.Module):
    """
    Early fusion = fuse at feature level before prediction.
    Here: concatenate embeddings then MLP.
    """
    def __init__(self, d_g: int, d_t: int, d_out: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_g + d_t, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, z_g: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_g, z_t], dim=-1)
        return self.mlp(x)


class LateFusion(nn.Module):
    """
    Late fusion = separate unimodal predictors, then combine at decision level.
    Here: weighted sum of logits (learned scalar gate).
    """
    def __init__(self, d_g: int, d_t: int, d_out: int, dropout: float = 0.2):
        super().__init__()
        self.head_g = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_g, d_out))
        self.head_t = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_t, d_out))
        # learn a mixing weight alpha in (0,1)
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, z_g: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        lg = self.head_g(z_g)
        lt = self.head_t(z_t)
        alpha = torch.sigmoid(self.logit_alpha)
        return alpha * lg + (1 - alpha) * lt


class TensorFusion(nn.Module):
    """
    Tensor Fusion (Zadeh et al. style):
      z~ = [1; z]
      fused = outer(z_g~, z_t~) flattened
    Then linear classifier.

    Uses einsum for the outer product.
    Complexity: O((d_g+1)*(d_t+1)) features.
    """
    def __init__(self, d_g: int, d_t: int, d_out: int, dropout: float = 0.2):
        super().__init__()
        self.d_g = d_g
        self.d_t = d_t
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear((d_g + 1) * (d_t + 1), d_out)

    def forward(self, z_g: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        B = z_g.size(0)
        ones = torch.ones(B, 1, device=z_g.device, dtype=z_g.dtype)
        zg = torch.cat([ones, z_g], dim=-1)  # [B, d_g+1]
        zt = torch.cat([ones, z_t], dim=-1)  # [B, d_t+1]
        # outer per batch: [B, d_g+1, d_t+1]
        outer = torch.einsum("bi,bj->bij", self.dropout(zg), self.dropout(zt))
        fused = outer.reshape(B, -1)
        return self.classifier(fused)


class LowRankTensorFusion(nn.Module):
    """
    Low-rank / LMF fusion: approximate tensor fusion with rank R.

    Idea: project each modality into rank-R factors and combine via elementwise product,
    then sum across rank.

    Uses einsum:
      g = einsum('bd,drm->brm')  (per-rank factors)
      t = einsum('bd,drm->brm')
      fused = sum_r (g_r * t_r)  -> [B, m]
      logits = fused @ W_out + b

    Here we produce fused dimension m = d_fused.
    """
    def __init__(self, d_g: int, d_t: int, d_out: int, rank: int = 8, d_fused: int = 256, dropout: float = 0.2):
        super().__init__()
        self.rank = rank
        self.d_fused = d_fused
        self.dropout = nn.Dropout(dropout)

        # include bias trick by augmenting inputs with 1
        self.U_g = nn.Parameter(torch.randn(d_g + 1, rank, d_fused) * 0.02)
        self.U_t = nn.Parameter(torch.randn(d_t + 1, rank, d_fused) * 0.02)

        self.W_out = nn.Linear(d_fused, d_out)

    def forward(self, z_g: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        B = z_g.size(0)
        ones = torch.ones(B, 1, device=z_g.device, dtype=z_g.dtype)
        zg = torch.cat([ones, z_g], dim=-1)  # [B, d_g+1]
        zt = torch.cat([ones, z_t], dim=-1)  # [B, d_t+1]

        zg = self.dropout(zg)
        zt = self.dropout(zt)

        # factors: [B, R, M]
        g_fac = torch.einsum("bd,drm->brm", zg, self.U_g)
        t_fac = torch.einsum("bd,drm->brm", zt, self.U_t)

        fused = (g_fac * t_fac).sum(dim=1)  # sum over rank -> [B, M]
        return self.W_out(fused)