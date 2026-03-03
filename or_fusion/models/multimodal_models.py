# models/multimodal_models.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

from models.fusions import EarlyFusion, LateFusion, TensorFusion, LowRankTensorFusion


@dataclass
class FusionConfig:
    name: str  # "early" | "late" | "tensor" | "lmf"
    hidden: int = 256
    dropout: float = 0.2
    rank: int = 8           # for LMF
    d_fused: int = 256      # for LMF


class GraphTextClassifier(nn.Module):
    def __init__(self, graph_encoder: nn.Module, text_encoder: nn.Module,
                 d_g: int, d_t: int, n_classes: int, cfg: FusionConfig):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder

        if cfg.name == "early":
            self.fusion = EarlyFusion(d_g, d_t, n_classes, hidden=cfg.hidden, dropout=cfg.dropout)
        elif cfg.name == "late":
            self.fusion = LateFusion(d_g, d_t, n_classes, dropout=cfg.dropout)
        elif cfg.name == "tensor":
            self.fusion = TensorFusion(d_g, d_t, n_classes, dropout=cfg.dropout)
        elif cfg.name == "lmf":
            self.fusion = LowRankTensorFusion(d_g, d_t, n_classes, rank=cfg.rank, d_fused=cfg.d_fused, dropout=cfg.dropout)
        else:
            raise ValueError(f"Unknown fusion: {cfg.name}")

    def forward(self, graph, text_list, device=None):
        z_g = self.graph_encoder(graph)                 # [B, d_g]
        z_t = self.text_encoder(text_list, device=device)  # [B, d_t]
        logits = self.fusion(z_g, z_t)
        return logits