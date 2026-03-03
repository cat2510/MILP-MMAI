# models/text_encoder.py
# Text encoder for graph-text matching. Supports sentence-transformers (stronger) or DistilBERT.
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging

""" usage example:
python -m train.train_contrastive --graphs_dir ./graphs_mps --logior_root ../ORThought/datasets/processed/LogiOR --batch_size 8 --epochs 20
"""


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-pad tokens (sentence-transformer style)."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@dataclass
class TextEncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    out_dim: int = 128
    max_length: int = 256
    dropout: float = 0.1
    use_mean_pool: bool = True  # True for sentence-transformers; False = CLS token


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        hf_logging.set_verbosity(prev_level)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.out_dim),
        )

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(device)

        out = self.backbone(**enc)
        h = out.last_hidden_state
        if self.cfg.use_mean_pool:
            pooled = _mean_pool(h, enc["attention_mask"])
        else:
            pooled = h[:, 0, :]
        z = self.proj(pooled)
        return z