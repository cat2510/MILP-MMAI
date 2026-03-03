# models/graph_encoder_gap.py
# OptiGuide-style bipartite GNN for MILP graphs.
# Adapted from: https://github.com/microsoft/OptiGuide/blob/main/milp-evolve/src/multi_class_learning/gap_model.py
# Uses bipartite message passing (constraint <-> variable) and self-attention pooling.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.nn import GATConv
except Exception as e:
    raise ImportError("Please install torch-geometric to use this encoder.") from e


def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BipartiteGraphConvolution(MessagePassing):
    """Message passing between variable (col) and constraint (row) nodes."""

    def __init__(
        self,
        emb_size: int = 64,
        edge_dim: int = 2,
        do_gat: bool = False,
        heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__(aggr="add")
        self.emb_size = emb_size
        self.feature_module_left = nn.Linear(emb_size, emb_size)
        self.feature_module_edge = nn.Linear(edge_dim, emb_size, bias=False)
        self.feature_module_right = nn.Linear(emb_size, emb_size, bias=False)
        self.do_gat = do_gat
        if do_gat:
            self.gat_conv = GATConv(
                emb_size, emb_size, heads=heads, edge_dim=edge_dim,
                concat=True, dropout=dropout
            )
        self.feature_module_final = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = nn.LayerNorm(emb_size)
        self.output_module = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_index, edge_features, right_features):
        out = self.propagate(
            edge_index,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(out), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        if self.do_gat:
            out = self.feature_module_final(
                self.gat_conv(node_features_i, node_features_j, edge_features)
            )
        else:
            out = self.feature_module_final(
                self.feature_module_left(node_features_i)
                + self.feature_module_edge(edge_features)
                + self.feature_module_right(node_features_j)
            )
        return out


@dataclass
class GAPConfig:
    row_dim: int
    col_dim: int
    edge_dim: int = 2
    emb_size: int = 64
    out_dim: int = 128
    n_gnn_iters: int = 2
    n_attn_iters: int = 1
    do_gat: bool = False
    heads: int = 8
    dropout: float = 0.3
    max_token_attn: int = 512


def _pyg_batch_to_bipartite(data) -> tuple:
    """
    Convert our PyG Batch (vars first, then constrs; edge constr->var)
    to OptiGuide format: x_rows (constrs), x_cols (vars), edge_index_rowcols,
    edge_vals, x_rows_batch, x_cols_batch.
    Requires node_type: 0=var, 1=constr (from graph_builders).
    """
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else torch.ones(edge_index.size(1), 1, device=x.device, dtype=x.dtype)
    batch = data.batch
    node_type = data.node_type  # 0=var, 1=constr (from graph_builders)

    var_mask = node_type == 0
    con_mask = node_type == 1

    x_cols = x[var_mask]
    x_rows = x[con_mask]
    x_cols_batch = batch[var_mask]
    x_rows_batch = batch[con_mask]

    # Build global -> row/col index maps
    global_to_col = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
    global_to_row = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
    global_to_col[var_mask] = torch.arange(x_cols.size(0), device=x.device)
    global_to_row[con_mask] = torch.arange(x_rows.size(0), device=x.device)

    # edge_index: src=constr, dst=var
    src, dst = edge_index[0], edge_index[1]
    row_idx = global_to_row[src]
    col_idx = global_to_col[dst]
    valid = (row_idx >= 0) & (col_idx >= 0)
    row_idx, col_idx = row_idx[valid], col_idx[valid]
    edge_index_rowcols = torch.stack([row_idx, col_idx], dim=0)
    edge_vals = edge_attr[valid]

    if edge_vals.size(-1) == 1:
        edge_vals = torch.cat([edge_vals, edge_vals.abs() / (edge_vals.abs().sum(dim=0, keepdim=True) + 1e-9)], dim=-1)
    if edge_vals.size(-1) > 2:
        edge_vals = edge_vals[:, :2]

    return x_rows, x_cols, edge_index_rowcols, edge_vals, x_rows_batch, x_cols_batch


class GraphEncoderGAP(nn.Module):
    """
    OptiGuide-style bipartite GNN for MILP graphs.
    Input: PyG Batch from our graph_builders (x, edge_index, edge_attr, batch, n_vars, n_cons).
    Output: [B, out_dim]
    """

    def __init__(self, cfg: GAPConfig):
        super().__init__()
        self.cfg = cfg
        norm_fn = nn.LayerNorm
        self.row_embedding = nn.Sequential(
            norm_fn(cfg.row_dim),
            nn.Linear(cfg.row_dim, cfg.emb_size),
            nn.ReLU(),
            nn.Linear(cfg.emb_size, cfg.emb_size),
            nn.ReLU(),
        )
        self.col_embedding = nn.Sequential(
            norm_fn(cfg.col_dim),
            nn.Linear(cfg.col_dim, cfg.emb_size),
            nn.ReLU(),
            nn.Linear(cfg.emb_size, cfg.emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential(norm_fn(cfg.edge_dim))

        ConvLayer = BipartiteGraphConvolution
        kw = dict(emb_size=cfg.emb_size, edge_dim=cfg.edge_dim, do_gat=cfg.do_gat, heads=cfg.heads, dropout=cfg.dropout)
        if cfg.n_gnn_iters == 1:
            self.conv_col_to_row = ConvLayer(**kw)
            self.conv_row_to_col = ConvLayer(**kw)
            self.conv_layers = None
        else:
            self.conv_col_to_row = None
            self.conv_row_to_col = None
            self.conv_layers = nn.ModuleList()
            for _ in range(cfg.n_gnn_iters):
                self.conv_layers.append(nn.ModuleDict({
                    "col_to_row": ConvLayer(**kw),
                    "row_to_col": ConvLayer(**kw),
                }))

        self.col_output = nn.Sequential(nn.Linear(cfg.emb_size, cfg.emb_size), nn.ReLU())
        self.row_output = nn.Sequential(nn.Linear(cfg.emb_size, cfg.emb_size), nn.ReLU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_size, nhead=8, batch_first=True, dropout=cfg.dropout
        )
        self.self_attn = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_attn_iters, norm=nn.LayerNorm(cfg.emb_size)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(cfg.emb_size, cfg.emb_size),
            nn.ReLU(),
            nn.Linear(cfg.emb_size, cfg.out_dim),
        )
        self.apply(_init_weights)

    def _special_attn_pool(
        self, row_embd, col_embd, x_rows_batch, x_cols_batch
    ) -> torch.Tensor:
        batch_size = x_rows_batch.max().item() + 1
        outputs = []
        for i in range(batch_size):
            row_mask = x_rows_batch == i
            col_mask = x_cols_batch == i
            row_tokens = row_embd[row_mask]
            col_tokens = col_embd[col_mask] + 0.5
            means_row = row_tokens.mean(dim=0, keepdim=True) * 10.0
            means_col = col_tokens.mean(dim=0, keepdim=True) * 10.0
            tokens = torch.cat([row_tokens, col_tokens], dim=0)
            if tokens.size(0) > self.cfg.max_token_attn:
                perm = torch.randperm(tokens.size(0), device=tokens.device)[: self.cfg.max_token_attn]
                tokens = tokens[perm]
            special = torch.zeros(1, row_embd.size(1), device=row_embd.device)
            agg = torch.cat([tokens, means_row, means_col, special], dim=0).unsqueeze(0)
            out = self.self_attn(agg)
            outputs.append(out[0, -1, :])
        return torch.stack(outputs, dim=0)

    def forward(self, data) -> torch.Tensor:
        x_rows, x_cols, edge_index_rowcols, edge_vals, x_rows_batch, x_cols_batch = _pyg_batch_to_bipartite(data)

        # NaN/Inf safety
        x_rows = torch.nan_to_num(x_rows, nan=0.0, posinf=0.0, neginf=0.0)
        x_cols = torch.nan_to_num(x_cols, nan=0.0, posinf=0.0, neginf=0.0)
        edge_vals = torch.nan_to_num(edge_vals, nan=0.0, posinf=0.0, neginf=0.0)

        if edge_vals.size(-1) < self.cfg.edge_dim:
            edge_vals = torch.nn.functional.pad(edge_vals, (0, self.cfg.edge_dim - edge_vals.size(-1)))
        edge_vals = edge_vals[:, : self.cfg.edge_dim]

        row_embd = self.row_embedding(x_rows)
        col_embd = self.col_embedding(x_cols)
        edge_embd = self.edge_embedding(edge_vals)

        r_edge = torch.stack([edge_index_rowcols[1], edge_index_rowcols[0]], dim=0)

        if self.conv_layers is not None:
            for layer in self.conv_layers:
                row_embd = row_embd + layer["col_to_row"](col_embd, r_edge, edge_embd, row_embd)
                col_embd = col_embd + layer["row_to_col"](row_embd, edge_index_rowcols, edge_embd, col_embd)
        else:
            row_embd = self.conv_col_to_row(col_embd, r_edge, edge_embd, row_embd)
            col_embd = self.conv_row_to_col(row_embd, edge_index_rowcols, edge_embd, col_embd)

        row_embd = self.row_output(row_embd)
        col_embd = self.col_output(col_embd)

        z = self._special_attn_pool(row_embd, col_embd, x_rows_batch, x_cols_batch)
        return self.out_proj(z)
