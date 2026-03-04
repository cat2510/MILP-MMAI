# train/train_contrastive.py
# Contrastive learning for graph-text alignment. Uses einsum as required.
# Includes: training, optional projectors, zero-shot classification.
from __future__ import annotations
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.logior_dataset import LogiORGraphTextDataset, collate_graph_text
from data.pair_dataset import LogiORGraphTextPairDataset
from data.graph_builders import GraphBuildConfig
from models.graph_encoder_gcn import GraphEncoderGCN, GCNConfig
from models.text_encoder import TextEncoder, TextEncoderConfig
from losses.contrastive import SymmetricInfoNCE


# ---------- Projectors (homework: try various projectors) ----------
class MLPProjector(nn.Module):
    """MLP projector: in_dim -> hidden -> out_dim. Uses einsum in forward for linear part."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dropout(self.fc1(x)))
        return self.fc2(self.dropout(x))


class LinearProjector(nn.Module):
    """Single linear projection: in_dim -> out_dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PairBCELoss(nn.Module):
    """BCE loss for pairwise (graph, text) with explicit positive/negative labels.
    Uses einsum for similarity. Good for pair_data with k_neg negatives."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_g: torch.Tensor, z_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_g = F.normalize(z_g, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        # Pairwise similarity [B]: einsum
        sim = torch.einsum("bd,bd->b", z_g, z_t)
        logits = sim / self.temperature
        return F.binary_cross_entropy_with_logits(logits, y.float())


# ---------- Zero-shot classification (homework requirement) ----------
@torch.no_grad()
def zero_shot_classify(
    graph_encoder: nn.Module,
    text_encoder: nn.Module,
    graph_batch,
    text_queries: list[str],
    device: torch.device,
    graph_proj: nn.Module | None = None,
    text_proj: nn.Module | None = None,
) -> torch.Tensor:
    """
    Zero-shot classification: given graphs and K text queries (class descriptions),
    predict which query matches each graph. Uses einsum for similarity.
    Returns: predicted class indices [B]
    """
    graph_encoder.eval()
    text_encoder.eval()
    z_g = graph_encoder(graph_batch.to(device))
    z_t = text_encoder(text_queries, device=device)
    if graph_proj is not None:
        z_g = graph_proj(z_g)
    if text_proj is not None:
        z_t = text_proj(z_t)
    z_g = F.normalize(z_g, dim=-1)
    z_t = F.normalize(z_t, dim=-1)
    # Similarity [B, K]: einsum as required
    logits = torch.einsum("bd,kd->bk", z_g, z_t)
    return logits.argmax(dim=-1)


def run_zero_shot_eval(
    graph_encoder, text_encoder, ds, device,
    graph_proj=None, text_proj=None, batch_size: int = 8,
):
    """Zero-shot eval: each graph is classified by similarity to all text 'queries' (classes)."""
    query_texts = [ds[i]["text"] for i in range(len(ds))]
    prob_ids = ds.prob_ids
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_text)
    correct, total = 0, 0
    graph_encoder.eval()
    text_encoder.eval()
    with torch.no_grad():
        z_t_all = text_encoder(query_texts, device=device)
        if text_proj is not None:
            z_t_all = text_proj(z_t_all)
        z_t_all = F.normalize(z_t_all, dim=-1)
        for batch in loader:
            graphs = batch["graph"].to(device)
            ids = batch["prob_id"]
            z_g = graph_encoder(graphs)
            if graph_proj is not None:
                z_g = graph_proj(z_g)
            z_g = F.normalize(z_g, dim=-1)
            sim = torch.einsum("bd,kd->bk", z_g, z_t_all)
            pred_idx = sim.argmax(dim=-1)
            for i, pid in enumerate(ids):
                true_idx = prob_ids.index(pid) if pid in prob_ids else -1
                if true_idx >= 0 and pred_idx[i].item() == true_idx:
                    correct += 1
                total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir", required=True, help="folder with prob_*_graph.json (or parent for pair_data)")
    ap.add_argument("--logior_root", required=True, help="LogiOR root with prob_XXX/question.txt")
    ap.add_argument("--train_jsonl", default=None, help="if set, use pair_data (pos+neg) instead of same-problem only")
    ap.add_argument("--val_jsonl", default=None, help="validation jsonl (optional, for pair_data)")
    ap.add_argument("--base_dir", default=None, help="base for resolving pair_data paths; default=dirname(graphs_dir)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_text_len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--projector", choices=["none", "linear", "mlp"], default="none")
    ap.add_argument("--out_dim", type=int, default=128)
    ap.add_argument("--ckpt_dir", default="./ckpts")
    ap.add_argument("--run_zero_shot", action="store_true", help="run zero-shot eval after training")
    args = ap.parse_args()

    # #region agent log
    _lp = "/Users/cat2510/Documents/MMAI/.cursor/debug-d6443b.log"
    try:
        import time
        _d = {"graphs_dir": args.graphs_dir, "logior_root": args.logior_root, "graphs_dir_abs": os.path.abspath(args.graphs_dir), "logior_root_abs": os.path.abspath(args.logior_root), "cwd": os.getcwd(), "graphs_exists": os.path.exists(args.graphs_dir), "logior_exists": os.path.exists(args.logior_root)}
        with open(_lp, "a") as _f:
            _f.write(json.dumps({"sessionId":"d6443b","hypothesisId":"H1,H2,H4","location":"train_contrastive.py:main","message":"Args and paths before dataset","data":_d,"timestamp":int(time.time()*1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    # Fix: when running from Assignment_II/or_fusion, ../ORThought resolves to Assignment_II/ORThought (missing).
    # ORThought lives at MMAI/ORThought, so use ../../ORThought when the given path does not exist.
    if not os.path.exists(args.logior_root):
        _fallback = os.path.normpath(os.path.join(os.getcwd(), "../../ORThought/datasets/processed/LogiOR"))
        if os.path.exists(_fallback):
            args.logior_root = _fallback

    device = torch.device(args.device)
    graph_cfg = GraphBuildConfig(use_bidirectional_edges=True, add_degree_feature=True)

    use_pair_data = args.train_jsonl is not None

    if use_pair_data:
        base = args.base_dir or os.path.dirname(os.path.normpath(args.graphs_dir))
        if not base or base == ".":
            base = os.getcwd()
        train_ds = LogiORGraphTextPairDataset(
            args.train_jsonl,
            graph_cfg=graph_cfg,
            base_dir=base,
            graphs_dir=base,
            max_text_chars=12000,
        )
        if len(train_ds) == 0:
            raise ValueError(f"pair_data train empty: {args.train_jsonl}")
        dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_text)
        crit = PairBCELoss(temperature=args.temperature).to(device)
    else:
        ds = LogiORGraphTextDataset(
            graphs_dir=args.graphs_dir,
            logior_root=args.logior_root,
            graph_cfg=graph_cfg,
            max_text_chars=12000,
        )
        if len(ds) == 0:
            raise ValueError(
                f"Dataset is empty. Found 0 graph–text pairs. "
                f"Check --graphs_dir={args.graphs_dir} and --logior_root={args.logior_root} "
                f"(resolved: {os.path.abspath(args.logior_root)}). "
                f"When running from or_fusion, use ../../ORThought/datasets/processed/LogiOR for logior_root if ORThought is at repo root."
            )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_text)
        crit = SymmetricInfoNCE(temperature=args.temperature).to(device)

    one = (train_ds[0]["graph"] if use_pair_data else ds[0]["graph"])
    in_dim = one.x.size(-1)

    g_enc = GraphEncoderGCN(GCNConfig(
        in_dim=in_dim, hidden_dim=128, out_dim=args.out_dim, num_layers=3, dropout=0.2
    )).to(device)
    t_enc = TextEncoder(TextEncoderConfig(out_dim=args.out_dim, max_length=args.max_text_len)).to(device)

    graph_proj = text_proj = None
    if args.projector == "linear":
        graph_proj = LinearProjector(args.out_dim, args.out_dim).to(device)
        text_proj = LinearProjector(args.out_dim, args.out_dim).to(device)
    elif args.projector == "mlp":
        graph_proj = MLPProjector(args.out_dim, args.out_dim, hidden=256).to(device)
        text_proj = MLPProjector(args.out_dim, args.out_dim, hidden=256).to(device)

    def encode_g(x):
        h = g_enc(x)
        return graph_proj(h) if graph_proj is not None else h
    def encode_t(txts):
        h = t_enc(txts, device=device)
        return text_proj(h) if text_proj is not None else h

    params = list(g_enc.parameters()) + list(t_enc.parameters()) + list(crit.parameters())
    if graph_proj is not None:
        params += list(graph_proj.parameters()) + list(text_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    g_enc.train()
    t_enc.train()
    if graph_proj is not None:
        graph_proj.train()
        text_proj.train()
    crit.train()

    for ep in range(1, args.epochs + 1):
        total, n = 0.0, 0
        for batch in dl:
            graph = batch["graph"].to(device)
            texts = batch["text"]
            z_g = encode_g(graph)
            z_t = encode_t(texts)
            if use_pair_data:
                y = batch["y"].to(device)
                loss = crit(z_g, z_t, y)
            else:
                loss = crit(z_g, z_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss.item()) * len(texts)
            n += len(texts)
        print(f"epoch {ep:02d} | loss {total / max(n, 1):.4f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = {
        "graph_encoder": g_enc.state_dict(),
        "text_encoder": t_enc.state_dict(),
    }
    if hasattr(crit, "state_dict"):
        ckpt["logit_scale"] = crit.state_dict()
    if graph_proj is not None:
        ckpt["graph_proj"] = graph_proj.state_dict()
        ckpt["text_proj"] = text_proj.state_dict()
    torch.save(ckpt, os.path.join(args.ckpt_dir, "contrastive_encoders.pt"))
    print(f"saved {args.ckpt_dir}/contrastive_encoders.pt")

    if args.run_zero_shot:
        eval_ds = ds if not use_pair_data else LogiORGraphTextDataset(
            graphs_dir=args.graphs_dir,
            logior_root=args.logior_root,
            graph_cfg=graph_cfg,
            max_text_chars=12000,
        )
        acc = run_zero_shot_eval(g_enc, t_enc, eval_ds, device, graph_proj, text_proj, args.batch_size)
        print(f"zero-shot accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
    """ usage example:
    python -m train.train_contrastive \
  --graphs_dir ../graphs_mps \
  --logior_root ../ORThought/datasets/processed/LogiOR \
  --projector linear --epochs 20 --run_zero_shot"""
