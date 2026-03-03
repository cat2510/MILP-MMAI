"""
Pilot fusion classification with flexible hyperparameters and metrics logging.
Logs: params, memory, convergence time, val_acc, val_auc.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.logior_dataset import collate_graph_text
from data.graph_builders import GraphBuildConfig
from models.graph_encoder_gcn import GraphEncoderGCN, GCNConfig
from models.graph_encoder_gap import GraphEncoderGAP, GAPConfig
from models.text_encoder import TextEncoder, TextEncoderConfig
from models.multimodal_models import GraphTextClassifier, FusionConfig
from data.pair_dataset import LogiORGraphTextPairDataset


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        graph = batch["graph"].to(device)
        texts = batch["text"]
        labels = batch["y"].to(device)
        logits = model(graph, texts, device=device)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    acc = ((y_prob >= 0.5).astype(float) == y_true).mean()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5
    return acc, auc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", required=True, choices=["early", "late", "tensor", "lmf"])
    ap.add_argument("--graph_encoder", choices=["gcn", "gap"], default="gcn")

    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--test_jsonl", default=None)
    ap.add_argument("--graphs_dir", required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--freeze_text", action="store_true", default=True, help="freeze text encoder backbone")
    ap.add_argument("--no_freeze_text", action="store_false", dest="freeze_text")

    ap.add_argument("--d_g", type=int, default=128)
    ap.add_argument("--d_t", type=int, default=128)

    # Fusion-specific
    ap.add_argument("--fusion_hidden", type=int, default=256, help="early fusion MLP hidden")
    ap.add_argument("--lmf_rank", type=int, default=8)
    ap.add_argument("--lmf_d_fused", type=int, default=256)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None, help="save run summary JSON here")
    ap.add_argument("--wandb", action="store_true", help="log to wandb (requires wandb installed)")
    return ap.parse_args()


def build_config(args):
    """Build FusionConfig from args."""
    return FusionConfig(
        name=args.fusion,
        hidden=args.fusion_hidden,
        dropout=args.dropout,
        rank=args.lmf_rank,
        d_fused=args.lmf_d_fused,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    graph_cfg = GraphBuildConfig()
    base = args.graphs_dir
    train_ds = LogiORGraphTextPairDataset(
        args.train_jsonl, graph_cfg=graph_cfg, base_dir=base, graphs_dir=base
    )
    val_ds = LogiORGraphTextPairDataset(
        args.val_jsonl, graph_cfg=graph_cfg, base_dir=base, graphs_dir=base
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_text
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graph_text
    )

    sample_graph = train_ds[0]["graph"]
    in_dim = sample_graph.x.size(-1)

    # Graph encoder
    if args.graph_encoder == "gap":
        graph_encoder = GraphEncoderGAP(GAPConfig(
            row_dim=in_dim, col_dim=in_dim, edge_dim=2,
            emb_size=64, out_dim=args.d_g, n_gnn_iters=2, n_attn_iters=1,
        )).to(device)
    else:
        graph_encoder = GraphEncoderGCN(GCNConfig(
            in_dim=in_dim, hidden_dim=128, out_dim=args.d_g, dropout=args.dropout
        )).to(device)

    text_encoder = TextEncoder(TextEncoderConfig(
        out_dim=args.d_t, dropout=args.dropout
    )).to(device)

    fusion_cfg = build_config(args)
    model = GraphTextClassifier(
        graph_encoder=graph_encoder,
        text_encoder=text_encoder,
        d_g=args.d_g,
        d_t=args.d_t,
        n_classes=train_ds.num_classes,
        cfg=fusion_cfg,
    ).to(device)

    if args.freeze_text:
        for p in model.text_encoder.backbone.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    # Param counts
    n_graph = count_params(model.graph_encoder)
    n_text = count_params(model.text_encoder)
    n_fusion = count_params(model.fusion)
    n_total = count_params(model)
    n_trainable = count_trainable_params(model)

    hparams = {
        "fusion": args.fusion,
        "fusion_hidden": args.fusion_hidden,
        "lmf_rank": args.lmf_rank,
        "lmf_d_fused": args.lmf_d_fused,
        "d_g": args.d_g,
        "d_t": args.d_t,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "freeze_text": args.freeze_text,
        "graph_encoder": args.graph_encoder,
    }
    params_summary = {
        "graph_encoder": n_graph,
        "text_encoder": n_text,
        "fusion": n_fusion,
        "total": n_total,
        "trainable": n_trainable,
    }

    print("=== Hyperparameters ===")
    for k, v in hparams.items():
        print(f"  {k}: {v}")
    print("=== Parameters ===")
    for k, v in params_summary.items():
        print(f"  {k}: {v:,}")

    n_train = len(train_ds)
    history = {"train_loss": [], "val_acc": [], "val_auc": [], "epoch_time": []}
    best_auc = 0.0
    epoch_to_converge = None
    target_auc = 0.75  # convergence threshold (optional)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="or_fusion", config={**hparams, **params_summary})
        except ImportError:
            pass

    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            graph = batch["graph"].to(device)
            texts = batch["text"]
            labels = batch["y"].to(device)
            logits = model(graph, texts, device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        val_acc, val_auc = evaluate(model, val_loader, device)
        t_epoch = time.perf_counter() - t_epoch

        history["train_loss"].append(total_loss / max(n_train, 1))
        history["val_acc"].append(float(val_acc))
        history["val_auc"].append(float(val_auc))
        history["epoch_time"].append(t_epoch)

        if val_auc > best_auc:
            best_auc = val_auc
        if epoch_to_converge is None and val_auc >= target_auc:
            epoch_to_converge = epoch

        print(f"[{args.fusion}] epoch {epoch:02d} | "
              f"train_loss={history['train_loss'][-1]:.4f} | "
              f"val_acc={val_acc:.4f} | val_auc={val_auc:.4f} | time={t_epoch:.1f}s")

        if wandb_run:
            wandb_run.log({"train_loss": history["train_loss"][-1], "val_acc": val_acc, "val_auc": val_auc, "epoch_time": t_epoch}, step=epoch)

    total_time = time.perf_counter() - t_start
    mem_mb = get_memory_mb(device)

    summary = {
        "hparams": hparams,
        "params": params_summary,
        "best_val_auc": float(best_auc),
        "best_val_acc": float(max(history["val_acc"])),
        "total_time_sec": total_time,
        "epoch_to_converge": epoch_to_converge,
        "memory_mb": mem_mb,
        "history": history,
    }

    print("\n=== Summary ===")
    print(f"  best_val_auc: {best_auc:.4f}")
    print(f"  total_time_sec: {total_time:.1f}")
    print(f"  memory_mb: {mem_mb:.1f}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        fname = f"run_{args.fusion}_{args.graph_encoder}.json"
        path = os.path.join(args.out_dir, fname)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to {path}")

    if wandb_run:
        wandb_run.log({"best_val_auc": best_auc, "total_time_sec": total_time, "memory_mb": mem_mb})
        wandb_run.finish()

    return summary


if __name__ == "__main__":
    main()
