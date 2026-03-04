#!/usr/bin/env python3
"""
Visual examples of graph-text alignment post training.
Generates: similarity heatmap, success/failure examples, embedding t-SNE.
Run after training contrastive encoders.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Add or_fusion to path so "data" and "models" resolve when run directly
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.logior_dataset import LogiORGraphTextDataset, collate_graph_text
from data.graph_builders import GraphBuildConfig
from models.graph_encoder_gcn import GraphEncoderGCN, GCNConfig
from models.text_encoder import TextEncoder, TextEncoderConfig


def load_encoders(ckpt_path: str, device: torch.device, out_dim: int = 128, max_length: int = 256):
    """Load trained encoders from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Infer in_dim from first layer
    g_state = ckpt["graph_encoder"]
    in_dim = g_state["convs.0.lin.weight"].shape[1]
    g_enc = GraphEncoderGCN(GCNConfig(
        in_dim=in_dim, hidden_dim=128, out_dim=out_dim, num_layers=3, dropout=0.0
    )).to(device)
    g_enc.load_state_dict(g_state)
    g_enc.eval()

    t_enc = TextEncoder(TextEncoderConfig(out_dim=out_dim, max_length=max_length)).to(device)
    t_enc.load_state_dict(ckpt["text_encoder"])
    t_enc.eval()

    graph_proj = text_proj = None
    if "graph_proj" in ckpt:
        from train.train_contrastive import LinearProjector, MLPProjector
        state = ckpt["graph_proj"]
        use_mlp = "fc1.weight" in state
        if use_mlp:
            graph_proj = MLPProjector(out_dim, out_dim, hidden=256).to(device)
            text_proj = MLPProjector(out_dim, out_dim, hidden=256).to(device)
        else:
            graph_proj = LinearProjector(out_dim, out_dim).to(device)
            text_proj = LinearProjector(out_dim, out_dim).to(device)
        graph_proj.load_state_dict(ckpt["graph_proj"])
        text_proj.load_state_dict(ckpt["text_proj"])
        graph_proj.eval()
        text_proj.eval()

    return g_enc, t_enc, graph_proj, text_proj


@torch.no_grad()
def compute_similarity_matrix(
    g_enc, t_enc, ds, device, batch_size: int = 16,
    graph_proj=None, text_proj=None,
):
    """Compute [N, N] similarity matrix: sim[i,j] = cosine(graph_i, text_j)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_text)
    all_z_g = []
    all_z_t = []
    for batch in loader:
        graphs = batch["graph"].to(device)
        texts = batch["text"]
        z_g = g_enc(graphs)
        z_t = t_enc(texts, device=device)
        if graph_proj is not None:
            z_g = graph_proj(z_g)
        if text_proj is not None:
            z_t = text_proj(z_t)
        z_g = F.normalize(z_g, dim=-1)
        z_t = F.normalize(z_t, dim=-1)
        all_z_g.append(z_g.cpu())
        all_z_t.append(z_t.cpu())
    z_g = torch.cat(all_z_g, dim=0)
    z_t = torch.cat(all_z_t, dim=0)
    sim = torch.einsum("nd,md->nm", z_g, z_t).numpy()
    return sim


def plot_heatmap(sim: np.ndarray, prob_ids: list[str], out_path: str):
    """Plot similarity heatmap (graphs vs texts). Diagonal = correct matches."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Text (problem index)")
    ax.set_ylabel("Graph (problem index)")
    ax.set_title("Graph–Text Similarity Matrix\n(diagonal = correct matches; bright diagonal = good alignment)")
    # Subsample ticks for readability
    n = len(prob_ids)
    step = max(1, n // 15)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([prob_ids[i] for i in ticks], rotation=45, ha="right")
    ax.set_yticklabels([prob_ids[i] for i in ticks])
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {out_path}")


def plot_examples(
    ds: LogiORGraphTextDataset,
    sim: np.ndarray,
    prob_ids: list[str],
    out_dir: str,
    n_success: int = 3,
    n_failure: int = 3,
):
    """Create example figures: successes (high diagonal) and failures (low diagonal or wrong match)."""
    n = len(prob_ids)
    diag = np.diag(sim)
    pred_idx = np.argmax(sim, axis=1)
    correct = pred_idx == np.arange(n)

    # Best successes: highest diagonal
    success_idx = np.argsort(diag)[::-1][:n_success]
    # Worst failures: lowest diagonal among incorrect, or incorrect with high wrong score
    fail_candidates = np.where(~correct)[0]
    if len(fail_candidates) > 0:
        fail_scores = diag[fail_candidates]
        failure_idx = fail_candidates[np.argsort(fail_scores)][:n_failure]
    else:
        # All correct: pick lowest diagonal as "near failure"
        failure_idx = np.argsort(diag)[:n_failure]

    os.makedirs(out_dir, exist_ok=True)

    def make_example(idx: int, tag: str, reason: str):
        pid = prob_ids[idx]
        item = ds[idx]
        wrong_idx = pred_idx[idx]
        has_mispred = wrong_idx != idx
        text = item["text"]
        max_len = 200 if (tag == "failure" and has_mispred) else 350
        if len(text) > max_len:
            text = text[: max_len - 3] + "..."
        graph = item["graph"]
        nv, nc = graph.num_nodes, graph.edge_index.shape[1] // 2
        true_sim = sim[idx, idx]
        wrong_sim = sim[idx, wrong_idx] if wrong_idx != idx else 0.0
        status = "CORRECT" if correct[idx] else "WRONG"
        fig_h = 9 if has_mispred else 7

        fig, ax = plt.subplots(figsize=(9, fig_h))
        ax.axis("off")
        y = 0.97
        ax.text(0.5, y, f"{tag}: {pid} ({status})", fontsize=14, fontweight="bold", ha="center")
        y -= 0.06
        ax.text(0.5, y, reason, fontsize=10, ha="center", style="italic", color="gray")
        y -= 0.05
        ax.text(0.1, y, "Graph stats:", fontsize=11, fontweight="bold")
        y -= 0.05
        ax.text(0.1, y, f"  Nodes: {nv}, Edges: ~{nc}", fontsize=10, family="monospace")
        y -= 0.06
        ax.text(0.1, y, "Correct question (text):", fontsize=11, fontweight="bold")
        y -= 0.02
        ax.text(0.1, y, text, fontsize=9, wrap=True, verticalalignment="top")
        y -= 0.12
        ax.text(0.1, y, f"Similarity with correct text: {true_sim:.4f}", fontsize=10)
        y -= 0.04
        if wrong_idx != idx:
            ax.text(0.1, y, f"Similarity with predicted text ({prob_ids[wrong_idx]}): {wrong_sim:.4f}", fontsize=10)
            y -= 0.06
            # Add mispredicted problem's question.txt
            wrong_text = ds[wrong_idx]["text"]
            if len(wrong_text) > 250:
                wrong_text = wrong_text[:247] + "..."
            ax.text(0.1, y, f"Mispredicted ({prob_ids[wrong_idx]}) question:", fontsize=11, fontweight="bold")
            y -= 0.02
            ax.text(0.1, y, wrong_text, fontsize=9, wrap=True, verticalalignment="top", color="darkred")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        path = os.path.join(out_dir, f"example_{tag}_{pid}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    for i, idx in enumerate(success_idx):
        r = "High similarity between graph and its matching text."
        make_example(idx, "success", r)
    for i, idx in enumerate(failure_idx):
        if correct[idx]:
            r = "Lowest diagonal (near-failure); model barely matched correctly."
        else:
            r = f"Model predicted {prob_ids[pred_idx[idx]]} instead of {prob_ids[idx]}."
        make_example(idx, "failure", r)

    # Write summary JSON for report
    summary = {
        "successes": [{"prob_id": prob_ids[i], "sim_correct": float(diag[i]), "correct": bool(correct[i])} for i in success_idx],
        "failures": [{"prob_id": prob_ids[i], "sim_correct": float(diag[i]), "correct": bool(correct[i]), "predicted": prob_ids[pred_idx[i]] if not correct[i] else None} for i in failure_idx],
    }
    with open(os.path.join(out_dir, "alignment_examples_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {n_success} success + {n_failure} failure examples to {out_dir}")


# Keywords for problem-type analysis (OR/logistics domain)
KEYWORDS = [
    "warehouse", "transport", "allocation", "assignment", "routing", "scheduling",
    "delivery", "minimize", "maximize", "inventory", "demand", "supply",
    "cost", "distance", "location", "customer", "product", "capacity",
]


def extract_keywords(text: str) -> list[str]:
    """Return keywords present in text (case-insensitive)."""
    t = text.lower()
    return [k for k in KEYWORDS if k in t]


def plot_keyword_analysis(
    ds: LogiORGraphTextDataset,
    sim: np.ndarray,
    prob_ids: list[str],
    out_dir: str,
    min_count: int = 3,
):
    """Plot success/failure rates by problem type (keyword presence)."""
    n = len(prob_ids)
    pred_idx = np.argmax(sim, axis=1)
    correct = pred_idx == np.arange(n)

    keyword_to_indices: dict[str, list[int]] = {k: [] for k in KEYWORDS}
    for idx in range(n):
        text = ds[idx]["text"]
        for k in extract_keywords(text):
            keyword_to_indices[k].append(idx)

    # Per keyword: (n_correct, n_total)
    rows = []
    for k in KEYWORDS:
        indices = keyword_to_indices[k]
        if len(indices) < min_count:
            continue
        n_correct = sum(1 for i in indices if correct[i])
        acc = n_correct / len(indices)
        rows.append((k, len(indices), n_correct, acc))

    if not rows:
        print("No keywords with enough samples for keyword analysis")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [r[0] for r in rows]
    totals = [r[1] for r in rows]
    accs = [r[3] * 100 for r in rows]
    colors = [plt.cm.RdYlGn(acc / 100) for acc in accs]
    bars = ax.barh(labels, accs, color=colors)
    ax.axvline(x=100 * correct.mean(), color="gray", linestyle="--", label=f"Overall ({100*correct.mean():.1f}%)")
    ax.set_xlabel("Accuracy (%) among problems containing keyword")
    ax.set_title("Alignment success by problem type (keyword in question)\nGreen=better, Red=worse")
    ax.set_xlim(0, 105)
    for bar, tot in zip(bars, totals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"n={tot}", va="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "keyword_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved keyword analysis to {path}")

    # Also save JSON for report
    summary = {
        k: {"n": n_tot, "n_correct": n_corr, "accuracy": acc}
        for k, n_tot, n_corr, acc in rows
    }
    with open(os.path.join(out_dir, "keyword_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)


def plot_tsne(
    g_enc, t_enc, ds, device, out_path: str,
    graph_proj=None, text_proj=None, batch_size: int = 16,
):
    """t-SNE of graph and text embeddings (colored by problem)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn not installed; skipping t-SNE")
        return
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_text)
    all_z_g, all_z_t, all_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            graphs = batch["graph"].to(device)
            texts = batch["text"]
            ids = batch["prob_id"]
            z_g = g_enc(graphs)
            z_t = t_enc(texts, device=device)
            if graph_proj is not None:
                z_g = graph_proj(z_g)
            if text_proj is not None:
                z_t = text_proj(z_t)
            z_g = F.normalize(z_g, dim=-1).cpu().numpy()
            z_t = F.normalize(z_t, dim=-1).cpu().numpy()
            all_z_g.append(z_g)
            all_z_t.append(z_t)
            all_ids.extend(ids)
    z_g = np.vstack(all_z_g)
    z_t = np.vstack(all_z_t)
    combined = np.vstack([z_g, z_t])
    n = len(z_g)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
    emb = tsne.fit_transform(combined)
    emb_g, emb_t = emb[:n], emb[n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.scatter(emb_g[i, 0], emb_g[i, 1], c=f"C{i % 10}", marker="o", s=30, alpha=0.7)
        ax.scatter(emb_t[i, 0], emb_t[i, 1], c=f"C{i % 10}", marker="x", s=30, alpha=0.7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Embedding space: ○ graph, × text (same color = same problem)\nGood alignment: ○ and × of same color close")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize graph-text alignment")
    ap.add_argument("--ckpt", default="./ckpts/contrastive_encoders.pt", help="checkpoint path")
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--logior_root", required=True)
    ap.add_argument("--out_dir", default="./alignment_vis")
    ap.add_argument("--out_dim", type=int, default=128)
    ap.add_argument("--max_text_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_tsne", action="store_true", help="skip t-SNE (needs sklearn)")
    args = ap.parse_args()

    # Path fallback (same as train_contrastive)
    if not os.path.exists(args.logior_root):
        fallback = os.path.normpath(os.path.join(os.getcwd(), "../../ORThought/datasets/processed/LogiOR"))
        if os.path.exists(fallback):
            args.logior_root = fallback

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_cfg = GraphBuildConfig(use_bidirectional_edges=True, add_degree_feature=True)
    ds = LogiORGraphTextDataset(
        graphs_dir=args.graphs_dir,
        logior_root=args.logior_root,
        graph_cfg=graph_cfg,
        max_text_chars=12000,
    )
    prob_ids = ds.prob_ids

    g_enc, t_enc, graph_proj, text_proj = load_encoders(
        args.ckpt, device, out_dim=args.out_dim, max_length=args.max_text_len
    )
    sim = compute_similarity_matrix(
        g_enc, t_enc, ds, device, args.batch_size, graph_proj, text_proj
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Heatmap
    plot_heatmap(sim, prob_ids, os.path.join(args.out_dir, "similarity_heatmap.png"))

    # 2. Success/failure examples
    plot_examples(ds, sim, prob_ids, args.out_dir)

    # 3. Keyword-type analysis
    plot_keyword_analysis(ds, sim, prob_ids, args.out_dir)

    # 4. t-SNE
    if not args.no_tsne:
        plot_tsne(g_enc, t_enc, ds, device, os.path.join(args.out_dir, "embeddings_tsne.png"),
                  graph_proj, text_proj, args.batch_size)

    # Summary stats
    diag = np.diag(sim)
    acc = (np.argmax(sim, axis=1) == np.arange(len(prob_ids))).mean()
    stats = {
        "zero_shot_accuracy": float(acc),
        "mean_diagonal_sim": float(np.mean(diag)),
        "min_diagonal_sim": float(np.min(diag)),
        "max_diagonal_sim": float(np.max(diag)),
    }
    with open(os.path.join(args.out_dir, "alignment_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Alignment stats: {stats}")


if __name__ == "__main__":
    main()
