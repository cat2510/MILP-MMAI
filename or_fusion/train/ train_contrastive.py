# train/train_contrastive.py
from __future__ import annotations
import argparse
import os
import torch
from torch.utils.data import DataLoader

from data.logior_dataset import LogiORGraphTextDataset, collate_graph_text
from data.graph_builders import GraphBuildConfig

from models.graph_encoder_gcn import GraphEncoderGCN, GCNConfig
from models.text_encoder import TextEncoder, TextEncoderConfig
from losses.contrastive import SymmetricInfoNCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir", required=True, help="folder with prob_XXX_graph.json (from mps2graph)")
    ap.add_argument("--logior_root", required=True, help="folder with prob_XXX/question.txt")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_text_len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)

    ds = LogiORGraphTextDataset(
        graphs_dir=args.graphs_dir,
        logior_root=args.logior_root,
        graph_cfg=GraphBuildConfig(use_bidirectional_edges=True, add_degree_feature=True),
        max_text_chars=12000,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_text)

    # infer graph in_dim from one item
    one = ds[0]["graph"]
    in_dim = one.x.size(-1)

    g_enc = GraphEncoderGCN(GCNConfig(in_dim=in_dim, hidden_dim=128, out_dim=128, num_layers=3, dropout=0.2)).to(device)
    t_enc = TextEncoder(TextEncoderConfig(out_dim=128, max_length=args.max_text_len)).to(device)

    crit = SymmetricInfoNCE(temperature=0.07).to(device)
    params = list(g_enc.parameters()) + list(t_enc.parameters()) + list(crit.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    g_enc.train(); t_enc.train(); crit.train()

    for ep in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for batch in dl:
            graph = batch["graph"].to(device)
            texts = batch["text"]

            z_g = g_enc(graph)                 # [B,128]
            z_t = t_enc(texts, device=device)  # [B,128]
            loss = crit(z_g, z_t)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            total += float(loss.item()) * len(texts)
            n += len(texts)

        print(f"epoch {ep:02d} | loss {total / max(n,1):.4f}")

    # Optionally save encoders
    os.makedirs("./ckpts", exist_ok=True)
    torch.save({"graph_encoder": g_enc.state_dict(),
                "text_encoder": t_enc.state_dict(),
                "logit_scale": crit.state_dict()},
               "./ckpts/contrastive_encoders.pt")
    print("saved ./ckpts/contrastive_encoders.pt")


if __name__ == "__main__":
    main()