#!/usr/bin/env python3
import argparse, json, os, random, re
from pathlib import Path
from typing import Dict, List, Tuple

PROB_RE = re.compile(r"prob_\d{3}")

def find_problems(graphs_dir: Path, logior_root: Path) -> List[str]:
    # graphs look like prob_080_graph.json
    probs = []
    for p in sorted(graphs_dir.glob("prob_*_graph.json")):
        m = PROB_RE.search(p.name)
        if not m:
            continue
        prob = m.group(0)  # prob_080
        txt = logior_root / prob / "question.txt"
        if txt.exists():
            probs.append(prob)
    return probs

def split_problems(probs: List[str], seed: int, train: float, val: float) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    probs = probs[:]
    rng.shuffle(probs)
    n = len(probs)
    n_train = int(round(train * n))
    n_val   = int(round(val * n))
    train_ids = probs[:n_train]
    val_ids   = probs[n_train:n_train+n_val]
    test_ids  = probs[n_train+n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}

def sample_negatives(prob: str, pool: List[str], k: int, rng: random.Random) -> List[str]:
    # choose k distinct negatives from pool excluding prob
    candidates = [p for p in pool if p != prob]
    if not candidates:
        return []
    if k >= len(candidates):
        # if k too large, just return all
        return candidates
    return rng.sample(candidates, k)

def build_examples_for_split(
    split_probs: List[str],
    graphs_dir: Path,
    logior_root: Path,
    k_neg: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    examples = []
    pool = split_probs[:]  # IMPORTANT: negatives sampled within split to avoid leakage

    for prob in split_probs:
        gpath = graphs_dir / f"{prob}_graph.json"
        tpath = logior_root / prob / "question.txt"

        # Positive
        examples.append({
            "graph_path": str(gpath),
            "text_path": str(tpath),
            "prob_graph": prob,
            "prob_text": prob,
            "label": 1
        })

        # Negatives
        negs = sample_negatives(prob, pool, k_neg, rng)
        for neg_prob in negs:
            neg_tpath = logior_root / neg_prob / "question.txt"
            examples.append({
                "graph_path": str(gpath),
                "text_path": str(neg_tpath),
                "prob_graph": prob,
                "prob_text": neg_prob,
                "label": 0
            })

    rng.shuffle(examples)
    return examples

def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--logior_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--k_neg", type=int, default=5, help="negatives per positive (default 5 for ~6x more examples)")
    args = ap.parse_args()

    graphs_dir = Path(args.graphs_dir)
    logior_root = Path(args.logior_root)
    out_dir = Path(args.out_dir)

    probs = find_problems(graphs_dir, logior_root)
    if not probs:
        raise SystemExit("No problems found with both graph json and question.txt")

    splits = split_problems(probs, args.seed, args.train_frac, args.val_frac)

    summary = {}
    for split_name, split_probs in splits.items():
        rows = build_examples_for_split(
            split_probs=split_probs,
            graphs_dir=graphs_dir,
            logior_root=logior_root,
            k_neg=args.k_neg,
            seed=args.seed + (0 if split_name == "train" else (1 if split_name == "val" else 2))
        )
        out_path = out_dir / f"{split_name}.jsonl"
        write_jsonl(out_path, rows)
        n_pos = sum(r["label"] == 1 for r in rows)
        n_neg = sum(r["label"] == 0 for r in rows)
        summary[split_name] = {"problems": len(split_probs), "examples": len(rows), "pos": n_pos, "neg": n_neg, "path": str(out_path)}

    print(json.dumps({"found_problems": len(probs), "splits": summary}, indent=2))

if __name__ == "__main__":
    main()