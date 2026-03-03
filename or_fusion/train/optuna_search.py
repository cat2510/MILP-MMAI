"""
Optuna hyperparameter search for fusion classification.
Run: python -m train.optuna_search --train_jsonl ../pair_data/train.jsonl --val_jsonl ../pair_data/val.jsonl --graphs_dir ..
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

# Add project root for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.trial import Trial


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--out_dir", default="./optuna_results")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--study_name", default="or_fusion")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def objective(trial: Trial, args) -> float:
    fusion = trial.suggest_categorical("fusion", ["early", "late", "tensor", "lmf"])
    graph_encoder = trial.suggest_categorical("graph_encoder", ["gcn", "gap"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    d_g = trial.suggest_categorical("d_g", [64, 128, 256])
    d_t = trial.suggest_categorical("d_t", [64, 128, 256])
    freeze_text = trial.suggest_categorical("freeze_text", [True, False])
    epochs = trial.suggest_int("epochs", 10, 40)

    trial_out = os.path.join(args.out_dir, f"trial_{trial.number}")
    os.makedirs(trial_out, exist_ok=True)

    cmd = [
        sys.executable, "-m", "train.pilot_fusion_classification",
        "--fusion", fusion,
        "--graph_encoder", graph_encoder,
        "--train_jsonl", args.train_jsonl,
        "--val_jsonl", args.val_jsonl,
        "--graphs_dir", args.graphs_dir,
        "--out_dir", trial_out,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--d_g", str(d_g),
        "--d_t", str(d_t),
        "--seed", str(args.seed + trial.number),
    ]
    if freeze_text:
        cmd.append("--freeze_text")
    else:
        cmd.append("--no_freeze_text")

    if fusion == "early":
        fusion_hidden = trial.suggest_categorical("fusion_hidden", [128, 256, 512])
        cmd += ["--fusion_hidden", str(fusion_hidden)]
    elif fusion == "lmf":
        lmf_rank = trial.suggest_int("lmf_rank", 4, 16)
        lmf_d_fused = trial.suggest_categorical("lmf_d_fused", [128, 256])
        cmd += ["--lmf_rank", str(lmf_rank), "--lmf_d_fused", str(lmf_d_fused)]

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)  # show pilot output so it doesn't look stalled
    if result.returncode != 0:
        raise RuntimeError(f"Trial {trial.number} failed (exit {result.returncode})")

    # Parse best_val_auc from JSON output
    run_path = os.path.join(trial_out, f"run_{fusion}_{graph_encoder}.json")
    if os.path.exists(run_path):
        with open(run_path) as f:
            data = json.load(f)
        auc = data.get("best_val_auc", 0.0)
    else:
        auc = 0.0

    return auc


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(
        lambda t: objective(t, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\n=== Best trial ===")
    print(f"  Value (val_auc): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    best_path = os.path.join(args.out_dir, "optuna_best.json")
    with open(best_path, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
        }, f, indent=2)
    print(f"  Saved to {best_path}")


if __name__ == "__main__":
    main()
