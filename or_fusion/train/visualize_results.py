import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def load_results(results_dir: str):
    runs = []
    for p in glob.glob(os.path.join(results_dir, "**/run_*.json"), recursive=True):
        try:
            with open(p) as f:
                runs.append(json.load(f))
        except Exception:
            pass
    return runs

def aggregate_by_fusion(runs, field_path, time_field=None):
    """
    Groups data by fusion type and aligns sequences of different lengths.
    Returns: {fusion_type: (mean_array, std_array, time_axis)}
    """
    grouped = {}
    for r in runs:
        f = r["hparams"]["fusion"]
        # Navigate nested dict (e.g., "history.val_auc")
        data = r
        for key in field_path.split("."):
            data = data.get(key, [])
        
        times = r.get("history", {}).get(time_field, []) if time_field else None
        if data:
            grouped.setdefault(f, []).append((np.array(data), np.array(times) if times is not None else None))

    stats = {}
    for f, results in grouped.items():
        # Find max length to align arrays
        max_len = max(len(x[0]) for x in results)
        aligned_data = []
        aligned_times = []
        
        for data, times in results:
            # Linear interpolation/padding to align different run lengths
            xp = np.linspace(0, 1, len(data))
            xnew = np.linspace(0, 1, max_len)
            aligned_data.append(np.interp(xnew, xp, data))
            if times is not None:
                # For cumulative time, we interpolate the cumsum
                cum_times = np.cumsum(times)
                aligned_times.append(np.interp(xnew, xp, cum_times))

        mean_data = np.mean(aligned_data, axis=0)
        std_data = np.std(aligned_data, axis=0)
        mean_time = np.mean(aligned_times, axis=0) if aligned_times else np.arange(max_len)
        
        stats[f] = (mean_data, std_data, mean_time)
    return stats

def plot_learning_curves(runs, out_path):
    """Plot Mean Val AUC vs Cumulative Training Time."""
    stats = aggregate_by_fusion(runs, "history.val_auc", time_field="epoch_time")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"early": "C0", "late": "C1", "tensor": "C2", "lmf": "C3"}

    for f, (mean, std, time) in stats.items():
        color = colors.get(f, None)
        line = ax.plot(time, mean, label=f, color=color, lw=2)
        ax.fill_between(time, mean - std, mean + std, color=line[0].get_color(), alpha=0.15)

    ax.set_xlabel("Cumulative Training Time (seconds)")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Fusion Efficiency: Accuracy vs. Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_loss_convergence(runs, out_path):
    """Plot Mean Training Loss vs Epochs."""
    stats = aggregate_by_fusion(runs, "history.train_loss")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"early": "C0", "late": "C1", "tensor": "C2", "lmf": "C3"}

    for f, (mean, std, epochs) in stats.items():
        color = colors.get(f, None)
        # 'epochs' here is just 0 to max_len
        line = ax.plot(epochs, mean, label=f, color=color, lw=2)
        ax.fill_between(epochs, mean - std, mean + std, color=line[0].get_color(), alpha=0.15)

    ax.set_xlabel("Training Steps (Normalized Epochs)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Convergence by Fusion Type")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_params(runs, out_path):
    """Bar chart: params by fusion type."""
    by_fusion = {}
    for r in runs:
        f = r["hparams"]["fusion"]
        by_fusion.setdefault(f, []).append(r["params"])

    fusions = sorted(by_fusion.keys())
    graph_p = [sum(x["graph_encoder"] for x in by_fusion[f]) / len(by_fusion[f]) for f in fusions]
    text_p = [sum(x["text_encoder"] for x in by_fusion[f]) / len(by_fusion[f]) for f in fusions]
    fusion_p = [sum(x["fusion"] for x in by_fusion[f]) / len(by_fusion[f]) for f in fusions]
    total_p = [sum(x["total"] for x in by_fusion[f]) / len(by_fusion[f]) for f in fusions]

    x = range(len(fusions))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 1.5 * w for i in x], graph_p, w, label="Graph", color="C0")
    ax.bar([i - 0.5 * w for i in x], text_p, w, label="Text", color="C1")
    ax.bar([i + 0.5 * w for i in x], fusion_p, w, label="Fusion", color="C2")
    ax.bar([i + 1.5 * w for i in x], total_p, w, label="Total", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(fusions)
    ax.set_ylabel("Parameters")
    ax.set_title("Parameters by Fusion Type")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_auc_vs_params(runs, out_path):
    """Scatter: best_val_auc vs total params."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"early": "C0", "late": "C1", "tensor": "C2", "lmf": "C3"}
    for r in runs:
        f = r["hparams"]["fusion"]
        ax.scatter(
            r["params"]["total"] / 1e6,
            r["best_val_auc"],
            c=colors.get(f, "gray"),
            label=f if f not in [x.get_label() for x in ax.collections] else "",
            alpha=0.7,
        )
    ax.set_xlabel("Total parameters (M)")
    ax.set_ylabel("Best val AUC")
    ax.set_title("Performance vs Model Size")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    runs = load_results(args.results_dir)
    if not runs:
        print(f"No runs found in {args.results_dir}")
        return

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    # Generate the requested plots
    plot_learning_curves(runs, os.path.join(out_dir, "auc_vs_time.png"))
    plot_loss_convergence(runs, os.path.join(out_dir, "loss_vs_epochs.png"))
    plot_params(runs, os.path.join(out_dir, "params_by_fusion.png"))
    plot_auc_vs_params(runs, os.path.join(out_dir, "auc_vs_params.png"))
    
    print(f"Aggregated plots saved to {out_dir}")


if __name__ == "__main__":
    main()