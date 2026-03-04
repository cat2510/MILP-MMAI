# Multimodal Optimization Problem Matching

### Graph–Text Fusion on LogiOR (ORThought)

This project implements multimodal learning on structured optimization problems from the **LogiOR** dataset (ORThought repository).

Each problem instance consists of:

- `question.txt` → natural language description
- `model/code.py` → Gurobi model
- exported `.mps` file → standardized linear model
- `*_graph.json` → bipartite MILP graph representation

The goal is to learn whether a graph representation of an optimization model matches its corresponding natural language description.

---

# 1. Dataset Construction

## 1.1 Raw Source

Repository:
`ORThought/datasets/processed/LogiOR`

Contains 92 logistics and supply chain optimization problems. Each problem folder includes:

```
prob_XXX/
  ├── question.txt
  ├── model.txt
  ├── code.py
```

---

## 1.2 Model → MPS Export

Each problem’s `code.py` was executed (without solving) to export a `.mps` file:

```
python export_all_mps.py \
  --root ../ORThought/datasets/processed/LogiOR \
  --out ./all_mps
```

Output:

```
all_mps/prob_XXX.mps
```

The `.mps` format provides a solver-standard representation of:

- Variables
- Constraints
- Linear coefficient matrix
- Objective
- Bounds

This avoids fragile text parsing and ensures structural correctness.

---

## 1.3 MPS → Bipartite Graph

Each `.mps` file is converted into a bipartite graph:

```
python mps2graph.py \
  --probs 080 019 054 \
  --root ../ORThought/datasets/processed/LogiOR/all_mps \
  --out ./graphs_mps
```

Graph structure:

- Variable nodes
- Constraint nodes
- Edges for each nonzero in A matrix
- Edge attribute: coefficient
- Node attributes:
  - Variable: vtype, lb, ub, objective coefficient
  - Constraint: sense, rhs

Meta information:

```
{
  "num_vars": ...,
  "num_constrs": ...,
  "num_edges": ...,
  "model_sense": "min" | "max"
}
```

All 92 graphs were successfully generated.

---

# 2. Supervised Matching Dataset

We frame graph–text matching as binary classification.

## Positive examples

Matched pairs:

```
(graph_i, text_i, label=1)
```

## Negative examples

Mismatched pairs:

```
(graph_i, text_j, label=0),  i ≠ j
```

Dataset built using:

```
python build_pair_labels.py
```

Produces:

```
pair_data/
  ├── train.jsonl
  ├── val.jsonl
  ├── test.jsonl
```

Each entry:

```
{
  "graph_path": "graphs_mps/prob_088_graph.json",
  "text_path": "../ORThought/.../prob_088/question.txt",
  "prob_graph": "prob_088",
  "prob_text": "prob_088",
  "label": 1
}
```

Splits:

- 64 train problems
- 14 validation
- 14 test
- Balanced positive/negative pairs

---

# 3. Model Architecture

Two modalities:

### Graph Encoder

- GCN (Kipf & Welling 2017)
- Input: bipartite MILP graph
- Output: fixed-size embedding vector

### Text Encoder

- Transformer-based encoder
- Input: raw `question.txt`
- Output: pooled embedding

---

# 4. Fusion Techniques Implemented

Homework requirement:

1. Early Fusion
2. Late Fusion
3. Tensor Fusion
4. Low-Rank Multimodal Fusion (LMF)

Each implemented in:

```
models/multimodal_models.py
```

### Early Fusion

Concatenate embeddings:

```
z = [g || t]
```

### Late Fusion

Independent unimodal classifiers combined via averaging/logits sum.

### Tensor Fusion

Full outer product:

```
T = g ⊗ t
```

Implemented using `torch.einsum`.

### Low-Rank Multimodal Fusion (LMF)

Low-rank tensor approximation:

```
f = Σ_r (Wg_r g) ⊙ (Wt_r t)
```

Efficient parameterization.

---

# 5. Training

Script:

```
train/pilot_fusion_classification.py
```

Run example (from `or_fusion`):

```
python -m train.pilot_fusion_classification \
  --fusion early \
  --train_jsonl ../pair_data/train.jsonl \
  --val_jsonl ../pair_data/val.jsonl \
  --graphs_dir .. \
  --out_dir ./runs \
  --epochs 20
```

Loss: `CrossEntropyLoss`. Metrics: validation accuracy, **val AUC** (class-imbalance robust).

**Hyperparameters** (all configurable):


| Arg                                             | Description              | Default          |
| ----------------------------------------------- | ------------------------ | ---------------- |
| `--fusion`                                      | early, late, tensor, lmf | —                |
| `--d_g`, `--d_t`                                | Embedding dims           | 128              |
| `--fusion_hidden`                               | Early fusion MLP hidden  | 256              |
| `--lmf_rank`, `--lmf_d_fused`                   | LMF params               | 8, 256           |
| `--lr`, `--batch_size`, `--epochs`, `--dropout` | Training                 | 1e-3, 8, 20, 0.2 |
| `--freeze_text` / `--no_freeze_text`            | Freeze text backbone     | freeze           |


**Optuna search**:

```bash
pip install optuna
python -m train.optuna_search \
  --train_jsonl ../pair_data/train.jsonl \
  --val_jsonl ../pair_data/val.jsonl \
  --graphs_dir .. \
  --n_trials 20 \
  --out_dir ./optuna_results
```

**Visualization** (params, memory, convergence time):

```bash
pip install matplotlib
python -m train.visualize_results --results_dir ./runs
```

## **6. Contrastive training**

Script:

```
train_contrastive.py
```

**Dataset:** LogiORGraphTextDataset

Note this is different from the `pilot_fusion_classification.py` used in `optuna_search.py` which use explicitly, manually labeled pairs LogiORGraphTextPairDataset from `pair_data/train.jsonl`

For each `prob_XXX`:

- Graph: `prob_XXX_graph.json`
- Text: `prob_XXX/question.txt` (same problem)

So you get 92 **same-problem** pairs (graph and text always match). **Negatives come from InfoNCE via in-batch negatives.**

**Apple-to-apple with fusion (use pair_data):**

Train on explicit positive + negative pairs (same data as optuna/fusion):

```bash
# Regenerate pair_data with k_neg>=5
python data/build_pair_labels.py --graphs_dir .. --logior_root ../../ORThought/datasets/processed/LogiOR --out_dir ../pair_data --k_neg 5

# Contrastive on pair_data
python -m train.train_contrastive \
  --graphs_dir ../graphs_mps \
  --logior_root ../ORThought/datasets/processed/LogiOR \
  --train_jsonl ../pair_data/train.jsonl \
  --base_dir .. \
  --epochs 20 --run_zero_shot
```

Uses BCE loss over pairwise similarities (einsum); explicit negatives improve alignment.

---

## **7. Alignment visualization**

Visual examples of data post alignment (homework requirement):

```bash
python -m train.visualize_alignment \
  --ckpt ./ckpts/contrastive_encoders.pt \
  --graphs_dir ../graphs_mps \
  --logior_root ../ORThought/datasets/processed/LogiOR \
  --out_dir ./alignment_vis
```

Produces:

- `similarity_heatmap.png` — graph vs text similarity matrix (diagonal = correct matches)
- `example_success_*.png`, `example_failure_*.png` — success/failure cases with graph stats and text
- `embeddings_tsne.png` — t-SNE of graph (○) and text (×) embeddings
- `alignment_stats.json`, `alignment_examples_summary.json` — for report

---

