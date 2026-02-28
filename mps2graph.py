#!/usr/bin/env python3
import argparse
import json
import os
import glob
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import gurobipy as gp
import math

def _safe_float(x: float, *, inf_as_none: bool = True, thresh: float = 1e19):
    """
    Convert Gurobi numeric to JSON-safe numeric.
    - If bound is +/- infinity (or huge sentinel), return None (-> null) by default.
    """
    if x is None:
        return None
    x = float(x)

    # True infinities
    if math.isinf(x) or math.isnan(x):
        return None if inf_as_none else (thresh if x > 0 else -thresh)

    # Gurobi sometimes uses large sentinels for infinity
    if abs(x) >= thresh:
        return None if inf_as_none else (thresh if x > 0 else -thresh)

    return x


def mps_to_bipartite_graph(mps_path: str) -> Dict[str, Any]:
    m = gp.read(mps_path)
    m.update()

    vars_ = m.getVars()
    cons_ = m.getConstrs()
    A = m.getA()  # scipy sparse (csr) in recent gurobi; supports .tocoo()

    # Map objects to indices
    var_idx = {v.VarName: i for i, v in enumerate(vars_)}
    con_idx = {c.ConstrName: i for i, c in enumerate(cons_)}

    # Node lists
    var_nodes: List[Dict[str, Any]] = []
    for v in vars_:
        var_nodes.append({
            "id": v.VarName,
            "type": "var",
            "vtype": v.VType,                 # 'C','B','I',...
            "lb": _safe_float(v.LB),
            "ub": _safe_float(v.UB),
            "obj": float(v.Obj),
        })

    con_nodes: List[Dict[str, Any]] = []
    for c in cons_:
        con_nodes.append({
            "id": c.ConstrName,
            "type": "constr",
            "sense": c.Sense,                 # '<', '>', '='
            "rhs": float(c.RHS),
        })

    # Edges = nonzeros of A
    edges: List[Dict[str, Any]] = []
    Acoo = A.tocoo()
    # A row = constraint index, col = var index
    for r, c, val in zip(Acoo.row, Acoo.col, Acoo.data):
        edges.append({
            "src": cons_[r].ConstrName,
            "dst": vars_[c].VarName,
            "coef": float(val),
        })

    # (Optional) objective node as a hyperedge-like row
    # If you want it, uncomment below:
    # obj_edges = []
    # for v in vars_:
    #     if abs(v.Obj) > 0:
    #         obj_edges.append({"src": "__OBJ__", "dst": v.VarName, "coef": float(v.Obj)})
    # con_nodes.append({"id": "__OBJ__", "type": "obj", "sense": "min", "rhs": 0.0})
    # edges.extend(obj_edges)

    meta = {
        "mps_path": mps_path,
        "num_vars": len(vars_),
        "num_constrs": len(cons_),
        "num_edges": len(edges),
        "model_sense": "min" if m.ModelSense == 1 else "max",
    }

    return {
        "meta": meta,
        "nodes": {
            "vars": var_nodes,
            "constrs": con_nodes,
        },
        "edges": edges,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True, help="e.g. 080 019 054")
    ap.add_argument("--root", required=True, help="dataset root containing prob_XXX folders")
    ap.add_argument("--out", required=True, help="output directory for graph jsons")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.probs:
        for p in args.probs:
            prob_id = f"prob_{int(p):03d}" if p.isdigit() else p
            prob_dir = os.path.join(args.root, prob_id)
            mps_path = os.path.join(args.root, f"{prob_id}.mps")
            if not os.path.exists(mps_path):
                print(f"[MISS] {prob_id}")
                continue
            g = mps_to_bipartite_graph(mps_path)
            out_path = os.path.join(args.out, f"{prob_id}_graph.json")
    else:
        for mps_path in glob.glob(os.path.join(args.root, "prob_*.mps")):
            g = mps_to_bipartite_graph(mps_path)
            out_path = os.path.join(args.out, f"{prob_id}_graph.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(g, f, indent=2)
            print(f"[OK] {prob_id}: wrote {out_path} | {g['meta']}")

if __name__ == "__main__":
    main()
    """ python mps2graph.py --probs 080 \
  --root ../ORThought/datasets/processed/LogiOR/all_mps \
  --out ./graphs_mps"""