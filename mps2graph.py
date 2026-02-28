#!/usr/bin/env python3
import argparse
import json
import os
import glob
from typing import Dict, Any, List

import gurobipy as gp
import math


def _safe_float(x: float, *, inf_as_none: bool = True, thresh: float = 1e19):
    """
    Convert Gurobi numeric to JSON-safe numeric.
    - By default, maps +/-inf and huge sentinel bounds to None (-> null in JSON).
    """
    if x is None:
        return None
    x = float(x)

    # True infinities / NaNs
    if math.isinf(x) or math.isnan(x):
        return None if inf_as_none else (thresh if x > 0 else -thresh)

    # Gurobi often uses large sentinels for infinity in bounds
    if abs(x) >= thresh:
        return None if inf_as_none else (thresh if x > 0 else -thresh)

    return x


def mps_to_bipartite_graph(mps_path: str, *, add_objective_node: bool = True) -> Dict[str, Any]:
    """
    Build a bipartite graph (constraints <-> variables) from an .mps file using Gurobi.

    Notes:
    - Uses m.getA() which captures ONLY the linear constraint matrix.
      Quadratic objective/constraints (QP/QCP) are not encoded in edges; we record flags in meta.
    - Optionally adds an explicit __OBJ__ node connected to variables with nonzero objective coeffs.
    """
    m = gp.read(mps_path)
    m.update()

    vars_ = m.getVars()
    cons_ = m.getConstrs()
    A = m.getA()  # sparse matrix of linear constraints

    # ---- Nodes: variables ----
    var_nodes: List[Dict[str, Any]] = []
    for v in vars_:
        var_nodes.append({
            "id": v.VarName,
            "type": "var",
            "vtype": v.VType,                 # 'C','B','I',...
            "lb": _safe_float(v.LB),
            "ub": _safe_float(v.UB),
            "obj": float(v.Obj),              # linear obj coefficient
        })

    # ---- Nodes: constraints ----
    con_nodes: List[Dict[str, Any]] = []
    for c in cons_:
        con_nodes.append({
            "id": c.ConstrName,
            "type": "constr",
            "sense": c.Sense,                 # '<', '>', '='
            "rhs": float(c.RHS),
        })

    # ---- Edges: linear matrix nonzeros ----
    edges: List[Dict[str, Any]] = []
    Acoo = A.tocoo()
    for r, c, val in zip(Acoo.row, Acoo.col, Acoo.data):
        edges.append({
            "src": cons_[r].ConstrName,
            "dst": vars_[c].VarName,
            "coef": float(val),
        })

    # ---- Objective as a node (recommended for uniform "row-like" treatment) ----
    if add_objective_node:
        obj_id = "__OBJ__"
        con_nodes.append({
            "id": obj_id,
            "type": "obj",
            "sense": "min" if m.ModelSense == 1 else "max",
            "rhs": 0.0,
        })
        for v in vars_:
            if v.Obj != 0.0:
                edges.append({
                    "src": obj_id,
                    "dst": v.VarName,
                    "coef": float(v.Obj),
                })

    # ---- Quadratic/nonlinear caveat flags ----
    # These attributes exist on Gurobi models; they remain 0 for pure LP/MILP.
    # We do NOT encode quadratic terms into edges here; we just record what would be missing.
    has_quad_obj = bool(getattr(m, "NumQNZs", 0) > 0)
    has_qconstr = bool(getattr(m, "NumQConstrs", 0) > 0)
    has_quad_constr_terms = bool(getattr(m, "NumQCNZs", 0) > 0)

    meta = {
        "mps_path": mps_path,
        "num_vars": len(vars_),
        "num_constrs": len(cons_) + (1 if add_objective_node else 0),
        "num_edges": len(edges),
        "model_sense": "min" if m.ModelSense == 1 else "max",
        # Caveat flags: if any are true, your current graph misses quadratic structure.
        "has_quad_objective": has_quad_obj,
        "has_quadratic_constraints": has_qconstr,
        "has_quad_constr_terms": has_quad_constr_terms,
        "linear_only_graph": not (has_quad_obj or has_qconstr or has_quad_constr_terms),
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
    ap.add_argument("--probs", nargs="*", help="e.g. 080 019 054; omit to process all prob_*.mps")
    ap.add_argument("--root", required=True, help="folder containing prob_XXX.mps files")
    ap.add_argument("--out", required=True, help="output directory for graph jsons")
    ap.add_argument("--no_obj_node", action="store_true", help="do not add __OBJ__ node/edges")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    add_obj = not args.no_obj_node

    # ---- Case A: explicit list ----
    if args.probs:
        for p in args.probs:
            prob_id = f"prob_{int(p):03d}" if p.isdigit() else p
            mps_path = os.path.join(args.root, f"{prob_id}.mps")
            if not os.path.exists(mps_path):
                print(f"[MISS] {prob_id}: {mps_path} not found")
                continue

            g = mps_to_bipartite_graph(mps_path, add_objective_node=add_obj)
            out_path = os.path.join(args.out, f"{prob_id}_graph.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(g, f, indent=2, allow_nan=False)

            caveat = ""
            if not g["meta"]["linear_only_graph"]:
                caveat = " | [WARN] quadratic/nonlinear parts not encoded (linear-only graph)"
            print(f"[OK] {prob_id}: wrote {out_path} | {g['meta']}{caveat}")

    # ---- Case B: process all ----
    else:
        for mps_path in sorted(glob.glob(os.path.join(args.root, "prob_*.mps"))):
            prob_id = os.path.splitext(os.path.basename(mps_path))[0]  # prob_XXX
            g = mps_to_bipartite_graph(mps_path, add_objective_node=add_obj)
            out_path = os.path.join(args.out, f"{prob_id}_graph.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(g, f, indent=2, allow_nan=False)

            caveat = ""
            if not g["meta"]["linear_only_graph"]:
                caveat = " | [WARN] quadratic/nonlinear parts not encoded (linear-only graph)"
            print(f"[OK] {prob_id}: wrote {out_path} | {g['meta']}{caveat}")


if __name__ == "__main__":
    main()
    """
    Examples:

    python mps2graph.py --probs 080 019 054 \
      --root ../ORThought/datasets/processed/LogiOR/all_mps \
      --out ./graphs_mps

    python mps2graph.py \
      --root ../ORThought/datasets/processed/LogiOR/all_mps \
      --out ./graphs_mps
    """