# data/graph_builders.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import torch

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("Please install torch-geometric to use this file.") from e


@dataclass
class GraphBuildConfig:
    use_bidirectional_edges: bool = True
    add_degree_feature: bool = True
    # Normalize large coefficients a bit; you can refine later
    coef_clip: float = 1e6
    coef_log1p: bool = True


def _safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        x = float(x)
    except Exception:
        return default
    if x != x:  # NaN
        return default
    if x == float("inf") or x == float("-inf"):
        return default
    return x


def build_pyg_from_mps_graph_json(g: Dict[str, Any], cfg: Optional[GraphBuildConfig] = None) -> Data:
    """
    Input: graph dict from mps2graph.py
    Output: PyG Data with x, edge_index, edge_attr, node_type, and masks for bipartite types.
    """
    if cfg is None:
        cfg = GraphBuildConfig()

    var_nodes = g["nodes"]["vars"]
    con_nodes = g["nodes"]["constrs"]
    edges = g["edges"]

    n_vars = len(var_nodes)
    n_cons = len(con_nodes)
    n_nodes = n_vars + n_cons

    # Indexing: vars [0..n_vars-1], constrs [n_vars..n_vars+n_cons-1]
    var_id_to_idx = {v["id"]: i for i, v in enumerate(var_nodes)}
    con_id_to_idx = {c["id"]: n_vars + i for i, c in enumerate(con_nodes)}

    # ----- Node features -----
    # Minimal numeric features:
    # vars: [lb, ub, obj]
    # cons: [sense(-1/0/1), rhs, 0]
    # plus a node-type one-hot (var/constr) embedded as scalar flags
    x = torch.zeros((n_nodes, 6), dtype=torch.float32)

    # node_type: 0=var, 1=constr
    node_type = torch.zeros((n_nodes,), dtype=torch.long)
    node_type[n_vars:] = 1

    # vars
    for i, v in enumerate(var_nodes):
        lb = _safe_float(v.get("lb"), 0.0)
        ub = _safe_float(v.get("ub"), 0.0)
        obj = _safe_float(v.get("obj"), 0.0)
        vtype = v.get("vtype", "C")
        # vtype encoding: C/I/B -> (0,1,2)
        vt = {"C": 0.0, "I": 1.0, "B": 2.0}.get(vtype, 0.0)

        x[i, 0] = lb
        x[i, 1] = ub
        x[i, 2] = obj
        x[i, 3] = vt
        x[i, 4] = 1.0  # is_var flag
        x[i, 5] = 0.0  # is_constr flag

    # constrs
    for j, c in enumerate(con_nodes):
        idx = n_vars + j
        sense = c.get("sense", "=")
        s = {"<": -1.0, ">": 1.0, "=": 0.0}.get(sense, 0.0)
        rhs = _safe_float(c.get("rhs"), 0.0)
        x[idx, 0] = s
        x[idx, 1] = rhs
        x[idx, 4] = 0.0
        x[idx, 5] = 1.0

    # ----- Edges -----
    src_list: List[int] = []
    dst_list: List[int] = []
    edge_attr_list: List[float] = []

    for e in edges:
        src = e["src"]
        dst = e["dst"]
        coef = _safe_float(e.get("coef"), 0.0)

        # Objective node optional: "__OBJ__" is in constrs list if you included it
        if src in con_id_to_idx and dst in var_id_to_idx:
            u = con_id_to_idx[src]
            v = var_id_to_idx[dst]
        else:
            # If something doesn't match, skip
            continue

        src_list.append(u)
        dst_list.append(v)
        edge_attr_list.append(coef)

        if cfg.use_bidirectional_edges:
            src_list.append(v)
            dst_list.append(u)
            edge_attr_list.append(coef)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).view(-1, 1)

    # Edge attr transform
    if cfg.coef_log1p:
        edge_attr = torch.sign(edge_attr) * torch.log1p(torch.clamp(edge_attr.abs(), max=cfg.coef_clip))
    else:
        edge_attr = torch.clamp(edge_attr, min=-cfg.coef_clip, max=cfg.coef_clip)

    # Degree feature (optional)
    if cfg.add_degree_feature:
        deg = torch.zeros((n_nodes,), dtype=torch.float32)
        for u in src_list:
            deg[u] += 1.0
        deg = torch.log1p(deg)
        # append as another feature channel
        x = torch.cat([x, deg.view(-1, 1)], dim=1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type,
        n_vars=n_vars,
        n_cons=n_cons,
    )
    return data