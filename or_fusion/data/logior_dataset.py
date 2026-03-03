# data/logior_dataset.py
"""Dataset that pairs prob_XXX_graph.json with question.txt in the original problem folder."""
from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from .graph_builders import build_pyg_from_mps_graph_json, GraphBuildConfig

try:
    from torch_geometric.data import Data
except Exception as e:
    raise ImportError("Please install torch-geometric to use this dataset.") from e


class LogiORGraphTextDataset(Dataset):
    """
    Expects:
      graphs_dir: contains prob_XXX_graph.json files (from mps2graph.py)
      logior_root: contains prob_XXX/ folders each with question.txt
    """
    def __init__(
        self,
        graphs_dir: str,
        logior_root: str,
        probs: Optional[List[str]] = None,
        graph_cfg: Optional[GraphBuildConfig] = None,
        max_text_chars: int = 8000,
    ):
        self.graphs_dir = graphs_dir
        self.logior_root = logior_root
        self.graph_cfg = graph_cfg or GraphBuildConfig()
        self.max_text_chars = max_text_chars

        if probs is None:
            files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("prob_") and f.endswith("_graph.json")])
            self.prob_ids = [f.replace("_graph.json", "") for f in files]  # prob_XXX
        else:
            self.prob_ids = [f"prob_{int(p):03d}" if p.isdigit() else p for p in probs]

        # verify existence (soft)
        self.items = []
        for pid in self.prob_ids:
            gj = os.path.join(graphs_dir, f"{pid}_graph.json")
            qt = os.path.join(logior_root, pid, "question.txt")
            if os.path.exists(gj) and os.path.exists(qt):
                self.items.append((pid, gj, qt))

        # #region agent log
        import sys
        _d = {"graphs_dir":graphs_dir,"logior_root":logior_root,"graphs_dir_abs":os.path.abspath(graphs_dir),"logior_root_abs":os.path.abspath(logior_root),"graphs_dir_exists":os.path.exists(graphs_dir),"logior_root_exists":os.path.exists(logior_root),"prob_ids_count":len(self.prob_ids),"items_count":len(self.items),"cwd":os.getcwd(),"sample_gj":os.path.join(graphs_dir, f"{self.prob_ids[0]}_graph.json") if self.prob_ids else None,"sample_qt":os.path.join(logior_root, self.prob_ids[0], "question.txt") if self.prob_ids else None,"sample_gj_exists":os.path.exists(os.path.join(graphs_dir, f"{self.prob_ids[0]}_graph.json")) if self.prob_ids else None,"sample_qt_exists":os.path.exists(os.path.join(logior_root, self.prob_ids[0], "question.txt")) if self.prob_ids else None,"listdir_graphs_count":len([f for f in (os.listdir(graphs_dir) if os.path.exists(graphs_dir) else []) if f.startswith("prob_") and f.endswith("_graph.json")]) if os.path.exists(graphs_dir) else -1}
        print("[DEBUG] LogiORGraphTextDataset:", json.dumps(_d, indent=2), file=sys.stderr)
        _lp = "/Users/cat2510/Documents/MMAI/.cursor/debug-d6443b.log"
        try:
            import time
            with open(_lp, "a") as _f:
                _f.write(json.dumps({"sessionId":"d6443b","hypothesisId":"H1,H3","location":"logior_dataset.py:init","message":"LogiORGraphTextDataset init","data":_d,"timestamp":int(time.time()*1000)}) + "\n")
        except Exception:
            pass
        # #endregion

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid, graph_path, q_path = self.items[idx]

        with open(graph_path, "r", encoding="utf-8") as f:
            g = json.load(f)

        pyg: Data = build_pyg_from_mps_graph_json(g, cfg=self.graph_cfg)

        with open(q_path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if len(txt) > self.max_text_chars:
            txt = txt[: self.max_text_chars]

        return {"prob_id": pid, "graph": pyg, "text": txt}


def collate_graph_text(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # PyG batching
    from torch_geometric.loader import DataLoader as PyGDataLoader
    graphs = [b["graph"] for b in batch]
    # hack: use pyg's internal collate via DataLoader
    pyg_loader = PyGDataLoader(graphs, batch_size=len(graphs))
    graph_batch = next(iter(pyg_loader))

    texts = [b["text"] for b in batch]
    prob_ids = [b.get("prob_id") or b.get("prob_graph") for b in batch]
    out: Dict[str, Any] = {"prob_id": prob_ids, "graph": graph_batch, "text": texts}
    if "y" in batch[0]:
        out["y"] = torch.stack([b["y"] for b in batch])
    return out