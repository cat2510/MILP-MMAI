# data/pair_dataset.py
"""Dataset that pairs prob_XXX_graph.json with question.txt in the original problem folder."""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import torch
from torch.utils.data import Dataset

from data.graph_builders import GraphBuildConfig, build_pyg_from_mps_graph_json  


@dataclass
class PairExample:
    graph_path: str
    text_path: str
    prob_graph: str
    prob_text: str
    label: int


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class LogiORGraphTextPairDataset(Dataset):
    """
    Loads graph-text pairs from jsonl lines like:
      {"graph_path": "graphs_mps/prob_088_graph.json",
       "text_path": "../ORThought/.../prob_088/question.txt",
       "prob_graph": "prob_088",
       "prob_text": "prob_088",
       "label": 1}

    Returns dict:
      {
        "graph": pyg_data,
        "text": str,
        "y": LongTensor scalar,
        "prob_graph": str,
        "prob_text": str,
      }
    """

    def __init__(
        self,
        jsonl_path: str,
        graph_cfg: Optional[GraphBuildConfig] = None,
        base_dir: Optional[str] = None,
        graphs_dir: Optional[str] = None,
        max_text_chars: Optional[int] = 8000,
    ):
        self.jsonl_path = jsonl_path
        self.base_dir = base_dir  # if set, resolves relative paths against this directory
        self.graphs_dir = graphs_dir  # if set, resolves graph_path against this directory
        self.graph_cfg = graph_cfg or GraphBuildConfig()
        self.max_text_chars = max_text_chars

        raw = _read_jsonl(jsonl_path)
        self.items: List[PairExample] = []
        for r in raw:
            # Validate required keys exist
            for k in ("graph_path", "text_path", "prob_graph", "prob_text", "label"):
                if k not in r:
                    raise KeyError(f"Missing key '{k}' in {jsonl_path}: {r}")

            self.items.append(
                PairExample(
                    graph_path=r["graph_path"],
                    text_path=r["text_path"],
                    prob_graph=r["prob_graph"],
                    prob_text=r["prob_text"],
                    label=int(r["label"]),
                )
            )

        # Binary by construction, but keep flexible
        self.num_classes = 2

    def __len__(self) -> int:
        return len(self.items)

    def _resolve(self, p: str, use_graphs_dir: bool = False) -> str:
        # Many of your jsonl entries mix relative ("graphs_mps/...") and absolute/../ paths.
        # If graphs_dir is set and use_graphs_dir, resolve graph paths against it.
        # Else if base_dir is set, resolve relative paths to it; else use jsonl file directory.
        if os.path.isabs(p):
            return p
        if use_graphs_dir and self.graphs_dir is not None:
            root = self.graphs_dir
        else:
            root = self.base_dir or os.path.dirname(os.path.abspath(self.jsonl_path))
        if root and not os.path.isabs(root):
            root = os.path.abspath(root)
        return os.path.normpath(os.path.join(root, p))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]

        gpath = self._resolve(ex.graph_path, use_graphs_dir=True)
        tpath = self._resolve(ex.text_path)

        # ---- graph ----
        # Expecting your MPS-derived JSON schema (vars/constrs/edges with coef, rhs, etc.)
        with open(gpath, "r", encoding="utf-8") as f:
            g_dict = json.load(f)
        graph = build_pyg_from_mps_graph_json(g_dict, cfg=self.graph_cfg)


        # ---- text ----
        with open(tpath, "r", encoding="utf-8") as f:
            text = f.read()
        if self.max_text_chars is not None and len(text) > self.max_text_chars:
            text = text[: self.max_text_chars]

        y = torch.tensor(ex.label, dtype=torch.long)

        return {
            "graph": graph,
            "text": text,
            "y": y,
            "prob_graph": ex.prob_graph,
            "prob_text": ex.prob_text,
        }
   