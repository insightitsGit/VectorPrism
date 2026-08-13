"""
causal_graph.py - Document-level causal DAG for Stage-1 candidate expansion.

Edges point cause → effect (earlier_doc_id → later_doc_id).
Expansion from dense seeds walks upstream (effects → causes) and optionally
downstream so root-cause golds enter the Stage-2 pool even when dense rank > 100.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


class CausalDocGraph:
    def __init__(self) -> None:
        self.upstream: Dict[str, Set[str]] = defaultdict(set)  # effect -> causes
        self.downstream: Dict[str, Set[str]] = defaultdict(set)  # cause -> effects

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CausalDocGraph":
        g = cls()
        path = Path(path)
        if not path.exists():
            return g
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            earlier = row.get("earlier_doc_id") or row.get("cause_doc_id")
            later = row.get("later_doc_id") or row.get("effect_doc_id")
            if earlier and later:
                g.add_edge(str(earlier), str(later))
        return g

    def add_edge(self, earlier_doc_id: str, later_doc_id: str) -> None:
        self.downstream[earlier_doc_id].add(later_doc_id)
        self.upstream[later_doc_id].add(earlier_doc_id)

    def expand(
        self,
        seed_ids: Iterable[str],
        hops: int = 2,
        *,
        upstream: bool = True,
        downstream: bool = False,
    ) -> Dict[str, int]:
        """Return {doc_id: min_hops_from_any_seed} including seeds at hop 0."""
        dist: Dict[str, int] = {}
        q: deque[tuple[str, int]] = deque()
        for s in seed_ids:
            s = str(s)
            dist[s] = 0
            q.append((s, 0))
        while q:
            node, d = q.popleft()
            if d >= hops:
                continue
            neighbors: List[str] = []
            if upstream:
                neighbors.extend(self.upstream.get(node, ()))
            if downstream:
                neighbors.extend(self.downstream.get(node, ()))
            for nb in neighbors:
                if nb not in dist:
                    dist[nb] = d + 1
                    q.append((nb, d + 1))
        return dist

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.downstream.values())
