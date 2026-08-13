"""
dense_dataset.py - Phase 1 labeled data loader for the dense channel.

Expected JSONL schema (one object per line):
  {"query": "...", "passage": "..."}

Optional fields are ignored. This loader does not fabricate pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from base_encoder import BaseTextEncoder
from training import PSMBatch


class DensePairRecord:
    __slots__ = ("query", "passage")

    def __init__(self, query: str, passage: str):
        self.query = query
        self.passage = passage


def load_dense_pairs_jsonl(path: str | Path) -> List[DensePairRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dense pair file not found: {path}")
    records: List[DensePairRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}") from e
            if "query" not in obj or "passage" not in obj:
                raise ValueError(
                    f"Line {lineno} missing required keys 'query' and 'passage': {obj!r}"
                )
            q, p = str(obj["query"]).strip(), str(obj["passage"]).strip()
            if not q or not p:
                raise ValueError(f"Line {lineno} has empty query or passage")
            records.append(DensePairRecord(q, p))
    if not records:
        raise ValueError(f"No dense pairs loaded from {path}")
    return records


class DensePairDataset(Dataset):
    """Encodes (query, passage) pairs with a frozen BaseTextEncoder."""

    def __init__(self, pairs: Sequence[DensePairRecord], encoder: BaseTextEncoder):
        self.pairs = list(pairs)
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.pairs[idx]
        emb = self.encoder.encode([rec.query, rec.passage])
        return emb[0], emb[1]


def collate_dense_batch(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> PSMBatch:
    anchors = torch.stack([a for a, _ in items], dim=0)
    positives = torch.stack([p for _, p in items], dim=0)
    return PSMBatch(
        dense_anchor=anchors,
        dense_positive=positives,
        header=torch.zeros(anchors.shape[0], 16),
    )


def make_dense_dataloader(
    jsonl_path: str | Path,
    encoder: BaseTextEncoder,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    pairs = load_dense_pairs_jsonl(jsonl_path)
    ds = DensePairDataset(pairs, encoder)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dense_batch,
    )
