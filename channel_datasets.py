"""
channel_datasets.py - Labeled JSONL loaders for every VectorPrism channel.

Schemas (one JSON object per line):

  dense:         {"query": str, "passage": str}
  relational:    {"subject": str, "relation": str, "object": str, "negative_object": str}
  disentangled:  {"text": str, "label": str|int}
  hyperbolic:    {"parent": str, "child": str, "negatives": [str, ...]}
  identity:      {"text": str, "in_domain": true}
  causal:        {"earlier": str, "later": str}
  documents:     {"document_id": str, "chunk_text": str, "epistemic_truth"?: float}
  eval:          {"query": str, "relevant_doc_ids": [str, ...]}
  truth:         {"text": str, "is_true": 0|1|bool}
  ood:           {"text": str, "is_ood": 0|1|bool}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from base_encoder import BaseTextEncoder
from jsonl_utils import load_jsonl
from training import PSMBatch
from dense_dataset import (
    DensePairRecord,
    DensePairDataset,
    collate_dense_batch,
    load_dense_pairs_jsonl,
    make_dense_dataloader,
)


# Re-export dense helpers
__all__ = [
    "load_dense_pairs_jsonl",
    "make_dense_dataloader",
    "RelationVocab",
    "make_channel_dataloader",
    "load_eval_examples_raw",
    "load_documents_jsonl",
]


@dataclass
class RelationVocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_relations(cls, relations: Sequence[str]) -> "RelationVocab":
        uniq = sorted(set(relations))
        if not uniq:
            raise ValueError("RelationVocab requires at least one relation string")
        stoi = {r: i for i, r in enumerate(uniq)}
        return cls(stoi=stoi, itos=uniq)

    def encode(self, relation: str) -> int:
        if relation not in self.stoi:
            raise KeyError(f"Unknown relation {relation!r}; known={self.itos}")
        return self.stoi[relation]

    @property
    def size(self) -> int:
        return len(self.itos)


class LabelVocab:
    def __init__(self, labels: Sequence[str]):
        uniq = sorted(set(str(x) for x in labels))
        if not uniq:
            raise ValueError("LabelVocab empty")
        self.stoi = {x: i for i, x in enumerate(uniq)}
        self.itos = uniq

    def encode(self, label) -> int:
        key = str(label)
        if key not in self.stoi:
            raise KeyError(f"Unknown label {key!r}")
        return self.stoi[key]

    @property
    def size(self) -> int:
        return len(self.itos)


def load_relational_jsonl(path: str | Path) -> Tuple[List[dict], RelationVocab]:
    rows = load_jsonl(path)
    for i, r in enumerate(rows, start=1):
        for k in ("subject", "relation", "object", "negative_object"):
            if k not in r or not str(r[k]).strip():
                raise ValueError(f"{path}:{i} invalid relational row: {r!r}")
    vocab = RelationVocab.from_relations([str(r["relation"]) for r in rows])
    return rows, vocab


def load_disentangled_jsonl(path: str | Path) -> Tuple[List[dict], LabelVocab]:
    rows = load_jsonl(path)
    for i, r in enumerate(rows, start=1):
        if "text" not in r or "label" not in r or not str(r["text"]).strip():
            raise ValueError(f"{path}:{i} invalid disentangled row: {r!r}")
    vocab = LabelVocab([r["label"] for r in rows])
    return rows, vocab


def load_hyperbolic_jsonl(path: str | Path) -> List[dict]:
    rows = load_jsonl(path)
    for i, r in enumerate(rows, start=1):
        if "parent" not in r or "child" not in r:
            raise ValueError(f"{path}:{i} invalid hyperbolic row: {r!r}")
        negs = r.get("negatives", [])
        if not isinstance(negs, list) or not negs:
            raise ValueError(f"{path}:{i} hyperbolic row needs non-empty negatives list")
    return rows


def load_identity_jsonl(path: str | Path) -> List[str]:
    rows = load_jsonl(path)
    texts = []
    for i, r in enumerate(rows, start=1):
        if "text" not in r or not str(r["text"]).strip():
            raise ValueError(f"{path}:{i} invalid identity row: {r!r}")
        texts.append(str(r["text"]).strip())
    return texts


def load_causal_jsonl(path: str | Path) -> List[dict]:
    rows = load_jsonl(path)
    for i, r in enumerate(rows, start=1):
        if "earlier" not in r or "later" not in r:
            raise ValueError(f"{path}:{i} invalid causal row: {r!r}")
    return rows


def load_documents_jsonl(path: str | Path) -> List[dict]:
    rows = load_jsonl(path)
    for i, r in enumerate(rows, start=1):
        if "document_id" not in r or "chunk_text" not in r:
            raise ValueError(f"{path}:{i} invalid document row: {r!r}")
    return rows


def load_eval_examples_raw(path: str | Path) -> List[dict]:
    rows = load_jsonl(path)
    for i, r in enumerate(rows, start=1):
        if "query" not in r or "relevant_doc_ids" not in r:
            raise ValueError(f"{path}:{i} invalid eval row: {r!r}")
        if not isinstance(r["relevant_doc_ids"], list) or not r["relevant_doc_ids"]:
            raise ValueError(f"{path}:{i} relevant_doc_ids must be a non-empty list")
    return rows


def load_truth_jsonl(path: str | Path) -> List[Tuple[str, int]]:
    rows = load_jsonl(path)
    out = []
    for i, r in enumerate(rows, start=1):
        if "text" not in r or "is_true" not in r:
            raise ValueError(f"{path}:{i} invalid truth row: {r!r}")
        y = r["is_true"]
        label = int(bool(y)) if not isinstance(y, str) else int(y in ("1", "true", "True"))
        out.append((str(r["text"]).strip(), label))
    return out


def load_ood_jsonl(path: str | Path) -> List[Tuple[str, int]]:
    rows = load_jsonl(path)
    out = []
    for i, r in enumerate(rows, start=1):
        if "text" not in r or "is_ood" not in r:
            raise ValueError(f"{path}:{i} invalid ood row: {r!r}")
        y = r["is_ood"]
        label = int(bool(y)) if not isinstance(y, str) else int(y in ("1", "true", "True"))
        out.append((str(r["text"]).strip(), label))
    return out


class _EncodedChannelDataset(Dataset):
    """Encodes text fields on the fly into a channel-specific PSMBatch fragment."""

    def __init__(self, channel: str, rows: List[dict], encoder: BaseTextEncoder,
                 relation_vocab: Optional[RelationVocab] = None,
                 label_vocab: Optional[LabelVocab] = None):
        self.channel = channel
        self.rows = rows
        self.encoder = encoder
        self.relation_vocab = relation_vocab
        self.label_vocab = label_vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        enc = self.encoder
        if self.channel == "relational":
            emb = enc.encode([r["subject"], r["object"], r["negative_object"]])
            rid = self.relation_vocab.encode(str(r["relation"]))
            return {
                "rel_subject": emb[0],
                "rel_object": emb[1],
                "rel_neg_object": emb[2],
                "rel_relation_id": torch.tensor(rid, dtype=torch.long),
            }
        if self.channel == "disentangled":
            emb = enc.encode([r["text"]])[0]
            y = self.label_vocab.encode(r["label"])
            return {
                "dis_input": emb,
                "dis_label": torch.tensor(y, dtype=torch.long),
            }
        if self.channel == "hyperbolic":
            texts = [r["parent"], r["child"], *list(r["negatives"])]
            emb = enc.encode(texts)
            return {
                "hyp_parent": emb[0],
                "hyp_child": emb[1],
                "hyp_negatives": emb[2:],
            }
        if self.channel == "identity":
            emb = enc.encode([r if isinstance(r, str) else r["text"]])[0]
            return {"identity_in_domain": emb}
        if self.channel == "causal":
            emb = enc.encode([r["earlier"], r["later"]])
            return {"causal_earlier": emb[0], "causal_later": emb[1]}
        raise ValueError(f"Unsupported channel dataset: {self.channel}")


def _collate_channel(channel: str, items: List[dict]) -> PSMBatch:
    B = len(items)
    header = torch.zeros(B, 16)
    if channel == "relational":
        return PSMBatch(
            rel_subject=torch.stack([x["rel_subject"] for x in items]),
            rel_object=torch.stack([x["rel_object"] for x in items]),
            rel_neg_object=torch.stack([x["rel_neg_object"] for x in items]),
            rel_relation_id=torch.stack([x["rel_relation_id"] for x in items]),
            header=header,
        )
    if channel == "disentangled":
        return PSMBatch(
            dis_input=torch.stack([x["dis_input"] for x in items]),
            dis_label=torch.stack([x["dis_label"] for x in items]),
            header=header,
        )
    if channel == "hyperbolic":
        # Pad negatives to max-k in batch
        ks = [x["hyp_negatives"].shape[0] for x in items]
        k = max(ks)
        d = items[0]["hyp_negatives"].shape[-1]
        negs = torch.zeros(B, k, d)
        for i, x in enumerate(items):
            n_i = x["hyp_negatives"].shape[0]
            negs[i, :n_i] = x["hyp_negatives"]
            if n_i < k:
                # repeat last negative to fill (keeps loss well-defined)
                negs[i, n_i:] = x["hyp_negatives"][-1]
        return PSMBatch(
            hyp_parent=torch.stack([x["hyp_parent"] for x in items]),
            hyp_child=torch.stack([x["hyp_child"] for x in items]),
            hyp_negatives=negs,
            header=header,
        )
    if channel == "identity":
        return PSMBatch(
            identity_in_domain=torch.stack([x["identity_in_domain"] for x in items]),
            header=header,
        )
    if channel == "causal":
        return PSMBatch(
            causal_earlier=torch.stack([x["causal_earlier"] for x in items]),
            causal_later=torch.stack([x["causal_later"] for x in items]),
            header=header,
        )
    raise ValueError(channel)


def make_channel_dataloader(
    channel: str,
    path: str | Path,
    encoder: BaseTextEncoder,
    batch_size: int = 16,
    shuffle: bool = True,
    relation_vocab: Optional[RelationVocab] = None,
    label_vocab: Optional[LabelVocab] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, dict]:
    """Returns (dataloader, aux) where aux may include relation_vocab / label_vocab."""
    channel = channel.lower()
    aux: dict = {}

    if channel == "dense":
        return make_dense_dataloader(path, encoder, batch_size=batch_size, shuffle=shuffle), aux

    if channel == "relational":
        rows, vocab = load_relational_jsonl(path)
        if relation_vocab is not None:
            vocab = relation_vocab
        aux["relation_vocab"] = vocab
        ds = _EncodedChannelDataset("relational", rows, encoder, relation_vocab=vocab)
    elif channel == "disentangled":
        rows, vocab = load_disentangled_jsonl(path)
        if label_vocab is not None:
            vocab = label_vocab
        aux["label_vocab"] = vocab
        aux["num_classes"] = vocab.size
        ds = _EncodedChannelDataset("disentangled", rows, encoder, label_vocab=vocab)
    elif channel == "hyperbolic":
        rows = load_hyperbolic_jsonl(path)
        ds = _EncodedChannelDataset("hyperbolic", rows, encoder)
    elif channel == "identity":
        texts = load_identity_jsonl(path)
        rows = [{"text": t} for t in texts]
        ds = _EncodedChannelDataset("identity", rows, encoder)
    elif channel == "causal":
        rows = load_causal_jsonl(path)
        ds = _EncodedChannelDataset("causal", rows, encoder)
    else:
        raise ValueError(
            f"Unknown channel {channel!r}. "
            "Use dense|relational|disentangled|hyperbolic|identity|causal"
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda items: _collate_channel(channel, items),
    )
    return loader, aux
