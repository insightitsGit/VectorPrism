"""
intrinsic_runner.py - Per-channel structural DoD checks (Phase 2/3).
"""

from __future__ import annotations

from typing import Dict, Any, List
import json
from pathlib import Path

import numpy as np
import torch

from vectorprism.base_encoder import BaseTextEncoder
from vectorprism.checkpointing import load_checkpoint
from vectorprism.channel_datasets import (
    load_causal_jsonl,
    load_relational_jsonl,
    load_hyperbolic_jsonl,
    load_disentangled_jsonl,
    load_ood_jsonl,
    load_identity_jsonl,
)
from vectorprism.intrinsic_validation import (
    causal_order_accuracy,
    link_prediction_eval,
    embedding_distortion,
    disentanglement_probe,
    ood_detection_auroc,
)
from vectorprism.losses import anchor_distance_score, poincare_distance
from vectorprism.tensor_contract import PSMTensorContract as C


@torch.no_grad()
def _project(adapter, encoder, texts: List[str], key: str) -> torch.Tensor:
    base = encoder.encode(texts)
    header = torch.zeros(base.shape[0], 16)
    _, raw = adapter(base, header)
    return raw[key]


def eval_causal_intrinsic(checkpoint: str, encoder: BaseTextEncoder, data_path: str) -> Dict[str, Any]:
    ckpt = load_checkpoint(checkpoint)
    adapter = ckpt["adapter"]
    M = torch.from_numpy(ckpt["causal_matrix"])
    rows = load_causal_jsonl(data_path)
    earlier = _project(adapter, encoder, [r["earlier"] for r in rows], "causal")
    later = _project(adapter, encoder, [r["later"] for r in rows], "causal")
    fwd = torch.einsum("bi,ij,bj->b", earlier, M, later).numpy()
    bwd = torch.einsum("bi,ij,bj->b", later, M, earlier).numpy()
    acc = causal_order_accuracy(fwd, bwd)
    return {
        "channel": "causal",
        "causal_order_accuracy": acc,
        "pass_intrinsic": acc >= 0.7,
        "n": len(rows),
    }


def eval_relational_intrinsic(checkpoint: str, encoder: BaseTextEncoder, data_path: str,
                              n_corruptions: int = 10) -> Dict[str, Any]:
    ckpt = load_checkpoint(checkpoint)
    adapter = ckpt["adapter"]
    table = ckpt["relation_table"]
    vocab = ckpt["relation_vocab"]
    if table is None or vocab is None:
        raise ValueError("Checkpoint missing relation_table/vocab for relational intrinsic eval")
    rows, _ = load_relational_jsonl(data_path)
    # Restrict to known relations
    rows = [r for r in rows if str(r["relation"]) in vocab.stoi]
    if not rows:
        raise ValueError("No overlapping relations between eval data and checkpoint vocab")

    subjects = _project(adapter, encoder, [r["subject"] for r in rows], "relational")
    objects = _project(adapter, encoder, [r["object"] for r in rows], "relational")
    neg_pool = _project(adapter, encoder, [r["negative_object"] for r in rows], "relational")
    rel_ids = torch.tensor([vocab.encode(str(r["relation"])) for r in rows], dtype=torch.long)
    rel = table(rel_ids)

    true_dist = torch.norm(subjects + rel - objects, p=2, dim=-1).numpy()
    rng = np.random.default_rng(0)
    corrupt = []
    for i in range(len(rows)):
        # Sample corruptions from negative pool with replacement
        idx = rng.integers(0, len(rows), size=n_corruptions)
        neg = neg_pool[idx]
        d = torch.norm(subjects[i] + rel[i] - neg, p=2, dim=-1).numpy()
        corrupt.append(d)
    report = link_prediction_eval(true_dist, np.stack(corrupt))
    hits10 = report.get("Hits@10", 0.0)
    return {
        "channel": "relational",
        **report,
        "pass_intrinsic": hits10 >= 0.3,
        "n": len(rows),
    }


def eval_hyperbolic_intrinsic(checkpoint: str, encoder: BaseTextEncoder, data_path: str) -> Dict[str, Any]:
    ckpt = load_checkpoint(checkpoint)
    adapter = ckpt["adapter"]
    rows = load_hyperbolic_jsonl(data_path)
    parents = _project(adapter, encoder, [r["parent"] for r in rows], "hyperbolic")
    children = _project(adapter, encoder, [r["child"] for r in rows], "hyperbolic")
    # Graph distance = 1 for edges; embedding distance = Poincare
    graph = np.ones((len(rows), len(rows)), dtype=np.float64)
    # Use only edge list distortion proxy: parent-child should be closer than parent-negative
    emb_edge = poincare_distance(parents, children).numpy()
    neg_texts = [r["negatives"][0] for r in rows]
    negs = _project(adapter, encoder, neg_texts, "hyperbolic")
    emb_neg = poincare_distance(parents, negs).numpy()
    edge_closer_rate = float(np.mean(emb_edge < emb_neg))
    # Build small distortion matrices for the metric helper
    n = min(len(rows), 32)
    g = np.ones((n, n))
    e = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e[i, j] = float(poincare_distance(parents[i:i+1], children[j:j+1])[0])
    dist = embedding_distortion(g, e)
    return {
        "channel": "hyperbolic",
        "edge_closer_than_negative_rate": edge_closer_rate,
        **dist,
        "pass_intrinsic": edge_closer_rate >= 0.6,
        "n": len(rows),
    }


def eval_disentangled_intrinsic(checkpoint: str, encoder: BaseTextEncoder, data_path: str) -> Dict[str, Any]:
    ckpt = load_checkpoint(checkpoint)
    adapter = ckpt["adapter"]
    rows, label_vocab = load_disentangled_jsonl(data_path)
    z = _project(adapter, encoder, [r["text"] for r in rows], "disentangled_z").numpy()
    intended = np.array([label_vocab.encode(r["label"]) for r in rows])
    # Synthetic nuisance: hash of text length parity (should NOT be encoded if disentangled)
    nuisance = np.array([len(str(r["text"])) % 2 for r in rows])
    probe = disentanglement_probe(z, intended, nuisance)
    return {
        "channel": "disentangled",
        **probe,
        "pass_intrinsic": probe["intended_label_accuracy"] >= 0.5,
        "n": len(rows),
    }


def eval_identity_intrinsic(
    checkpoint: str,
    encoder: BaseTextEncoder,
    in_domain_jsonl: str,
    ood_jsonl: str,
) -> Dict[str, Any]:
    ckpt = load_checkpoint(checkpoint)
    adapter = ckpt["adapter"]
    in_texts = load_identity_jsonl(in_domain_jsonl)
    ood_rows = load_ood_jsonl(ood_jsonl)
    ood_texts = [t for t, y in ood_rows if y == 1]
    if not ood_texts:
        ood_texts = [t for t, _ in ood_rows]
    in_id = _project(adapter, encoder, in_texts, "identity")
    ood_id = _project(adapter, encoder, ood_texts, "identity")
    v0 = adapter.identity_anchor_v0
    in_dist = anchor_distance_score(in_id, v0).numpy()
    ood_dist = anchor_distance_score(ood_id, v0).numpy()
    auc = ood_detection_auroc(in_dist, ood_dist)
    return {
        "channel": "identity",
        "ood_auroc": auc,
        "pass_intrinsic": auc >= 0.7,
        "n_in": len(in_texts),
        "n_ood": len(ood_texts),
    }


def run_intrinsic(channel: str, checkpoint: str, encoder: BaseTextEncoder, data_path: str,
                  ood_path: str | None = None) -> Dict[str, Any]:
    channel = channel.lower()
    if channel == "causal":
        return eval_causal_intrinsic(checkpoint, encoder, data_path)
    if channel == "relational":
        return eval_relational_intrinsic(checkpoint, encoder, data_path)
    if channel == "hyperbolic":
        return eval_hyperbolic_intrinsic(checkpoint, encoder, data_path)
    if channel == "disentangled":
        return eval_disentangled_intrinsic(checkpoint, encoder, data_path)
    if channel == "identity":
        if not ood_path:
            raise ValueError("identity intrinsic eval requires --ood JSONL")
        return eval_identity_intrinsic(checkpoint, encoder, data_path, ood_path)
    raise ValueError(f"No intrinsic runner for channel {channel}")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
