"""
eval_runner.py - Phase 1+ retrieval quality evaluation and baseline comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import json

import numpy as np
import torch

from base_encoder import BaseTextEncoder
from channel_datasets import load_eval_examples_raw, load_documents_jsonl
from checkpointing import load_checkpoint
from eval_harness import EvalExample, evaluate
from ablation_harness import run_ablation, print_ablation_report
from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
from retrieval_engine import PSMRetrievalEngine
from db_client import VectorDBClient
from tensor_contract import PSMTensorContract as C


class InMemoryCorpusDB(VectorDBClient):
    def __init__(self):
        self.rows: List[dict] = []

    def upsert(self, doc_id, chunk_text, tensor_1024d, meta):
        # Replace existing id if present
        self.rows = [r for r in self.rows if r["document_id"] != doc_id]
        self.rows.append({
            "document_id": doc_id,
            "chunk_text": chunk_text,
            "tensor_1024d": np.asarray(tensor_1024d, dtype=np.float32).copy(),
            "epistemic_truth": float(meta["epistemic_truth"]),
            "anchor_dist": float(meta["anchor_dist"]),
            "valid_timestamp": int(meta["valid_timestamp"]),
            "model_version": int(meta.get("model_version", 0)),
        })

    def query_dense_slice(self, vector_slice, min_truth, max_anchor_dist, limit):
        scored = []
        for r in self.rows:
            if r["epistemic_truth"] < min_truth or r["anchor_dist"] > max_anchor_dist:
                continue
            dense = r["tensor_1024d"][C.DENSE_CORE.start:C.DENSE_CORE.end]
            score = float(dense @ vector_slice)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]


@dataclass
class BaselineReport:
    vectorprism: Dict[str, float]
    dense_cosine_baseline: Dict[str, float]
    beats_or_ties_baseline: bool


@torch.no_grad()
def build_eval_set(
    eval_jsonl: str,
    encoder: BaseTextEncoder,
    pipeline: VectorPrismIngestPipeline,
) -> List[EvalExample]:
    raw = load_eval_examples_raw(eval_jsonl)
    out = []
    for row in raw:
        q = pipeline.encode_query(str(row["query"]))
        out.append(
            EvalExample(
                query_text=str(row["query"]),
                query_1024d=q,
                relevant_doc_ids=set(str(x) for x in row["relevant_doc_ids"]),
            )
        )
    return out


def dense_cosine_baseline(
    eval_set: List[EvalExample],
    corpus_tensors: Dict[str, np.ndarray],
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Plain cosine on dense_core_slice only (no Stage-2 multi-channel fusion)."""
    doc_ids = list(corpus_tensors.keys())
    mat = np.stack([corpus_tensors[i][C.DENSE_CORE.start:C.DENSE_CORE.end] for i in doc_ids])
    recalls = {k: [] for k in k_values}
    rr = []
    max_k = max(k_values)
    for ex in eval_set:
        q = ex.query_1024d[C.DENSE_CORE.start:C.DENSE_CORE.end]
        scores = mat @ q
        order = np.argsort(scores)[::-1][:max_k]
        retrieved = [doc_ids[i] for i in order]
        for k in k_values:
            top = set(retrieved[:k])
            recalls[k].append(len(top & ex.relevant_doc_ids) / max(len(ex.relevant_doc_ids), 1))
        rank = 0.0
        for i, d in enumerate(retrieved, start=1):
            if d in ex.relevant_doc_ids:
                rank = 1.0 / i
                break
        rr.append(rank)
    report = {f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()}
    report["MRR"] = float(np.mean(rr))
    report["n_eval_examples"] = float(len(eval_set))
    return report


def run_phase1_eval(
    checkpoint: str,
    encoder: BaseTextEncoder,
    documents_jsonl: str,
    eval_jsonl: str,
    hard_truth_filter: bool = False,
) -> BaselineReport:
    ckpt = load_checkpoint(checkpoint)
    db = InMemoryCorpusDB()
    pipe = VectorPrismIngestPipeline(
        encoder,
        ckpt["adapter"],
        db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
    )
    docs = load_documents_jsonl(documents_jsonl)
    pipe.upsert_documents([
        IngestDocument(
            document_id=str(d["document_id"]),
            chunk_text=str(d["chunk_text"]),
            epistemic_truth=float(d.get("epistemic_truth", 1.0)),
        )
        for d in docs
    ])
    eval_set = build_eval_set(eval_jsonl, encoder, pipe)
    engine = PSMRetrievalEngine(
        db,
        causal_matrix=ckpt["causal_matrix"],
        hard_truth_filter=hard_truth_filter,
    )
    # Force dense-only weights for Phase-1 DoD comparison fairness
    engine.classifier.classify = lambda _q: (
        np.array([1.0, 0, 0, 0, 0], dtype=np.float32),
        {"min_truth": 0.0, "max_anchor_dist": 1.0},
    )
    vp = evaluate(engine, eval_set)
    corpus = {r["document_id"]: r["tensor_1024d"] for r in db.rows}
    base = dense_cosine_baseline(eval_set, corpus)
    beats = vp["recall@10"] >= base["recall@10"] - 1e-9
    return BaselineReport(vectorprism=vp, dense_cosine_baseline=base, beats_or_ties_baseline=beats)


def run_ablation_eval(
    checkpoint: str,
    encoder: BaseTextEncoder,
    documents_jsonl: str,
    eval_jsonl: str,
) -> Dict[str, Dict[str, float]]:
    ckpt = load_checkpoint(checkpoint)
    db = InMemoryCorpusDB()
    pipe = VectorPrismIngestPipeline(
        encoder, ckpt["adapter"], db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
    )
    docs = load_documents_jsonl(documents_jsonl)
    pipe.upsert_documents([
        IngestDocument(str(d["document_id"]), str(d["chunk_text"]),
                       epistemic_truth=float(d.get("epistemic_truth", 1.0)))
        for d in docs
    ])
    eval_set = build_eval_set(eval_jsonl, encoder, pipe)
    engine = PSMRetrievalEngine(db, causal_matrix=ckpt["causal_matrix"])
    results = run_ablation(engine, eval_set)
    print_ablation_report(results)
    return results


def write_report(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
