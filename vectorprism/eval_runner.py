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

from vectorprism.base_encoder import BaseTextEncoder
from vectorprism.channel_datasets import load_eval_examples_raw, load_documents_jsonl
from vectorprism.checkpointing import load_checkpoint
from vectorprism.eval_harness import EvalExample, evaluate
from vectorprism.ablation_harness import run_ablation, print_ablation_report
from vectorprism.ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
from vectorprism.retrieval_engine import PSMRetrievalEngine
from vectorprism.db_client import VectorDBClient
from vectorprism.tensor_contract import PSMTensorContract as C


class InMemoryCorpusDB(VectorDBClient):
    """In-memory corpus with optional NPZ persistence across CLI processes."""

    DEFAULT_STORE = "checkpoints/memory_corpus.npz"

    def __init__(self, store_path: Optional[str | Path] = None, *, autoload: bool = True):
        self.rows: List[dict] = []
        self.store_path: Optional[Path] = Path(store_path) if store_path else None
        if autoload and self.store_path is not None and self.store_path.is_file():
            self.load(self.store_path)

    def upsert(self, doc_id, chunk_text, tensor_1024d, meta):
        # Replace existing id if present
        self.rows = [r for r in self.rows if r["document_id"] != doc_id]
        vt = meta.get("valid_to_timestamp")
        tx = meta.get("transaction_timestamp")
        self.rows.append({
            "document_id": doc_id,
            "chunk_text": chunk_text,
            "tensor_1024d": np.asarray(tensor_1024d, dtype=np.float32).copy(),
            "epistemic_truth": float(meta["epistemic_truth"]),
            "anchor_dist": float(meta["anchor_dist"]),
            "valid_timestamp": int(meta["valid_timestamp"]),
            "valid_to_timestamp": None if vt is None else int(vt),
            "transaction_timestamp": None if tx is None else int(tx),
            "model_version": int(meta.get("model_version", 0)),
        })

    def query_dense_slice(
        self,
        vector_slice,
        min_truth,
        max_anchor_dist,
        limit,
        model_version: Optional[int] = None,
        as_of: Optional[int] = None,
        as_of_transaction: Optional[int] = None,
    ):
        from vectorprism.bitemporal import passes_bitemporal_filters

        scored = []
        for r in self.rows:
            if r["epistemic_truth"] < min_truth or r["anchor_dist"] > max_anchor_dist:
                continue
            if model_version is not None and int(r.get("model_version", 0)) != int(model_version):
                continue
            if not passes_bitemporal_filters(
                r, as_of=as_of, as_of_transaction=as_of_transaction
            ):
                continue
            dense = r["tensor_1024d"][C.DENSE_CORE.start:C.DENSE_CORE.end]
            score = float(dense @ vector_slice)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    def get_by_ids(self, doc_ids):
        want = set(str(x) for x in doc_ids)
        return [r for r in self.rows if r["document_id"] in want]

    def save(self, path: Optional[str | Path] = None) -> Path:
        out = Path(path or self.store_path or self.DEFAULT_STORE)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            np.savez_compressed(
                out,
                tensors=np.zeros((0, 1024), dtype=np.float32),
                meta=np.zeros((0, 3), dtype=np.float64),
                valid_timestamps=np.zeros((0,), dtype=np.int64),
                valid_to_timestamps=np.full((0,), -1, dtype=np.int64),
                transaction_timestamps=np.full((0,), -1, dtype=np.int64),
                ids=np.array([], dtype=object),
                texts=np.array([], dtype=object),
            )
        else:
            tensors = np.stack([r["tensor_1024d"] for r in self.rows]).astype(np.float32)
            meta = np.asarray(
                [
                    [
                        r["epistemic_truth"],
                        r["anchor_dist"],
                        float(r["model_version"]),
                    ]
                    for r in self.rows
                ],
                dtype=np.float64,
            )
            valid_timestamps = np.asarray(
                [int(r["valid_timestamp"]) for r in self.rows], dtype=np.int64
            )
            # -1 sentinel = NULL / open-ended / unset (int64 arrays cannot hold None)
            valid_to_timestamps = np.asarray(
                [
                    -1 if r.get("valid_to_timestamp") is None else int(r["valid_to_timestamp"])
                    for r in self.rows
                ],
                dtype=np.int64,
            )
            transaction_timestamps = np.asarray(
                [
                    -1
                    if r.get("transaction_timestamp") is None
                    else int(r["transaction_timestamp"])
                    for r in self.rows
                ],
                dtype=np.int64,
            )
            ids = np.asarray([r["document_id"] for r in self.rows], dtype=object)
            texts = np.asarray([r["chunk_text"] for r in self.rows], dtype=object)
            np.savez_compressed(
                out,
                tensors=tensors,
                meta=meta,
                valid_timestamps=valid_timestamps,
                valid_to_timestamps=valid_to_timestamps,
                transaction_timestamps=transaction_timestamps,
                ids=ids,
                texts=texts,
            )
        self.store_path = out
        return out

    def load(self, path: str | Path) -> int:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        tensors = data["tensors"]
        meta = data["meta"]
        ids = data["ids"]
        texts = data["texts"]
        # New format: valid_timestamps int64 array. Legacy: meta[:, 2] was timestamp.
        if "valid_timestamps" in data.files:
            valid_timestamps = np.asarray(data["valid_timestamps"], dtype=np.int64)
            legacy_ts_in_meta = False
        else:
            valid_timestamps = None
            legacy_ts_in_meta = True
        self.rows = []
        has_to = "valid_to_timestamps" in data.files
        has_tx = "transaction_timestamps" in data.files
        for i in range(len(ids)):
            if legacy_ts_in_meta:
                # float64 can hold unix-seconds exactly; cast via int
                ts = int(meta[i, 2])
                ver = int(meta[i, 3]) if meta.shape[1] > 3 else 0
                vt = None
                tx = None
            else:
                ts = int(valid_timestamps[i])
                ver = int(meta[i, 2]) if meta.shape[1] > 2 else 0
                if has_to:
                    raw_to = int(data["valid_to_timestamps"][i])
                    vt = None if raw_to < 0 else raw_to
                else:
                    vt = None
                if has_tx:
                    raw_tx = int(data["transaction_timestamps"][i])
                    tx = None if raw_tx < 0 else raw_tx
                else:
                    tx = None
            self.rows.append(
                {
                    "document_id": str(ids[i]),
                    "chunk_text": str(texts[i]),
                    "tensor_1024d": np.asarray(tensors[i], dtype=np.float32),
                    "epistemic_truth": float(meta[i, 0]),
                    "anchor_dist": float(meta[i, 1]),
                    "valid_timestamp": ts,
                    "valid_to_timestamp": vt,
                    "transaction_timestamp": tx,
                    "model_version": ver,
                }
            )
        self.store_path = path
        return len(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


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
