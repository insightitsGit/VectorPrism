"""
eval_harness.py - Retrieval QUALITY evaluation (recall@k, MRR).

This is the piece that was entirely missing before: benchmark_harness.py
only ever measured speed. A latency number tells you nothing about
whether the system retrieves the right documents. This harness requires
a labeled eval set you provide — a list of (query_text, query_1024d,
relevant_document_ids) triples — because relevance judgments cannot be
synthesized; they encode what a human considers correct for that query.

Usage sketch:
    eval_set = [
        EvalExample(query_text="...", query_1024d=<np.ndarray (1024,)>,
                    relevant_doc_ids={"doc_17", "doc_42"}),
        ...
    ]
    report = evaluate(engine, eval_set, k_values=[1, 5, 10])
    print(report)
"""

from dataclasses import dataclass
from typing import List, Dict, Set
import numpy as np

from vectorprism.retrieval_engine import PSMRetrievalEngine


@dataclass
class EvalExample:
    query_text: str
    query_1024d: np.ndarray
    relevant_doc_ids: Set[str]


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return float("nan")
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(
    engine: PSMRetrievalEngine, eval_set: List[EvalExample], k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    if not eval_set:
        raise ValueError("eval_set is empty — cannot report meaningful metrics on zero examples")

    recalls = {k: [] for k in k_values}
    rr = []

    max_k = max(k_values)
    for ex in eval_set:
        results = engine.search(ex.query_1024d, ex.query_text, top_k=max_k)
        retrieved_ids = [r["document_id"] for r in results]
        for k in k_values:
            recalls[k].append(recall_at_k(retrieved_ids, ex.relevant_doc_ids, k))
        rr.append(reciprocal_rank(retrieved_ids, ex.relevant_doc_ids))

    report = {f"recall@{k}": float(np.nanmean(v)) for k, v in recalls.items()}
    report["MRR"] = float(np.mean(rr))
    report["n_eval_examples"] = len(eval_set)
    return report
