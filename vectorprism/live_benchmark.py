"""
live_benchmark.py - Phase 5 end-to-end latency against a real VectorDBClient.

Unlike benchmark_harness.py (synthetic Stage-2 only), this measures
encode_query + db.query_dense_slice + Stage-2 rescoring.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from vectorprism.base_encoder import HashingEncoder, SentenceTransformerEncoder
from vectorprism.checkpointing import load_checkpoint
from vectorprism.channel_datasets import load_documents_jsonl
from vectorprism.db_factory import make_db
from vectorprism.ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
from vectorprism.retrieval_engine import PSMRetrievalEngine


def build_encoder(name: str, device: str):
    if name in {"hash", "hashing", "test"}:
        return HashingEncoder(768)
    return SentenceTransformerEncoder(name, device=device)


def main(argv: Optional[list] = None) -> dict:
    p = argparse.ArgumentParser(description="VectorPrism live e2e latency benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--documents", required=True)
    p.add_argument("--encoder", default="hash")
    p.add_argument("--backend", default="memory", choices=["memory", "pgvector", "postgres", "qdrant"])
    p.add_argument("--dsn", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--collection", default="psm_document_embeddings")
    p.add_argument("--queries", default=None, help="Optional JSONL with {query}")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--p95-budget-ms", type=float, default=50.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-ingest", action="store_true")
    args = p.parse_args(argv)

    ckpt = load_checkpoint(args.checkpoint)
    encoder = build_encoder(args.encoder, args.device)
    if args.backend in {"memory", "mem", "inmemory"}:
        from vectorprism.eval_runner import InMemoryCorpusDB

        # Ephemeral in-process corpus — do not share CLI --store NPZ
        db = InMemoryCorpusDB(store_path=None, autoload=False)
    else:
        db = make_db(
            args.backend, dsn=args.dsn, qdrant_url=args.qdrant_url, collection=args.collection
        )
        if args.backend in {"pgvector", "postgres"}:
            from vectorprism.paths import schema_sql_path

            db.ensure_schema(str(schema_sql_path()))
    pipe = VectorPrismIngestPipeline(
        encoder, ckpt["adapter"], db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
    )
    if not args.skip_ingest:
        docs = [
            IngestDocument(str(d["document_id"]), str(d["chunk_text"]))
            for d in load_documents_jsonl(args.documents)
        ]
        n = pipe.upsert_documents(docs)
        print(f"[live_benchmark] ingested {n} docs into {args.backend}")

    if args.queries:
        from vectorprism.jsonl_utils import load_jsonl
        queries = [str(r["query"]) for r in load_jsonl(args.queries)]
    else:
        docs = load_documents_jsonl(args.documents)
        queries = [str(d["chunk_text"])[:80] for d in docs[: max(1, args.n_trials)]]

    engine = PSMRetrievalEngine(
        db,
        causal_matrix=ckpt["causal_matrix"],
        model_version=int(ckpt["model_version"]),
    )
    lat = []
    for i in range(args.n_trials):
        qtext = queries[i % len(queries)]
        t0 = time.perf_counter()
        q = pipe.encode_query(qtext)
        _ = engine.search(q, qtext, top_k=args.top_k)
        lat.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(lat, dtype=np.float64)
    report = {
        "backend": args.backend,
        "n_trials": args.n_trials,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "p95_budget_ms": args.p95_budget_ms,
        "pass_sla": bool(np.percentile(arr, 95) <= args.p95_budget_ms),
    }
    print(
        f"[live_benchmark] mean={report['mean_ms']:.3f}ms "
        f"p50={report['p50_ms']:.3f}ms p95={report['p95_ms']:.3f}ms "
        f"max={report['max_ms']:.3f}ms pass_sla={report['pass_sla']}"
    )
    return report


if __name__ == "__main__":
    main()
