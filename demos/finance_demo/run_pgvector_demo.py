"""
Finance demo on live Postgres + pgvector (production path).

Steps:
  1. Ensure checkpoint exists (train dense if missing)
  2. Apply schema.sql
  3. Ingest demos/finance_demo/documents.jsonl
  4. Run sample searches against Stage-1 HNSW + Stage-2 rescoring
  5. Write a short results report

Usage (Docker Compose — recommended):
  docker compose up -d db
  docker compose run --rm finance-pg

Host DSN (compose maps 5433 -> 5432):
  postgresql://vectorprism:vectorprism@localhost:5433/vectorprism
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_DSN = os.environ.get(
    "VECTORPRISM_PG_DSN",
    "postgresql://vectorprism:vectorprism@localhost:5433/vectorprism",
)

SAMPLE_QUERIES = [
    "Why are international wires delayed after callback voids?",
    "When must a maintenance margin call be met before liquidation?",
    "What time is official fund NAV struck?",
    "Do large international wires require callback verification?",
    "What happens if weekly fund outflows exceed fifteen percent of AUM?",
]


def wait_for_db(dsn: str, timeout_s: float = 60.0) -> None:
    import psycopg
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with psycopg.connect(dsn, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            print(f"[pg-demo] DB ready: {dsn}")
            return
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise RuntimeError(f"Database not ready after {timeout_s}s: {last_err}")


def ensure_checkpoint(path: Path, encoder: str, epochs: int) -> None:
    if path.exists():
        print(f"[pg-demo] Using existing checkpoint: {path}")
        return
    print(f"[pg-demo] Checkpoint missing — training dense first...")
    code = subprocess.call(
        [
            sys.executable, str(ROOT / "train.py"),
            "--channel", "dense",
            "--data", str(DEMO / "dense_pairs.jsonl"),
            "--out", str(path),
            "--encoder", encoder,
            "--epochs", str(epochs),
            "--batch-size", "16",
        ],
        cwd=str(ROOT),
    )
    if code != 0:
        raise SystemExit(f"Training failed with exit {code}")


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description="Finance demo on pgvector")
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "finance_demo.pt"))
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if not args.skip_train:
        ensure_checkpoint(ckpt_path, args.encoder, args.epochs)
    elif not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    wait_for_db(args.dsn)

    from base_encoder import HashingEncoder, SentenceTransformerEncoder
    from checkpointing import load_checkpoint
    from channel_datasets import load_documents_jsonl
    from db_client import PgVectorClient
    from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
    from retrieval_engine import PSMRetrievalEngine

    if args.encoder in {"hash", "hashing", "test"}:
        encoder = HashingEncoder(768)
    else:
        encoder = SentenceTransformerEncoder(args.encoder)

    ckpt = load_checkpoint(str(ckpt_path))
    db = PgVectorClient(args.dsn)
    db.ensure_schema(str(ROOT / "schema.sql"))

    pipe = VectorPrismIngestPipeline(
        encoder,
        ckpt["adapter"],
        db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
    )
    docs = [
        IngestDocument(str(d["document_id"]), str(d["chunk_text"]))
        for d in load_documents_jsonl(DEMO / "documents.jsonl")
    ]
    n = pipe.upsert_documents(docs, batch_size=16)
    count = db.count_documents()
    print(f"[pg-demo] Upserted {n} docs; table count={count}")

    engine = PSMRetrievalEngine(db, causal_matrix=ckpt["causal_matrix"])
    results = []
    print("\n[pg-demo] Live pgvector search:")
    for q in SAMPLE_QUERIES:
        qv = pipe.encode_query(q)
        hits = engine.search(qv, q, top_k=3)
        print(f"\nQ: {q}")
        row = {"query": q, "hits": []}
        for i, h in enumerate(hits, 1):
            print(f"  {i}. {h['document_id']}  score={h['final_score']:.3f}")
            print(f"     {h.get('chunk_text', '')[:140]}")
            row["hits"].append({
                "document_id": h["document_id"],
                "final_score": h["final_score"],
                "chunk_text": h.get("chunk_text", ""),
            })
        results.append(row)

    out = DEMO / "results" / "pgvector_live_search.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dsn_host_hint": "localhost:5433 when using docker compose db",
        "document_count": count,
        "checkpoint": str(ckpt_path),
        "encoder": args.encoder,
        "model_version": ckpt["model_version"],
        "queries": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[pg-demo] Wrote {out}")
    print("[pg-demo] Done — finance corpus is live on Postgres/pgvector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
