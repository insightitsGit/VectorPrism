"""search_cli.py - Phase 5 interactive / one-shot VectorPrism search."""

from __future__ import annotations

import argparse
import json
from typing import Optional

from base_encoder import HashingEncoder, SentenceTransformerEncoder
from checkpointing import load_checkpoint
from db_factory import make_db
from epistemic_truth import load_truth_classifier
from ingest_pipeline import VectorPrismIngestPipeline
from retrieval_engine import PSMRetrievalEngine


def build_encoder(name: str, device: str):
    if name in {"hash", "hashing", "test"}:
        return HashingEncoder(768)
    return SentenceTransformerEncoder(name, device=device)


def main(argv: Optional[list] = None) -> list:
    p = argparse.ArgumentParser(description="VectorPrism search")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--encoder", default="hash")
    p.add_argument("--backend", default="memory", choices=["memory", "pgvector", "qdrant"])
    p.add_argument("--dsn", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--collection", default="psm_document_embeddings")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--hard-truth-filter", action="store_true")
    p.add_argument("--truth-checkpoint", default=None,
                   help="Required with --hard-truth-filter; must have hard_filter_allowed")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    ckpt = load_checkpoint(args.checkpoint)
    encoder = build_encoder(args.encoder, args.device)
    db = make_db(args.backend, dsn=args.dsn, qdrant_url=args.qdrant_url, collection=args.collection)

    hard = bool(args.hard_truth_filter)
    if hard:
        if not args.truth_checkpoint:
            raise SystemExit("--hard-truth-filter requires --truth-checkpoint")
        _, meta = load_truth_classifier(args.truth_checkpoint)
        if not meta.get("hard_filter_allowed"):
            raise SystemExit(
                "Truth classifier ECE not good enough for hard filtering "
                f"(ece={meta.get('ece')}). Keep soft mode."
            )

    # Encoder path for query tensor (DB already holds corpus)
    pipe = VectorPrismIngestPipeline(
        encoder, ckpt["adapter"], db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
    )
    engine = PSMRetrievalEngine(db, causal_matrix=ckpt["causal_matrix"], hard_truth_filter=hard)
    q = pipe.encode_query(args.query)
    hits = engine.search(q, args.query, top_k=args.top_k)
    for i, h in enumerate(hits, start=1):
        print(f"{i}. {h['document_id']} score={h['final_score']:.4f}")
        text = h.get("chunk_text", "")
        if text:
            print(f"   {text[:200]}")
    return hits


if __name__ == "__main__":
    main()
