"""ingest_cli.py - Phase 5 production document upsert."""

from __future__ import annotations

import argparse
from typing import Optional

from base_encoder import HashingEncoder, SentenceTransformerEncoder
from checkpointing import load_checkpoint
from channel_datasets import load_documents_jsonl
from db_factory import make_db
from epistemic_truth import load_truth_classifier
from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument


def build_encoder(name: str, device: str):
    if name in {"hash", "hashing", "test"}:
        return HashingEncoder(768)
    return SentenceTransformerEncoder(name, device=device)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="VectorPrism ingest documents JSONL")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--documents", required=True, help="JSONL {document_id, chunk_text}")
    p.add_argument("--encoder", default="hash")
    p.add_argument("--backend", default="memory", choices=["memory", "pgvector", "qdrant"])
    p.add_argument("--dsn", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--collection", default="psm_document_embeddings")
    p.add_argument("--truth-checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)

    ckpt = load_checkpoint(args.checkpoint)
    encoder = build_encoder(args.encoder, args.device)
    db = make_db(args.backend, dsn=args.dsn, qdrant_url=args.qdrant_url, collection=args.collection)
    truth = None
    if args.truth_checkpoint:
        truth, meta = load_truth_classifier(args.truth_checkpoint)
        print(f"[ingest] truth classifier loaded ece={meta.get('ece')} "
              f"hard_filter_allowed={meta.get('hard_filter_allowed')}")

    pipe = VectorPrismIngestPipeline(
        encoder,
        ckpt["adapter"],
        db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
        truth_classifier=truth,
    )
    docs = [
        IngestDocument(
            document_id=str(d["document_id"]),
            chunk_text=str(d["chunk_text"]),
            epistemic_truth=float(d["epistemic_truth"]) if "epistemic_truth" in d else None,
        )
        for d in load_documents_jsonl(args.documents)
    ]
    n = pipe.upsert_documents(docs, batch_size=args.batch_size)
    print(f"[ingest] upserted {n} documents into backend={args.backend} "
          f"model_version={ckpt['model_version']}")
    return n


if __name__ == "__main__":
    main()
