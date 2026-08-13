"""
reingest.py - Phase 6 versioned corpus re-encode.

Any adapter retrain invalidates old vectors. This tool re-reads a documents
JSONL (or exports from a prior ingest file) and upserts with the new
model_version from the checkpoint.
"""

from __future__ import annotations

import argparse
from typing import Optional

from ingest_cli import main as ingest_main


def main(argv: Optional[list] = None) -> int:
    """Thin wrapper: reingest == ingest with an explicit new checkpoint."""
    p = argparse.ArgumentParser(description="VectorPrism versioned re-ingest")
    p.add_argument("--checkpoint", required=True, help="NEW trained checkpoint")
    p.add_argument("--documents", required=True, help="Full corpus JSONL to re-encode")
    p.add_argument("--encoder", default="hash")
    p.add_argument("--backend", default="memory", choices=["memory", "pgvector", "qdrant"])
    p.add_argument("--dsn", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--collection", default="psm_document_embeddings")
    p.add_argument("--truth-checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)

    print(f"[reingest] Re-encoding corpus with checkpoint={args.checkpoint}")
    # Delegate to ingest_cli with reconstructed argv
    forwarded = [
        "--checkpoint", args.checkpoint,
        "--documents", args.documents,
        "--encoder", args.encoder,
        "--backend", args.backend,
        "--collection", args.collection,
        "--device", args.device,
        "--batch-size", str(args.batch_size),
    ]
    if args.dsn:
        forwarded += ["--dsn", args.dsn]
    if args.qdrant_url:
        forwarded += ["--qdrant-url", args.qdrant_url]
    if args.truth_checkpoint:
        forwarded += ["--truth-checkpoint", args.truth_checkpoint]
    return ingest_main(forwarded)


if __name__ == "__main__":
    main()
