"""db_factory.py - Construct VectorDBClient from CLI/env settings."""

from __future__ import annotations

import os
from typing import Optional

from db_client import VectorDBClient, PgVectorClient, QdrantVectorClient
from eval_runner import InMemoryCorpusDB


def make_db(
    backend: str = "memory",
    dsn: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    collection: str = "psm_document_embeddings",
) -> VectorDBClient:
    backend = backend.lower()
    if backend in {"memory", "mem", "inmemory"}:
        return InMemoryCorpusDB()
    if backend in {"pg", "postgres", "pgvector"}:
        dsn = dsn or os.environ.get("VECTORPRISM_PG_DSN")
        if not dsn:
            raise ValueError("Postgres backend requires --dsn or VECTORPRISM_PG_DSN")
        return PgVectorClient(dsn)
    if backend in {"qdrant"}:
        url = qdrant_url or os.environ.get("VECTORPRISM_QDRANT_URL", "http://localhost:6333")
        return QdrantVectorClient(url, collection_name=collection)
    raise ValueError(f"Unknown DB backend {backend!r}; use memory|pgvector|qdrant")
