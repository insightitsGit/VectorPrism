"""
db_client.py - Vector DB clients for VectorPrism (pgvector + Qdrant).

Shared VectorDBClient interface used by PSMRetrievalEngine.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class VectorDBClient(ABC):
    @abstractmethod
    def upsert(self, doc_id: str, chunk_text: str, tensor_1024d: np.ndarray, meta: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def query_dense_slice(
        self, vector_slice: np.ndarray, min_truth: float, max_anchor_dist: float, limit: int
    ) -> List[Dict[str, Any]]:
        """Returns list of dicts, each containing at least
        {'tensor_1024d': np.ndarray(1024,), 'chunk_text': str, 'document_id': str}."""
        ...

    def get_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Optional fetch-by-id for graph-expanded Stage-1 candidates. Default: unsupported."""
        raise NotImplementedError(f"{type(self).__name__}.get_by_ids is not implemented")


class PgVectorClient(VectorDBClient):
    """
    Requires: pip install 'psycopg[binary]' pgvector
    Assumes schema.sql has already been applied.
    """
    def __init__(self, dsn: str):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self.conn)

    def upsert(self, doc_id: str, chunk_text: str, tensor_1024d: np.ndarray, meta: Dict[str, Any]) -> None:
        assert tensor_1024d.shape == (1024,), f"expected 1024d vector, got {tensor_1024d.shape}"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO psm_document_embeddings
                    (document_id, chunk_text, tensor_1024d,
                     epistemic_truth, anchor_dist, valid_timestamp, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    tensor_1024d = EXCLUDED.tensor_1024d,
                    epistemic_truth = EXCLUDED.epistemic_truth,
                    anchor_dist = EXCLUDED.anchor_dist,
                    valid_timestamp = EXCLUDED.valid_timestamp,
                    model_version = EXCLUDED.model_version
                """,
                (
                    doc_id,
                    chunk_text,
                    tensor_1024d.astype(np.float32),
                    float(meta["epistemic_truth"]),
                    float(meta["anchor_dist"]),
                    int(meta["valid_timestamp"]),
                    int(meta.get("model_version", 0)),
                ),
            )

    def ensure_schema(self, schema_sql_path: str) -> None:
        """Apply schema.sql (idempotent CREATE IF NOT EXISTS statements)."""
        from pathlib import Path
        sql = Path(schema_sql_path).read_text(encoding="utf-8")
        with self.conn.cursor() as cur:
            cur.execute(sql)
        # Older DBs may lack UNIQUE(document_id) — add if missing
        with self.conn.cursor() as cur:
            cur.execute(
                """
                DO $$ BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'psm_document_embeddings_document_id_key'
                  ) THEN
                    ALTER TABLE psm_document_embeddings
                      ADD CONSTRAINT psm_document_embeddings_document_id_key UNIQUE (document_id);
                  END IF;
                END $$;
                """
            )

    def count_documents(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM psm_document_embeddings")
            return int(cur.fetchone()[0])

    def query_dense_slice(
        self, vector_slice: np.ndarray, min_truth: float, max_anchor_dist: float, limit: int
    ) -> List[Dict[str, Any]]:
        assert vector_slice.shape == (368,), f"expected 368d dense-core slice, got {vector_slice.shape}"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, chunk_text, tensor_1024d
                FROM psm_document_embeddings
                WHERE epistemic_truth >= %s AND anchor_dist <= %s
                ORDER BY dense_core_slice <=> %s
                LIMIT %s
                """,
                (min_truth, max_anchor_dist, vector_slice.astype(np.float32), limit),
            )
            rows = cur.fetchall()
        return [
            {
                "document_id": r[0],
                "chunk_text": r[1],
                "tensor_1024d": _as_float32_vector(r[2]),
            }
            for r in rows
        ]


def _as_float32_vector(value) -> np.ndarray:
    """Convert pgvector Vector / list / ndarray to float32 numpy vector."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if hasattr(value, "to_numpy"):
        return value.to_numpy().astype(np.float32)
    if hasattr(value, "tolist"):
        return np.asarray(value.tolist(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


class QdrantVectorClient(VectorDBClient):
    """
    Requires: pip install qdrant-client

    Named vectors:
      - dense_core_slice (368d, HNSW) for Stage 1
      - full_tensor (1024d, flat) for Stage 2 rescoring
    """
    def __init__(self, url: str, collection_name: str = "psm_document_embeddings"):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        self.client = QdrantClient(url=url)
        self.collection = collection_name
        self._qmodels = qmodels

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense_core_slice": qmodels.VectorParams(
                        size=368, distance=qmodels.Distance.COSINE,
                        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128),
                    ),
                    "full_tensor": qmodels.VectorParams(
                        size=1024, distance=qmodels.Distance.COSINE,
                        hnsw_config=qmodels.HnswConfigDiff(m=0),
                    ),
                },
            )

    def upsert(self, doc_id: str, chunk_text: str, tensor_1024d: np.ndarray, meta: Dict[str, Any]) -> None:
        from tensor_contract import PSMTensorContract as C
        assert tensor_1024d.shape == (1024,)
        dense_slice = tensor_1024d[C.DENSE_CORE.start:C.DENSE_CORE.end]
        point_id = doc_id
        # Qdrant point ids must be uuid or unsigned int; hash string ids stably.
        if not _is_qdrant_id(doc_id):
            import uuid
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))

        self.client.upsert(
            collection_name=self.collection,
            points=[
                self._qmodels.PointStruct(
                    id=point_id,
                    vector={
                        "dense_core_slice": dense_slice.tolist(),
                        "full_tensor": tensor_1024d.tolist(),
                    },
                    payload={
                        "document_id": doc_id,
                        "chunk_text": chunk_text,
                        "epistemic_truth": float(meta["epistemic_truth"]),
                        "anchor_dist": float(meta["anchor_dist"]),
                        "valid_timestamp": int(meta["valid_timestamp"]),
                        "model_version": int(meta.get("model_version", 0)),
                    },
                )
            ],
        )

    def query_dense_slice(
        self, vector_slice: np.ndarray, min_truth: float, max_anchor_dist: float, limit: int
    ) -> List[Dict[str, Any]]:
        qfilter = self._qmodels.Filter(
            must=[
                self._qmodels.FieldCondition(key="epistemic_truth", range=self._qmodels.Range(gte=min_truth)),
                self._qmodels.FieldCondition(key="anchor_dist", range=self._qmodels.Range(lte=max_anchor_dist)),
            ]
        )
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector_slice.tolist(),
            using="dense_core_slice",
            query_filter=qfilter,
            limit=limit,
            with_vectors=["full_tensor"],
        ).points
        return [
            {
                "document_id": str(h.payload.get("document_id", h.id)),
                "chunk_text": h.payload["chunk_text"],
                "tensor_1024d": np.asarray(h.vector["full_tensor"], dtype=np.float32),
            }
            for h in hits
        ]


def _is_qdrant_id(value: str) -> bool:
    import uuid
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return str(value).isdigit()
