"""
db_client.py - Vector DB clients for VectorPrism (pgvector + Qdrant).

Shared VectorDBClient interface used by PSMRetrievalEngine.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class VectorDBClient(ABC):
    @abstractmethod
    def upsert(self, doc_id: str, chunk_text: str, tensor_1024d: np.ndarray, meta: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def query_dense_slice(
        self,
        vector_slice: np.ndarray,
        min_truth: float,
        max_anchor_dist: float,
        limit: int,
        model_version: Optional[int] = None,
        as_of: Optional[int] = None,
        as_of_transaction: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Returns list of dicts, each containing at least
        {'tensor_1024d': np.ndarray(1024,), 'chunk_text': str, 'document_id': str}.

        Optional bitemporal Stage-1 gates (unix seconds, exact int):
          as_of — valid-time membership [valid_from, valid_to)
          as_of_transaction — transaction_time <= as_of_transaction
        """
        ...

    def get_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Optional fetch-by-id for graph-expanded Stage-1 candidates. Default: unsupported."""
        raise NotImplementedError(f"{type(self).__name__}.get_by_ids is not implemented")


class PgVectorClient(VectorDBClient):
    """
    Requires: pip install 'psycopg[binary]' pgvector
    Call ensure_schema() on first use against an empty database.
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
                     epistemic_truth, anchor_dist, valid_timestamp,
                     valid_to_timestamp, transaction_timestamp, model_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    chunk_text = EXCLUDED.chunk_text,
                    tensor_1024d = EXCLUDED.tensor_1024d,
                    epistemic_truth = EXCLUDED.epistemic_truth,
                    anchor_dist = EXCLUDED.anchor_dist,
                    valid_timestamp = EXCLUDED.valid_timestamp,
                    valid_to_timestamp = EXCLUDED.valid_to_timestamp,
                    transaction_timestamp = EXCLUDED.transaction_timestamp,
                    model_version = EXCLUDED.model_version
                """,
                (
                    doc_id,
                    chunk_text,
                    tensor_1024d.astype(np.float32),
                    float(meta["epistemic_truth"]),
                    float(meta["anchor_dist"]),
                    int(meta["valid_timestamp"]),
                    (
                        None
                        if meta.get("valid_to_timestamp") is None
                        else int(meta["valid_to_timestamp"])
                    ),
                    (
                        None
                        if meta.get("transaction_timestamp") is None
                        else int(meta["transaction_timestamp"])
                    ),
                    int(meta.get("model_version", 0)),
                ),
            )

    def ensure_schema(self, schema_sql_path: Optional[str] = None) -> None:
        """Apply schema.sql (idempotent CREATE IF NOT EXISTS statements)."""
        from pathlib import Path

        if schema_sql_path:
            sql = Path(schema_sql_path).read_text(encoding="utf-8")
        else:
            from vectorprism.paths import read_schema_sql

            sql = read_schema_sql()
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
        # Bitemporal columns (0.1.3+) — idempotent for existing deployments
        with self.conn.cursor() as cur:
            cur.execute(
                """
                ALTER TABLE psm_document_embeddings
                  ADD COLUMN IF NOT EXISTS valid_to_timestamp BIGINT;
                ALTER TABLE psm_document_embeddings
                  ADD COLUMN IF NOT EXISTS transaction_timestamp BIGINT;
                CREATE INDEX IF NOT EXISTS idx_psm_valid_timestamp
                  ON psm_document_embeddings (valid_timestamp);
                CREATE INDEX IF NOT EXISTS idx_psm_valid_to_timestamp
                  ON psm_document_embeddings (valid_to_timestamp);
                CREATE INDEX IF NOT EXISTS idx_psm_transaction_timestamp
                  ON psm_document_embeddings (transaction_timestamp);
                """
            )

    def count_documents(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM psm_document_embeddings")
            return int(cur.fetchone()[0])

    def get_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        if not doc_ids:
            return []
        ids = [str(x) for x in doc_ids]
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, chunk_text, tensor_1024d, epistemic_truth, anchor_dist
                FROM psm_document_embeddings
                WHERE document_id = ANY(%s)
                """,
                (ids,),
            )
            rows = cur.fetchall()
        return [
            {
                "document_id": r[0],
                "chunk_text": r[1],
                "tensor_1024d": _as_float32_vector(r[2]),
                "epistemic_truth": float(r[3]),
                "anchor_dist": float(r[4]),
            }
            for r in rows
        ]

    def query_dense_slice(
        self,
        vector_slice: np.ndarray,
        min_truth: float,
        max_anchor_dist: float,
        limit: int,
        model_version: Optional[int] = None,
        as_of: Optional[int] = None,
        as_of_transaction: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        assert vector_slice.shape == (368,), f"expected 368d dense-core slice, got {vector_slice.shape}"
        where = "epistemic_truth >= %s AND anchor_dist <= %s"
        params: List[Any] = [min_truth, max_anchor_dist]
        if model_version is not None:
            where += " AND model_version = %s"
            params.append(int(model_version))
        if as_of is not None:
            # [valid_from, valid_to) with NULL valid_to = open-ended
            where += (
                " AND valid_timestamp <= %s"
                " AND (valid_to_timestamp IS NULL OR valid_to_timestamp > %s)"
            )
            params.extend([int(as_of), int(as_of)])
        if as_of_transaction is not None:
            where += (
                " AND (transaction_timestamp IS NULL OR transaction_timestamp <= %s)"
            )
            params.append(int(as_of_transaction))
        params.extend([vector_slice.astype(np.float32), limit])
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT document_id, chunk_text, tensor_1024d, epistemic_truth, anchor_dist,
                       model_version, valid_timestamp, valid_to_timestamp, transaction_timestamp
                FROM psm_document_embeddings
                WHERE {where}
                ORDER BY dense_core_slice <=> %s
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        return [
            {
                "document_id": r[0],
                "chunk_text": r[1],
                "tensor_1024d": _as_float32_vector(r[2]),
                "epistemic_truth": float(r[3]),
                "anchor_dist": float(r[4]),
                "model_version": int(r[5]),
                "valid_timestamp": int(r[6]),
                "valid_to_timestamp": None if r[7] is None else int(r[7]),
                "transaction_timestamp": None if r[8] is None else int(r[8]),
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
        from vectorprism.tensor_contract import PSMTensorContract as C
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
                        "valid_to_timestamp": (
                            None
                            if meta.get("valid_to_timestamp") is None
                            else int(meta["valid_to_timestamp"])
                        ),
                        "transaction_timestamp": (
                            None
                            if meta.get("transaction_timestamp") is None
                            else int(meta["transaction_timestamp"])
                        ),
                        "model_version": int(meta.get("model_version", 0)),
                    },
                )
            ],
        )

    def get_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch by payload document_id (string ids are uuid5-hashed as point ids)."""
        if not doc_ids:
            return []
        ids = [str(x) for x in doc_ids]
        qfilter = self._qmodels.Filter(
            must=[
                self._qmodels.FieldCondition(
                    key="document_id",
                    match=self._qmodels.MatchAny(any=ids),
                )
            ]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qfilter,
            limit=max(len(ids), 1),
            with_payload=True,
            with_vectors=["full_tensor"],
        )
        out: List[Dict[str, Any]] = []
        for h in points:
            payload = h.payload or {}
            vec = h.vector
            if isinstance(vec, dict):
                full = vec.get("full_tensor")
            else:
                full = vec
            if full is None:
                continue
            out.append(
                {
                    "document_id": str(payload.get("document_id", h.id)),
                    "chunk_text": str(payload.get("chunk_text", "")),
                    "tensor_1024d": np.asarray(full, dtype=np.float32),
                    "epistemic_truth": float(payload.get("epistemic_truth", 1.0)),
                    "anchor_dist": float(payload.get("anchor_dist", 0.0)),
                    "model_version": int(payload.get("model_version", 0)),
                }
            )
        return out

    def query_dense_slice(
        self,
        vector_slice: np.ndarray,
        min_truth: float,
        max_anchor_dist: float,
        limit: int,
        model_version: Optional[int] = None,
        as_of: Optional[int] = None,
        as_of_transaction: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        from vectorprism.bitemporal import passes_bitemporal_filters

        must = [
            self._qmodels.FieldCondition(key="epistemic_truth", range=self._qmodels.Range(gte=min_truth)),
            self._qmodels.FieldCondition(key="anchor_dist", range=self._qmodels.Range(lte=max_anchor_dist)),
        ]
        if model_version is not None:
            must.append(
                self._qmodels.FieldCondition(
                    key="model_version",
                    match=self._qmodels.MatchValue(value=int(model_version)),
                )
            )
        if as_of is not None:
            must.append(
                self._qmodels.FieldCondition(
                    key="valid_timestamp",
                    range=self._qmodels.Range(lte=int(as_of)),
                )
            )
        qfilter = self._qmodels.Filter(must=must)
        fetch_n = limit * 3 if (as_of is not None or as_of_transaction is not None) else limit
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector_slice.tolist(),
            using="dense_core_slice",
            query_filter=qfilter,
            limit=max(fetch_n, limit),
            with_vectors=["full_tensor"],
        ).points
        out: List[Dict[str, Any]] = []
        for h in hits:
            payload = h.payload or {}
            vec = h.vector
            full = vec.get("full_tensor") if isinstance(vec, dict) else vec
            if full is None:
                continue
            row = {
                "document_id": str(payload.get("document_id", h.id)),
                "chunk_text": payload.get("chunk_text", ""),
                "tensor_1024d": np.asarray(full, dtype=np.float32),
                "epistemic_truth": float(payload.get("epistemic_truth", 1.0)),
                "anchor_dist": float(payload.get("anchor_dist", 0.0)),
                "model_version": int(payload.get("model_version", 0)),
                "valid_timestamp": int(payload.get("valid_timestamp", 0)),
                "valid_to_timestamp": (
                    None
                    if payload.get("valid_to_timestamp") is None
                    else int(payload["valid_to_timestamp"])
                ),
                "transaction_timestamp": (
                    None
                    if payload.get("transaction_timestamp") is None
                    else int(payload["transaction_timestamp"])
                ),
            }
            if not passes_bitemporal_filters(
                row, as_of=as_of, as_of_transaction=as_of_transaction
            ):
                continue
            out.append(row)
            if len(out) >= limit:
                break
        return out


def _is_qdrant_id(value: str) -> bool:
    import uuid
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return str(value).isdigit()
