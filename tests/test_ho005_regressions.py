"""Regression tests for HO-VectorPrism-005 (BUG-001–008)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vectorprism.checkpointing import load_checkpoint, save_checkpoint
from vectorprism.eval_runner import InMemoryCorpusDB
from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.retrieval_engine import PSMRetrievalEngine
from vectorprism.tensor_contract import PSMTensorContract as C


ROOT = Path(__file__).resolve().parents[1]


def test_memory_store_roundtrip(tmp_path):
    store = tmp_path / "corpus.npz"
    db = InMemoryCorpusDB(store, autoload=False)
    t = np.zeros(1024, dtype=np.float32)
    t[C.DENSE_CORE.start : C.DENSE_CORE.end] = 1.0
    t[C.DENSE_CORE.start : C.DENSE_CORE.end] /= np.linalg.norm(
        t[C.DENSE_CORE.start : C.DENSE_CORE.end]
    )
    db.upsert(
        "doc_a",
        "hello vectorprism",
        t,
        {
            "epistemic_truth": 1.0,
            "anchor_dist": 0.0,
            "valid_timestamp": 1,
            "model_version": 3,
        },
    )
    db.save(store)

    db2 = InMemoryCorpusDB(store, autoload=True)
    assert len(db2) == 1
    hits = db2.query_dense_slice(
        t[C.DENSE_CORE.start : C.DENSE_CORE.end],
        min_truth=0.0,
        max_anchor_dist=1.0,
        limit=5,
        model_version=3,
    )
    assert hits[0]["document_id"] == "doc_a"
    empty = db2.query_dense_slice(
        t[C.DENSE_CORE.start : C.DENSE_CORE.end],
        min_truth=0.0,
        max_anchor_dist=1.0,
        limit=5,
        model_version=99,
    )
    assert empty == []


def test_model_version_filter_on_inmemory():
    db = InMemoryCorpusDB(autoload=False)
    for ver, did in [(1, "v1"), (2, "v2")]:
        t = np.random.default_rng(ver).standard_normal(1024).astype(np.float32)
        dense = t[C.DENSE_CORE.start : C.DENSE_CORE.end]
        t[C.DENSE_CORE.start : C.DENSE_CORE.end] = dense / (np.linalg.norm(dense) + 1e-9)
        db.upsert(
            did,
            f"doc {did}",
            t,
            {
                "epistemic_truth": 1.0,
                "anchor_dist": 0.0,
                "valid_timestamp": 1,
                "model_version": ver,
            },
        )
    q = db.rows[0]["tensor_1024d"][C.DENSE_CORE.start : C.DENSE_CORE.end]
    only_v2 = db.query_dense_slice(q, 0.0, 1.0, 10, model_version=2)
    assert [r["document_id"] for r in only_v2] == ["v2"]


def test_get_by_ids_missing_warns(caplog):
    class _NoFetch(InMemoryCorpusDB):
        def get_by_ids(self, doc_ids):
            raise NotImplementedError("missing fetch")

    broken = _NoFetch(autoload=False)
    engine = PSMRetrievalEngine(broken, model_version=1)
    with caplog.at_level("WARNING"):
        out = engine._fetch_missing(
            ["nbr"], {"min_truth": 0.0, "max_anchor_dist": 1.0, "model_version": 1}
        )
    assert out == []
    assert "get_by_ids" in caplog.text.lower() or "structured neighbors" in caplog.text.lower()


def test_get_by_ids_inmemory():
    db = InMemoryCorpusDB(autoload=False)
    t = np.zeros(1024, dtype=np.float32)
    t[C.DENSE_CORE.start] = 1.0
    db.upsert(
        "nbr",
        "neighbor text",
        t,
        {"epistemic_truth": 1.0, "anchor_dist": 0.0, "valid_timestamp": 1, "model_version": 1},
    )
    got = db.get_by_ids(["nbr"])
    assert got[0]["document_id"] == "nbr"


def test_checkpoint_safe_load_roundtrip(tmp_path):
    adapter = MultiTaskProjectionAdapter(base_dim=768)
    path = tmp_path / "safe.pt"
    save_checkpoint(path, adapter, model_version=11, meta={"k": 1})
    loaded = load_checkpoint(path)
    assert loaded["model_version"] == 11
    # Default path must reject arbitrary non-weight pickle payloads
    bad = tmp_path / "bad.pt"
    torch.save({"evil": object()}, bad)
    with pytest.raises(Exception):
        load_checkpoint(bad)


def test_packaged_schema_and_examples_exist():
    from vectorprism.paths import example_jsonl, schema_sql_path

    sql = schema_sql_path().read_text(encoding="utf-8")
    assert "UNIQUE" in sql.upper() or "document_id VARCHAR(255) NOT NULL UNIQUE" in sql
    assert example_jsonl("dense_pairs.example.jsonl").is_file()
    assert example_jsonl("documents.example.jsonl").is_file()


def test_cli_memory_two_process_ingest_search(tmp_path):
    """BUG-001: separate processes share --store NPZ."""
    docs = ROOT / "vectorprism" / "data" / "documents.example.jsonl"
    if not docs.is_file():
        docs = ROOT / "data" / "documents.example.jsonl"
    assert docs.is_file(), f"missing example documents at {docs}"
    ckpt = tmp_path / "smoke.pt"
    store = tmp_path / "mem.npz"
    adapter = MultiTaskProjectionAdapter(768)
    save_checkpoint(ckpt, adapter, model_version=1)

    env = os.environ.copy()
    # Prefer installed / repo package; do not put tests/ on path ahead of package
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    r1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "vectorprism",
            "ingest",
            "--checkpoint",
            str(ckpt),
            "--documents",
            str(docs),
            "--encoder",
            "hash",
            "--backend",
            "memory",
            "--store",
            str(store),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0, r1.stdout + r1.stderr
    assert store.is_file()

    r2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "vectorprism",
            "search",
            "--checkpoint",
            str(ckpt),
            "--query",
            "VectorPrism uses a 1024-dimensional tensor",
            "--encoder",
            "hash",
            "--backend",
            "memory",
            "--store",
            str(store),
            "--top-k",
            "3",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "score=" in r2.stdout

    r3 = subprocess.run(
        [
            sys.executable,
            "-m",
            "vectorprism",
            "search",
            "--checkpoint",
            str(ckpt),
            "--query",
            "anything",
            "--encoder",
            "hash",
            "--backend",
            "memory",
            "--store",
            str(tmp_path / "missing.npz"),
            "--top-k",
            "3",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert r3.returncode == 2
    assert "empty" in (r3.stderr + r3.stdout).lower()


@pytest.mark.skipif(not os.environ.get("VECTORPRISM_PG_DSN"), reason="VECTORPRISM_PG_DSN not set")
def test_pgvector_get_by_ids_and_schema():
    from vectorprism.db_client import PgVectorClient
    from vectorprism.paths import schema_sql_path as ssp

    dsn = os.environ["VECTORPRISM_PG_DSN"]
    db = PgVectorClient(dsn)
    db.ensure_schema(str(ssp()))
    t = np.zeros(1024, dtype=np.float32)
    t[C.DENSE_CORE.start] = 1.0
    db.upsert(
        "doc_dense",
        "pg get_by_ids probe",
        t,
        {
            "epistemic_truth": 1.0,
            "anchor_dist": 0.0,
            "valid_timestamp": 1,
            "model_version": 1,
        },
    )
    rows = db.get_by_ids(["doc_dense"])
    assert len(rows) == 1
    assert rows[0]["tensor_1024d"].shape == (1024,)


def test_qdrant_get_by_ids_unit():
    """Unit-level get_by_ids without a live Qdrant server."""
    from vectorprism.db_client import QdrantVectorClient
    import types

    class _FakeClient:
        def __init__(self):
            self.points = {}

        def collection_exists(self, name):
            return True

        def upsert(self, collection_name, points):
            for p in points:
                self.points[p.id] = p

        def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
            want = set(scroll_filter.must[0].match.any)
            out = []
            for p in self.points.values():
                if p.payload.get("document_id") in want:
                    out.append(p)
            return out, None

        def query_points(self, **kwargs):
            class _R:
                points = []

            return _R()

    client = object.__new__(QdrantVectorClient)
    fc = _FakeClient()
    client.client = fc
    client.collection = "psm_document_embeddings"

    class _MatchAny:
        def __init__(self, any):
            self.any = any

    class _FieldCondition:
        def __init__(self, key, match=None, range=None):
            self.key = key
            self.match = match
            self.range = range

    class _Filter:
        def __init__(self, must):
            self.must = must

    class _PointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    client._qmodels = types.SimpleNamespace(
        MatchAny=_MatchAny,
        FieldCondition=_FieldCondition,
        Filter=_Filter,
        PointStruct=_PointStruct,
        MatchValue=lambda value: types.SimpleNamespace(value=value),
        Range=lambda **kw: types.SimpleNamespace(**kw),
    )

    t = np.zeros(1024, dtype=np.float32)
    t[0] = 1.0
    client.upsert(
        "doc_dense",
        "qdrant text",
        t,
        {
            "epistemic_truth": 1.0,
            "anchor_dist": 0.0,
            "valid_timestamp": 1,
            "model_version": 7,
        },
    )
    rows = client.get_by_ids(["doc_dense"])
    assert len(rows) == 1
    assert rows[0]["document_id"] == "doc_dense"
    assert rows[0]["model_version"] == 7
