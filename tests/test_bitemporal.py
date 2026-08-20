"""Bitemporal Stage-1 filters + packing (release 0.1.3)."""

from __future__ import annotations

import numpy as np
import pytest

from vectorprism.bitemporal import (
    passes_bitemporal_filters,
    row_in_transaction_time,
    row_in_valid_time,
)
from vectorprism.eval_runner import InMemoryCorpusDB
from vectorprism.ingest_pipeline import IngestDocument, VectorPrismIngestPipeline
from vectorprism.base_encoder import HashingEncoder
from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.retrieval_engine import PSMRetrievalEngine
from vectorprism.tensor_contract import PSMTensorContract as C


def _mk_tensor(valid: int, tx: int | None = None) -> np.ndarray:
    t = np.zeros(1024, dtype=np.float32)
    t[:16] = C.pack_header(1, 1.0, 0.0, valid, model_version=1, transaction_time=tx)
    # Non-zero dense so ANN scoring is deterministic-ish
    t[C.DENSE_CORE.start : C.DENSE_CORE.start + 8] = 0.1
    n = np.linalg.norm(t[C.DENSE_CORE.start : C.DENSE_CORE.end]) + 1e-9
    t[C.DENSE_CORE.start : C.DENSE_CORE.end] /= n
    return t


def test_valid_time_half_open_interval():
    row = {"valid_timestamp": 100, "valid_to_timestamp": 200}
    assert row_in_valid_time(row, 100) is True
    assert row_in_valid_time(row, 199) is True
    assert row_in_valid_time(row, 200) is False
    assert row_in_valid_time(row, 99) is False
    open_row = {"valid_timestamp": 100, "valid_to_timestamp": None}
    assert row_in_valid_time(open_row, 10_000) is True


def test_transaction_time_legacy_passthrough():
    assert row_in_transaction_time({"transaction_timestamp": None}, 50) is True
    assert row_in_transaction_time({"transaction_timestamp": 40}, 50) is True
    assert row_in_transaction_time({"transaction_timestamp": 60}, 50) is False


def test_passes_bitemporal_both_clocks():
    row = {
        "valid_timestamp": 100,
        "valid_to_timestamp": 200,
        "transaction_timestamp": 110,
    }
    assert passes_bitemporal_filters(row, as_of=150, as_of_transaction=120)
    assert not passes_bitemporal_filters(row, as_of=150, as_of_transaction=100)
    assert not passes_bitemporal_filters(row, as_of=50, as_of_transaction=120)


def test_memory_db_as_of_filters_stage1():
    db = InMemoryCorpusDB(autoload=False)
    # Policy v1: [100, 200)
    db.upsert(
        "policy_v1",
        "old wire limit 10k",
        _mk_tensor(100, tx=105),
        {
            "epistemic_truth": 1.0,
            "anchor_dist": 0.0,
            "valid_timestamp": 100,
            "valid_to_timestamp": 200,
            "transaction_timestamp": 105,
            "model_version": 1,
        },
    )
    # Policy v2: [200, open)
    db.upsert(
        "policy_v2",
        "new wire limit 25k",
        _mk_tensor(200, tx=205),
        {
            "epistemic_truth": 1.0,
            "anchor_dist": 0.0,
            "valid_timestamp": 200,
            "valid_to_timestamp": None,
            "transaction_timestamp": 205,
            "model_version": 1,
        },
    )
    q = np.zeros(368, dtype=np.float32)
    q[:8] = 0.1
    q /= np.linalg.norm(q) + 1e-9

    at_150 = db.query_dense_slice(q, 0.0, 1.0, 10, as_of=150)
    assert [r["document_id"] for r in at_150] == ["policy_v1"]

    at_250 = db.query_dense_slice(q, 0.0, 1.0, 10, as_of=250)
    assert [r["document_id"] for r in at_250] == ["policy_v2"]

    # Before tx of v2 was recorded, as_of_transaction hides v2 even if valid_from passed
    known = db.query_dense_slice(q, 0.0, 1.0, 10, as_of=250, as_of_transaction=150)
    assert known == []


def test_engine_search_as_of(tmp_path):
    enc = HashingEncoder(768)
    adapter = MultiTaskProjectionAdapter(base_dim=768)
    db = InMemoryCorpusDB(tmp_path / "c.npz", autoload=False)
    pipe = VectorPrismIngestPipeline(enc, adapter, db, model_version=1)
    pipe.upsert_documents(
        [
            IngestDocument(
                "old",
                "limit ten thousand dollars",
                timestamp=100,
                valid_to=200,
                transaction_time=105,
            ),
            IngestDocument(
                "new",
                "limit twenty five thousand dollars",
                timestamp=200,
                valid_to=None,
                transaction_time=205,
            ),
        ]
    )
    engine = PSMRetrievalEngine(db, model_version=1)
    q = pipe.encode_query("wire transfer limit")
    hits_old = engine.search(q, "wire transfer limit", top_k=5, as_of=150)
    assert [h["document_id"] for h in hits_old] == ["old"]
    hits_new = engine.search(q, "wire transfer limit", top_k=5, as_of=250)
    assert [h["document_id"] for h in hits_new] == ["new"]


def test_ingest_rejects_invalid_interval():
    enc = HashingEncoder(768)
    adapter = MultiTaskProjectionAdapter(base_dim=768)
    db = InMemoryCorpusDB(autoload=False)
    pipe = VectorPrismIngestPipeline(enc, adapter, db, model_version=1)
    with pytest.raises(ValueError, match="valid_to"):
        pipe.upsert_documents(
            [IngestDocument("bad", "x", timestamp=200, valid_to=100)]
        )


def test_npz_roundtrip_bitemporal(tmp_path):
    store = tmp_path / "bt.npz"
    db = InMemoryCorpusDB(store, autoload=False)
    db.upsert(
        "d1",
        "text",
        _mk_tensor(1_700_000_000, tx=1_700_000_050),
        {
            "epistemic_truth": 1.0,
            "anchor_dist": 0.0,
            "valid_timestamp": 1_700_000_000,
            "valid_to_timestamp": 1_800_000_000,
            "transaction_timestamp": 1_700_000_050,
            "model_version": 2,
        },
    )
    db.save()
    db2 = InMemoryCorpusDB(store, autoload=True)
    r = db2.rows[0]
    assert r["valid_timestamp"] == 1_700_000_000
    assert r["valid_to_timestamp"] == 1_800_000_000
    assert r["transaction_timestamp"] == 1_700_000_050


def _two_era_corpus(tmp_path):
    """Shared fixture data: expired v1 + current v2, same query text."""
    enc = HashingEncoder(768)
    adapter = MultiTaskProjectionAdapter(base_dim=768)
    db = InMemoryCorpusDB(tmp_path / "era.npz", autoload=False)
    pipe = VectorPrismIngestPipeline(enc, adapter, db, model_version=1)
    pipe.upsert_documents(
        [
            IngestDocument(
                "policy_v1",
                "wire transfer limit ten thousand",
                timestamp=100,
                valid_to=200,
                transaction_time=105,
            ),
            IngestDocument(
                "policy_v2",
                "wire transfer limit twenty five thousand",
                timestamp=200,
                valid_to=None,
                transaction_time=205,
            ),
        ]
    )
    engine = PSMRetrievalEngine(db, model_version=1)
    q = pipe.encode_query("wire transfer limit")
    return engine, q, db


def test_opt_in_default_includes_both_eras(tmp_path):
    """WITHOUT as_of: both versions remain eligible (feature off)."""
    engine, q, db = _two_era_corpus(tmp_path)
    assert len(db.rows) == 2

    hits_default = engine.search(q, "wire transfer limit", top_k=5)
    ids_default = {h["document_id"] for h in hits_default}
    assert ids_default == {"policy_v1", "policy_v2"}, (
        "Default search must not apply temporal gates — got "
        f"{ids_default}. Pass as_of explicitly to filter by era."
    )


def test_with_vs_without_as_of_comparison(tmp_path):
    """Side-by-side: default vs opt-in as_of must differ on a two-era corpus."""
    engine, q, db = _two_era_corpus(tmp_path)
    query = "wire transfer limit"

    without = engine.search(q, query, top_k=5)
    with_150 = engine.search(q, query, top_k=5, as_of=150)
    with_250 = engine.search(q, query, top_k=5, as_of=250)

    ids_without = [h["document_id"] for h in without]
    ids_150 = [h["document_id"] for h in with_150]
    ids_250 = [h["document_id"] for h in with_250]

    assert set(ids_without) == {"policy_v1", "policy_v2"}
    assert ids_150 == ["policy_v1"]
    assert ids_250 == ["policy_v2"]
    # Opt-in must change the result set vs default
    assert ids_150 != ids_without
    assert ids_250 != ids_without
    assert ids_150 != ids_250


def test_stage1_with_vs_without_as_of_same_query_vector(tmp_path):
    """Stage-1 pool size: default returns both; as_of returns one."""
    _, q, db = _two_era_corpus(tmp_path)
    dense = q[C.DENSE_CORE.start : C.DENSE_CORE.end]

    default_pool = db.query_dense_slice(dense, 0.0, 1.0, 10)
    gated_pool = db.query_dense_slice(dense, 0.0, 1.0, 10, as_of=150)

    assert {r["document_id"] for r in default_pool} == {"policy_v1", "policy_v2"}
    assert {r["document_id"] for r in gated_pool} == {"policy_v1"}
    assert len(gated_pool) < len(default_pool)
