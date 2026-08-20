"""Temporal / header packing safety — epochs must stay int64, not float embeddings."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from vectorprism.temporal_types import (
    coerce_unix_epoch_seconds,
    pack_int64_as_2f32,
    unpack_2f32_as_int64,
)
from vectorprism.tensor_contract import PSMTensorContract as C
from vectorprism.eval_runner import InMemoryCorpusDB


def test_int64_2f32_roundtrip_exact():
    for ts in (0, 1, -1, 1_700_000_000, 2_000_000_000, 4102444800):
        pair = pack_int64_as_2f32(ts)
        assert pair.dtype == np.float32 and pair.shape == (2,)
        assert unpack_2f32_as_int64(pair) == ts


def test_header_timestamp_not_float_value():
    """Epoch must survive pack/unpack exactly (bit reinterpret, not cast)."""
    ts = 1_735_689_600
    hdr = C.pack_header(1, 1.0, 0.0, ts, model_version=3)
    # Writing ts as float32 *value* would destroy low bits; our slots must differ
    as_float_value = np.float32(ts)
    assert not np.allclose(hdr[3:5], np.array([as_float_value, 0], dtype=np.float32))
    t = np.zeros(1024, dtype=np.float32)
    t[:16] = hdr
    assert C.unpack_header(t)["timestamp"] == ts
    assert C.unpack_header(t)["transaction_time"] is None
    assert np.allclose(hdr[8:16], 0.0)


def test_optional_transaction_time_roundtrip():
    valid = 1_700_000_000
    tx = 1_700_000_100
    hdr = C.pack_header(1, 1.0, 0.0, valid, model_version=1, transaction_time=tx)
    t = np.zeros(1024, dtype=np.float32)
    t[:16] = hdr
    out = C.unpack_header(t)
    assert out["timestamp"] == valid
    assert out["transaction_time"] == tx
    assert np.allclose(hdr[8:16], 0.0)


def test_coerce_rejects_fractional_and_bool():
    with pytest.raises(ValueError, match="whole unix seconds"):
        coerce_unix_epoch_seconds(1.5)
    with pytest.raises(TypeError, match="bool"):
        coerce_unix_epoch_seconds(True)


def test_coerce_iso_and_int_string():
    assert coerce_unix_epoch_seconds("1700000000") == 1700000000
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert coerce_unix_epoch_seconds(dt) == int(dt.timestamp())
    assert coerce_unix_epoch_seconds("2024-01-15T12:00:00Z") == int(dt.timestamp())


def test_npz_store_keeps_int64_timestamps(tmp_path):
    store = tmp_path / "corpus.npz"
    db = InMemoryCorpusDB(store, autoload=False)
    t = np.zeros(1024, dtype=np.float32)
    t[:16] = C.pack_header(1, 1.0, 0.0, 1_735_689_600, model_version=9)
    db.upsert(
        "d1",
        "text",
        t,
        {
            "epistemic_truth": 1.0,
            "anchor_dist": 0.0,
            "valid_timestamp": 1_735_689_600,
            "model_version": 9,
        },
    )
    db.save()
    data = np.load(store, allow_pickle=True)
    assert "valid_timestamps" in data.files
    assert data["valid_timestamps"].dtype == np.int64
    assert int(data["valid_timestamps"][0]) == 1_735_689_600

    db2 = InMemoryCorpusDB(store, autoload=True)
    assert db2.rows[0]["valid_timestamp"] == 1_735_689_600
    assert db2.rows[0]["model_version"] == 9


def test_legacy_npz_meta_still_loads(tmp_path):
    """Old 4-column meta layout (ts in col 2) must still load."""
    store = tmp_path / "legacy.npz"
    tensors = np.zeros((1, 1024), dtype=np.float32)
    meta = np.asarray([[1.0, 0.0, 1_600_000_000.0, 2.0]], dtype=np.float64)
    np.savez_compressed(
        store,
        tensors=tensors,
        meta=meta,
        ids=np.array(["legacy"], dtype=object),
        texts=np.array(["x"], dtype=object),
    )
    db = InMemoryCorpusDB(store, autoload=True)
    assert db.rows[0]["valid_timestamp"] == 1_600_000_000
    assert db.rows[0]["model_version"] == 2
