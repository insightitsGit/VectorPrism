"""Tests for encode / integration guards (Onyx-style misuse)."""

from __future__ import annotations

import warnings

import pytest

from vectorprism.encode_guards import (
    INTEGRATION_BANNER,
    check_encoder_matches_checkpoint,
    expected_encoder_from_checkpoint,
)


def test_banner_mentions_foreign_embeddings():
    assert "NOT a document chunker" in INTEGRATION_BANNER
    assert "Onyx" in INTEGRATION_BANNER
    assert "1024d" in INTEGRATION_BANNER


def test_expected_encoder_from_meta():
    assert expected_encoder_from_checkpoint({"meta": {"encoder": "sentence-transformers/x"}}) == (
        "sentence-transformers/x"
    )
    assert expected_encoder_from_checkpoint({}) is None


def test_encoder_mismatch_warns():
    ckpt = {"meta": {"encoder": "sentence-transformers/all-mpnet-base-v2"}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_encoder_matches_checkpoint("other-model", ckpt, context="test")
        assert any("encoder mismatch" in str(x.message) for x in w)
        assert any("Onyx" in str(x.message) for x in w)


def test_encoder_mismatch_strict_exits():
    ckpt = {"meta": {"encoder": "sentence-transformers/all-mpnet-base-v2"}}
    with pytest.raises(SystemExit, match="encoder mismatch"):
        check_encoder_matches_checkpoint("other-model", ckpt, strict=True, context="test")


def test_hash_encoder_warns_not_production():
    ckpt = {"meta": {"encoder": "sentence-transformers/all-mpnet-base-v2"}}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_encoder_matches_checkpoint("hash", ckpt, context="test")
        assert any("HashingEncoder" in str(x.message) for x in w)
