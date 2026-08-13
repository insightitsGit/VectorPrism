"""
test_phases.py - Phase 1-6 plumbing tests (example JSONL, hash encoder).

These assert CODE paths work end-to-end. They do NOT claim quality DoDs.
"""

from pathlib import Path
import numpy as np
import torch

from base_encoder import HashingEncoder
from train import main as train_main
from checkpointing import load_checkpoint
from epistemic_truth import train_truth_classifier, load_truth_classifier, shadow_filter_report, score_texts
from eval_runner import run_phase1_eval, run_ablation_eval, InMemoryCorpusDB
from intrinsic_runner import run_intrinsic
from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
from channel_datasets import load_documents_jsonl, make_channel_dataloader
from retrieval_engine import PSMRetrievalEngine
from live_benchmark import main as live_bench_main

DATA = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"


def test_all_channel_dataloaders_smoke():
    enc = HashingEncoder(768)
    mapping = {
        "dense": DATA / "dense_pairs.example.jsonl",
        "causal": DATA / "causal.example.jsonl",
        "relational": DATA / "relational.example.jsonl",
        "hyperbolic": DATA / "hyperbolic.example.jsonl",
        "disentangled": DATA / "disentangled.example.jsonl",
        "identity": DATA / "identity.example.jsonl",
    }
    for channel, path in mapping.items():
        loader, aux = make_channel_dataloader(channel, path, enc, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.header.shape[0] <= 4


def test_train_dense_and_causal_checkpoint(tmp_path):
    out = tmp_path / "vp.pt"
    train_main([
        "--channel", "dense",
        "--data", str(DATA / "dense_pairs.example.jsonl"),
        "--out", str(out),
        "--encoder", "hash",
        "--epochs", "1",
        "--batch-size", "8",
    ])
    train_main([
        "--channel", "causal",
        "--data", str(DATA / "causal.example.jsonl"),
        "--init", str(out),
        "--out", str(out),
        "--encoder", "hash",
        "--epochs", "1",
        "--batch-size", "8",
    ])
    ckpt = load_checkpoint(out)
    assert ckpt["enabled_channels"]["dense"] is True
    assert ckpt["enabled_channels"]["causal"] is True
    assert ckpt["causal_matrix"].shape == (128, 128)


def test_phase1_eval_and_ablation(tmp_path):
    out = tmp_path / "vp.pt"
    train_main([
        "--channel", "dense",
        "--data", str(DATA / "dense_pairs.example.jsonl"),
        "--out", str(out),
        "--encoder", "hash",
        "--epochs", "1",
        "--batch-size", "8",
    ])
    enc = HashingEncoder(768)
    report = run_phase1_eval(
        str(out), enc,
        str(DATA / "documents.example.jsonl"),
        str(DATA / "eval.example.jsonl"),
    )
    assert "recall@10" in report.vectorprism
    assert "recall@10" in report.dense_cosine_baseline
    ablation = run_ablation_eval(
        str(out), enc,
        str(DATA / "documents.example.jsonl"),
        str(DATA / "eval.example.jsonl"),
    )
    assert "dense_only" in ablation


def test_intrinsic_causal(tmp_path):
    out = tmp_path / "vp.pt"
    train_main([
        "--channel", "dense",
        "--data", str(DATA / "dense_pairs.example.jsonl"),
        "--out", str(out), "--encoder", "hash", "--epochs", "1", "--batch-size", "8",
    ])
    train_main([
        "--channel", "causal",
        "--data", str(DATA / "causal.example.jsonl"),
        "--init", str(out), "--out", str(out), "--encoder", "hash", "--epochs", "2", "--batch-size", "8",
    ])
    enc = HashingEncoder(768)
    report = run_intrinsic("causal", str(out), enc, str(DATA / "causal.example.jsonl"))
    assert "causal_order_accuracy" in report
    assert 0.0 <= report["causal_order_accuracy"] <= 1.0


def test_truth_classifier_and_shadow(tmp_path):
    enc = HashingEncoder(768)
    out = tmp_path / "truth.pt"
    report = train_truth_classifier(
        str(DATA / "truth.example.jsonl"),
        enc,
        str(out),
        epochs=3,
        batch_size=8,
    )
    model, meta = load_truth_classifier(out)
    probs = score_texts(model, enc, ["VectorPrism uses a 1024-dimensional tensor."])
    shadow = shadow_filter_report(probs, threshold=0.5)
    assert "would_reject" in shadow
    assert out.exists()
    assert "ece" in meta


def test_ingest_search_live_benchmark(tmp_path):
    out = tmp_path / "vp.pt"
    train_main([
        "--channel", "dense",
        "--data", str(DATA / "dense_pairs.example.jsonl"),
        "--out", str(out), "--encoder", "hash", "--epochs", "1", "--batch-size", "8",
    ])
    report = live_bench_main([
        "--checkpoint", str(out),
        "--documents", str(DATA / "documents.example.jsonl"),
        "--encoder", "hash",
        "--backend", "memory",
        "--n-trials", "5",
        "--p95-budget-ms", "200",
    ])
    assert report["pass_sla"] is True
    assert report["p95_ms"] < 200


def test_truth_scores_written_into_header(tmp_path):
    from tensor_contract import PSMTensorContract as C

    enc = HashingEncoder(768)
    out = tmp_path / "vp.pt"
    train_main([
        "--channel", "dense",
        "--data", str(DATA / "dense_pairs.example.jsonl"),
        "--out", str(out), "--encoder", "hash", "--epochs", "1", "--batch-size", "8",
    ])
    ckpt = load_checkpoint(out)
    tpath = tmp_path / "truth.pt"
    train_truth_classifier(str(DATA / "truth.example.jsonl"), enc, str(tpath), epochs=2, batch_size=8)
    model, _ = load_truth_classifier(tpath)

    db = InMemoryCorpusDB()
    pipe = VectorPrismIngestPipeline(
        enc, ckpt["adapter"], db, model_version=1, truth_classifier=model
    )
    pipe.upsert_documents([
        IngestDocument("x", "VectorPrism uses a 1024-dimensional multi-channel tensor contract.")
    ])
    hdr = C.unpack_header(db.rows[0]["tensor_1024d"])
    assert 0.0 <= hdr["epistemic_truth"] <= 1.0
