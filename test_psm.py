"""
test_psm.py - Automated regression suite. Run with: pytest test_psm.py -v

Covers: tensor contract round-trips, loss function numerical stability,
adapter forward pass / slice alignment, retrieval engine correctness,
intrinsic validation metrics, ablation harness, ingest + dense dataset plumbing.

This tests that the CODE is correct — it uses synthetic data throughout
and asserts shapes/ranges/invariants, not accuracy numbers.
"""

import time
import numpy as np
import pytest
import torch

from tensor_contract import PSMTensorContract as C
from ingestion_adapter import MultiTaskProjectionAdapter
from losses import (
    info_nce_loss, transe_margin_loss, VIBHead, vib_loss,
    poincare_distance, poincare_negative_sampling_loss,
    center_loss, anchor_distance_score, causal_asymmetric_loss,
)
from retrieval_engine import PSMRetrievalEngine, IntentClassifier
from db_client import VectorDBClient
from intrinsic_validation import (
    ndcg_at_k, disentanglement_probe, ood_detection_auroc, causal_order_accuracy,
    expected_calibration_error,
)
from eval_harness import EvalExample
from ablation_harness import run_ablation
from base_encoder import HashingEncoder
from dense_dataset import load_dense_pairs_jsonl, make_dense_dataloader
from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
from checkpointing import save_checkpoint, load_checkpoint
from pathlib import Path


# --------------------------------------------------------------------------
# Tensor contract
# --------------------------------------------------------------------------
class TestTensorContract:
    def test_header_roundtrip_exact(self):
        ts = int(time.time())
        hdr = C.pack_header(
            bitmask=0b1011,
            epistemic_truth=0.77,
            anchor_distance=0.33,
            timestamp=ts,
            model_version=42,
        )
        t = np.zeros(1024, dtype=np.float32)
        t[0:16] = hdr
        out = C.unpack_header(t)
        assert out["bitmask"] == 0b1011
        assert abs(out["epistemic_truth"] - 0.77) < 1e-5
        assert abs(out["anchor_distance"] - 0.33) < 1e-5
        assert out["timestamp"] == ts
        assert out["model_version"] == 42

    def test_slices_are_contiguous_and_cover_1024(self):
        slices = [C.HEADER, C.DENSE_CORE, C.RELATIONAL, C.DISENTANGLED,
                  C.HYPERBOLIC, C.IDENTITY, C.CAUSAL_TIME]
        pos = 0
        for s in slices:
            assert s.start == pos, f"{s.name} starts at {s.start}, expected {pos}"
            pos = s.end
        assert pos == 1024

    def test_truth_score_clipped_to_valid_range(self):
        hdr = C.pack_header(bitmask=0, epistemic_truth=5.0, anchor_distance=0.0, timestamp=0)
        t = np.zeros(1024, dtype=np.float32)
        t[0:16] = hdr
        out = C.unpack_header(t)
        assert 0.0 <= out["epistemic_truth"] <= 1.0

    def test_header_reserved_tail_is_zero(self):
        hdr = C.pack_header(1, 0.5, 0.1, 123, model_version=7)
        assert np.allclose(hdr[6:16], 0.0)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
class TestLosses:
    @pytest.fixture
    def dims(self):
        return dict(B=8, D=128)

    def test_all_losses_finite(self, dims):
        B, D = dims["B"], dims["D"]
        assert torch.isfinite(info_nce_loss(torch.randn(B, D), torch.randn(B, D)))
        assert torch.isfinite(transe_margin_loss(*[torch.randn(B, D) for _ in range(4)]))

        vib = VIBHead(D, 32, 5)
        z, mu, logvar, logits = vib(torch.randn(B, D))
        assert torch.isfinite(vib_loss(logits, torch.randint(0, 5, (B,)), mu, logvar))

        def to_ball(x):
            n = x.norm(dim=-1, keepdim=True)
            return x / (1 + n) * 0.9
        p, c, negs = to_ball(torch.randn(B, D)), to_ball(torch.randn(B, D)), to_ball(torch.randn(B, 5, D))
        assert torch.isfinite(poincare_negative_sampling_loss(p, c, negs))

        assert torch.isfinite(center_loss(torch.randn(B, D), torch.randn(D)))

        M = torch.randn(D, D) * 0.01
        assert torch.isfinite(causal_asymmetric_loss(torch.randn(B, D), torch.randn(B, D), M))

    def test_poincare_distance_symmetric_and_nonnegative(self, dims):
        D = dims["D"]
        def to_ball(x):
            n = x.norm(dim=-1, keepdim=True)
            return x / (1 + n) * 0.9
        u, v = to_ball(torch.randn(5, D)), to_ball(torch.randn(5, D))
        d_uv = poincare_distance(u, v)
        d_vu = poincare_distance(v, u)
        assert torch.allclose(d_uv, d_vu, atol=1e-4)
        assert (d_uv >= 0).all()

    def test_poincare_distance_zero_for_identical_points(self, dims):
        D = dims["D"]
        u = torch.full((3, D), 0.1)
        d = poincare_distance(u, u)
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-3)


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------
class TestAdapter:
    def test_forward_shape_and_slice_alignment(self):
        model = MultiTaskProjectionAdapter(base_dim=768)
        B = 4
        base_emb = torch.randn(B, 768)
        headers = torch.zeros(B, 16)
        tensor_1024d, raw = model(base_emb, headers)
        assert tensor_1024d.shape == (B, 1024)
        assert torch.allclose(tensor_1024d[:, C.DENSE_CORE.start:C.DENSE_CORE.end], raw["dense"])
        assert torch.allclose(tensor_1024d[:, C.HYPERBOLIC.start:C.HYPERBOLIC.end], raw["hyperbolic"])

    def test_hyperbolic_output_stays_in_ball(self):
        model = MultiTaskProjectionAdapter(base_dim=768)
        base_emb = torch.randn(50, 768) * 10
        _, raw = model(base_emb, torch.zeros(50, 16))
        norms = torch.norm(raw["hyperbolic"], dim=-1)
        assert (norms < 1.0).all(), "hyperbolic output escaped the Poincare ball"

    def test_dense_output_is_unit_normalized(self):
        model = MultiTaskProjectionAdapter(base_dim=768)
        base_emb = torch.randn(20, 768)
        _, raw = model(base_emb, torch.zeros(20, 16))
        norms = torch.norm(raw["dense"], dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_identity_anchor_is_frozen_buffer_not_parameter(self):
        model = MultiTaskProjectionAdapter(base_dim=768)
        param_names = [n for n, _ in model.named_parameters()]
        assert "identity_anchor_v0" not in param_names


# --------------------------------------------------------------------------
# Retrieval engine
# --------------------------------------------------------------------------
class _FakeDB(VectorDBClient):
    def __init__(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        self.docs = []
        for i in range(n):
            t = rng.standard_normal(1024).astype(np.float32)
            t[C.DENSE_CORE.start:C.DENSE_CORE.end] /= np.linalg.norm(t[C.DENSE_CORE.start:C.DENSE_CORE.end])
            hyp = t[C.HYPERBOLIC.start:C.HYPERBOLIC.end]
            t[C.HYPERBOLIC.start:C.HYPERBOLIC.end] = hyp / (1 + np.linalg.norm(hyp)) * 0.9
            self.docs.append({"document_id": f"doc_{i}", "chunk_text": "", "tensor_1024d": t,
                               "epistemic_truth": 0.9, "anchor_dist": 0.1})

    def upsert(self, *a, **kw):
        pass

    def query_dense_slice(self, vector_slice, min_truth, max_anchor_dist, limit, model_version=None):
        return self.docs[:limit]


class TestRetrievalEngine:
    def test_search_returns_top_k(self):
        db = _FakeDB(n=50)
        engine = PSMRetrievalEngine(db)
        q = db.docs[0]["tensor_1024d"].copy()
        results = engine.search(q, "test query", top_k=5)
        assert len(results) == 5

    def test_exact_match_query_ranks_itself_first(self):
        db = _FakeDB(n=50)
        engine = PSMRetrievalEngine(db)
        q = db.docs[7]["tensor_1024d"].copy()
        results = engine.search(q, "test query", top_k=1)
        assert results[0]["document_id"] == "doc_7"

    def test_causal_slice_uses_full_128_dims(self):
        assert C.CAUSAL_TIME.end - C.CAUSAL_TIME.start == 128
        assert C.CAUSAL_TIME.start == 896 and C.CAUSAL_TIME.end == 1024

    def test_intent_classifier_weights_sum_to_one(self):
        clf = IntentClassifier()
        for q in ["why did this happen", "show me the category tree", "generic query"]:
            w, _ = clf.classify(q)
            assert abs(w.sum() - 1.0) < 1e-5
            assert w.shape == (5,)

    def test_truth_filter_soft_by_default(self):
        clf = IntentClassifier()
        _, filters = clf.classify("why did this happen")
        assert filters["min_truth"] == 0.0

    def test_hard_truth_filter_opt_in(self):
        clf = IntentClassifier(hard_truth_filter=True)
        _, filters = clf.classify("why did this happen")
        assert filters["min_truth"] >= 0.70

    def test_causal_uses_asymmetric_matrix(self):
        """Train/serve parity: retrieval score is q^T M c (not plain dot product)."""
        db = _FakeDB(n=20, seed=1)
        # Unit-normalize causal slices so outer-product M has a unique maximizer.
        for doc in db.docs:
            c = doc["tensor_1024d"][C.CAUSAL_TIME.start:C.CAUSAL_TIME.end]
            doc["tensor_1024d"][C.CAUSAL_TIME.start:C.CAUSAL_TIME.end] = c / (np.linalg.norm(c) + 1e-7)

        q = db.docs[0]["tensor_1024d"].copy()
        t_c = db.docs[3]["tensor_1024d"][C.CAUSAL_TIME.start:C.CAUSAL_TIME.end]
        q_c = q[C.CAUSAL_TIME.start:C.CAUSAL_TIME.end]
        q_c = q_c / (np.linalg.norm(q_c) + 1e-7)
        q[C.CAUSAL_TIME.start:C.CAUSAL_TIME.end] = q_c

        M = np.outer(q_c, t_c).astype(np.float32)
        engine = PSMRetrievalEngine(db, causal_matrix=M)
        engine.classifier.classify = lambda _q: (
            np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            {"min_truth": 0.0, "max_anchor_dist": 1.0},
        )
        results = engine.search(q, "why cause", top_k=1)
        assert results[0]["document_id"] == "doc_3"

        # Direct score check: target beats a different candidate under M
        cands = np.stack([
            db.docs[1]["tensor_1024d"][C.CAUSAL_TIME.start:C.CAUSAL_TIME.end],
            db.docs[3]["tensor_1024d"][C.CAUSAL_TIME.start:C.CAUSAL_TIME.end],
        ])
        scores = engine.causal_score(q_c, cands)
        assert scores[1] > scores[0]


# --------------------------------------------------------------------------
# Intrinsic validation metrics
# --------------------------------------------------------------------------
class TestIntrinsicValidation:
    def test_ndcg_perfect_ranking_is_one(self):
        assert ndcg_at_k([3, 2, 1], k=3) == pytest.approx(1.0)

    def test_ece_zero_for_perfectly_calibrated(self):
        rng = np.random.default_rng(0)
        probs = np.repeat([0.1, 0.5, 0.9], 1000)
        truth = rng.uniform(0, 1, len(probs)) < probs
        result = expected_calibration_error(probs, truth.astype(int))
        assert result["ECE"] < 0.05

    def test_disentanglement_probe_detects_leakage(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, (300, 8))
        nuisance = (z[:, 0] > 0).astype(int)
        intended = rng.integers(0, 2, 300)
        result = disentanglement_probe(z, intended, nuisance)
        assert result["nuisance_label_accuracy"] > 0.8

    def test_ood_auroc_range(self):
        rng = np.random.default_rng(0)
        auc = ood_detection_auroc(rng.uniform(0, 0.3, 50), rng.uniform(0.5, 1.0, 50))
        assert 0.0 <= auc <= 1.0

    def test_causal_order_accuracy_range(self):
        rng = np.random.default_rng(0)
        acc = causal_order_accuracy(rng.uniform(0, 1, 30), rng.uniform(0, 1, 30))
        assert 0.0 <= acc <= 1.0


# --------------------------------------------------------------------------
# Ablation harness
# --------------------------------------------------------------------------
class TestAblationHarness:
    def test_restores_original_classifier_after_run(self):
        db = _FakeDB(n=20)
        engine = PSMRetrievalEngine(db)
        original = engine.classifier.classify
        eval_set = [EvalExample("q", db.docs[0]["tensor_1024d"].copy(), {"doc_0"})]
        run_ablation(engine, eval_set)
        assert engine.classifier.classify == original

    def test_all_configs_present(self):
        db = _FakeDB(n=20)
        engine = PSMRetrievalEngine(db)
        eval_set = [EvalExample("q", db.docs[0]["tensor_1024d"].copy(), {"doc_0"})]
        results = run_ablation(engine, eval_set)
        expected = {"dense_only", "dense+relational", "dense+disentangled",
                    "dense+hyperbolic", "dense+causal", "all_channels_equal"}
        assert expected.issubset(results.keys())


# --------------------------------------------------------------------------
# Ingest / dataset / checkpoint plumbing
# --------------------------------------------------------------------------
class _MemoryDB(VectorDBClient):
    def __init__(self):
        self.rows = []

    def upsert(self, doc_id, chunk_text, tensor_1024d, meta):
        self.rows.append({
            "document_id": doc_id,
            "chunk_text": chunk_text,
            "tensor_1024d": tensor_1024d.copy(),
            **meta,
        })

    def query_dense_slice(self, vector_slice, min_truth, max_anchor_dist, limit, model_version=None):
        out = []
        for r in self.rows:
            if model_version is not None and int(r.get("model_version", 0)) != int(model_version):
                continue
            if r["epistemic_truth"] >= min_truth and r["anchor_dist"] <= max_anchor_dist:
                out.append(r)
            if len(out) >= limit:
                break
        return out

    def get_by_ids(self, doc_ids):
        want = set(str(x) for x in doc_ids)
        return [r for r in self.rows if r["document_id"] in want]


class TestPlumbing:
    def test_example_dense_pairs_load(self):
        path = Path(__file__).parent / "data" / "dense_pairs.example.jsonl"
        pairs = load_dense_pairs_jsonl(path)
        assert len(pairs) >= 10

    def test_dense_dataloader_yields_psm_batch(self):
        path = Path(__file__).parent / "data" / "dense_pairs.example.jsonl"
        enc = HashingEncoder(768)
        loader = make_dense_dataloader(path, enc, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch.dense_anchor.shape[0] == 4
        assert batch.dense_anchor.shape[1] == 768
        assert batch.dense_positive.shape == batch.dense_anchor.shape

    def test_ingest_pipeline_upserts_1024d(self):
        enc = HashingEncoder(768)
        adapter = MultiTaskProjectionAdapter(base_dim=768)
        db = _MemoryDB()
        pipe = VectorPrismIngestPipeline(enc, adapter, db, model_version=3)
        n = pipe.upsert_documents([
            IngestDocument("d1", "VectorPrism stores multi-channel tensors."),
            IngestDocument("d2", "Dense retrieval remains the backbone."),
        ])
        assert n == 2
        assert db.rows[0]["tensor_1024d"].shape == (1024,)
        assert db.rows[0]["model_version"] == 3
        hdr = C.unpack_header(db.rows[0]["tensor_1024d"])
        assert hdr["model_version"] == 3

    def test_checkpoint_roundtrip(self, tmp_path):
        adapter = MultiTaskProjectionAdapter(base_dim=768)
        path = tmp_path / "vp.pt"
        save_checkpoint(path, adapter, model_version=9, meta={"phase": 1})
        loaded = load_checkpoint(path)
        assert loaded["model_version"] == 9
        assert loaded["causal_matrix"].shape == (128, 128)
        assert isinstance(loaded["adapter"], MultiTaskProjectionAdapter)
