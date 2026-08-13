"""
retrieval_engine.py - 2-Stage Intent-Gated Hybrid Retrieval (VectorPrism)

Pillars unchanged:
  - Stage 1: HNSW on dense_core_slice + header filters
  - Stage 2: 5-channel weighted rescoring (identity is Stage-1 gate only)
  - Intent → w_intent over dense/rel/dis/hyp/causal

Improvements:
  1. Causal scored with learned asymmetric matrix M: score = q^T M c
     (matches causal_asymmetric_loss train-time objective).
  2. Epistemic truth defaults to soft (min_truth=0.0); hard filter is opt-in.
  3. Channel fusion uses z-score normalization (stable vs per-query min-max).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from tensor_contract import PSMTensorContract as C
from db_client import VectorDBClient


class IntentClassifier:
    """Maps human prompts to channel weight vectors W_intent (5 channels:
    dense, relational, disentangled, hyperbolic, causal).

    Keyword routing is a bootstrap heuristic. Prefer fixed dense-heavy
    weights or a learned router once labeled intent data exists.
    """

    def __init__(self, hard_truth_filter: bool = False, default_min_truth: float = 0.0):
        # Soft by default: hard truth gating is unsafe until ECE-calibrated.
        self.hard_truth_filter = hard_truth_filter
        self.default_min_truth = float(default_min_truth)

    def classify(self, query: str) -> Tuple[np.ndarray, Dict[str, float]]:
        q = query.lower()
        # Dense-heavy default — exotic channels start near-off until earned.
        w_intent = np.array([1.0, 0.05, 0.05, 0.0, 0.0], dtype=np.float32)
        filters = {
            "min_truth": self.default_min_truth,
            "max_anchor_dist": 0.85,
        }

        if any(w in q for w in ["why", "cause", "reason", "due to", "resulted"]):
            w_intent = np.array([0.35, 0.1, 0.1, 0.0, 0.45], dtype=np.float32)
            if self.hard_truth_filter:
                filters["min_truth"] = max(filters["min_truth"], 0.70)
        elif any(w in q for w in ["category", "parent", "tree", "type of", "hierarchy"]):
            w_intent = np.array([0.4, 0.1, 0.1, 0.4, 0.0], dtype=np.float32)

        if not self.hard_truth_filter:
            filters["min_truth"] = self.default_min_truth

        w_intent = w_intent / (np.sum(w_intent) + 1e-7)
        return w_intent, filters


class PSMRetrievalEngine:
    def __init__(
        self,
        db_client: VectorDBClient,
        causal_matrix: Optional[np.ndarray] = None,
        hard_truth_filter: bool = False,
    ):
        self.db = db_client
        self.classifier = IntentClassifier(hard_truth_filter=hard_truth_filter)
        # Train/serve parity: use learned M when available; identity ≈ old dot product.
        d = C.CAUSAL_TIME.length
        if causal_matrix is None:
            self.causal_matrix = np.eye(d, dtype=np.float32)
        else:
            m = np.asarray(causal_matrix, dtype=np.float32)
            assert m.shape == (d, d), f"causal_matrix must be ({d},{d}), got {m.shape}"
            self.causal_matrix = m

    def set_causal_matrix(self, causal_matrix: np.ndarray) -> None:
        d = C.CAUSAL_TIME.length
        m = np.asarray(causal_matrix, dtype=np.float32)
        assert m.shape == (d, d), f"causal_matrix must be ({d},{d}), got {m.shape}"
        self.causal_matrix = m

    def poincare_distance(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        sq_dist = np.sum((u - v) ** 2, axis=-1)
        u_norm = np.clip(np.sum(u ** 2, axis=-1), 0, 0.98)
        v_norm = np.clip(np.sum(v ** 2, axis=-1), 0, 0.98)
        gamma = 1 + 2 * sq_dist / ((1 - u_norm) * (1 - v_norm) + 1e-7)
        return np.arccosh(np.clip(gamma, 1.0 + 1e-7, 1e7))

    def zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Per-candidate-set z-score. More stable than min-max for fusion."""
        mu = np.mean(scores)
        sigma = np.std(scores)
        if sigma < 1e-7:
            return np.zeros_like(scores)
        return (scores - mu) / sigma

    def causal_score(self, q_causal: np.ndarray, c_causal: np.ndarray) -> np.ndarray:
        """Directional score q^T M c for each candidate row in c_causal (N, d)."""
        # (N, d) @ (d,) after q^T M  ->  c @ (M^T q)
        mq = self.causal_matrix.T @ q_causal
        return c_causal @ mq

    def search(self, query_1024d: np.ndarray, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        assert query_1024d.shape == (1024,), f"expected 1024d query vector, got {query_1024d.shape}"

        w_intent, filters = self.classifier.classify(query_text)

        dense_query_slice = query_1024d[C.DENSE_CORE.start:C.DENSE_CORE.end]
        candidates = self.db.query_dense_slice(
            vector_slice=dense_query_slice,
            min_truth=filters["min_truth"],
            max_anchor_dist=filters["max_anchor_dist"],
            limit=100,
        )
        if not candidates:
            return []

        candidate_matrix = np.stack([c["tensor_1024d"] for c in candidates])  # (N, 1024)

        c_dense = candidate_matrix[:, C.DENSE_CORE.start:C.DENSE_CORE.end]
        s_dense = c_dense @ dense_query_slice

        c_rel = candidate_matrix[:, C.RELATIONAL.start:C.RELATIONAL.end]
        q_rel = query_1024d[C.RELATIONAL.start:C.RELATIONAL.end]
        s_rel = -np.linalg.norm(c_rel - q_rel, axis=-1)

        c_dis = candidate_matrix[:, C.DISENTANGLED.start:C.DISENTANGLED.end]
        q_dis = query_1024d[C.DISENTANGLED.start:C.DISENTANGLED.end]
        s_dis = c_dis @ q_dis

        c_hyp = candidate_matrix[:, C.HYPERBOLIC.start:C.HYPERBOLIC.end]
        q_hyp = query_1024d[C.HYPERBOLIC.start:C.HYPERBOLIC.end]
        s_hyp = -self.poincare_distance(c_hyp, q_hyp)

        c_causal = candidate_matrix[:, C.CAUSAL_TIME.start:C.CAUSAL_TIME.end]
        q_causal = query_1024d[C.CAUSAL_TIME.start:C.CAUSAL_TIME.end]
        s_causal = self.causal_score(q_causal, c_causal)

        norm_scores = np.column_stack([
            self.zscore_normalize(s_dense),
            self.zscore_normalize(s_rel),
            self.zscore_normalize(s_dis),
            self.zscore_normalize(s_hyp),
            self.zscore_normalize(s_causal),
        ])  # (N, 5)

        final_scores = norm_scores @ w_intent

        ranked_indices = np.argsort(final_scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            res = dict(candidates[idx])
            res["final_score"] = float(final_scores[idx])
            results.append(res)
        return results


# Product-facing alias
VectorPrismRetrievalEngine = PSMRetrievalEngine
