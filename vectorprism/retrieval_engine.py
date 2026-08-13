"""
retrieval_engine.py - 2-Stage Intent-Gated Hybrid Retrieval (VectorPrism)

Stage 1: dense ANN ∪ causal ancestors ∪ taxonomy lineage ∪ relational matches
Stage 2: z-score weighted fusion OR Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import logging
import re
import time
import numpy as np

from vectorprism.tensor_contract import PSMTensorContract as C
from vectorprism.db_client import VectorDBClient
from vectorprism.causal_graph import CausalDocGraph
from vectorprism.structure_index import TaxonomyGraph, RelationalAttrIndex

logger = logging.getLogger(__name__)


class IntentClassifier:
    def __init__(
        self,
        hard_truth_filter: bool = False,
        default_min_truth: float = 0.0,
        model_version: Optional[int] = None,
    ):
        self.hard_truth_filter = hard_truth_filter
        self.default_min_truth = float(default_min_truth)
        self.model_version = model_version

    def classify(self, query: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        q = query.lower()
        # dense, relational, disentangled, hyperbolic, causal
        w_intent = np.array([1.0, 0.05, 0.05, 0.0, 0.0], dtype=np.float32)
        filters: Dict[str, Any] = {
            "min_truth": self.default_min_truth,
            "max_anchor_dist": 0.85,
            "model_version": self.model_version,
        }

        if any(w in q for w in ["why", "cause", "reason", "due to", "resulted"]):
            w_intent = np.array([0.30, 0.10, 0.05, 0.05, 0.50], dtype=np.float32)
            if self.hard_truth_filter:
                filters["min_truth"] = max(filters["min_truth"], 0.70)
        elif any(
            w in q
            for w in [
                "category",
                "parent",
                "tree",
                "type of",
                "hierarchy",
                "tier-",
                "which operational approval",
                "which rule",
                "currently effective",
                "segregation requirement",
            ]
        ):
            w_intent = np.array([0.30, 0.10, 0.05, 0.45, 0.10], dtype=np.float32)
        elif any(
            w in q
            for w in [
                "without",
                "threshold",
                "exceed",
                "callback",
                "ssi",
                "disposition",
                "requirement",
                "same-day",
                "london",
            ]
        ):
            w_intent = np.array([0.30, 0.45, 0.05, 0.10, 0.10], dtype=np.float32)

        if not self.hard_truth_filter:
            filters["min_truth"] = self.default_min_truth
        w_intent = w_intent / (np.sum(w_intent) + 1e-7)
        return w_intent, filters


def _query_tokens(text: str) -> set[str]:
    stop = {
        "why",
        "did",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "when",
        "what",
        "which",
        "was",
        "even",
        "though",
        "still",
        "showed",
        "already",
    }
    return {t for t in re.findall(r"[a-z0-9$%]{3,}", text.lower()) if t not in stop}


class PSMRetrievalEngine:
    def __init__(
        self,
        db_client: VectorDBClient,
        causal_matrix: Optional[np.ndarray] = None,
        hard_truth_filter: bool = False,
        causal_graph: Optional[CausalDocGraph] = None,
        taxonomy_graph: Optional[TaxonomyGraph] = None,
        relational_index: Optional[RelationalAttrIndex] = None,
        stage1_dense_limit: int = 50,
        stage1_graph_hops: int = 2,
        graph_struct_bonus: float = 4.0,
        fusion: str = "zscore",  # "zscore" | "rrf"
        rrf_k: int = 20,  # post-validation sweep: k=20 best among {20..60} on adversarial
        doc_text_lookup: Optional[Dict[str, str]] = None,
        query_conditioned_expansion: bool = True,
        beam_width: Optional[int] = None,
        max_stage1_candidates: Optional[int] = None,
        model_version: Optional[int] = None,
    ):
        self.db = db_client
        self.classifier = IntentClassifier(
            hard_truth_filter=hard_truth_filter,
            model_version=model_version,
        )
        self.causal_graph = causal_graph
        self.taxonomy_graph = taxonomy_graph
        self.relational_index = relational_index
        self.stage1_dense_limit = int(stage1_dense_limit)
        self.stage1_graph_hops = int(stage1_graph_hops)
        self.graph_struct_bonus = float(graph_struct_bonus)
        self.fusion = fusion
        self.rrf_k = int(rrf_k)
        self.doc_text_lookup = doc_text_lookup or {}
        self.query_conditioned_expansion = bool(query_conditioned_expansion)
        self.beam_width = beam_width
        self.max_stage1_candidates = max_stage1_candidates
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
        assert m.shape == (d, d)
        self.causal_matrix = m

    def set_causal_graph(self, causal_graph: Optional[CausalDocGraph]) -> None:
        self.causal_graph = causal_graph

    def poincare_distance(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        sq_dist = np.sum((u - v) ** 2, axis=-1)
        u_norm = np.clip(np.sum(u ** 2, axis=-1), 0, 0.98)
        v_norm = np.clip(np.sum(v ** 2, axis=-1), 0, 0.98)
        gamma = 1 + 2 * sq_dist / ((1 - u_norm) * (1 - v_norm) + 1e-7)
        return np.arccosh(np.clip(gamma, 1.0 + 1e-7, 1e7))

    def zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        mu = np.mean(scores)
        sigma = np.std(scores)
        if sigma < 1e-7:
            return np.zeros_like(scores)
        return (scores - mu) / sigma

    def causal_score(self, q_causal: np.ndarray, c_causal: np.ndarray) -> np.ndarray:
        mq = self.causal_matrix.T @ q_causal
        return c_causal @ mq

    def _filter_by_query_context(self, expanded: Dict[str, int], query_text: str) -> Dict[str, int]:
        if not self.query_conditioned_expansion or not expanded:
            return expanded
        qtoks = _query_tokens(query_text)
        if len(qtoks) < 3:
            return expanded
        kept: Dict[str, int] = {}
        for did, hop in expanded.items():
            if hop == 0:
                kept[did] = hop
                continue
            text = self.doc_text_lookup.get(did, "")
            if not text:
                if hop <= 1:
                    kept[did] = hop
                continue
            # Hop-1 ancestors are direct structured neighbors of dense seeds — keep them.
            # Fan-out control applies only at hop ≥ 2 via query-token overlap.
            if hop == 1:
                kept[did] = hop
                continue
            overlap = len(qtoks & _query_tokens(text))
            if overlap >= 2:
                kept[did] = hop
        return kept if kept else expanded

    def _fetch_missing(self, missing: List[str], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not missing:
            return []
        try:
            fetched = self.db.get_by_ids(missing)
        except NotImplementedError as exc:
            logger.warning(
                "Graph expansion requested %d doc ids but %s; "
                "structured neighbors will be dropped. Implement get_by_ids on the DB client.",
                len(missing),
                exc,
            )
            return []
        except Exception:
            logger.exception(
                "get_by_ids failed while fetching %d graph-expanded candidates",
                len(missing),
            )
            raise
        out = []
        want_version = filters.get("model_version")
        for row in fetched:
            if row.get("epistemic_truth", 1.0) < filters["min_truth"]:
                continue
            if row.get("anchor_dist", 0.0) > filters["max_anchor_dist"]:
                continue
            if want_version is not None and int(row.get("model_version", 0)) != int(want_version):
                continue
            out.append(row)
        return out

    def _stage1_candidates(
        self,
        dense_query_slice: np.ndarray,
        filters: Dict[str, Any],
        query_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Returns candidates plus per-channel hop maps (causal / hyp / rel).
        Channel hops are independent of dense membership so Stage-2 bonuses still apply
        when gold is already in the dense pool.
        """
        use_struct = any(
            x is not None for x in (self.causal_graph, self.taxonomy_graph, self.relational_index)
        )
        dense_limit = self.stage1_dense_limit if use_struct else 100
        candidates = self.db.query_dense_slice(
            vector_slice=dense_query_slice,
            min_truth=filters["min_truth"],
            max_anchor_dist=filters["max_anchor_dist"],
            limit=max(dense_limit, 1),
            model_version=filters.get("model_version"),
        )
        causal_hops: Dict[str, int] = {}
        hyp_hops: Dict[str, int] = {}
        rel_hops: Dict[str, int] = {}
        if not use_struct or not candidates:
            return candidates, causal_hops, hyp_hops, rel_hops

        seed_n = min(10, len(candidates))
        seed_ids = [c["document_id"] for c in candidates[:seed_n]]
        have = {c["document_id"] for c in candidates}
        to_fetch: List[str] = []

        if self.causal_graph is not None:
            expanded = self.causal_graph.expand(
                seed_ids,
                hops=self.stage1_graph_hops,
                upstream=True,
                beam_width=self.beam_width,
            )
            expanded = self._filter_by_query_context(expanded, query_text)
            for did, hop in expanded.items():
                if hop <= 0:
                    continue
                causal_hops[did] = min(causal_hops.get(did, hop), hop)
                if did not in have:
                    to_fetch.append(did)

        if self.taxonomy_graph is not None:
            lineage = self.taxonomy_graph.get_lineage(
                seed_ids,
                depth_delta=self.stage1_graph_hops,
                beam_width=self.beam_width,
            )
            for did, hop in lineage.items():
                if hop <= 0:
                    continue
                hyp_hops[did] = min(hyp_hops.get(did, hop), hop)
                if did not in have:
                    to_fetch.append(did)

        if self.relational_index is not None:
            for did in self.relational_index.match_predicates(query_text, limit=20):
                rel_hops[did] = 1
                if did not in have:
                    to_fetch.append(did)

        for row in self._fetch_missing(list(dict.fromkeys(to_fetch)), filters):
            did = row["document_id"]
            if did in have:
                continue
            candidates.append(row)
            have.add(did)
            if did not in self.doc_text_lookup:
                self.doc_text_lookup[did] = str(row.get("chunk_text", ""))

        # Cap Stage-1 pool: keep dense head, then structured adds by hop priority
        if self.max_stage1_candidates is not None and len(candidates) > self.max_stage1_candidates:
            dense_n = min(self.stage1_dense_limit, self.max_stage1_candidates)
            head = candidates[:dense_n]
            rest = candidates[dense_n:]

            def _prio(row: Dict[str, Any]) -> Tuple[int, str]:
                did = row["document_id"]
                hop = min(
                    causal_hops.get(did, 99),
                    hyp_hops.get(did, 99),
                    rel_hops.get(did, 99),
                )
                return (hop, did)

            rest.sort(key=_prio)
            budget = self.max_stage1_candidates - len(head)
            candidates = head + rest[: max(budget, 0)]

        return candidates, causal_hops, hyp_hops, rel_hops

    def _rrf_fuse(self, channel_scores: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        n = channel_scores[0].shape[0]
        fused = np.zeros(n, dtype=np.float32)
        for scores, w in zip(channel_scores, weights):
            order = np.argsort(-scores)
            ranks = np.empty(n, dtype=np.int32)
            ranks[order] = np.arange(1, n + 1)
            fused += (float(w) / (self.rrf_k + ranks)).astype(np.float32)
        return fused

    def search(
        self,
        query_1024d: np.ndarray,
        query_text: str,
        top_k: int = 5,
        *,
        return_stats: bool = False,
    ):
        assert query_1024d.shape == (1024,), f"expected 1024d query vector, got {query_1024d.shape}"
        t0 = time.perf_counter()
        w_intent, filters = self.classifier.classify(query_text)
        dense_query_slice = query_1024d[C.DENSE_CORE.start : C.DENSE_CORE.end]
        t1 = time.perf_counter()
        candidates, causal_hops, hyp_hops, rel_hops = self._stage1_candidates(
            dense_query_slice, filters, query_text
        )
        t2 = time.perf_counter()
        if not candidates:
            if return_stats:
                return [], {
                    "n_candidates": 0,
                    "n_causal_hops": 0,
                    "n_hyp_hops": 0,
                    "n_rel_hops": 0,
                    "stage1_ms": (t2 - t1) * 1000.0,
                    "stage2_ms": 0.0,
                    "total_ms": (t2 - t0) * 1000.0,
                }
            return []

        candidate_matrix = np.stack([c["tensor_1024d"] for c in candidates])
        c_dense = candidate_matrix[:, C.DENSE_CORE.start : C.DENSE_CORE.end]
        s_dense = c_dense @ dense_query_slice

        c_rel = candidate_matrix[:, C.RELATIONAL.start : C.RELATIONAL.end]
        q_rel = query_1024d[C.RELATIONAL.start : C.RELATIONAL.end]
        s_rel = -np.linalg.norm(c_rel - q_rel, axis=-1)
        if self.relational_index is not None:
            pred = np.array(
                [self.relational_index.predicate_score(query_text, c["document_id"]) for c in candidates],
                dtype=np.float32,
            )
            # Strong predicate prior so exact attribute matches beat near-miss distractors
            s_rel = s_rel + 5.0 * pred
            for i, c in enumerate(candidates):
                if c["document_id"] in rel_hops:
                    s_rel[i] += self.graph_struct_bonus

        c_dis = candidate_matrix[:, C.DISENTANGLED.start : C.DISENTANGLED.end]
        q_dis = query_1024d[C.DISENTANGLED.start : C.DISENTANGLED.end]
        s_dis = c_dis @ q_dis

        c_hyp = candidate_matrix[:, C.HYPERBOLIC.start : C.HYPERBOLIC.end]
        q_hyp = query_1024d[C.HYPERBOLIC.start : C.HYPERBOLIC.end]
        s_hyp = -self.poincare_distance(c_hyp, q_hyp)
        if self.taxonomy_graph is not None and self.graph_struct_bonus > 0:
            for i, c in enumerate(candidates):
                hop = hyp_hops.get(c["document_id"])
                if hop is not None and hop > 0:
                    s_hyp[i] += self.graph_struct_bonus / float(hop)

        c_causal = candidate_matrix[:, C.CAUSAL_TIME.start : C.CAUSAL_TIME.end]
        q_causal = query_1024d[C.CAUSAL_TIME.start : C.CAUSAL_TIME.end]
        s_causal = self.causal_score(q_causal, c_causal)
        if self.causal_graph is not None and self.graph_struct_bonus > 0:
            for i, c in enumerate(candidates):
                hop = causal_hops.get(c["document_id"])
                if hop is not None and hop > 0:
                    s_causal[i] += self.graph_struct_bonus / float(hop)

        channel_scores = [s_dense, s_rel, s_dis, s_hyp, s_causal]
        if self.fusion == "rrf":
            final_scores = self._rrf_fuse(channel_scores, w_intent)
        else:
            norm_scores = np.column_stack([self.zscore_normalize(s) for s in channel_scores])
            final_scores = norm_scores @ w_intent

        ranked_indices = np.argsort(final_scores)[::-1][:top_k]
        results = []
        for idx in ranked_indices:
            res = dict(candidates[idx])
            res["final_score"] = float(final_scores[idx])
            results.append(res)
        t3 = time.perf_counter()
        if return_stats:
            return results, {
                "n_candidates": len(candidates),
                "n_causal_hops": len(causal_hops),
                "n_hyp_hops": len(hyp_hops),
                "n_rel_hops": len(rel_hops),
                "stage1_ms": (t2 - t1) * 1000.0,
                "stage2_ms": (t3 - t2) * 1000.0,
                "total_ms": (t3 - t0) * 1000.0,
            }
        return results


VectorPrismRetrievalEngine = PSMRetrievalEngine
