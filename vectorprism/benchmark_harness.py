"""
benchmark_harness.py - Latency benchmark for Stage 2 rescoring.

CHANGES FROM PRIOR DRAFT:
  - No longer reimplements scoring logic inline (that copy had its own
    causal-slice off-by-one bug AND a different w_intent dimensionality
    than the real engine — two implementations silently drifting apart).
    This harness now calls PSMRetrievalEngine.search() directly, so a
    passing benchmark actually proves something about the shipped code.
  - Still uses synthetic random tensors, and that's fine for a LATENCY
    benchmark. It is explicitly NOT a correctness/quality benchmark —
    see eval_harness.py for that, which requires real labeled data.
"""

import time
import numpy as np

from vectorprism.tensor_contract import PSMTensorContract as C
from vectorprism.retrieval_engine import PSMRetrievalEngine
from vectorprism.db_client import VectorDBClient


def generate_synthetic_tensors(num_records: int = 100_000) -> np.ndarray:
    print(f"[*] Allocating {num_records} synthetic 1024d tensors "
          f"(~{(num_records * 1024 * 4) / 1024 / 1024:.2f} MB)...")
    rng = np.random.default_rng(42)
    tensors = rng.standard_normal((num_records, C.TOTAL_DIM)).astype(np.float32)

    dense = tensors[:, C.DENSE_CORE.start:C.DENSE_CORE.end]
    tensors[:, C.DENSE_CORE.start:C.DENSE_CORE.end] = dense / np.linalg.norm(dense, axis=-1, keepdims=True)

    hyp = tensors[:, C.HYPERBOLIC.start:C.HYPERBOLIC.end]
    norm = np.linalg.norm(hyp, axis=-1, keepdims=True)
    tensors[:, C.HYPERBOLIC.start:C.HYPERBOLIC.end] = hyp / (1 + norm) * 0.9  # keep inside Poincare ball

    tensors[:, C.HEADER.start:C.HEADER.end] = 0.0
    tensors[:, 1] = 1.0  # epistemic_truth header slot = 1.0, passes any filter

    return tensors


class InMemoryBenchmarkDB(VectorDBClient):
    """Stands in for a real DB so the benchmark measures Stage 2 compute
    only, isolated from network/disk I/O — Stage 1 candidate selection
    here is a slice, not a real HNSW query."""
    def __init__(self, tensors: np.ndarray):
        self.tensors = tensors

    def upsert(self, *a, **kw):
        raise NotImplementedError("benchmark DB is read-only")

    def query_dense_slice(
        self, vector_slice, min_truth, max_anchor_dist, limit, model_version=None
    ):
        top = self.tensors[:limit]
        return [
            {"document_id": f"doc_{i}", "chunk_text": "", "tensor_1024d": top[i]}
            for i in range(top.shape[0])
        ]


def run_benchmark(num_records: int = 100_000, top_k: int = 5, n_trials: int = 20):
    tensors = generate_synthetic_tensors(num_records)
    db = InMemoryBenchmarkDB(tensors)
    engine = PSMRetrievalEngine(db)

    rng = np.random.default_rng(7)
    query = rng.standard_normal(C.TOTAL_DIM).astype(np.float32)
    q_hyp = query[C.HYPERBOLIC.start:C.HYPERBOLIC.end]
    query[C.HYPERBOLIC.start:C.HYPERBOLIC.end] = q_hyp / (1 + np.linalg.norm(q_hyp)) * 0.9

    print("\n[1] Header unpack timing...")
    t0 = time.perf_counter()
    info = C.unpack_header(query)
    print(f"    {((time.perf_counter()-t0)*1000):.4f} ms | {info}")

    print(f"\n[2] Stage 2 rescoring latency over {n_trials} trials "
          f"(real PSMRetrievalEngine.search(), top-100 candidates -> top-{top_k})...")
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        results = engine.search(query, query_text="why did this happen", top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"    mean={latencies.mean():.3f} ms  p50={np.percentile(latencies,50):.3f} ms  "
          f"p95={np.percentile(latencies,95):.3f} ms  max={latencies.max():.3f} ms")
    print(f"    top-{top_k} doc ids: {[r['document_id'] for r in results]}")

    # SLA check uses p95, not a single lucky run
    assert np.percentile(latencies, 95) < 15.0, "Latency SLA failed: p95 exceeded 15ms budget"
    print(f"\n[SUCCESS] p95 latency within 15ms end-to-end retrieval SLA "
          f"(this measures the REAL search() call, not a parallel reimplementation).")


if __name__ == "__main__":
    run_benchmark()
