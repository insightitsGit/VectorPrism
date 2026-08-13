# PSM Architecture — Gap Resolution Matrix (v2)

All code below has been written and verified to run (unit-tested per-function,
smoke-tested end-to-end for adapter forward pass, training loop with all 6
losses active, retrieval engine against a fake DB, latency benchmark, and
eval harness). Nothing in this matrix is marked resolved without a passing
test behind it in this session.

Two categories, kept distinct on purpose — collapsing them is what caused
the original doc's "9/9 RESOLVED" table to overstate readiness:

- **CODE-RESOLVED**: a bug or missing implementation, now fixed and tested.
- **DATA-DEPENDENT**: correct code is now in place, but the component only
  becomes meaningful once trained/calibrated on real data that only you
  can provide. No further code work changes this category.

| # | Item | Category | What changed |
|---|------|----------|--------------|
| 1 | Timestamp precision loss (float32 cast of Unix time) | **CODE-RESOLVED** | Bit-reinterpreted as int64 across 2 float32 slots in `tensor_contract.py`. Exact round-trip verified. |
| 2 | Header bitmask broadcast-fragility (4 slots for a 1-slot value) | **CODE-RESOLVED** | Fixed 1-slot packing, 11 slots explicitly reserved and zero-filled. |
| 3 | Poincaré projection numerical stability | **CODE-RESOLVED** (carried over, already correct) | `tanh(norm)/norm` squash retained — verified boundary-safe. |
| 4 | Benchmark causal-slice off-by-one (`896:1023` drops last dim) | **CODE-RESOLVED** | Benchmark now calls the real `PSMRetrievalEngine.search()` instead of a parallel reimplementation, so this class of drift can't recur. |
| 5 | Benchmark/engine used different w_intent dimensionality (5 vs 6) | **CODE-RESOLVED** | Single scoring path; both now share 5 channels (dense, relational, disentangled, hyperbolic, causal). |
| 6 | `db.query_dense_slice()` called but never implemented | **CODE-RESOLVED** | `db_client.py`: real `PgVectorClient` and `QdrantVectorClient`, both implementing a common `VectorDBClient` interface. |
| 7 | Qdrant config stored full tensor as unindexed payload | **CODE-RESOLVED** | Stored as a second named vector (`full_tensor`, flat/unindexed) so Stage 2 retrieves it through the normal vector API. |
| 8 | Identity Anchor scored against origin, not query, inside Stage 2 relevance sum (effectively double-counting the Stage-1 gate under a broken formula) | **CODE-RESOLVED** | Removed from Stage 2 ranking entirely; it now does its one job (OOD/injection-risk gating) only in Stage 1, via the header's `anchor_dist` filter. |
| 9 | Relational scoring didn't implement the S+R≈O structure its own loss claimed | **CODE-RESOLVED** (semantics), **DATA-DEPENDENT** (validity) | `losses.py`'s `transe_margin_loss` now takes explicit `(subject, relation, object, negative)` args with a real relation-embedding table (`RelationEmbeddingTable` in `training.py`). Retrieval-time scoring is documented as valid *only if* training data pairs real query text against target chunks in that triple format — this is a data requirement, not something further code can resolve. |
| 10 | No training loop existed for the 6 non-dense heads | **CODE-RESOLVED** (loop), **DATA-DEPENDENT** (outcome) | `losses.py` + `training.py`: real InfoNCE, TransE, VIB, Poincaré negative-sampling, center-loss, and asymmetric-causal losses, wired into one training loop that backprops each channel independently when its data is present. All 6 losses verified numerically finite on synthetic batches. **This does not mean the heads are trained** — running this loop requires real labeled data per the schema in `PSMBatch`'s docstring (dense: query/passage pairs; relational: KG triples; disentangled: input+label pairs; hyperbolic: taxonomy edges; identity: in-domain reference set; causal: ordered event pairs). |
| 11 | Epistemic Truth Score had no classifier behind it, yet was used as a hard Stage-1 filter | **PARTIALLY OPEN** | Not fixed by code. This needs either (a) a trained factuality/quality classifier with its own labeled data and eval, or (b) a decision to drop it as a hard filter and use it only as a soft re-ranking signal with a conservative default. Recommend (b) until (a) has an eval showing it's reliable — a miscalibrated hard filter silently drops good results. |
| 12 | No retrieval-quality evaluation existed, only latency | **CODE-RESOLVED** (harness), **DATA-DEPENDENT** (numbers) | `eval_harness.py`: recall@k and MRR against a labeled `EvalExample` set. Smoke-tested and working. You still need to build the labeled eval set — no eval harness can manufacture ground-truth relevance judgments. |
| 13 | Prompt-injection defense claimed via anchor-distance alone | **OPEN — design limitation, not a bug** | `anchor_distance_score()` in `losses.py` is now explicitly documented as one soft feature into a real safety pipeline, not a standalone gate. Embedding-distance-to-anchor is not a validated injection detector in the literature; recommend pairing with a classifier trained on known injection examples before relying on this in production. |

## Files delivered

- `tensor_contract.py` — memory layout + fixed header pack/unpack
- `losses.py` — 6 real, unit-verified loss functions
- `ingestion_adapter.py` — the 6-head model, now training-ready (returns raw per-head tensors)
- `training.py` — full training loop, explicit `PSMBatch` data contract
- `db_client.py` — pgvector + Qdrant clients implementing a shared interface
- `retrieval_engine.py` — fixed 2-stage search, single source of truth for scoring
- `benchmark_harness.py` — latency-only benchmark, now calls real `search()`
- `eval_harness.py` — recall@k / MRR against a labeled eval set (new)
- `schema.sql` — unchanged; offsets re-verified correct against the fixed contract

## What "production-ready" honestly means from here

The code is ready to receive data. It is not yet a working retrieval system,
because 6 of 7 channels have never seen a training example. The single
highest-leverage next step is: pick ONE non-dense channel you have real data
for, train it, and eval it with `eval_harness.py` before investing in the
other five. Shipping 7 subspaces where 1 is validated and 6 are hopeful is
strictly worse than shipping 2 that are both proven to work.

## v4 follow-up (VectorPrism build kickoff)

Additional CODE-RESOLVED items without changing pillars:

| # | Item | Fix |
|---|---|---|
| 15 | Causal scored with dot product while trained with `uᵀMv` | `retrieval_engine.causal_score` = `qᵀ M c` |
| 16 | Min-max fusion instability | Z-score normalize each channel |
| 17 | No model versioning | Header slot `[5]` + `schema.sql.model_version` |
| 11 (partial) | Hard truth filter with no classifier | Soft default `min_truth=0.0`; hard filter opt-in |
| 18 | Missing Phase-1 path | `train_dense.py`, JSONL loader, encoder interface, ingest pipeline |

Product name is VectorPrism; `PSM*` identifiers remain as internal aliases.
