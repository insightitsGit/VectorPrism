# VectorPrism Architecture — Implementation Specification (v4, Build-Ready)

**Document Identifier:** `SPEC-VECTORPRISM-2026-V4` (supersedes `SPEC-PSM-2026-V3`)
**Product name:** VectorPrism (internal `PSM*` type names retained for compatibility)
**Status:** Phases 0–6 are code-complete (train/eval/intrinsic/ablation/truth/
ingest/search/live-benchmark/reingest). Model *quality* still depends on your
labeled JSONL + a real sentence encoder — example data is plumbing only.
**Audience:** an engineer/agent picking this up fresh, with no prior
context on how this design evolved.

**Pillars (immutable):** 1024d contract · 6 channels · 2-stage retrieval ·
identity Stage-1-only · earn channels via ablation.

---

## 0. What this document is, and the one rule that matters most

This spec replaces `SPEC-PSM-2026-V1`. All code referenced below has been
written and passed a 21-test automated suite (`test_psm.py`) plus targeted
numerical checks. **Nothing in this document should ever again claim a
gap is "RESOLVED" based on code existing alone.** The prior draft did that
and it was wrong — code correctness and model correctness are different
claims. This spec keeps them in two separate columns everywhere. If you
add a feature, add it to the matrix in §6 in the correct column.

---

## 1. Executive Summary

PSM replaces flat cosine-similarity retrieval with a **1024-dimensional
tensor contract** encoding 6 independently-trained relevance channels
(dense semantics, relational structure, disentangled topic latents,
hierarchical/taxonomy structure, source-trust/OOD gating, temporal-causal
order) plus a 16-float metadata header, in one contiguous buffer per
document chunk.

**Cognitive grounding (use this framing externally, not "brain=cosine" claims):**

| Channel | Cognitive model | Evidence strength | ML technique used |
|---|---|---|---|
| Dense Core | Semantic memory (Tulving 1972) | Strong, uncontroversial | InfoNCE contrastive |
| Relational | Spreading activation (Collins & Loftus 1975) | Moderate | TransE margin loss |
| Hyperbolic Taxonomy | Prototype/category theory (Rosch 1973) | Strong — trees embed with low distortion in hyperbolic, not Euclidean, space | Poincaré ball + negative sampling |
| Time/Causal | Temporal Context Model (Howard & Kahana 2002) | Strong | Asymmetric bilinear ranking |
| Identity Anchor | Source monitoring (Johnson et al. 1993) | Phenomenon well-evidenced; distance-to-fixed-point is a rough mechanism, not a direct implementation of the theory | Center loss |
| Disentangled Latent | No strong cognitive analog — frame as pragmatic ML, not cognition | N/A | Variational Information Bottleneck |
| Epistemic Truth Score | Metacognitive confidence (Koriat 1997) | Phenomenon real; current implementation is an unsupervised linear head with no calibration — do not claim cognitive grounding until it's trained and calibrated | Requires a trained classifier (not yet built — see §6 item 11) |

Do not present all 6 channels as equally cognitively grounded in
external material — the table above is accurate; a flattened "inspired by
human cognition" claim across all 6 is not.

---

## 2. System Architecture (unchanged shape, fixed internals)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: INGESTION                                                          │
│ Raw Text ──► Frozen Base Encoder (768d) ──► MultiTaskProjectionAdapter      │
│                                                    │                        │
│                                                    ▼                        │
│                                  1024d Tensor Buffer (tensor_contract.py)   │
└──────────────────────────┬───────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: STORAGE  (db_client.py: PgVectorClient | QdrantVectorClient)      │
│  • pgvector: dense_core_slice GENERATED column, HNSW-indexed               │
│  • Qdrant:  "dense_core_slice" (HNSW) + "full_tensor" (flat) named vectors │
│  • epistemic_truth / anchor_dist / valid_timestamp as filterable scalars   │
└──────────────────────────┬───────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: RETRIEVAL (retrieval_engine.py)                                   │
│  IntentClassifier ──► w_intent (5-dim: dense/rel/dis/hyp/causal)           │
│  Stage 1 (DB): HNSW on dense_core_slice + truth/anchor filter ──► top 100 │
│  Stage 2 (RAM): per-channel scoring, min-max normalize, weighted sum      │
│                 ──► top-k                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Identity Anchor is Stage-1-only** (a gate, not a Stage-2 relevance
term) — this was a bug in earlier drafts (double-counted, and not
query-dependent when used in Stage 2). Do not re-add it to the Stage 2
weighted sum without re-deriving why; see §6 item 8.

---

## 3. Memory Contract (`tensor_contract.py`)

| Offset | Dims | Channel | Header sub-layout (if applicable) |
|---|---|---|---|
| `[0..15]` | 16 | Header | `[0]`=bitmask(uint32), `[1]`=truth(f32), `[2]`=anchor_dist(f32), `[3:5]`=timestamp(int64, bit-reinterpreted), `[5]`=model_version(uint32), `[6:16]`=reserved(zero) |
| `[16..383]` | 368 | Dense Core | cosine space, L2-normalized |
| `[384..511]` | 128 | Relational | TransE-style, needs a relation embedding table (not in this slice) |
| `[512..639]` | 128 | Disentangled | VIB latent `z` (mu/logvar live in the model, not the stored tensor) |
| `[640..767]` | 128 | Hyperbolic | Poincaré ball, norm strictly < 1 |
| `[768..895]` | 128 | Identity | distance-to-fixed-`v0`, `v0` is a frozen buffer, not learned |
| `[896..1023]` | 128 | Causal | scored via learned asymmetric matrix `M` as `qᵀ M c` at retrieval (train/serve parity) |

**Do not** revert the header packing to a per-field float cast — a Unix
timestamp exceeds float32's exact-integer range (2^24) and will silently
lose precision. Use `PSMTensorContract.pack_header` / `unpack_header` only.

---

## 4. File Manifest (all code-complete and tested; nothing here is a stub)

| File | Purpose | Test coverage |
|---|---|---|
| `tensor_contract.py` | Memory layout, header pack/unpack (+ model_version) | `TestTensorContract` |
| `losses.py` | 6 real loss functions + `poincare_distance` | `TestLosses` |
| `ingestion_adapter.py` | `MultiTaskProjectionAdapter` (the trainable model) | `TestAdapter` |
| `training.py` | `PSMBatch` data contract + training loop | smoke-tested (all 6 channels backprop) |
| `db_client.py` | `PgVectorClient`, `QdrantVectorClient` | integration-shaped; needs a live DB to run for real |
| `retrieval_engine.py` | `PSMRetrievalEngine` with `qᵀMc` causal + soft truth + z-score fusion | `TestRetrievalEngine` |
| `base_encoder.py` | Frozen encoder interface (`HashingEncoder`, `SentenceTransformerEncoder`) | used by plumbing tests |
| `dense_dataset.py` | JSONL `{query,passage}` loader for Phase 1 | `TestPlumbing` |
| `ingest_pipeline.py` | Text → encoder → 1024d → DB upsert | `TestPlumbing` |
| `checkpointing.py` | Save/load adapter + causal_matrix + model_version | `TestPlumbing` |
| `train_dense.py` | Phase-1 CLI entrypoint | manual / smoke |
| `benchmark_harness.py` | Stage-2 **latency** benchmark (p50/p95/max) | run, passing |
| `eval_harness.py` | **Quality** eval: recall@k, MRR against labeled data | smoke-tested |
| `intrinsic_validation.py` | Per-channel structural metrics | `TestIntrinsicValidation` |
| `ablation_harness.py` | dense-only vs dense+channel vs all-channels comparison | `TestAblationHarness` |
| `test_psm.py` | Full regression suite | **run this first**, `pytest test_psm.py -v` |
| `schema.sql` | pgvector DDL (+ model_version) | offsets re-verified against fixed contract |
| `GAP_RESOLUTION.md` | Historical record of what was fixed and why | reference only |

Run `pytest test_psm.py -v` before starting any new work and after any
change. All tests must pass. If you change `losses.py`'s
`poincare_distance`, re-check `test_poincare_distance_zero_for_identical_points`
specifically — this one has already broken once from an over-aggressive
epsilon clamp (`min=1+eps` inflated true-zero distances to ~0.0045; fixed
to `min=1.0`).

---

## 5. Phased Implementation Plan (for the agent doing training/deployment)

**Do not skip phases or reorder them.** Each phase has an explicit
Definition of Done (DoD). If a DoD isn't met, stop and report back rather
than continuing to the next phase — this is what the earlier drafts
skipped, which is how "9/9 gaps resolved" got claimed without evidence.

### Phase 0 — Environment
- `pip install -r requirements.txt`
- Run `pytest test_psm.py -v`. **DoD: all tests pass.**

### Phase 1 — Dense channel only
This is the one channel with cheap, available data (any (query, relevant
passage) pairs — search logs, FAQ pairs, existing sentence datasets).
- Put pairs in JSONL as `{"query":"...","passage":"..."}` (see `data/dense_pairs.example.jsonl`).
- Train with `python train_dense.py --pairs YOUR.jsonl --encoder sentence-transformers/all-mpnet-base-v2`.
- **DoD:** `eval_harness.evaluate()` on a held-out labeled query set shows recall@10 competitive with a standard sentence-embedding baseline (e.g. off-the-shelf `nomic-embed-text` cosine search on the same eval set). If VectorPrism's dense channel alone doesn't beat or match that baseline, do not proceed to Phase 2 — debug the dense channel first.

### Phase 2 — Second channel (pick ONE based on available data)
Recommended order by data availability, not by architectural interest:
1. **Causal** if you have any ordered-event data (support tickets with resolution timelines, changelogs, news event sequences).
2. **Relational** if you have or can extract (subject, relation, object) triples.
3. **Hyperbolic** if you have an explicit category/taxonomy tree.
- Train per `training.py`.
- **DoD (intrinsic):** run the matching function in `intrinsic_validation.py`
  (`causal_order_accuracy`, `link_prediction_eval`, or `embedding_distortion`
  respectively) on held-out labeled pairs. Report the number — there is no
  universal pass/fail threshold, but causal order accuracy below ~0.7 or
  Hits@10 below ~0.3 on link prediction indicates the channel has not
  learned useful structure and should not be shipped.
- **DoD (system-level):** run `ablation_harness.run_ablation()` comparing
  `dense_only` vs `dense+<channel>` on the SAME eval set from Phase 1.
  Ship the channel only if it improves recall@10 or MRR over dense_only.
  A channel can pass the intrinsic DoD and fail this one — that's a valid
  outcome (structurally correct but not useful for this retrieval task).
  Do not ship a channel that fails the ablation DoD regardless of how
  clean its intrinsic metrics look.

### Phase 3 — Remaining channels
Repeat Phase 2's process one channel at a time. Do not train all
remaining channels in parallel and evaluate once at the end — you will
not be able to attribute a system-level score change to a specific
channel, which defeats the purpose of doing this incrementally.

### Phase 4 — Epistemic Truth Score (separate track, do last)
This is currently unimplemented as a trained classifier — it's a header
field with no supervision behind it (§6 item 11).
- Collect labeled (passage, is_actually_true) data.
- Train a classifier, run `intrinsic_validation.expected_calibration_error`.
- **DoD:** ECE below ~0.05 before this is used as anything stronger than a
  soft re-ranking signal. **Do not enable it as a hard Stage-1 filter
  (`epistemic_truth >= threshold`) until ECE has been measured** — an
  uncalibrated hard filter silently drops correct results with no error
  signal. Run it in shadow mode (log what it would reject, human-review a
  sample weekly) before promoting it to a gate.

### Phase 5 — Production DB deployment
- Apply `schema.sql` (pgvector) or let `QdrantVectorClient.__init__` create
  the collection (Qdrant).
- Write a real upsert pipeline calling `MultiTaskProjectionAdapter.forward()`
  then `db_client.upsert()` per document chunk.
- **DoD:** `benchmark_harness.py`-style latency check against the LIVE db
  client (not the in-memory fake) — confirm p95 end-to-end latency budget
  is still met with real network/disk I/O, not just in-RAM compute.

### Phase 6 — Ongoing
- Any retraining of a channel invalidates previously-ingested vectors for
  that slice (they're not comparable to newly-encoded ones). Track a
  model version alongside `valid_timestamp` and plan re-ingestion, not
  just re-training.
- Re-run `ablation_harness.py` periodically as data grows — a channel
  that didn't help at launch may help once it has more training data, and
  vice versa.

---

## 6. Gap Matrix (carried forward from GAP_RESOLUTION.md, kept current)

| # | Item | Category | Status |
|---|---|---|---|
| 1-8 | Timestamp precision, header packing, Poincaré stability, benchmark/engine drift, missing DB client, Qdrant payload mismatch, identity-anchor double-counting | **CODE-RESOLVED** | Done, tested |
| 9 | Relational S+R≈O semantics | CODE-RESOLVED (mechanism) / **DATA-DEPENDENT** (validity) | Needs real triples per Phase 2/3 |
| 10 | Training loop for 6 heads | CODE-RESOLVED (loop) / **DATA-DEPENDENT** (outcome) | Needs real data per Phase 1-3 |
| 11 | Epistemic Truth Score has no trained classifier | **CODE-RESOLVED (classifier)** / **DATA-DEPENDENT (ECE)** | `epistemic_truth.py` + `vectorprism.py train-truth`. Hard filter remains blocked until `hard_filter_allowed` (ECE gate). |
| 12 | No retrieval-quality eval existed | CODE-RESOLVED (harness) / **DATA-DEPENDENT** (numbers) | Use per Phase 1+ DoDs |
| 13 | Prompt-injection defense via anchor-distance alone | **OPEN — design limitation** | Treat as one soft feature into a real safety pipeline, not a standalone gate |
| 14 | Poincaré distance inflated true-zero distances (`min=1+eps` clamp bug) | **CODE-RESOLVED** | Fixed to `min=1.0`; regression test added |
| 15 | Causal train/serve mismatch (loss used `M`, retrieval used dot product) | **CODE-RESOLVED** | Retrieval now scores `qᵀ M c`; engine loads `causal_matrix` from checkpoint |
| 16 | Unstable min-max fusion over Stage-1 candidates | **CODE-RESOLVED** | Z-score normalization per channel before weighted sum |
| 17 | No model_version in header/schema | **CODE-RESOLVED** | Header slot `[5]` + DB column; set via ingest pipeline |
| 18 | No Phase-1 data path / encoder / upsert pipeline | **CODE-RESOLVED (plumbing)** / **DATA-DEPENDENT (quality)** | `base_encoder.py`, `dense_dataset.py`, `train_dense.py`, `ingest_pipeline.py` |
| 19 | Phases 2–6 not implemented as runnable paths | **CODE-RESOLVED** | `train.py`, `channel_datasets.py`, `intrinsic_runner.py`, `eval_runner.py`, `epistemic_truth.py`, `ingest_cli.py`, `search_cli.py`, `live_benchmark.py`, `reingest.py`, `vectorprism.py` |
| 20 | Relational train TransE vs serve L2 (no query-time relation id) | **DOCUMENTED / OPEN for full S+R≈O serve** | README + Stage-2 currently use \(-\|q-c\|\); recovery claim path is causal first (`run_causal_recovery.py`) |
| 21 | Hard-eval never exercised non-dense channels | **CODE-RESOLVED (harness)** / **DATA-DEPENDENT (lift)** | `run_hard_eval.py` + `run_causal_recovery.py`; small measured causal lift on GPT miss set |

---

## 7. Explicit non-goals of this spec

- This document does not include a labeled dataset. None can be
  fabricated; sourcing each channel's data is Phase 1-4's actual work.
- This document does not claim the 6-channel design beats a strong
  dense+BM25+reranker baseline. That is an empirical question Phase 1-3's
  ablation results will answer, not something to assert beforehand.
- Cognitive-science grounding (§1 table) motivates *why* each channel's
  geometry was chosen; it is not evidence that the trained system will
  perform well. Only the DoDs in §5 are evidence.
