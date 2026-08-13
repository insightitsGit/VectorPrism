# VectorPrism

Multi-channel retrieval: each chunk is a fixed **1024d tensor** (header + six channels). Stage 1 = HNSW on dense core; Stage 2 = intent-weighted multi-channel rescoring.

**Pillars (immutable):** 1024d contract · 6 channels · 2-stage retrieval · identity Stage-1-only · earn channels via ablation.

## Install

```bash
pip install -r requirements.txt
python vectorprism.py phase0
```

## Phases 0–6 (complete implementation)

| Phase | Command | What it does |
|---|---|---|
| 0 | `python vectorprism.py phase0` | Run full test suite |
| 1 | `python train.py --channel dense --data pairs.jsonl --encoder sentence-transformers/all-mpnet-base-v2` | Train dense |
| 1 DoD | `python vectorprism.py eval --checkpoint ckpt.pt --documents docs.jsonl --eval eval.jsonl --encoder ...` | vs dense baseline |
| 2–3 | `python train.py --channel causal --data causal.jsonl --init ckpt.pt --out ckpt.pt` | One channel at a time |
| 2–3 DoD | `python vectorprism.py intrinsic --channel causal --checkpoint ckpt.pt --data causal.jsonl` | Intrinsic gate |
| 2–3 DoD | `python vectorprism.py ablation --checkpoint ckpt.pt --documents docs.jsonl --eval eval.jsonl` | System lift gate |
| 4 | `python vectorprism.py train-truth --data truth.jsonl --out truth.pt` | Epistemic truth + ECE |
| 5 | `python vectorprism.py ingest --checkpoint ckpt.pt --documents docs.jsonl --backend pgvector --dsn ...` | Upsert |
| 5 | `python vectorprism.py search --checkpoint ckpt.pt --query "..." --backend pgvector --dsn ...` | Query |
| 5 | `python vectorprism.py live-benchmark --checkpoint ckpt.pt --documents docs.jsonl --backend ...` | E2E latency |
| 6 | `python vectorprism.py reingest --checkpoint new.pt --documents docs.jsonl --backend ...` | Versioned re-encode |

### Full plumbing smoke (example data only)

```bash
python vectorprism.py run-all-smoke
```

This exercises Phases 1–6 on `data/*.example.jsonl`. It is **not** a quality claim — DoD gates may exit code `2` on tiny hash-encoder examples.

## Data schemas (`data/*.example.jsonl`)

- `dense`: `{"query","passage"}`
- `causal`: `{"earlier","later"}`
- `relational`: `{"subject","relation","object","negative_object"}`
- `hyperbolic`: `{"parent","child","negatives":[...]}`
- `disentangled`: `{"text","label"}`
- `identity`: `{"text","in_domain"}`
- `ood`: `{"text","is_ood"}`
- `truth`: `{"text","is_true"}`
- `documents`: `{"document_id","chunk_text"}`
- `eval`: `{"query","relevant_doc_ids":[...]}`

## Production backends

```bash
# Postgres / pgvector
psql "$VECTORPRISM_PG_DSN" -f schema.sql
python vectorprism.py ingest --backend pgvector --dsn "$VECTORPRISM_PG_DSN" ...

# Qdrant
python vectorprism.py ingest --backend qdrant --qdrant-url http://localhost:6333 ...
```

## Honest readiness

| Layer | Status |
|---|---|
| Code / CLIs / tests for Phases 0–6 | Complete |
| Trained quality on your corpus | Requires your labeled JSONL + real encoder |
| Shipping all 6 channels | Only after each channel passes intrinsic + ablation DoDs |

See `IMPLEMENTATION_SPEC.md` for DoD thresholds and gap matrix.
