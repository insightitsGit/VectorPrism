# What VectorPrism needs for Phase-1 (exact contract)

## 1) Dense pairs JSONL — **required**

**File:** `dense_pairs.jsonl`  
**One object per line:**

```json
{"query": "When must a maintenance margin call be met?", "passage": "Maintenance margin calls must be met by 2 PM ET on T+1 or positions are subject to liquidation."}
```

| Field | Type | Rules |
|---|---|---|
| `query` | string | Non-empty natural language question / search |
| `passage` | string | Non-empty answer chunk (usually a corpus chunk) |

**Scale**

| Goal | Minimum | Better |
|---|---:|---:|
| Plumbing / demo | ~100 | — |
| Honest Phase-1 DoD | **300+** | **1,000–10,000+** |
| Production finance | — | search logs + FAQ + policy Q&A |

**Where clients get this**

- Internal search/click logs → `(query, clicked_doc_chunk)`
- Support / compliance FAQ pairs
- Synthetic paraphrases of each policy chunk (what our finance demo generator does)

Optional extra field (ignored by trainer, useful for debugging): `source_doc_id`.

---

## 2) Document corpus JSONL — **required for eval/search**

**File:** `documents.jsonl`

```json
{"document_id": "pol_margin_01", "chunk_text": "Maintenance margin calls must be met by 2 PM ET on T+1 or positions are subject to liquidation."}
```

| Field | Type | Rules |
|---|---|---|
| `document_id` | string | Stable unique id (used by eval) |
| `chunk_text` | string | Indexed passage |
| `epistemic_truth` | float 0–1 | Optional |

**Scale:** dozens for a demo; thousands+ for a real client KB.

---

## 3) Eval set JSONL — **required for Phase-1 DoD**

**File:** `eval.jsonl`  
**Held-out** queries (do not train on these exact strings if you want a clean DoD).

```json
{"query": "By when must a maintenance margin call be met?", "relevant_doc_ids": ["pol_margin_01"]}
```

| Field | Type | Rules |
|---|---|---|
| `query` | string | Question a finance user would type |
| `relevant_doc_ids` | string[] | One or more `document_id`s that are correct |

**Scale**

| Goal | Minimum |
|---|---:|
| Demo | 20–60 |
| Honest DoD | **50+** labeled queries |
| Enterprise sign-off | 100–300+ with SME review |

**DoD:** VectorPrism dense recall@10 ≥ plain dense-cosine baseline on the **same** eval set.

---

## 4) Real encoder — **required for client-facing quality**

| Encoder | Use |
|---|---|
| `hash` | Offline CI / plumbing only — **not** for client demos |
| `sentence-transformers/all-mpnet-base-v2` | Near-real demo / Phase-1 baseline |
| Client’s production embedder (768d) | Best — wire via `SentenceTransformerEncoder` or a custom `BaseTextEncoder` |

Adapter expects **768d** base embeddings today.

---

## 5) Optional next (after Phase-1 passes)

| Channel | JSONL schema | Finance example |
|---|---|---|
| Causal | `{"earlier","later"}` | Incident timelines (wire delay → callback void) |
| Relational | `{"subject","relation","object","negative_object"}` | Entity–control–regulation triples |
| Hyperbolic | `{"parent","child","negatives":[...]}` | Product / policy taxonomy |
| Truth | `{"text","is_true":0\|1}` | Verified vs outdated policy text |
| DB | pgvector DSN or Qdrant URL | Production deploy |

---

## Finance demo included here

This folder ships a **near-real** synthetic wealth/capital-markets knowledge base:

```bash
python demos/finance_demo/generate_corpus.py
python demos/finance_demo/run_demo.py --encoder sentence-transformers/all-mpnet-base-v2
```

Replace these JSONL files with the client’s real policies + labeled pairs when available. The code path stays identical.
