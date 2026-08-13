# Real-world pilot playbook (external users, not internal evals)

A **pilot** means someone outside your laptop installs VectorPrism and runs it on **their** documents, queries, and (ideally) labels. Packaging (`pip` / git clone) is what makes that possible.

---

## What a pilot partner needs from you

1. **Installable software** — `pip install "vectorprism[all]"` from PyPI, or `pip install -e ".[all]"` from git for latest main / full adversarial packs
2. **A 30–60 minute path to first search** — ingest → search on their corpus
3. **Clear success criteria** — e.g. “recover root-cause docs dense misses” with R@10 / MRR on *their* eval set
4. **Support channel** — Discord / email / shared Slack for 2–4 weeks

You do **not** need a perfect auto-KG on day one. You need: install works, search works, they can drop in JSONL, and you can iterate on their graph/attrs together.

---

## How to get a real pilot (recruitment)

| Channel | Who | Pitch |
|---------|-----|--------|
| Existing customers / warm intros | Ops, risk, compliance, SRE | “Dense RAG returns funny neighbors on why/policy queries — we recover with causal/taxonomy channels” |
| Communities | RAG Discord, local AI meetups, LinkedIn | Offer a **free 2-week pilot** with white-glove ingest help |
| Design partners | 1–3 fintech / infra teams | Trade discounted/free usage for labeled misses + quote for case study |

**Ask for:** 200–5,000 document chunks, 20–50 labeled queries (or incident tickets), optional cause/policy links if they have them.

**Offer:** install support, one shared eval report, no lock-in, Apache-2.0.

---

## Partner install (git + pip)

```bash
git clone https://github.com/insightitsGit/VectorPrism.git
cd VectorPrism
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip
pip install -e ".[all]"   # encoder + postgres + qdrant + pytest

vectorprism --help
vectorprism pilot-check
pytest test_psm.py test_phases.py -q
```

Windows + sentence-transformers: prefer Docker (`DOCKER.md`) if native wheels crash.

```bash
docker compose run --rm vectorprism vectorprism pilot-check
```

---

## Partner data contract (minimum)

Put three files in a folder (e.g. `pilot_data/`):

| File | Format |
|------|--------|
| `documents.jsonl` | `{"document_id", "chunk_text", "epistemic_truth"?}` |
| `dense_pairs.jsonl` | `{"query", "passage"}` for dense training |
| `eval.jsonl` | `{"query", "relevant_doc_ids": ["..."]}` |

Optional structure (multipliers for hard queries):

| File | Format |
|------|--------|
| `causal_graph.jsonl` | `{"earlier_doc_id", "later_doc_id"}` cause→symptom |
| `hyperbolic_graph.jsonl` | `{"parent_doc_id", "child_doc_id"}` |
| `relational_attrs.jsonl` | `{"doc_id", "attributes": {...}}` |

Auto-extract a starter graph from text:

```bash
python demos/finance_demo/extract_structure_auto.py \
  --pack pilot_data --out pilot_data_auto --backend heuristic
# or --backend llm with OPENAI_API_KEY / VECTORPRISM_LLM_API_KEY
```

---

## Pilot runbook (week 1)

```bash
# 1) Train dense on their pairs
vectorprism train --channel dense \
  --data pilot_data/dense_pairs.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --out checkpoints/pilot_dense.pt --epochs 3

# 2) Ingest (memory backend for laptop demos — persists to --store NPZ)
vectorprism ingest --checkpoint checkpoints/pilot_dense.pt \
  --documents pilot_data/documents.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --backend memory \
  --store checkpoints/memory_corpus.npz

# 3) Search (same --store; empty store exits non-zero)
vectorprism search --checkpoint checkpoints/pilot_dense.pt \
  --query "Why did outbound wires freeze after MFA passed?" \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --backend memory --store checkpoints/memory_corpus.npz --top-k 10

# 4) Eval vs dense baseline
vectorprism eval --checkpoint checkpoints/pilot_dense.pt \
  --documents pilot_data/documents.jsonl \
  --eval pilot_data/eval.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --out pilot_results/phase1.json
```

Postgres path (shared staging):

```bash
docker compose up -d db
export VECTORPRISM_PG_DSN=postgresql://vectorprism:vectorprism@localhost:5433/vectorprism
vectorprism ingest --checkpoint checkpoints/pilot_dense.pt \
  --documents pilot_data/documents.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --backend postgres --dsn "$VECTORPRISM_PG_DSN"
```

Week 2: add causal/hyp/rel training + graphs; score recovery on dense misses (reuse finance demo scripts as templates).

---

## Success criteria (agree up front)

| Tier | Bar |
|------|-----|
| **Smoke** | Install + ingest + search returns sensible docs |
| **Dense DoD** | R@10 / MRR on their eval ≥ agreed baseline |
| **Channel win** | On confirmed dense misses@10, multi-channel recovers ≥30–50% (domain-dependent) |
| **Ops** | p95 search latency budget met on their backend |

Publish only claims measured on **their** data — not the adversarial finance pack alone.

---

## Git + PyPI checklist (enables pilots)

| Step | Status in repo |
|------|----------------|
| `pyproject.toml` + `pip install -e .` | Added |
| Console script `vectorprism` | Added |
| Public git clone install docs | This file + README |
| Push main with packaging | **You push when ready** |
| PyPI publish (`twine upload`) | **0.1.0 live**; bump for bugfix |
| Example checkpoint download | Optional (HF / release assets) — don’t put huge `.pt` on PyPI |

### Publish to PyPI (newer releases)

```bash
pip install -e ".[dev]"
python -m build
twine check dist/*
# twine upload dist/*   # requires PyPI token; 0.1.0 already live
```

---

## Your next actions (this week)

1. **Push** packaging commits to GitHub (so partners can `git clone` + `pip install -e ".[all]"`).
2. **Message 5 warm contacts** with PILOT.md + a 15-min Loom of finance demo.
3. **Book one design partner** who can share JSONL under NDA.
4. Run `vectorprism pilot-check` with them on a call — if it fails, fix install before any model talk.
