# VectorPrism

**Positional Subspace Multiplexing (PSM) & Intent-Gated 2-Stage Retrieval Engine for High-Scale RAG.**

[![PyPI](https://img.shields.io/badge/PyPI-vectorprism-blue?logo=pypi&logoColor=white)](https://pypi.org/project/vectorprism/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Build](https://img.shields.io/badge/tests-36%20passing-brightgreen)](https://github.com/insightitsGit/VectorPrism/actions)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white)](https://discord.gg/vectorprism)
[![GitHub](https://img.shields.io/badge/GitHub-insightitsGit%2FVectorPrism-181717?logo=github)](https://github.com/insightitsGit/VectorPrism)

> One contiguous **1024d** tensor. Six independently trained relevance subspaces. Stage-1 HNSW + Stage-2 intent-gated rescoring. Baseline vector-DB storage cost — not 6× multi-vector inflation.

---

## The Core Problem (Why VectorPrism?)

Enterprise RAG is stuck between two bad defaults:

1. **Flat cosine over a single embedding** — semantically “close” neighbors that are causally wrong, taxonomically wrong, or temporally expired. Teams call them *funny neighbors*; production calls them **hallucination fuel**.
2. **Multi-vector indexing** (one ANN index per representation) — better signal, but **500%–1,000%** storage and query fan-out on pgvector / Qdrant bills.

**VectorPrism** multiplexes **six specialized representation subspaces** plus a **16-float Control Header** into a **single 1024-dimensional contiguous buffer** per chunk:

| Constraint | VectorPrism answer |
|---|---|
| Storage | **1×** vector footprint (one `vector(1024)` / named full tensor) |
| Stage 1 | HNSW only on the **368d dense core** slice |
| Stage 2 | In-RAM zero-copy slice scoring with intent weights |
| Early exit | Header filters (`epistemic_truth`, `anchor_dist`, timestamp, `model_version`) before heavy math |
| Latency target | **&lt; 15ms** end-to-end search SLA (see benchmarks) |

Philosophically grounded channel design. Engineering-grounded memory contract. Production path for pgvector and Qdrant.

---

## High-Value Enterprise Use Cases

### 1. Root-Cause Causal Analysis & Incident Logs
**Keywords:** causal retrieval, incident log RAG, DevOps root-cause analysis, “why did the service fail”

When on-call asks *“Why did Server X crash at 3 AM?”*, cosine-only RAG returns symptom-adjacent text. VectorPrism’s **Time ODE & Directional Causality** slice (`[896:1024)`) is trained with an asymmetric bilinear score \(q^{\top} M c\) (`PSMRetrievalEngine.causal_score`). Intent routing up-weights the causal channel on “why / cause / reason” queries so Stage-2 rescoring prefers **cause→effect order**, not merely lexical neighbors.

### 2. Enterprise Knowledge Graphs & Taxonomy Search
**Keywords:** hyperbolic embeddings RAG, taxonomy search, medical ontology retrieval, legal hierarchy search

Parent–child trees distort badly in Euclidean space. The **Hyperbolic Taxonomy (Porphyry)** slice (`[640:768)`) lives in a Poincaré ball (norm &lt; 1) and is scored with Poincaré distance in Stage 2. Hierarchy intents (“category”, “parent”, “type of”, “tree”) shift `IntentClassifier` weights toward hyperbolic structure for medical, legal, and product taxonomies.

### 3. Bitemporal & Compliance Audit Trail Retrieval
**Keywords:** bitemporal retrieval, compliance RAG, healthcare audit trail, finance document expiry filter

Before Stage-2 matrix math, the **16d Control Header Manifest** (`[0:16)`) exposes O(1) metadata:

- Epistemic truth score (soft by default; hard filter opt-in after ECE calibration)
- Identity anchor distance (OOD / injection-risk gate in Stage 1)
- Exact int64 timestamp (bit-reinterpreted across two float32 slots — no float32 rounding of Unix time)
- Model version for safe re-ingest after retrains

Stage 1 can reject expired, unverified, or out-of-domain chunks **before** rescoring.

### 4. Cost-Optimized Scale for pgvector & Qdrant
**Keywords:** multi-vector RAG cost reduction, pgvector HNSW, Qdrant named vectors, high-scale vector search

Instead of six ANN indexes, VectorPrism stores **one 1024d tensor**. Stage 1 indexes only the generated **368d `dense_core_slice`**. Stage 2 pulls the full tensor for the top-~100 candidates and rescored slices in RAM. AI SaaS platforms keep multi-signal retrieval without multi-vector sticker shock.

---

## 1024-Dimensional Tensor Memory Map (Code Contract)

Ground truth: `PSMTensorContract` / `VectorPrismTensorContract` in [`tensor_contract.py`](tensor_contract.py).

```
1024-d VectorPrism Tensor (float32)
┌──────────────────────────────────────────────────────────────────────────┐
│ [  0 ..  15]  16d   Control Header Manifest                              │
│ [ 16 .. 383] 368d   Dense Semantic Core          (Hume / Wittgenstein)   │
│ [384 .. 511] 128d   Relational Group Algebra     (Aristotle / Al-Khwarizmi)│
│ [512 .. 639] 128d   Disentangled Latent Space    (Jabir)                 │
│ [640 .. 767] 128d   Hyperbolic Taxonomy          (Porphyry)              │
│ [768 .. 895] 128d   Identity Consistency         (Ibn Sina)              │
│ [896 ..1023] 128d   Time ODE & Causality         (Mulla Sadra / Spinoza) │
└──────────────────────────────────────────────────────────────────────────┘
         ▲ Stage-1 HNSW indexes ONLY dense_core [16:384) → 368 dims
```

| Inclusive range | Code slice (`start:end`) | Dims | Channel | Role |
|---|---|---:|---|---|
| `[0000..0015]` | `HEADER` `[0:16)` | 16 | Control Header Manifest | Bitmask, truth, anchor dist, timestamp, model version |
| `[0016..0383]` | `DENSE_CORE` `[16:384)` | 368 | Dense Semantic Core | L2-normalized cosine space; Stage-1 ANN |
| `[0384..0511]` | `RELATIONAL` `[384:512)` | 128 | Relational Group Algebra | Train: TransE \(S+R\approx O\); **serve today: L2 proximity** \(-\|q_{\mathrm{rel}}-c_{\mathrm{rel}}\|\) (no query-time relation id yet) |
| `[0512..0639]` | `DISENTANGLED` `[512:640)` | 128 | Disentangled Latent (Jabir) | VIB latent \(z\) |
| `[0640..0767]` | `HYPERBOLIC` `[640:768)` | 128 | Hyperbolic Taxonomy (Porphyry) | Poincaré ball |
| `[0768..0895]` | `IDENTITY` `[768:896)` | 128 | Identity Consistency (Ibn Sina) | Distance-to-frozen \(v_0\); **Stage-1 gate only** |
| `[0896..1023]` | `CAUSAL_TIME` `[896:1024)` | 128 | Time ODE & Causality (Spinoza) | Scored as \(q^{\top} M c\) |

**Header sub-layout** (exact packing via `PSMTensorContract.pack_header` / `unpack_header`):

| Slot | Field | Encoding |
|---|---|---|
| `[0]` | Channel bitmask | `uint32` ↔ `float32` bit reinterpret |
| `[1]` | Epistemic truth | `float32` in `[0, 1]` |
| `[2]` | Identity anchor distance | `float32` |
| `[3:5]` | Bitemporal timestamp | `int64` ↔ `2×float32` bit reinterpret |
| `[5]` | Model version | `uint32` ↔ `float32` bit reinterpret |
| `[6:16]` | Reserved | zero-filled |

---

## Quickstart & Code Examples

### Installation

```bash
# From source (current)
git clone https://github.com/insightitsGit/VectorPrism.git
cd VectorPrism
pip install -r requirements.txt

# Verify
pytest test_psm.py test_phases.py -v

# PyPI (when published)
# pip install vectorprism
```

Core deps: `torch`, `numpy`, `scipy`, `scikit-learn`, `pytest`. Optional: `psycopg[binary]`, `pgvector`, `qdrant-client`, `sentence-transformers`.

### Production path (Docker + Postgres/pgvector) — recommended on Windows

```bash
docker compose up -d db
docker compose run --rm test
docker compose run --rm finance-pg
docker compose run --rm production-smoke
```

- DB: `localhost:5433` · DSN `postgresql://vectorprism:vectorprism@localhost:5433/vectorprism`
- Results: `demos/finance_demo/results/` (`PRODUCTION_RESULTS.md`, eval, live search JSON)
- Full checklist: [`PRODUCTION.md`](PRODUCTION.md) · Docker notes: [`DOCKER.md`](DOCKER.md)

### Example 1 — Multi-Task Ingestion Adapter

Encode raw text with a frozen 768d encoder → `MultiTaskProjectionAdapter` → contiguous **1024d** tensor (matches `ingestion_adapter.py` + `ingest_pipeline.py`).

```python
import time
import torch
import numpy as np

from base_encoder import SentenceTransformerEncoder
from ingestion_adapter import MultiTaskProjectionAdapter, VectorPrismProjectionAdapter
from tensor_contract import PSMTensorContract as C, VectorPrismTensorContract
from losses import anchor_distance_score

# Frozen base encoder (768d) + trainable 6-head adapter
encoder = SentenceTransformerEncoder("sentence-transformers/all-mpnet-base-v2")
adapter = MultiTaskProjectionAdapter(base_dim=768)  # alias: VectorPrismProjectionAdapter
adapter.eval()

texts = ["Cache eviction storm preceded the 3 AM outage on Server X."]
base = encoder.encode(texts)  # (1, 768)

header = C.pack_header(
    bitmask=C.default_channel_bitmask({"dense": True, "identity": True, "causal": True}),
    epistemic_truth=1.0,
    anchor_distance=0.0,
    timestamp=int(time.time()),
    model_version=1,
)
header_t = torch.from_numpy(header).unsqueeze(0)  # (1, 16)

with torch.no_grad():
    tensor_1024d, raw = adapter(base, header_t)
    # Fill identity distance into header slot [2]
    dist = anchor_distance_score(raw["identity"], adapter.identity_anchor_v0)
    out = tensor_1024d.cpu().numpy().astype(np.float32)
    out[0, C.HDR_ANCHOR.start] = float(dist[0].item())

assert out.shape == (1, 1024)
assert out[0, C.DENSE_CORE.start:C.DENSE_CORE.end].shape == (368,)
meta = C.unpack_header(out[0])
print(meta)  # bitmask, epistemic_truth, anchor_distance, timestamp, model_version
```

Production shorthand (upsert path):

```python
from checkpointing import load_checkpoint
from db_client import PgVectorClient  # or QdrantVectorClient
from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
from base_encoder import SentenceTransformerEncoder

ckpt = load_checkpoint("checkpoints/vectorprism.pt")
encoder = SentenceTransformerEncoder("sentence-transformers/all-mpnet-base-v2")
db = PgVectorClient("postgresql://user:pass@localhost:5432/vectorprism")

pipe = VectorPrismIngestPipeline(
    encoder=encoder,
    adapter=ckpt["adapter"],
    db=db,
    model_version=ckpt["model_version"],
    enabled_channels=ckpt.get("enabled_channels"),
)
pipe.upsert_documents([
    IngestDocument(document_id="inc-42", chunk_text="Cache eviction preceded the outage."),
])
```

### Example 2 — Stage 1 / Stage 2 Hybrid Search

Intent classification → HNSW on dense core → zero-copy slice rescoring (`PSMRetrievalEngine.search`).

```python
from checkpointing import load_checkpoint
from base_encoder import SentenceTransformerEncoder
from db_client import PgVectorClient
from ingest_pipeline import VectorPrismIngestPipeline
from retrieval_engine import PSMRetrievalEngine, IntentClassifier, VectorPrismRetrievalEngine
from tensor_contract import PSMTensorContract as C

ckpt = load_checkpoint("checkpoints/vectorprism.pt")
encoder = SentenceTransformerEncoder("sentence-transformers/all-mpnet-base-v2")
db = PgVectorClient("postgresql://user:pass@localhost:5432/vectorprism")

pipe = VectorPrismIngestPipeline(encoder, ckpt["adapter"], db, model_version=ckpt["model_version"])
engine = PSMRetrievalEngine(  # alias: VectorPrismRetrievalEngine
    db_client=db,
    causal_matrix=ckpt["causal_matrix"],  # learned M for qᵀ M c
    hard_truth_filter=False,              # keep soft until ECE-calibrated
)

query_text = "Why did Server X crash at 3 AM?"
query_1024d = pipe.encode_query(query_text)

# Optional: inspect intent weights (dense, relational, disentangled, hyperbolic, causal)
w_intent, filters = engine.classifier.classify(query_text)
print("w_intent=", w_intent, "filters=", filters)

hits = engine.search(query_1024d, query_text, top_k=5)
for h in hits:
    print(h["document_id"], h["final_score"], h.get("chunk_text", "")[:120])

# Stage-2 scoring uses exact slices, e.g. causal:
#   q_c = query_1024d[C.CAUSAL_TIME.start:C.CAUSAL_TIME.end]
#   s_causal = engine.causal_score(q_c, candidate_causal_matrix)
```

CLI equivalents:

```bash
python train.py --channel dense --data data/dense_pairs.example.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 --out checkpoints/vectorprism.pt

python vectorprism.py ingest --checkpoint checkpoints/vectorprism.pt \
  --documents data/documents.example.jsonl --backend pgvector --dsn "$VECTORPRISM_PG_DSN"

python vectorprism.py search --checkpoint checkpoints/vectorprism.pt \
  --query "Why did Server X crash at 3 AM?" --backend pgvector --dsn "$VECTORPRISM_PG_DSN"
```

---

## Database Setup & Schema

### PostgreSQL + pgvector

Exact DDL from [`schema.sql`](schema.sql):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS psm_document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL,
    chunk_text TEXT NOT NULL,

    -- Full 1024-Dimensional Composite Tensor Payload
    tensor_1024d vector(1024) NOT NULL,

    -- Generated Column for Stage 1 Dense Core Slice [16..383] (368d)
    -- pgvector subvector() is 1-indexed: (17, 368) == zero-indexed [16:384)
    dense_core_slice vector(368) GENERATED ALWAYS AS (
        subvector(tensor_1024d, 17, 368)
    ) STORED,

    epistemic_truth FLOAT NOT NULL DEFAULT 1.0,
    anchor_dist FLOAT NOT NULL DEFAULT 0.0,
    valid_timestamp BIGINT NOT NULL,
    model_version INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_psm_dense_core_hnsw
ON psm_document_embeddings
USING hnsw (dense_core_slice vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_psm_epistemic_truth ON psm_document_embeddings (epistemic_truth);
CREATE INDEX IF NOT EXISTS idx_psm_anchor_dist ON psm_document_embeddings (anchor_dist);
CREATE INDEX IF NOT EXISTS idx_psm_model_version ON psm_document_embeddings (model_version);
CREATE INDEX IF NOT EXISTS idx_psm_document_id ON psm_document_embeddings (document_id);
```

Apply:

```bash
psql "$VECTORPRISM_PG_DSN" -f schema.sql
```

### Qdrant named-vector collection

Matches `QdrantVectorClient` in [`db_client.py`](db_client.py):

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

client = QdrantClient(url="http://localhost:6333")
collection = "psm_document_embeddings"

if not client.collection_exists(collection):
    client.create_collection(
        collection_name=collection,
        vectors_config={
            # Stage 1 — HNSW on dense core only
            "dense_core_slice": qmodels.VectorParams(
                size=368,
                distance=qmodels.Distance.COSINE,
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128),
            ),
            # Stage 2 — full 1024d tensor (flat / m=0, not a second ANN tax)
            "full_tensor": qmodels.VectorParams(
                size=1024,
                distance=qmodels.Distance.COSINE,
                hnsw_config=qmodels.HnswConfigDiff(m=0),
            ),
        },
    )
```

Payload fields used for Stage-1 filters: `epistemic_truth`, `anchor_dist`, `valid_timestamp`, `model_version`, `chunk_text`, `document_id`.

---

## Architecture Benchmarks & SLA

Targets enforced by the design and `benchmark_harness.py` / `live_benchmark.py` budgets:

| Stage | Operation | Budget |
|---|---|---|
| Header | Bitmask / header unpack `[0:16)` via `PSMTensorContract.unpack_header` | **&lt; 0.05 ms** |
| Stage 1 | HNSW coarse search on `dense_core_slice` (368d) + header filters → top 100 | **&lt; 10 ms** |
| Stage 2 | RAM zero-copy slice rescoring (z-score fuse × `w_intent`) | **&lt; 2 ms** |
| **E2E** | **Encode path excluded in pure Stage-2 harness; search SLA** | **&lt; 15 ms** |

```bash
# Stage-2 focused latency (synthetic corpus, real PSMRetrievalEngine.search)
python benchmark_harness.py

# End-to-end against a live backend (ingest + encode_query + search)
python vectorprism.py live-benchmark \
  --checkpoint checkpoints/vectorprism.pt \
  --documents data/documents.example.jsonl \
  --backend memory --n-trials 20 --p95-budget-ms 15
```

**Architecture path (unchanged pillars):**

```
Text ─► Frozen 768d Encoder ─► MultiTaskProjectionAdapter ─► 1024d tensor
                              │
                              ▼
              pgvector / Qdrant (dense HNSW + full tensor)
                              │
         IntentClassifier ─► w_intent + filters
                              │
         Stage 1: HNSW(dense_core_slice) + truth/anchor filters
                              │
         Stage 2: dense / rel / dis / hyp / causal scores → top-k
                  (Identity is Stage-1 gate only — not double-counted)
```

---

## Train Channels the Right Way

Channels are **earned**, not assumed. One channel at a time:

```bash
python train.py --channel dense --data your_pairs.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 --out checkpoints/vectorprism.pt

python vectorprism.py eval --checkpoint checkpoints/vectorprism.pt \
  --documents your_docs.jsonl --eval your_eval.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2

# Only after dense DoD: add causal / relational / hyperbolic / ...
python train.py --channel causal --data your_causal.jsonl \
  --init checkpoints/vectorprism.pt --out checkpoints/vectorprism.pt
```

See [`IMPLEMENTATION_SPEC.md`](IMPLEMENTATION_SPEC.md) for phased Definitions of Done and the gap matrix.

---

## Deploying VectorPrism at Scale?

Building a regulated RAG stack, a multi-tenant AI SaaS retrieval plane, or a private compliance-aware knowledge system?

Insight ITS works with enterprise architects on:

- Custom multi-task adapter fine-tuning for your ontology / incident / audit corpora  
- Private compliance connectors (bitemporal filters, calibrated epistemic truth, HITL review)  
- Managed control planes for versioned re-ingest across pgvector & Qdrant fleets  

**Talk to us**

- Email: `enterprise@insightits.com`  
- Book a technical deep-dive: `https://cal.com/insightits/vectorprism`  
- GitHub Discussions: [insightitsGit/VectorPrism](https://github.com/insightitsGit/VectorPrism/discussions)  
- Discord: [Join the VectorPrism community](https://discord.gg/vectorprism)

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) (or repository license metadata).

---

## Citation

If VectorPrism informs your research or production retrieval stack:

```bibtex
@software{vectorprism2026,
  title  = {VectorPrism: Positional Subspace Multiplexing for Intent-Gated Retrieval},
  author = Amin Parva,
  year   = {2026},
  url    = {https://github.com/insightitsGit/VectorPrism}
}
```

---

**VectorPrism** — six signals, one tensor, baseline storage cost, intent-gated speed.
