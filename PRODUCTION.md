# VectorPrism — Production Readiness

## What “production-ready” means here

| Layer | Status |
|---|---|
| Library code (Phases 0–6) | Ready |
| Docker runtime (Linux + sentence-transformers) | Ready |
| Finance demo corpus + eval | Ready (synthetic / pre-client) |
| Postgres + pgvector ingest/search | Ready (Compose `db` on `:5433`) |
| Real client labeled data | **Not yet** — swap JSONL when a client arrives |
| Extra channels (causal/relational/…) in prod | **Earn via ablation** after dense DoD |
| PyPI package `pip install vectorprism` | **Live** — [vectorprism 0.1.0](https://pypi.org/project/vectorprism/0.1.0/); wheel includes `schema.sql` + example JSONL |
| Real external pilot | See [`PILOT.md`](PILOT.md) — recruit a partner; not another internal benchmark |

## Partner install

```bash
pip install "vectorprism[all]"
vectorprism pilot-check

# Or from git (latest main / full adversarial packs):
# git clone https://github.com/insightitsGit/VectorPrism.git && cd VectorPrism
# pip install -e ".[all]"
```

## One-command production smoke

```bash
# From repo root, Docker Desktop running
docker compose up -d db
docker compose run --rm test
docker compose run --rm finance-pg
docker compose run --rm vectorprism python demos/finance_demo/run_production_smoke.py
```

Or:

```bash
docker compose run --rm vectorprism python demos/finance_demo/run_production_smoke.py --full
```

Artifacts land in:

- `demos/finance_demo/results/phase1_eval.json`
- `demos/finance_demo/results/pgvector_live_search.json`
- `demos/finance_demo/results/production_readiness.json`
- `demos/finance_demo/results/PRODUCTION_RESULTS.md`

## Deploy checklist

1. [ ] `docker compose up -d db` healthy  
2. [ ] Checkpoint trained (`checkpoints/finance_demo.pt` or your own)  
3. [ ] `schema.sql` applied (automatic via `ensure_schema` / init mount)  
4. [ ] Documents upserted via `ingest_cli` / `run_pgvector_demo.py`  
5. [ ] Sample search returns correct `document_id`s  
6. [ ] Phase-1 eval beats or ties dense baseline on **your** eval set  
7. [ ] `model_version` bumped + reingest after any adapter retrain  
8. [ ] Epistemic hard filter **off** until ECE ≤ ~0.05  
9. [ ] Secrets only in `.env` (from `.env.example`)  

## Point at your own Postgres

```bash
export VECTORPRISM_PG_DSN="postgresql://USER:PASS@host.docker.internal:5432/YOURDB"
docker compose run --rm -e VECTORPRISM_PG_DSN finance-pg \
  python demos/finance_demo/run_pgvector_demo.py --dsn "$VECTORPRISM_PG_DSN" --skip-train
```

## Client handoff

Replace:

- `demos/finance_demo/documents.jsonl`
- `demos/finance_demo/dense_pairs.jsonl`
- `demos/finance_demo/eval.jsonl`

Retrain dense, re-ingest, re-eval. Same code path.
