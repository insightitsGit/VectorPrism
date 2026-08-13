# Run VectorPrism in Docker (recommended on Windows)

Windows can crash on `import sentence_transformers` (native DLL / pyarrow conflicts).
Linux containers avoid that.

## Prerequisites

- Docker Desktop running
- From repo root: `c:\code\VectorPrism`

## Build

```bash
docker compose build
```

## Finance demo (real encoder, in-memory)

```bash
docker compose run --rm vectorprism
```

This trains dense on the finance corpus with `sentence-transformers/all-mpnet-base-v2`,
writes `checkpoints/finance_demo.pt`, prints Phase-1 eval + sample queries.

First run downloads the model into a Docker volume (cached for next runs).

## Finance demo on Postgres + pgvector (production path)

```bash
docker compose up -d db
docker compose run --rm finance-pg
```

- Starts `pgvector/pgvector:pg16` on **localhost:5433**
- DSN: `postgresql://vectorprism:vectorprism@localhost:5433/vectorprism`
- Applies `schema.sql`, ingests finance docs, runs live Stage-1/2 search
- Writes `demos/finance_demo/results/pgvector_live_search.json`

Point at your own local Postgres instead:

```bash
docker compose run --rm -e VECTORPRISM_PG_DSN="postgresql://USER:PASS@host.docker.internal:5432/DB" finance-pg \
  python demos/finance_demo/run_pgvector_demo.py --dsn "postgresql://USER:PASS@host.docker.internal:5432/DB"
```

## Tests

```bash
docker compose run --rm test
```

## One-off commands

```bash
# Verify encoder imports inside the container
docker compose run --rm vectorprism python -c "import sentence_transformers; print(sentence_transformers.__version__)"

# Train only
docker compose run --rm vectorprism python train.py --channel dense \
  --data demos/finance_demo/dense_pairs.jsonl \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --out checkpoints/finance_demo.pt --epochs 3

# Shell
docker compose run --rm vectorprism bash
```

## Notes

- Checkpoints are bind-mounted to `./checkpoints` on the host.
- HuggingFace / sentence-transformers caches persist in named volumes.
- For GPU later: add a CUDA base image + `deploy.resources` — CPU is enough for the finance demo.
