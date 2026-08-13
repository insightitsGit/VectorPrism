# VectorPrism Production Results

Generated (UTC): `2026-08-13T18:54:31.866327+00:00`  
Git: `84715ad`  
Encoder: `sentence-transformers/all-mpnet-base-v2`  
Checkpoint: `checkpoints/finance_demo.pt`

## Gate summary

| Step | Pass |
|---|---|
| pytest | True |
| pgvector live demo | True |
| live benchmark | True |
| **Pre-client production ready** | **True** |

## Phase-1 eval (finance demo)

```json
{
  "vectorprism": {
    "recall@1": 1.0,
    "recall@5": 1.0,
    "recall@10": 1.0,
    "MRR": 1.0,
    "n_eval_examples": 62
  },
  "dense_cosine_baseline": {
    "recall@1": 1.0,
    "recall@5": 1.0,
    "recall@10": 1.0,
    "MRR": 1.0,
    "n_eval_examples": 62.0
  },
  "beats_or_ties_baseline": true
}
```

## pgvector live search

- Document count: `65`
- Model version: `1`
- Queries exercised: `5`

See `pgvector_live_search.json` for full hit lists.

## Notes

- Synthetic finance corpus — replace with client data before claiming client production quality.
- Hard epistemic truth filter remains opt-in until ECE-calibrated.

## Reproduce

```bash
docker compose up -d db
docker compose run --rm test
docker compose run --rm finance-pg
docker compose run --rm vectorprism python demos/finance_demo/run_production_smoke.py --full
```
