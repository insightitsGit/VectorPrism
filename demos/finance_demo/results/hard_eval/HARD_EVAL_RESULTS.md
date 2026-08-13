# Hard-eval dense baseline results

Encoder: `sentence-transformers/all-mpnet-base-v2`
Checkpoint: `checkpoints/finance_hard.pt`
Train pairs: `/app/demos/finance_demo/hard_combined/dense_pairs.jsonl` (798 rows)
Indexed corpus: `/app/demos/finance_demo/hard_combined/documents.jsonl` (288 docs)

## Run A — Gemini (primary)

```json
{
  "vectorprism_dense_only": {
    "recall@1": 0.5535714285714286,
    "recall@5": 0.875,
    "recall@10": 0.9285714285714286,
    "MRR": 0.6803287981859409,
    "n_eval_examples": 112
  },
  "dense_cosine_baseline": {
    "recall@1": 0.5535714285714286,
    "recall@5": 0.875,
    "recall@10": 0.9285714285714286,
    "MRR": 0.6803287981859409,
    "n_eval_examples": 112.0
  }
}
```

- Confirmed dense misses@10: **8** / 112 (7.1%)
- Labeled `dense_should_miss` confirmed: **8** / 112 (7.1%)

## Run B — GPT (secondary)

```json
{
  "vectorprism_dense_only": {
    "recall@1": 0.46,
    "recall@5": 0.68,
    "recall@10": 0.82,
    "MRR": 0.5558452380952381,
    "n_eval_examples": 100
  },
  "dense_cosine_baseline": {
    "recall@1": 0.46,
    "recall@5": 0.68,
    "recall@10": 0.82,
    "MRR": 0.5558452380952381,
    "n_eval_examples": 100.0
  }
}
```

- Confirmed dense misses@10: **18** / 100 (18.0%)
- Labeled `dense_should_miss` confirmed: **16** / 90 (17.8%)

## Interpretation

- High confirmation rate ⇒ pack is hard enough for channel-recovery work.
- Low confirmation rate ⇒ dense already solves most traps; need harder queries.
- Next: train causal (then others) and measure recovery on confirmed misses only.
