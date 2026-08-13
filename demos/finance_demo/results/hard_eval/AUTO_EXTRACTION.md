# Automated structure extraction vs curated graphs

- Backend: `heuristic`
- Output: `/app/demos/finance_demo/hard_adversarial_auto`

## Edge / attribute agreement vs curated

| Structure | Precision | Recall | F1 | TP | FP | FN |
|-----------|----------:|-------:|---:|---:|---:|---:|
| causal_edges | 0.106 | 0.580 | 0.179 | 65 | 550 | 47 |
| taxonomy_edges | 0.055 | 0.059 | 0.057 | 16 | 273 | 256 |
| relational_attrs | 0.288 | 0.239 | 0.261 | 88 | 218 | 280 |

## Recovery with auto graphs (multi z-score)

| Graph source | Causal edges | Tax edges | Rel attrs | Recovered@10 | R@10 | MRR |
|--------------|-------------:|----------:|----------:|-------------:|-----:|----:|
| auto | 615 | 289 | 182 | 11/13 | 0.786 | 0.696 |
| curated | 112 | 272 | 112 | 13/13 | 1.000 | 1.000 |

## Notes

- Extraction uses **document text only** (no eval gold id shortcuts).
- Heuristic backend is offline/deterministic; LLM backend needs `OPENAI_API_KEY` or `VECTORPRISM_LLM_API_KEY` (+ optional `VECTORPRISM_LLM_BASE_URL` / `VECTORPRISM_LLM_MODEL`).
- Low edge F1 with high recovery is possible when auto edges are *different but still useful*.
