# Adversarial pack — full retest results

Encoder: `sentence-transformers/all-mpnet-base-v2`  
Corpus: `hard_adversarial/` (306 docs, 14 calibrated eval queries, 7 causal pairs)

## Dense baseline (after fine-tune)

| Metric | Value |
|--------|------:|
| R@1 | **0.000** |
| R@5 | **0.071** |
| R@10 | **0.071** |
| MRR | **0.024** |
| Miss@10 | **13/14 (92.9%)** |
| Label confirm@10 | **92.9%** |

Compare prior packs: Gemini ~8% miss@10, GPT ~18% miss@10.

**Verdict:** Adversarial pack is a real dense stress test.

## Causal recovery

| Config | Recovered@10 on dense misses |
|--------|-------------------------------:|
| dense_only | 0/13 |
| dense+causal | 0/13 |
| causal_heavy | 0/13 |
| intent_router | 0/13 |

- Stage-1 top-100 coverage on misses: **11/13 (84.6%)** — 2 golds unreachable by Stage-2
- Causal train N=7 (too small / weakly aligned to doc golds)

**Verdict:** Pack hardness fixed; **channel recovery not yet earned**. Need more/better causal supervision linking symptom queries → root-cause docs (or wider Stage-1 for causal-heavy queries).

## Artifacts

- `dense_baseline_adversarial.json` / `HARD_EVAL_RESULTS_ADVERSARIAL.md`
- `causal_recovery.json` / `CAUSAL_RECOVERY_RESULTS.md`
- Checkpoints: `finance_hard_adversarial.pt`, `finance_hard_adversarial_causal.pt`
