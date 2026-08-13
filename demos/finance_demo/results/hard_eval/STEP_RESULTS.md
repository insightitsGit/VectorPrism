# Hard-eval fix steps — results

## Step 1 — Causal train + recovery scorecard

| Pack | Dense misses@10 | Stage-1 cov | dense+causal recovered@10 | causal_heavy@10 | Full R@10 dense → +causal |
|------|-----------------|-------------|---------------------------|-----------------|---------------------------|
| Gemini | 9/112 | 100% | **1** (11%) | 1 | 0.920 → 0.911 |
| GPT | 18/100 | 100% | **2** (11%) | **5** (28%) | 0.820 → 0.830 |

**Verdict:** Design is *partially* earned — small real lift on GPT miss set with causal-heavy weights; Gemini barely moves. Not enough for a strong product claim yet (tiny N + weak causal data alignment).

Artifacts: `CAUSAL_RECOVERY_RESULTS.md`, `causal_recovery.json`  
Checkpoints: `checkpoints/finance_hard_*_causal.pt`

## Step 2 — Harden labels

| Pack | `dense_should_miss` before → after |
|------|-------------------------------------|
| Gemini | 112 → **9** |
| GPT | 90 → **19** |

Confirmed miss lists: `confirmed_misses_gemini.jsonl`, `confirmed_misses_gpt.jsonl`  
`hard_combined` eval copies refreshed.

## Step 3 — Code honesty / CLI

- README relational channel: train TransE vs serve L2 proximity (documented).
- CLI: `vectorprism hard-eval`, `vectorprism causal-recovery`.
- Validator skips 50% `dense_should_miss` gate when `miss_confirmed` annotations exist.

## What’s still not proven

- Broad “channels beat dense” marketing
- Relational/hyperbolic recovery (no hard-pack JSONL + relational serve gap)
- Large miss-set statistics (need harder adversarial queries)
