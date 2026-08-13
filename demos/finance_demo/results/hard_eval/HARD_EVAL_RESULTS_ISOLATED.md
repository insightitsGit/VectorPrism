# Hard-eval dense baseline — ISOLATED packs

Encoder: `sentence-transformers/all-mpnet-base-v2`
Each pack was trained, indexed, and evaluated **alone** (no cross-pack mixing).

## Gemini

- Train pairs: 158 | Docs indexed: 128 | Eval: 112
- Checkpoint: `/app/checkpoints/finance_hard_gemini.pt`
- R@1 / R@5 / R@10: **0.625** / **0.857** / **0.920**
- MRR: **0.726**
- Misses@1: **42** / 112 (37.5%)
- Misses@10: **9** / 112 (8.0%)
- Label confirm@10: **9** / 112 (8.0%)

## GPT

- Train pairs: 640 | Docs indexed: 160 | Eval: 100
- Checkpoint: `/app/checkpoints/finance_hard_gpt.pt`
- R@1 / R@5 / R@10: **0.360** / **0.610** / **0.820**
- MRR: **0.483**
- Misses@1: **64** / 100 (64.0%)
- Misses@10: **18** / 100 (18.0%)
- Label confirm@10: **15** / 90 (16.7%)

## vs previous MIXED run

```json
{
  "note": "Prior mixed-corpus scorecard (train+index on hard_combined)",
  "gemini_r10": 0.9285714285714286,
  "gemini_miss10": 8,
  "gpt_r10": 0.82,
  "gpt_miss10": 18
}
```

## Interpretation

- Isolated = fair pack hardness (no foreign distractors / no cross-pack pair leakage).
- Higher miss@10 than mixed ⇒ mixing was making dense look stronger (or weaker) artificially.
