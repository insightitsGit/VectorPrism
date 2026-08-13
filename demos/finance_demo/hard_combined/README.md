# Combined hard-eval training pack

Merged from `hard_gpt` + `hard_gemini` for denser Stage-1 training.

| File | Rows | Role |
|------|------|------|
| documents.jsonl | 288 | Union corpus (GPT 160 + Gemini 128) |
| dense_pairs.jsonl | 798 | Union dense supervision |
| causal.jsonl | 108 | Union causal pairs |
| eval.jsonl | 112 | **Primary scorecard = Gemini** |
| eval_gemini.jsonl | 112 | Gemini hard eval |
| eval_gpt.jsonl | 100 | GPT hard eval |

## Recommended protocol (2 runs, 1 train)

1. Train dense / channels on `hard_combined` (documents + dense_pairs + causal).
2. **Run A (primary):** evaluate on `eval_gemini.jsonl` — realistic banking traps.
3. **Run B (secondary):** evaluate on `eval_gpt.jsonl` — templated cause/symptom traps.
4. Report both scorecards; do not average into one vanity number.

Do not naive-merge eval sets into one ranking without labeling which pack a query came from.
