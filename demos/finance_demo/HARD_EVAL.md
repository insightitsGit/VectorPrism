# Hard-eval packs (finance demo)

Goal: prove specialized channels recover evidence a strong dense retriever misses.

## Packs on disk

| Pack | Path | Docs | Dense pairs | Eval | Causal | Notes |
|------|------|------|-------------|------|--------|-------|
| Gemini (primary) | `hard_gemini/` | 128 | 158* | 112 | 48 | Rich banking narrative; labels hardened to **9** confirmed dense misses |
| GPT (secondary) | `hard_gpt/` | 160 | 640 | 100 | 60 | Templated traps; labels hardened to **19** confirmed dense misses |
| Combined train | `hard_combined/` | 288 | 798 | Gemini default | 108 | Union for training; **two separate eval scorecards** |
| Legacy alias | `hard/` | = GPT snapshot | | | | Same as `hard_gpt` unless you overwrite it |

\*Gemini chat summary claimed ~416 dense pairs; the paste only contained **158** valid JSONL rows.

## Protocol (completed steps)

```bash
# Dense baseline (isolated packs)
docker compose run --rm vectorprism python demos/finance_demo/run_hard_eval.py --mode isolated
# or: docker compose run --rm vectorprism python vectorprism.py hard-eval --mode isolated

# Step 1 — Causal recovery scorecard
docker compose run --rm vectorprism python demos/finance_demo/run_causal_recovery.py --pack both
# or: docker compose run --rm vectorprism python vectorprism.py causal-recovery --pack both

# Step 2 — Harden dense_should_miss to measured misses
python scripts/harden_dense_miss_labels.py

# Validate
python scripts/validate_hard_eval.py --pack-dir demos/finance_demo/hard_gemini --min-dense-pairs 150
python scripts/validate_hard_eval.py --pack-dir demos/finance_demo/hard_gpt --min-dense-pairs 400
```

Results rollup: `results/hard_eval/STEP_RESULTS.md`

## Adversarial pack (recommended next corpus)

Diagnosis of why Gemini/GPT packs were easy + calibrated pack:

- Theory: `HARD_EVAL_ADVERSARIAL.md`
- Generator: `generate_adversarial_pack.py`
- Dense calibration loop: `calibrate_adversarial_pack.py`
- Output: `hard_adversarial/` (cluster size ≥18; confirmed Miss@10 after calibration)

```bash
python demos/finance_demo/generate_adversarial_pack.py
docker compose run --rm vectorprism python demos/finance_demo/calibrate_adversarial_pack.py
```

## Why not one giant eval?

- ID namespaces already differ (`DOC_*` vs `vp_*`) — good for merge.
- Mixing eval queries without labels muddies which failure mode you fixed.
- Gemini is better for “real ops” narrative; GPT is better for dense-pair count.
