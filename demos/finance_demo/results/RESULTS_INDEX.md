# VectorPrism — Saved Results Index

All adversarial / multi-channel evaluation artifacts live under this tree.
Checkpoints (`.pt`) are **local-only** (gitignored); retrain via the commands below.

**Last milestone:** multi-channel recovery ≥70% on hard_adversarial + post/robustness/auto-extraction validation.

---

## Start here

| Doc | Purpose |
|-----|---------|
| [`demos/finance_demo/TECHNICAL_REPORT.md`](../TECHNICAL_REPORT.md) | Stakeholder-facing consolidated report |
| [`hard_eval/TECHNICAL_REPORT.md`](hard_eval/TECHNICAL_REPORT.md) | Same report mirrored under results |
| [`hard_eval/ADVERSARIAL_FULL_TEST.md`](hard_eval/ADVERSARIAL_FULL_TEST.md) | Primary adversarial verdict |

Rebuild the report from JSON artifacts:

```bash
python demos/finance_demo/run_stakeholder_demo.py
```

---

## Primary scorecards (JSON + Markdown)

| Artifact | What it proves |
|----------|----------------|
| `hard_eval/multichannel_recovery.json` + `MULTICHANNEL_RECOVERY_RESULTS.md` | Dense 7.1% → multi **13/13** (z-score) / **10–11/13** (RRF) |
| `hard_eval/causal_recovery.json` + `CAUSAL_RECOVERY_RESULTS.md` | Causal+graph recovery (hop-map fix) |
| `hard_eval/dense_baseline_adversarial.json` + `HARD_EVAL_RESULTS_ADVERSARIAL.md` | Dense stress baseline |
| `hard_eval/adversarial_calibration.json` + `ADVERSARIAL_CALIBRATION.md` | Miss@10 calibration |

## Post-validation & robustness

| Artifact | What it proves |
|----------|----------------|
| `hard_eval/post_validation.json` + `POST_VALIDATION.md` | OOD, +400 noise edges, latency, RRF-k |
| `hard_eval/robustness_validation.json` + `ROBUSTNESS_VALIDATION.md` | MRR/FP, 20–40% edge drop, 1000-doc scale |
| `hard_eval/auto_extraction.json` + `AUTO_EXTRACTION.md` | Auto graphs **11/13** vs curated **13/13** |

## Earlier / supporting

| Artifact | Notes |
|----------|-------|
| `hard_eval/HARD_EVAL_RESULTS*.md` | Isolated / mixed hard packs |
| `hard_eval/STEP_RESULTS.md` | Step trail |
| `production_readiness.json` + `PRODUCTION_RESULTS.md` | Easy-pack smoke + pgvector (not adversarial) |
| `phase1_eval.json`, `pgvector_live_search.json` | Phase-1 / live search |

## Data packs & auto graphs

| Path | Contents |
|------|----------|
| `demos/finance_demo/hard_adversarial/` | Curated docs, eval, causal/hyp/rel graphs |
| `demos/finance_demo/hard_adversarial_auto/` | Heuristic auto-extracted graphs |
| `demos/finance_demo/hard_adversarial_scale/` | 1000-doc scale corpus (local; regenerate via robustness script) |

## Local checkpoints (not in git)

| File | Role |
|------|------|
| `checkpoints/finance_hard_adversarial.pt` | Dense |
| `checkpoints/finance_hard_adversarial_causal.pt` | +causal |
| `checkpoints/finance_hard_adversarial_multi.pt` | +hyp+rel (preferred for demos) |

```bash
# Recreate multi checkpoint + scorecard
docker compose run --rm vectorprism python demos/finance_demo/run_multichannel_recovery.py
```

---

## Headline numbers (saved)

| Metric | Value |
|--------|------:|
| Dense R@10 (adversarial) | **0.071** |
| Dense Miss@10 | **13/14** |
| Multi z-score recovered@10 | **13/13 (100%)** |
| Multi RRF recovered@10 | **10–11/13 (77–85%)** |
| Auto-graph recovered@10 | **11/13 (85%)** |
| +400 noise-edge recovery | **11/13 (85%)** |
| 1000-doc multi recovered@10 | **13/13** |
| Clean-graph latency (in-mem) | **~1.1 ms / ~55 cands** |
