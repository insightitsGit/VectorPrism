# VectorPrism Benchmarks

**Live interactive demo:** [GitHub Pages](https://insightitsgit.github.io/VectorPrism/) · source [`docs/index.html`](docs/index.html)

Committed scorecards: [`demos/finance_demo/results/`](demos/finance_demo/results/) · index [`RESULTS_INDEX.md`](demos/finance_demo/results/RESULTS_INDEX.md)

---

## Headline result (adversarial finance pack)

**Scope:** calibrated `hard_adversarial` finance pack only. Dense is intentionally stressed here.  
**Transfer check:** public incident pack audit → [`demos/external_audit/results/incident_bench/CORPUS_RECOVERY_AUDIT.md`](demos/external_audit/results/incident_bench/CORPUS_RECOVERY_AUDIT.md) (dense already R@10=1.0 — no silent fails to recover).

| Metric | Value |
|--------|------:|
| Dense R@10 | **7.1%** |
| Dense Miss@10 | **13 / 14 (93%)** |
| Multi-channel recovered@10 (z-score) | **13 / 13 (100%)** |
| Multi-channel recovered@10 (RRF) | **10–11 / 13 (77–85%)** |
| Auto-extracted graphs recovered@10 | **11 / 13 (85%)** |
| Full R@10 multi z-score | **1.000** |
| Full R@10 RRF | **0.786** |

```
Dense R@10     ████░░░░░░░░░░░░░░░░  7%
RRF R@10       ████████████████░░░░ 79%
Z-score R@10   ████████████████████ 100%
```

---

## Multi-channel recovery scorecard

Source: `demos/finance_demo/results/hard_eval/multichannel_recovery.json`

| Config | Recovered@10 | Full R@10 | MRR |
|--------|-------------:|----------:|----:|
| dense_only | 0/13 | 0.071 | 0.024 |
| causal+graph | **13/13** | **1.000** | 0.937 |
| +hyp+rel zscore | **13/13** | **1.000** | **1.000** |
| +hyp+rel RRF | 10/13 | 0.786 | 0.514 |
| balanced RRF | 10/13 | 0.786 | 0.520 |
| intent_router RRF | 11/13 | 0.786 | 0.411 |

Checkpoint: `checkpoints/finance_hard_adversarial_multi.pt` (local; retrain via scripts).

---

## Post-validation

Source: `post_validation.json` / `POST_VALIDATION.md`

| Dimension | Outcome |
|-----------|---------|
| OOD transfer (no pack graphs) | P@10 within **±0.4pp** on soft packs |
| +400 cross-cluster noise edges | Recovery **100% → 85%** (still ≥70%) |
| Latency (clean graph) | ~**55** cands / **~1.1 ms** in-memory |
| RRF best \(k\) | **k=20** (10/13); z-score still leads |

---

## Robustness

Source: `robustness_validation.json` / `ROBUSTNESS_VALIDATION.md`

| Check | Outcome |
|-------|---------|
| MRR (multi z-score vs dense) | **1.000** vs **0.024** |
| Cross-cluster FP@10 | **0.043** (multi) |
| Edge dropout 20–40% | Still **13/13** recovery |
| 1000-doc scale | **13/13**, ~55 cands, ~**1.8 ms** |

---

## Automated graphs vs curated

Source: `auto_extraction.json`

| Graph source | Recovered@10 | R@10 | MRR |
|--------------|-------------:|-----:|----:|
| heuristic auto | **11/13** | 0.786 | 0.696 |
| curated | **13/13** | 1.000 | 1.000 |

---

## How to reproduce

**Checkpoints (`*.pt`) are gitignored** — a fresh clone does **not** include
`checkpoints/finance_hard_adversarial_multi.pt`. Do not rely on `--skip-train`
unless you already trained locally.

```bash
git clone https://github.com/insightitsGit/VectorPrism.git
cd VectorPrism
pip install -e ".[all]"   # or use Docker below

# Recommended: one script trains missing checkpoints then scores
docker compose run --rm vectorprism \
  python scripts/reproduce_adversarial_benchmarks.py --epochs 3

# Optional extras (needs multi checkpoint from the step above)
docker compose run --rm vectorprism \
  python scripts/reproduce_adversarial_benchmarks.py --skip-train --with-post --with-robustness
```

Fast path **only if** you already have local checkpoints:

```bash
docker compose run --rm vectorprism \
  python demos/finance_demo/run_multichannel_recovery.py --skip-train
```

Rebuild the stakeholder report (uses existing JSON artifacts):

```bash
python demos/finance_demo/run_stakeholder_demo.py
```

External (non-finance) transfer audits:

```bash
docker compose run --rm --no-deps vectorprism \
  python scripts/corpus_recovery_audit.py \
  --documents demos/external_audit/packs/scifact_subset/documents.jsonl \
  --eval demos/external_audit/packs/scifact_subset/eval.jsonl \
  --checkpoint checkpoints/finance_hard_adversarial_multi.pt \
  --vertical generic \
  --pack-meta demos/external_audit/packs/scifact_subset/meta.json \
  --out demos/external_audit/results/scifact_subset
```

---

## Caveats (read before quoting)

1. Curated structure indexes amplify recovery — auto graphs still hit **85%**, not 100%.
2. Prefer **RRF (~78–85%)** as the conservative public number; z-score 100% is calibrated-fusion.
3. Easy finance demo saturates dense R@10 ≈ 1.0 — use the **adversarial** pack for channel claims.
