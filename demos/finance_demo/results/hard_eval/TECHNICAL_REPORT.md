# VectorPrism Technical Report — Adversarial Multi-Channel Milestone

*Generated 2026-08-13 21:16 UTC by `demos/finance_demo/run_stakeholder_demo.py`*

## 1. Executive summary

VectorPrism’s multi-channel thesis is validated on a calibrated adversarial finance pack:

- Dense-only retrieval collapses to **R@10 = 7.1%** (13/14 Miss@10).
- Structured Stage-1 expansion + Stage-2 fusion recovers **10–13 / 13** dense misses (**77–100%**), depending on fusion (RRF vs z-score).
- The critical engineering fix was **per-channel hop maps**: topological bonuses apply even when gold is already inside the dense Stage-1 pool.

## 2. Architecture (what was measured)

```
Query
  ├─ Stage-1: Dense Top-K  ∪  Causal ancestors  ∪  Taxonomy lineage  ∪  Relational predicates
  └─ Stage-2: z-score weighted fusion  |  Reciprocal Rank Fusion (k=20 default)
```

| Channel | Role on this pack |
|---------|-------------------|
| Dense | Semantic neighborhood (fails under cohesive distractors) |
| Causal | Cause→symptom DAG; hop-scaled Stage-2 bonus |
| Hyperbolic | Policy tree parent/child/sibling expansion |
| Relational | Exact attribute predicates (amount, callback, timezone, …) |

## 3. Primary recovery scorecard

- Checkpoint: `/app/checkpoints/finance_hard_adversarial_multi.pt`
- Dense misses@10: **13** / 14

| Config | Recovered@10 | Full R@10 | MRR |
|--------|-------------:|----------:|----:|
| dense_only | 0/13 (0.0%) | 0.071 | 0.024 |
| causal+graph | 13/13 (100.0%) | 1.000 | 0.937 |
| +hyp+rel zscore | 13/13 (100.0%) | 1.000 | 1.000 |
| +hyp+rel RRF | 10/13 (76.9%) | 0.786 | 0.514 |
| balanced RRF | 10/13 (76.9%) | 0.786 | 0.520 |
| intent_router RRF | 11/13 (84.6%) | 0.786 | 0.411 |

## 4. Post-validation (noise · latency · OOD · RRF-k)

### 4.1 OOD transfer (no pack graphs)

| Pack | dense R@10 | multi R@10 | ΔP@10 |
|------|-----------:|-----------:|------:|
| easy_finance | 1.000 | 1.000 | +0.000 |
| hard_gemini | 0.955 | 0.920 | -0.004 |
| hard_gpt | 0.640 | 0.680 | +0.004 |
| hard_combined | 0.902 | 0.866 | -0.004 |

### 4.2 Graph density stress

| Noise edges | Recovered@10 | Full R@10 | Full P@10 |
|------------:|-------------:|----------:|----------:|
| 0 | 13 | 1.000 | 0.100 |
| 50 | 12 | 0.929 | 0.093 |
| 150 | 12 | 0.929 | 0.093 |
| 400 | 11 | 0.857 | 0.086 |

### 4.3 Latency envelope

- Clean graph: mean **54.57142857142857** cands / **1.1013026431620736** ms
- Noisy graph (+400 edges): mean **71.71428571428571** cands / **1.656065142794562** ms

### 4.4 RRF \(k\)

- Z-score recovered@10: **13**
- Best RRF: k=**20** → recovered@10=**10**

**Verdict:** OOD: multi fusion without graphs does not collapse precision on soft packs. Density: recovery 100% @0 noise → 85% @400 noise edges (P@10 0.100 → 0.086). Latency: pruning config recovered@10=11 at mean 72 candidates / 1.48 ms. RRF: best k=20 closes 3 recovered gap vs z-score (10 vs 13).

## 5. Robustness (precision · sparsity · scale)

### 5.1 Precision / MRR / false positives

| Config | R@10 | P@10 | MRR | Cross-cluster FP@10 |
|--------|-----:|-----:|----:|--------------------:|
| dense_only | 0.071 | 0.007 | 0.024 | 0.021 |
| causal+graph | 1.000 | 0.100 | 0.936 | 0.029 |
| multi zscore | 1.000 | 0.100 | 1.000 | 0.043 |
| multi RRF k20 | 0.786 | 0.079 | 0.505 | 0.079 |

### 5.2 Graph sparsity (edge dropout)

| Drop | Recovered@10 | MRR |
|-----:|-------------:|----:|
| 0% | 13/13 | 1.000 |
| 20% | 13/13 | 0.964 |
| 30% | 13/13 | 1.000 |
| 40% | 13/13 | 1.000 |

### 5.3 Scale (1000 docs)

| Config | Recovered@10 | mean cands | mean ms | Cross-cluster FP@10 |
|--------|-------------:|-----------:|--------:|--------------------:|
| dense_only | 0/13 | 100.0 | 1.07 | 0.164 |
| multi_full | 13/13 | 55.3 | 1.85 | 0.114 |
| multi_beam8_cap150 | 13/13 | 55.3 | 1.77 | 0.114 |

**Verdict:** Precision: multi MRR=1.000 vs dense 0.024; cross-cluster FP@10=0.043 (expansion does not flood unrelated clusters). Sparsity: recovery 100% @0% drop → 100% @40% drop. Scale: 1000 docs — multi recovered@10=13, mean 55 cands / 1.85 ms; pruned config 55 cands / 1.77 ms.

## 6. Automated graph ingestion (curated vs auto)

- Backend: `heuristic`

| Structure | P | R | F1 |
|-----------|--:|--:|---:|
| causal_edges | 0.106 | 0.580 | 0.179 |
| taxonomy_edges | 0.055 | 0.059 | 0.057 |
| relational_attrs | 0.288 | 0.239 | 0.261 |

| Graph source | Recovered@10 | R@10 | MRR |
|--------------|-------------:|-----:|----:|
| auto | 11/13 | 0.786 | 0.696 |
| curated | 13/13 | 1.000 | 1.000 |

## 7. Caveats for stakeholders

1. **Curated alignment:** High recovery assumes structure indexes reflect domain logic. Auto extraction is the path to production; expect lower edge F1 until LLM/human review loops land.
2. **RRF vs z-score:** Prefer **RRF (k=20)** as the conservative production default; z-score shines when channel score scales are stable.
3. **Open-world noise:** Cross-cluster edge injection still kept recovery ≥85% at +400 edges; use `beam_width` / `max_stage1_candidates` when graphs densify.

## 8. Reproducibility commands

```bash
# Multi-channel recovery scorecard
docker compose run --rm vectorprism python demos/finance_demo/run_multichannel_recovery.py --skip-train

# Post-validation (OOD / density / latency / RRF-k)
docker compose run --rm vectorprism python demos/finance_demo/run_post_validation.py

# Robustness (precision / sparsity / 1k scale)
docker compose run --rm vectorprism python demos/finance_demo/run_robustness_validation.py

# Automated structure extraction (+ recovery with auto graphs)
docker compose run --rm vectorprism python demos/finance_demo/extract_structure_auto.py --backend heuristic --score

# Rebuild this report from artifacts
python demos/finance_demo/run_stakeholder_demo.py
```

## 9. Artifact index

| File | Contents |
|------|----------|
| `results/hard_eval/multichannel_recovery.json` | Primary recovery scorecard |
| `results/hard_eval/post_validation.json` | OOD / density / latency / RRF |
| `results/hard_eval/robustness_validation.json` | MRR / sparsity / scale |
| `results/hard_eval/auto_extraction.json` | Auto vs curated graphs |
| `checkpoints/finance_hard_adversarial_multi.pt` | Trained multi-channel checkpoint |
| `hard_adversarial/` | Curated pack + graphs |
| `hard_adversarial_auto/` | Auto-extracted graphs |
