# Robustness validation: precision · sparsity · scale

Checkpoint: `/app/checkpoints/finance_hard_adversarial_multi.pt`

## 1. Precision / MRR / false-positive profile

Single-relevant eval ⇒ theoretical max P@10 ≈ 0.10 when gold is retrieved. **Cross-cluster FP@10** is the key noise signal (irrelevant clusters in top-10).

| Config | R@10 | P@10 | MRR | FP@10 | Cross-cluster FP@10 | Recovered@10 |
|--------|-----:|-----:|----:|------:|--------------------:|-------------:|
| dense_only | 0.071 | 0.007 | 0.024 | 0.993 | 0.021 | 0/13 |
| causal+graph | 1.000 | 0.100 | 0.936 | 0.900 | 0.029 | 13/13 |
| multi zscore | 1.000 | 0.100 | 1.000 | 0.900 | 0.043 | 13/13 |
| multi RRF k20 | 0.786 | 0.079 | 0.505 | 0.921 | 0.079 | 10/13 |

## 2. Graph sparsity stress (random edge dropout)

Causal + hyperbolic edges dropped independently; relational attrs kept (attribute index ≠ edge graph).

| Drop | Causal kept | Tax kept | Recovered@10 | R@10 | P@10 | MRR | Cross-cluster FP@10 |
|-----:|------------:|---------:|-------------:|-----:|-----:|----:|--------------------:|
| 0% | 112/112 | 272/272 | 13/13 | 1.000 | 0.100 | 1.000 | 0.043 |
| 20% | 90/112 | 218/272 | 13/13 | 1.000 | 0.100 | 0.964 | 0.043 |
| 30% | 78/112 | 190/272 | 13/13 | 1.000 | 0.100 | 1.000 | 0.036 |
| 40% | 67/112 | 163/272 | 13/13 | 1.000 | 0.100 | 1.000 | 0.050 |

## 3. Corpus scale (≥1000 docs)

- Documents: **1000** (ADV=306, merged OOD=641, fillers=53)
- Eval queries: 14 (same adversarial gold labels)

| Config | R@10 | Recovered@10 | MRR | P@10 | Cross-cluster FP@10 | mean |cands| | mean ms |
|--------|-----:|-------------:|----:|-----:|--------------------:|-------------:|--------:|
| dense_only | 0.071 | 0/13 | 0.014 | 0.007 | 0.164 | 100.0 | 1.07 |
| multi_full | 1.000 | 13/13 | 0.964 | 0.100 | 0.114 | 55.3 | 1.85 |
| multi_beam8_cap150 | 1.000 | 13/13 | 0.964 | 0.100 | 0.114 | 55.3 | 1.77 |

## Verdict

Precision: multi MRR=1.000 vs dense 0.024; cross-cluster FP@10=0.043 (expansion does not flood unrelated clusters). Sparsity: recovery 100% @0% drop → 100% @40% drop. Scale: 1000 docs — multi recovered@10=13, mean 55 cands / 1.85 ms; pruned config 55 cands / 1.77 ms.
