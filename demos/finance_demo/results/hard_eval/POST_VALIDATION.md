# Post-validation: OOD · density · latency · RRF-k

Checkpoint: `/app/checkpoints/finance_hard_adversarial_multi.pt`

## 1. OOD transfer (no pack graphs)

Adversarial-trained multi checkpoint scored on other finance corpora with channel fusion only (no Stage-1 graph).

| Pack | dense R@10 | multi R@10 | ΔR@10 | dense P@10 | multi P@10 | ΔP@10 |
|------|-----------:|-----------:|------:|-----------:|-----------:|------:|
| easy_finance | 1.000 | 1.000 | +0.000 | 0.100 | 0.100 | +0.000 |
| hard_gemini | 0.955 | 0.929 | -0.027 | 0.096 | 0.093 | -0.003 |
| hard_gpt | 0.640 | 0.670 | +0.030 | 0.064 | 0.067 | +0.003 |
| hard_combined | 0.902 | 0.866 | -0.036 | 0.090 | 0.087 | -0.004 |

## 2. Graph-density stress (cross-cluster noise edges)

Base causal edges: **112**. Noise edges are random cross-cluster cause→effect links.

| Noise edges | Total edges | Recovered@10 | Recovery | Full R@10 | Full P@10 |
|------------:|------------:|-------------:|---------:|----------:|----------:|
| 0 | 112 | 13 | 100.0% | 1.000 | 0.100 |
| 50 | 162 | 12 | 92.3% | 0.929 | 0.093 |
| 150 | 262 | 12 | 92.3% | 0.929 | 0.093 |
| 400 | 512 | 11 | 84.6% | 0.857 | 0.086 |

## 3. Latency & candidate pruning

Stress graph edges: **512** (base + 400 noise).

| Config | mean |cands| | max |cands| | mean Stage-1 ms | mean total ms | Recovered@10 |
|--------|-------------:|------------:|----------------:|--------------:|-------------:|
| baseline_no_cap | 71.7 | 89 | 1.44 | 2.97 | 12 (92%) |
| beam8 | 71.6 | 89 | 1.13 | 2.34 | 12 (92%) |
| beam4 | 69.8 | 87 | 1.65 | 3.32 | 12 (92%) |
| cap200 | 71.7 | 89 | 1.86 | 3.33 | 12 (92%) |
| cap100 | 71.7 | 89 | 1.02 | 2.33 | 12 (92%) |
| beam8_cap150 | 71.6 | 89 | 1.48 | 3.67 | 12 (92%) |
| clean_graph_baseline | 54.6 | 66 | 0.97 | 2.29 | 13 (100%) |

## 4. RRF \(k\) sweep vs z-score

Z-score recovered@10: **13** (100.0%)

| Config | Recovered@10 | Full R@10 | MRR |
|--------|-------------:|----------:|----:|
| zscore | 13 | 1.000 | — |
| rrf k=20 | 10 | 0.786 | 0.490 |
| rrf k=30 | 10 | 0.786 | 0.493 |
| rrf k=40 | 10 | 0.786 | 0.493 |
| rrf k=50 | 10 | 0.786 | 0.505 |
| rrf k=60 | 10 | 0.786 | 0.502 |

- Best RRF: `k=50` recovered@10=10
- Gap to z-score (recovered@10): **3**

## Verdict

OOD: multi fusion without graphs does not collapse precision on soft packs. Density: recovery 100% @0 noise → 85% @400 noise edges (P@10 0.100 → 0.086). Latency: pruning config recovered@10=12 at mean 72 candidates / 3.67 ms. RRF: best k=50 closes 3 recovered gap vs z-score (10 vs 13).
