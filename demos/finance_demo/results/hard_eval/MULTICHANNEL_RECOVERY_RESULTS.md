# Multi-channel recovery scorecard

Stage-1: dense ∪ causal ancestors ∪ taxonomy lineage ∪ relational predicates.
Stage-2: z-score fusion or Reciprocal Rank Fusion (RRF).

- Checkpoint: `/app/checkpoints/finance_hard_adversarial_multi.pt`
- Dense misses@10: **13** / 14
- Enabled channels: `{"dense": true, "relational": true, "disentangled": false, "hyperbolic": true, "identity": true, "causal": true}`

### Full-set recall

| Config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| dense_only | 0.000 | 0.071 | 0.071 | 0.024 |
| causal+graph | 0.929 | 0.929 | 1.000 | 0.938 |
| +hyp+rel zscore | 1.000 | 1.000 | 1.000 | 1.000 |
| +hyp+rel RRF | 0.429 | 0.786 | 0.786 | 0.538 |
| balanced RRF | 0.429 | 0.714 | 0.786 | 0.554 |
| intent_router RRF | 0.214 | 0.643 | 0.786 | 0.374 |

### Recovery on dense misses@10

| Config | recovered@1 | recovered@5 | recovered@10 |
|--------|-------------|-------------|--------------|
| dense_only | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| causal+graph | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) |
| +hyp+rel zscore | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) |
| +hyp+rel RRF | 6 (46.2%) | 10 (76.9%) | 10 (76.9%) |
| balanced RRF | 6 (46.2%) | 9 (69.2%) | 10 (76.9%) |
| intent_router RRF | 3 (23.1%) | 9 (69.2%) | 11 (84.6%) |

### Per-miss detail (`causal+graph`)

- `ADV_C01_gold` dense_rank=138 → **RECOVERED** top5=['ADV_C01_gold', 'ADV_C01_d01', 'ADV_C01_d12', 'ADV_C01_d06', 'ADV_C01_d08']
- `ADV_C02_gold` dense_rank=14 → **RECOVERED** top5=['ADV_C02_gold', 'ADV_C02_d01', 'ADV_C02_d09', 'ADV_C02_d03', 'ADV_C02_d02']
- `ADV_C03_gold` dense_rank=12 → **RECOVERED** top5=['ADV_C03_gold', 'ADV_R01_gold', 'ADV_C03_d01', 'ADV_C03_d07', 'ADV_C03_d04']
- `ADV_H01_gold` dense_rank=19 → **RECOVERED** top5=['ADV_H01_gold', 'ADV_H01_d01', 'ADV_H01_d11', 'ADV_H01_d15', 'ADV_H01_d03']
- `ADV_R01_gold` dense_rank=13 → **RECOVERED** top5=['ADV_R01_gold', 'ADV_C04_gold', 'ADV_R01_d14', 'ADV_R01_d03', 'ADV_R01_d06']
- `ADV_R02_gold` dense_rank=42 → **RECOVERED** top5=['ADV_R02_gold', 'ADV_R02_d02', 'ADV_R02_d07', 'ADV_R02_d05', 'ADV_R02_d06']
- `ADV_E01_gold` dense_rank=22 → **RECOVERED** top5=['ADV_E01_gold', 'ADV_E01_d02', 'ADV_E01_d15', 'ADV_E01_d13', 'ADV_E01_d09']
- `ADV_E02_gold` dense_rank=73 → **RECOVERED** top5=['ADV_E02_gold', 'ADV_E02_d09', 'ADV_E02_d04', 'ADV_E02_d03', 'ADV_E02_d10']
- `ADV_C04_gold` dense_rank=12 → **RECOVERED** top5=['ADV_C04_gold', 'ADV_C04_d01', 'ADV_C04_d13', 'ADV_C04_d07', 'ADV_C04_d02']
- `ADV_C05_gold` dense_rank=12 → **RECOVERED** top5=['ADV_C05_gold', 'ADV_C05_d05', 'ADV_C05_d14', 'ADV_C05_d13', 'ADV_C05_d15']
- `ADV_C06_gold` dense_rank=66 → **RECOVERED** top5=['ADV_C06_gold', 'ADV_C06_d04', 'ADV_C06_d01', 'ADV_C06_d07', 'ADV_C06_d08']
- `ADV_C07_gold` dense_rank=200 → **RECOVERED** top5=['ADV_C07_gold', 'ADV_C07_d03', 'ADV_C07_d01', 'ADV_C07_d08', 'ADV_C07_d14']
- `ADV_R03_gold` dense_rank=37 → **RECOVERED** top5=['ADV_R03_gold', 'ADV_R03_d13', 'ADV_R03_d08', 'ADV_R03_d01', 'ADV_R03_d11']
