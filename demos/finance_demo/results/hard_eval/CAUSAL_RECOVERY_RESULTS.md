# Causal recovery scorecard (Step 1)

Trained causal from dense checkpoint; scored recovery on confirmed dense misses.

## Pack: adversarial

- Causal pairs: 140
- Checkpoint: `/app/checkpoints/finance_hard_adversarial_causal.pt`
- Dense misses@10: **13** / 14
- Stage-1 top-100 coverage on those misses: **11** / 13 (84.6%)

### Full-set recall

| Config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| dense_only | 0.000 | 0.071 | 0.071 | 0.024 |
| dense+causal | 0.000 | 0.071 | 0.071 | 0.024 |
| dense+causal+graph | 0.571 | 0.929 | 1.000 | 0.689 |
| causal_heavy+graph | 0.929 | 0.929 | 1.000 | 0.936 |
| intent_router+graph | 0.714 | 0.714 | 0.714 | 0.714 |

### Recovery on dense misses@10

| Config | recovered@1 | recovered@5 | recovered@10 |
|--------|-------------|-------------|--------------|
| dense_only | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| dense+causal | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| dense+causal+graph | 8 (61.5%) | 12 (92.3%) | 13 (100.0%) |
| causal_heavy+graph | 13 (100.0%) | 13 (100.0%) | 13 (100.0%) |
| intent_router+graph | 10 (76.9%) | 10 (76.9%) | 10 (76.9%) |

- Lift@10 vs dense_only on miss set: `{"dense+causal": 0, "dense+causal+graph": 13, "causal_heavy+graph": 13, "intent_router+graph": 10}`

## How to read this

- **Recovery@10 > 0 on dense+causal** while dense_only stays ~0 ⇒ channel earns its keep.
- **Stage-1 coverage < 100%** ⇒ some golds never enter the rescoring pool; Stage-2 cannot fix those.
- Full-set R@10 moving up with dense+causal is secondary; the miss-set lift is the product claim.
