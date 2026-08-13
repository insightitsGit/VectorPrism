# Causal recovery scorecard (Step 1)

Trained causal from dense checkpoint; scored recovery on confirmed dense misses.

## Pack: adversarial

- Causal pairs: 7
- Checkpoint: `/app/checkpoints/finance_hard_adversarial_causal.pt`
- Dense misses@10: **13** / 14
- Stage-1 top-100 coverage on those misses: **11** / 13 (84.6%)

### Full-set recall

| Config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| dense_only | 0.000 | 0.071 | 0.071 | 0.024 |
| dense+causal | 0.000 | 0.071 | 0.071 | 0.024 |
| causal_heavy | 0.000 | 0.071 | 0.071 | 0.018 |
| intent_router | 0.000 | 0.071 | 0.071 | 0.018 |

### Recovery on dense misses@10

| Config | recovered@1 | recovered@5 | recovered@10 |
|--------|-------------|-------------|--------------|
| dense_only | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| dense+causal | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| causal_heavy | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| intent_router | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

- Lift@10 vs dense_only on miss set: `{"dense+causal": 0, "causal_heavy": 0, "intent_router": 0}`

## How to read this

- **Recovery@10 > 0 on dense+causal** while dense_only stays ~0 ⇒ channel earns its keep.
- **Stage-1 coverage < 100%** ⇒ some golds never enter the rescoring pool; Stage-2 cannot fix those.
- Full-set R@10 moving up with dense+causal is secondary; the miss-set lift is the product claim.
