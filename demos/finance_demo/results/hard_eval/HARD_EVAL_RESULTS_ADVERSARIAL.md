# Hard-eval dense baseline — ISOLATED packs

Encoder: `sentence-transformers/all-mpnet-base-v2`
Each pack was trained, indexed, and evaluated **alone** (no cross-pack mixing).

## Adversarial

- Train pairs: 153 | Docs indexed: 306 | Eval: 14
- Checkpoint: `/app/checkpoints/finance_hard_adversarial.pt`
- R@1 / R@5 / R@10: **0.000** / **0.071** / **0.071**
- MRR: **0.024**
- Misses@1: **14** / 14 (100.0%)
- Misses@10: **13** / 14 (92.9%)
- Label confirm@10: **13** / 14 (92.9%)

## Interpretation

- Isolated = fair pack hardness (no foreign distractors / no cross-pack pair leakage).
- Higher miss@10 than mixed ⇒ mixing was making dense look stronger (or weaker) artificially.
