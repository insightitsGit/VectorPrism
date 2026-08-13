# Why the first hard packs were “easy” — and the adversarial fix

## Failure modes (diagnosed)

1. **Cluster cardinality vs Top-K** — topics with 4–8 docs and K=10 made R@10 nearly automatic once dense found the neighborhood.
2. **Lexical / entity anchors** — rare tokens (`EOD-904`, `camt.056`, `871(m)`) pulled gold into ranks 1–3.
3. **Symptom↔gold leakage** — query symptom wording was closer to gold than to distractors.

## VectorPrism channel win conditions

| Pattern | Dense failure | Channel |
|---------|---------------|---------|
| Multi-hop causal disconnect | Symptom neighborhood ≠ root cause doc | Causal \(q^{\top}Mc\) |
| Hyperbolic sibling / level collision | Parent/sibling clauses outrank deep child | Poincaré distance |
| Relational boundary collision | Same keywords, different predicate | Relational constraints |
| Epistemic / version trap | Drafts & legacy outrank active policy | Truth / metadata filters |

## Pack + calibration

```bash
python demos/finance_demo/generate_adversarial_pack.py
docker compose run --rm vectorprism python demos/finance_demo/calibrate_adversarial_pack.py
```

Calibration **only keeps** eval rows where gold dense rank > 10 after iterative distractor hardening.

Targets: Miss@10 ≥ 50% of generated queries surviving as confirmed misses; then re-run causal recovery on `hard_adversarial/`.
