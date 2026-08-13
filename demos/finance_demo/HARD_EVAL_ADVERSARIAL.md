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

## Causal graph expansion (zero-recovery fix)

```bash
python demos/finance_demo/build_adversarial_causal_graph.py   # → 140 pairs, 112 edges
docker compose run --rm vectorprism python demos/finance_demo/run_causal_recovery.py --pack adversarial
```

Stage-1 unions dense top-50 with **upstream causal neighbors** of the top-10 seeds; Stage-2 adds a hop-scaled structural bonus into the causal score.

Latest measured lift: dense R@10 **0.071** → dense+causal+graph R@10 **0.357**; recovered **4/13** dense misses (was 0/13).
