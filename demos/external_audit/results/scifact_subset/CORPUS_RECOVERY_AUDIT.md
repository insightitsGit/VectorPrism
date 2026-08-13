# VectorPrism corpus recovery audit

- Documents: **500**
- Eval queries: **40**
- Dense misses@10: **0**
- Encoder: `sentence-transformers/all-mpnet-base-v2`
- Checkpoint: `checkpoints/finance_hard_adversarial_multi.pt`
- Structure indexes provided: **False**

## Honesty

This audit measures dense vs multi on *this* corpus. Finance adversarial 13/13 recovery does not automatically transfer.

Pack: {"source": "BeIR/scifact (subset)", "n_documents": 500, "n_eval": 40, "vertical": "scientific claim verification / compliance-adjacent", "why_this_pack": "Public BeIR SciFact subset; not VectorPrism-authored adversarial finance data."}

## Full-set metrics

| Config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| dense_only | 0.800 | 0.825 | 1.000 | 0.832 |
| multi_channels | 0.725 | 0.875 | 0.950 | 0.801 |

## Recovery on dense misses

| Config | recovered@10 | rate |
|--------|--------------|------|
| dense_only | 0/0 | 0.0% |
| multi_channels | 0/0 | 0.0% |

## Silent dense failures (queries)

_No dense Miss@10 on this pack._

## Liability / risk framing (illustrative)

- Failure mode: Silent wrong neighbor in top-k → hallucinated answer grounded on wrong doc
- Silent dense fail rate: **0.0%** (0/40)
- Unit cost assumption: **$5,000** (per high-severity wrong-answer incident)
- Placeholder unit cost until partner provides loss model.
- Disclaimer: Not a promise of savings. Scenario math for pilot scoping.

## Next step

Treat this as a **paid pilot audit** input: identify silent fails, add structure indexes (causal/rel/hyp), retrain channels on *their* labels, re-measure recovery.
