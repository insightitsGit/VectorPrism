# VectorPrism corpus recovery audit

- Documents: **12**
- Eval queries: **18**
- Dense misses@10: **0**
- Encoder: `sentence-transformers/all-mpnet-base-v2`
- Checkpoint: `checkpoints/finance_hard_adversarial_multi.pt`
- Structure indexes provided: **False**

## Honesty

This audit measures dense vs multi on *this* corpus. Finance adversarial 13/13 recovery does not automatically transfer.

Pack: {"source": "trentdoney/synthetic-incident-search-benchmark", "license_note": "Public Hugging Face dataset; synthetic incidents (not real customer data).", "n_documents": 12, "n_eval": 18, "vertical": "SRE / incident-response retrieval", "why_this_pack": "Public-safe operational incident notes with labeled queries and hard negatives. Not constructed by VectorPrism authors to force dense failure."}

## Full-set metrics

| Config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| dense_only | 0.889 | 1.000 | 1.000 | 0.944 |
| multi_channels | 0.889 | 1.000 | 1.000 | 0.931 |

## Recovery on dense misses

| Config | recovered@10 | rate |
|--------|--------------|------|
| dense_only | 0/0 | 0.0% |
| multi_channels | 0/0 | 0.0% |

## Silent dense failures (queries)

_No dense Miss@10 on this pack._

## Liability / risk framing (illustrative)

- Failure mode: Wrong runbook / root-cause incident cited → longer MTTR, repeat outage
- Silent dense fail rate: **0.0%** (0/18)
- Unit cost assumption: **$15,000** (per major incident with wrong root-cause retrieval)
- Industry SRE cost-of-downtime proxies vary widely; use as conversation starter only.
- Disclaimer: Not a promise of savings. Scenario math for pilot scoping.

## Verdict for this pack

**Dense already solves labeled retrieval (R@10 = 1.0).** Multi-channel does not add recovery value here because there are **zero** dense Miss@10 cases. That is an honest negative for the “dense fails on causal queries” slogan **outside** the finance adversarial pack.

## Next step

1. Run the same audit on a **partner** corpus where funny neighbors actually exist (or a harder public set with many near-miss distractors).
2. Sell **Audit only ($4,500)** when silent fails ≥1; upsell **Audit + fix ($12,000)** when structure/channels can move Miss@10.
3. Offer template: [`PILOT_ENGAGEMENT.md`](../../../PILOT_ENGAGEMENT.md).
