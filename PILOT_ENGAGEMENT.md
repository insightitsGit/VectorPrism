# VectorPrism paid pilot — audit + fix (not a subscription)

**Status:** Offer draft for first 2–3 design partners  
**Soft CTA:** email subject **RECOVER** → `insightits.info@gmail.com`

---

## What you buy

A **time-boxed engagement**, not SaaS seats:

1. **Silent-failure audit** on *your* document corpus + query log (or tickets)
2. **Measured** dense Miss@k vs VectorPrism multi-channel (same checkpoint protocol)
3. **Fix plan** — which structure indexes / channel labels to add, expected recovery delta
4. Optional **implementation sprint** to ship the fix on your staging retrieval path

Tooling shipped in-repo:

```bash
python scripts/import_external_pack.py ...   # or bring VectorPrism JSONL
python scripts/corpus_recovery_audit.py \
  --documents your/documents.jsonl \
  --eval your/eval.jsonl \
  --checkpoint checkpoints/....pt \
  --out reports/partner_audit/
```

---

## Pricing (founding)

| Tier | Price | Includes | Duration |
|------|------:|----------|----------|
| **Audit only** | **$4,500** | Ingest + scorecard + silent-fail list + liability framing + 90-min readout | ~1 week |
| **Audit + fix** | **$12,000** | Above + structure extraction / channel training on your labels + re-measure + staging integration guide | 2–3 weeks |
| **Boundary add-on** | +**$3,000** | Wire **PrismManifest** gate if money/tool args leave retrieval into a DAG (optional; separate product) | concurrent |

Not offered yet as annual subscription. Revisit after 2–3 completed pilots.

---

## You provide

- 200–5,000 document chunks (JSONL or export we convert)
- 20–50 labeled queries (or we draft labels from tickets with your review)
- Optional: known cause→effect / policy links
- Staging DSN or acceptance that in-memory / Compose pgvector is enough for the pilot

## We provide

- Written audit (`CORPUS_RECOVERY_AUDIT.md` + JSON)
- Explicit list of **silent dense fails** (query, wrong top-5, gold)
- Honest transfer note if finance demo checkpoint is used OOD
- Commercial-use Apache-2.0 software; no lock-in

---

## What success looks like

| Signal | Target for pilot |
|--------|------------------|
| Dense silent fails identified | ≥5 concrete queries (or proof none exist) |
| Multi recovery on those fails | Measured % — not promised 100% a priori |
| Partner quote | Written permission to anonymize 1 scorecard for case study |

---

## Out of scope (unless scoped in writing)

- Production SLA / 24×7 support
- Replacing your vector DB
- Guaranteeing exam / clinical outcomes
- Building PrismManifest money-gate (sold separately)

---

## Close

Reply **RECOVER** with vertical (SRE / finance / healthcare) and rough corpus size.
