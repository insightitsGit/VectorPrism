# External corpus audit

Honest generalization tests — **not** the finance `hard_adversarial` pack.

## Case studies (committed results)

| Pack | Source | Dense Miss@10 | Dense R@10 | Multi R@10 | Report |
|------|--------|---------------|------------|------------|--------|
| `incident_bench` | HF `trentdoney/synthetic-incident-search-benchmark` | 0 | 1.000 | ~1.0 | [`results/incident_bench/`](results/incident_bench/) |
| `scifact_subset` | BeIR/SciFact (500 docs / 40 queries) | **0** | **1.000** | **0.950** | [`results/scifact_subset/`](results/scifact_subset/) |

**Takeaway:** On these public packs, dense already recovers gold in top-10. Finance adversarial **13/13** does **not** transfer. The audit tool’s job is to say that honestly and scope a paid pilot where structure indexes can matter.

## Layout

| Path | Role |
|------|------|
| `raw_incident_bench/` | Downloaded public HF incident benchmark |
| `packs/incident_bench/` | Converted VectorPrism JSONL |
| `packs/scifact_subset/` | BeIR SciFact subset JSONL |
| `results/*/` | Audit JSON + Markdown |
| `../../scripts/import_external_pack.py` | Converter |
| `../../scripts/corpus_recovery_audit.py` | Dense vs multi scorecard for any pack |

## Reproduce SciFact (first non-finance case study)

```bash
docker compose run --rm --no-deps vectorprism \
  python scripts/corpus_recovery_audit.py \
  --documents demos/external_audit/packs/scifact_subset/documents.jsonl \
  --eval demos/external_audit/packs/scifact_subset/eval.jsonl \
  --checkpoint checkpoints/finance_hard_adversarial_multi.pt \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --vertical generic \
  --pack-meta demos/external_audit/packs/scifact_subset/meta.json \
  --out demos/external_audit/results/scifact_subset
```

## Reproduce incident pack

```bash
python scripts/import_external_pack.py \
  --preset incident_bench \
  --raw-dir demos/external_audit/raw_incident_bench \
  --out demos/external_audit/packs/incident_bench

docker compose run --rm --no-deps vectorprism \
  python scripts/corpus_recovery_audit.py \
  --documents demos/external_audit/packs/incident_bench/documents.jsonl \
  --eval demos/external_audit/packs/incident_bench/eval.jsonl \
  --checkpoint checkpoints/finance_hard_adversarial_multi.pt \
  --encoder sentence-transformers/all-mpnet-base-v2 \
  --vertical sre_incident \
  --pack-meta demos/external_audit/packs/incident_bench/meta.json \
  --out demos/external_audit/results/incident_bench
```

## Why these packs

Public-safe labeled retrieval sets **not authored by VectorPrism to force dense failure**.  
If dense already works, the report must say so — that is the credibility product.
