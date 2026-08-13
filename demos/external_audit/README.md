# External corpus audit

Honest generalization tests — **not** the finance `hard_adversarial` pack.

## Layout

| Path | Role |
|------|------|
| `raw_incident_bench/` | Downloaded public HF `trentdoney/synthetic-incident-search-benchmark` |
| `packs/incident_bench/` | Converted VectorPrism JSONL |
| `results/incident_bench/` | Audit JSON + Markdown |
| `../../scripts/import_external_pack.py` | Converter |
| `../../scripts/corpus_recovery_audit.py` | Dense vs multi scorecard for any pack |

## Reproduce

```bash
# 1) Convert public incident benchmark → VectorPrism JSONL
python scripts/import_external_pack.py \
  --preset incident_bench \
  --raw-dir demos/external_audit/raw_incident_bench \
  --out demos/external_audit/packs/incident_bench

# 2) Audit (Docker recommended for sentence-transformers)
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

## Why this pack

Public-safe synthetic SRE incidents with labeled queries and hard negatives.  
**Not authored by VectorPrism to force dense failure.** If dense already works well here, the report must say so.
