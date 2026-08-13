# Public claims audit (2026-08-13)

Prerequisite for talking to design partners / anyone with budget.

| Claim (before) | Reality | Fix |
|----------------|---------|-----|
| Badge `tests-36 passing` | Suite grew (HO-005 + skips); count drifts | Live CI status badge |
| Discord `discord.gg/vectorprism` | No verified public invite | Point to GitHub Discussions |
| Benchmarks table as general truth | Measured on **author-calibrated** `hard_adversarial` pack | README scope honesty + external audit path |
| PyPI unpublished / 404 language | **0.1.2 live** (README/claims sync) | PRODUCTION/PILOT/PUBLISH aligned |
| Wheel library-only | Wheel ships `schema.sql` + example JSONL | Docs updated in 0.1.1 |
| `BENCHMARKS.md --skip-train` on fresh clone | Breaks: `*.pt` gitignored | Documented + `scripts/reproduce_adversarial_benchmarks.py` |

## Verified live

- PyPI: https://pypi.org/project/vectorprism/ (target **0.1.2** — republish after this tag)
- GitHub public + Discussions enabled
- GitHub Pages demo: https://insightitsgit.github.io/VectorPrism/
- Release tag: `v0.1.2`

## Still do not claim

- That dense Miss@10 ≈ 93% on **arbitrary** corpora
- That multi-channel recovery is 13/13 without partner graphs / labels
- A Discord community that does not exist
