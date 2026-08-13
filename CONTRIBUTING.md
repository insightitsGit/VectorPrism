# Contributing

Thanks for your interest in VectorPrism.

## Ground rules

1. **Never push directly to `main`** from a fork — open a Pull Request.
2. **`main` is protected:** force-push and branch deletion are disabled; CI (`test`) must pass; stale review dismissals apply; conversations must be resolved.
3. **Do not commit secrets** (`.env`, PyPI tokens, API keys, `.pt` credentials). Use `.env.example` only.
4. **Security bugs:** use [GitHub private vulnerability reporting](https://github.com/insightitsGit/VectorPrism/security/advisories/new) — not public issues. See [`SECURITY.md`](SECURITY.md).

## Discussions

Product questions, pilot stories, and design ideas belong in
[Discussions](https://github.com/insightitsGit/VectorPrism/discussions) — not Issues.

Use **Issues** for reproducible bugs and concrete feature requests.

## Dev setup

```bash
git clone https://github.com/insightitsGit/VectorPrism.git
cd VectorPrism
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
vectorprism pilot-check
pytest test_psm.py test_phases.py -q
```

## PR checklist

- [ ] `pytest test_psm.py test_phases.py` passes
- [ ] No large binary artifacts / checkpoints
- [ ] Docs updated if behavior changes (`BENCHMARKS.md` / `README.md` when metrics change)
- [ ] Linked Discussion or Issue where relevant
