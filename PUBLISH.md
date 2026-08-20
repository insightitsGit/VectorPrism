# Publish VectorPrism to PyPI

Published: build **0.1.3** (bitemporal Stage-1). Upload to make [vectorprism 0.1.3](https://pypi.org/project/vectorprism/0.1.3/) live.

## 1. One-time: create a PyPI API token

1. https://pypi.org/manage/account/token/
2. Scope: project `vectorprism` (or entire account)
3. Copy token (starts with `pypi-`)

Never commit the token. Prefer env vars.

## 2. Build + upload (PowerShell)

```powershell
cd c:\code\VectorPrism

# Optional: TestPyPI first
# $env:TWINE_REPOSITORY = "testpypi"

$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-YOUR_TOKEN_HERE"   # paste your token

# Or: $env:TWINE_PASSWORD = $env:PYPI_TOKEN

powershell -ExecutionPolicy Bypass -File scripts\publish_pypi.ps1
```

## 3. Build + upload (bash)

```bash
cd /path/to/VectorPrism
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TOKEN_HERE
# export TWINE_REPOSITORY=testpypi   # optional dry-run registry
bash scripts/publish_pypi.sh
```

## 4. After publish — verify

```bash
pip install vectorprism==0.1.3
vectorprism version
vectorprism pilot-check
```

## 5. Git tag (recommended)

```bash
git tag v0.1.3
git push origin main --tags
```

## Notes

- Wheel includes library code + `schema.sql` + `data/*.example.jsonl`. Full adversarial packs stay in **git**, not PyPI.
- Torch is a hard dependency; partners with CUDA should install torch from pytorch.org first if needed.
- To yank a bad release: PyPI project → Releases → Yank.
