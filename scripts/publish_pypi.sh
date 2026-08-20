#!/usr/bin/env bash
# Publish vectorprism to PyPI
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-...
#   bash scripts/publish_pypi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"
if [[ -z "${TWINE_PASSWORD:-}" && -n "${PYPI_TOKEN:-}" ]]; then
  export TWINE_PASSWORD="$PYPI_TOKEN"
fi
if [[ -z "${TWINE_PASSWORD:-}" ]]; then
  echo "Set TWINE_PASSWORD or PYPI_TOKEN to your PyPI API token (pypi-...)." >&2
  exit 1
fi

python -m pip install -U pip build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*

if [[ "${TWINE_REPOSITORY:-pypi}" == "testpypi" ]]; then
  python -m twine upload --repository testpypi dist/*
else
  python -m twine upload dist/*
fi

echo "Done. Install with: pip install vectorprism==0.1.3"
