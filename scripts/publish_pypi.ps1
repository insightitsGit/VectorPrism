# Publish vectorprism to PyPI (PowerShell)
# Usage:
#   $env:TWINE_USERNAME = "__token__"
#   $env:TWINE_PASSWORD = "pypi-..."   # or set PYPI_TOKEN
#   powershell -ExecutionPolicy Bypass -File scripts\publish_pypi.ps1
#
# TestPyPI:
#   $env:TWINE_REPOSITORY = "testpypi"
#   ... then same script

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $env:TWINE_PASSWORD) {
  if ($env:PYPI_TOKEN) {
    $env:TWINE_PASSWORD = $env:PYPI_TOKEN
  }
}
if (-not $env:TWINE_USERNAME) {
  $env:TWINE_USERNAME = "__token__"
}
if (-not $env:TWINE_PASSWORD) {
  Write-Error "Set TWINE_PASSWORD or PYPI_TOKEN to your PyPI API token (pypi-...)."
}

Write-Host "==> Installing build tooling" -ForegroundColor Cyan
python -m pip install -U pip build twine

Write-Host "==> Cleaning dist/" -ForegroundColor Cyan
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }
Get-ChildItem -Directory -Filter "*.egg-info" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "==> Building sdist + wheel" -ForegroundColor Cyan
python -m build

Write-Host "==> Twine check" -ForegroundColor Cyan
python -m twine check dist/*

$repo = if ($env:TWINE_REPOSITORY) { $env:TWINE_REPOSITORY } else { "pypi" }
Write-Host "==> Uploading to $repo" -ForegroundColor Cyan
if ($repo -eq "testpypi") {
  python -m twine upload --repository testpypi dist/*
} else {
  python -m twine upload dist/*
}

Write-Host "==> Done. Install with: pip install vectorprism==0.1.3" -ForegroundColor Green
