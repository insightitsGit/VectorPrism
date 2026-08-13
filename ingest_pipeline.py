"""Shim: prefer `from vectorprism.ingest_pipeline import ...`."""
from vectorprism import ingest_pipeline as _mod
from vectorprism.ingest_pipeline import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
