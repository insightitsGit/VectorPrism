"""Shim: prefer `from vectorprism.ingestion_adapter import ...`."""
from vectorprism import ingestion_adapter as _mod
from vectorprism.ingestion_adapter import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
