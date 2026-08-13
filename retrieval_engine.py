"""Shim: prefer `from vectorprism.retrieval_engine import ...`."""
from vectorprism import retrieval_engine as _mod
from vectorprism.retrieval_engine import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
