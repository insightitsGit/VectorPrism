"""Shim: prefer `from vectorprism.reingest import ...`."""
from vectorprism import reingest as _mod
from vectorprism.reingest import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
