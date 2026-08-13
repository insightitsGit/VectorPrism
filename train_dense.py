"""Shim: prefer `from vectorprism.train_dense import ...`."""
from vectorprism import train_dense as _mod
from vectorprism.train_dense import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
